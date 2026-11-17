"""OpenBMB MiniCPM-o 2.6 as a turn-based speech-to-speech model (its own code, as Hugging Face remote code).

MiniCPM-o's model exists only as Python in its Hugging Face repository (``modeling_minicpmo.py`` and six more
files), so it loads only after the operator opts in for this model (``allow_remote_code``; :mod:`bolo_torch.remote_code`
checks the pinned commit and every file's sha256 first). Then it follows the model card:

* ``AutoModel.from_pretrained(..., trust_remote_code=True, attn_implementation="sdpa", init_vision, init_audio,
  init_tts)``, ``init_tts()`` (ChatTTS tokenizer and Vocos from the repository's ``assets``, which bolo downloads,
  so nothing is fetched at load), ``tts.float()``;
* the "audio assistant" system prompt of ``get_sys_prompt`` with a voice's reference recording (the card's
  ``assistant_female_voice``, ``assistant_male_voice`` and the less stable ``assistant_default_female_voice``), the
  session's instructions after it;
* ``mode: stream`` (default): the card's streaming API. A new session per answer; the system prompt, then the
  conversation is prefilled (``streaming_prefill``; user speech in 1 s pieces, as the card does), and
  ``streaming_generate`` yields text with 24 kHz audio piece by piece (``_generate_mel_spec_audio_streaming``, 25
  speech-codec frames a piece). The model's earlier answers are prefilled as text in the chat-template markup the
  streaming API writes itself (``<|im_end|>``, ``<|im_start|>assistant``);
* ``mode: chat``, and every text-only answer: ``chat(msgs, use_tts_template=True, generate_audio=...)``, the
  answer in one piece when generation ends.

A cancel takes effect between pieces of the stream.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from bolo.engines.base import ConversationChunk, ConversationRequest, ConversationTurn, SpeechToSpeechModel

from bolo_torch import remote_code
from bolo_torch.models import peak_safe, to_voices
from bolo_torch.models.s2s_util import no_grad, system_text, to_numpy
from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require, split_kwargs

log = logging.getLogger("bolo_torch.minicpm_o")

VOICES = {
    "female": "assets/input_examples/assistant_female_voice.wav",
    "male": "assets/input_examples/assistant_male_voice.wav",
    "default_female": "assets/input_examples/assistant_default_female_voice.wav",
}
MARKS = ("<|tts_eos|>", "<|im_end|>", "<|endoftext|>", "<|tts_bos|>")
PIECE = 16000  # the card prefills speech one second at a time


def _clean(text: str) -> str:
    for mark in MARKS:
        text = text.replace(mark, "")
    return text


class MiniCPMoS2S(SpeechToSpeechModel):
    sample_rate = 24000
    supports_tools = False
    placement_report: dict[str, Any] | None = None

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        voices: dict[str, np.ndarray],
        *,
        mode: str = "stream",
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        language: str = "en",
    ) -> None:
        if mode not in ("stream", "chat"):
            raise InvalidRequest(f"unknown MiniCPM-o mode {mode!r} (stream or chat)")
        self.model, self.tokenizer = model, tokenizer
        self.references = voices
        self.voices = to_voices({"id": v, "name": v.replace("_", " ").capitalize(), "gender": v[0]} for v in voices)
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.language = language
        self._lock = threading.Lock()

    def voice(self, name: str | None) -> str:
        if not name:
            return next(iter(self.references))
        key = name.lower().replace(" ", "_")
        if key not in self.references:
            raise InvalidRequest(f"unknown voice {name!r}; this model has {', '.join(self.references)}")
        return key

    def system(self, request: ConversationRequest, voice: str) -> dict[str, Any]:
        msg = self.model.get_sys_prompt(
            ref_audio=self.references[voice], mode="audio_assistant", language=self.language
        )
        extra = system_text(request.instructions)
        if extra:
            msg = {"role": msg["role"], "content": [*msg["content"], extra]}
        return dict(msg)

    def respond(
        self, request: ConversationRequest, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[ConversationChunk]:
        if cancelled is not None and cancelled():
            return
        voice = self.voice(request.voice)
        limit = min(request.max_output_tokens or self.max_new_tokens, self.max_new_tokens)
        with self._lock, no_grad():
            if self.mode == "chat" or not request.audio_out:
                yield from self._chat(request, voice, limit, cancelled)
            else:
                yield from self._stream(request, voice, limit, cancelled)

    def _chat(
        self, request: ConversationRequest, voice: str, limit: int, cancelled: Callable[[], bool] | None
    ) -> Iterator[ConversationChunk]:
        msgs = [self.system(request, voice)]
        for turn in request.turns:
            msgs.append(_chat_message(turn))
        out = self.model.chat(
            msgs=msgs, tokenizer=self.tokenizer, sampling=True, max_new_tokens=limit, use_tts_template=True,
            generate_audio=request.audio_out, temperature=self.temperature,
        )  # fmt: skip
        if cancelled is not None and cancelled():
            return
        text = _clean(str(getattr(out, "text", out)))
        wav = getattr(out, "audio_wav", None)
        audio = peak_safe(to_numpy(wav)) if request.audio_out and wav is not None else None
        yield ConversationChunk(text=text, audio=audio if audio is not None and audio.size else None)

    def _stream(
        self, request: ConversationRequest, voice: str, limit: int, cancelled: Callable[[], bool] | None
    ) -> Iterator[ConversationChunk]:
        model, tok = self.model, self.tokenizer
        model.reset_session()
        session = uuid.uuid4().hex
        model.streaming_prefill(session_id=session, msgs=[self.system(request, voice)], tokenizer=tok)
        for msg in _stream_messages(request.turns):
            if cancelled is not None and cancelled():
                return
            model.streaming_prefill(session_id=session, msgs=[msg], tokenizer=tok)
        pieces = model.streaming_generate(
            session_id=session, tokenizer=tok, max_new_tokens=limit, temperature=self.temperature, generate_audio=True
        )
        for piece in pieces:
            if cancelled is not None and cancelled():
                return
            text = _clean(str(getattr(piece, "text", "") or ""))
            wav = getattr(piece, "audio_wav", None)
            audio = peak_safe(to_numpy(wav)) if wav is not None else None
            if text or (audio is not None and audio.size):
                yield ConversationChunk(text=text, audio=audio if audio is not None and audio.size else None)


def _chat_message(turn: ConversationTurn) -> dict[str, Any]:
    if turn.role == "assistant":
        return {"role": "assistant", "content": turn.text}
    if turn.role == "tool":
        return {"role": "user", "content": [f"(result of a tool call: {turn.text})"]}
    content: list[Any] = []
    if turn.audio is not None and len(turn.audio):
        content.append(np.asarray(turn.audio, dtype=np.float32))
    if turn.text:
        content.append(turn.text)
    return {"role": "user", "content": content}


def _stream_messages(turns: list[ConversationTurn]) -> Iterator[dict[str, Any]]:
    """The conversation as ``streaming_prefill`` messages (one each): user speech in 1 s pieces; an earlier answer as
    an ``assistant`` message carrying the markup that ``streaming_prefill`` does not add for that role (it opens
    each new user message itself)."""
    for turn in turns:
        if turn.role == "assistant":
            body = turn.text.strip()
            yield {"role": "assistant", "content": [f"<|im_end|>\n<|im_start|>assistant\n{body}<|im_end|>\n"]}
            continue
        if turn.role == "tool":
            yield {"role": "user", "content": [f"(result of a tool call: {turn.text})"]}
            continue
        if turn.audio is not None and len(turn.audio):
            audio = np.asarray(turn.audio, dtype=np.float32)
            for start in range(0, len(audio), PIECE):
                yield {"role": "user", "content": [audio[start : start + PIECE]]}
        if turn.text:
            yield {"role": "user", "content": [turn.text]}


def load(ctx: LoadContext) -> MiniCPMoS2S:
    path = model_dir(ctx.path, ctx.options)
    trust = remote_code.trust_remote_code(ctx, path)
    references = {}
    librosa = require("librosa", "minicpm-o")
    for name, rel in VOICES.items():
        if not (path / rel).is_file():
            raise InvalidRequest(f"{path / rel} is missing: pull the model again (its voices are in assets/)")
        references[name] = np.asarray(librosa.load(str(path / rel), sr=16000, mono=True)[0], dtype=np.float32)
    for rel in ("assets/Vocos.pt", "assets/chattts_tokenizer/tokenizer.json"):
        if not (path / rel).is_file():  # init_tts would download it from the Hub (unpinned) instead
            raise InvalidRequest(f"{path / rel} is missing: pull the model again")
    transformers = require("transformers", "minicpm-o")
    split = split_kwargs(ctx)
    model = transformers.AutoModel.from_pretrained(
        str(path), trust_remote_code=trust, attn_implementation=str(ctx.option("attn_implementation", "sdpa")),
        torch_dtype=ctx.dtype, init_vision=bool(ctx.option("init_vision", True)), init_audio=True, init_tts=True,
        **split,
    )  # fmt: skip
    if not split:
        model = model.to(ctx.device)
    model = model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust)
    model.init_tts()
    model.tts.float()
    out = MiniCPMoS2S(
        model, tokenizer, references, mode=str(ctx.option("mode", "stream")),
        max_new_tokens=int(ctx.option("max_new_tokens", 512)), temperature=float(ctx.option("temperature", 0.5)),
        language=str(ctx.option("prompt_language", "en")),
    )  # fmt: skip
    if split:
        from bolo_torch.runtime import device_map_report

        out.placement_report = device_map_report(model)
    return out
