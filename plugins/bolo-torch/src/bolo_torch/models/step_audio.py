"""StepFun Step-Audio 2 mini as a turn-based speech-to-speech model (text, speech and tool calls, streamed).

What runs, in the authors' recipe (github.com/stepfun-ai/Step-Audio2, ``stepaudio2.py``, ``token2wav.py`` and
``examples.py`` at the pinned commit):

* the model repository's own code (``modeling_step_audio_2.py``, Hugging Face remote code: opt in per model with
  ``allow_remote_code``) loaded by transformers' ``AutoModelForCausalLM``;
* the prompt of ``StepAudio2.apply_chat_template``: ``<|BOT|>role\\n ... <|EOT|>`` per message, each 25 s piece
  of a user turn's 16 kHz audio as a 128-bin log-mel spectrogram with ``<audio_patch>`` placeholders, tools as a
  ``tool_json_schemas`` message, tool results as an ``input`` message, and ``<tts_start>`` to ask for speech;
* new ids below 151688 are text, ids above 151695 are speech tokens (minus 151696; 6561 and up are
  padding), turned into 24 kHz audio by the repository's ``Token2wav.stream`` (CosyVoice 2 flow matching and
  HiFT) in pieces of 25 tokens plus its 3-token look-ahead, as ``examples-vllm-stream.py`` does;
* the voice is the timbre of a prompt recording: ``female`` (default) or ``male``, the repository's two.

Streaming: text goes out as the model writes it; the first audio after the first 28 speech tokens (about 1 s of
speech). The model's ``forward`` keeps no key-value cache (each new token reprocesses the whole conversation), so a
long conversation slows every token down. token2wav calls ``.cuda()``: without an NVIDIA GPU the model answers in
text only (a warning says so). A cancel stops generation at the next token.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from bolo.engines.base import (
    ConversationChunk,
    ConversationRequest,
    ConversationTurn,
    FloatArray,
    SpeechToSpeechModel,
)

from bolo_torch import remote_code, upstream
from bolo_torch.models import to_voices
from bolo_torch.models.s2s_util import (
    GrowingText,
    TokenQueue,
    ToolCallFilter,
    generate_in_thread,
    no_grad,
    stopping_criteria,
    system_text,
    tool_schemas,
)
from bolo_torch.runtime import InvalidRequest, LoadContext, model_dir, require, split_kwargs

log = logging.getLogger("bolo_torch.step_audio")

TEXT_LIMIT = 151688  # ids below are text
AUDIO_OFFSET = 151696  # ids from here are speech tokens
AUDIO_CODES = 6561  # speech tokens from here are padding
CHUNK = 25
PIECE_S = 25  # the audio encoder takes 25 s at a time
DEFAULT_SYSTEM = "You are a helpful assistant."
MARKS = ("<|EOT|>", "<|BOT|>", "<|endoftext|>", "<tts_end>", "<tts_start>")
VOICES = {"female": "assets/default_female.wav", "male": "assets/default_male.wav"}


def messages(request: ConversationRequest, tools: bool) -> list[dict[str, Any]]:
    """The conversation in the shape ``StepAudio2.apply_chat_template`` takes (audio items hold arrays)."""
    out: list[dict[str, Any]] = [
        {"role": "system", "content": system_text(request.instructions) or DEFAULT_SYSTEM},
    ]
    if tools and request.tools and request.tool_choice != "none":
        import json

        out.append(
            {"role": "tool_json_schemas", "content": json.dumps(tool_schemas(request.tools), ensure_ascii=False)}
        )
    for turn in request.turns:
        out.append(_message(turn))
    if request.audio_out:
        out.append({"role": "assistant", "content": "<tts_start>", "eot": False})
    else:
        out.append({"role": "assistant", "content": None})
    return out


def _message(turn: ConversationTurn) -> dict[str, Any]:
    if turn.role == "assistant":
        text = turn.text
        for call in turn.tool_calls:
            text += f"<tool_call>function\n{call.get('name', '')}\n{call.get('arguments', '{}')}</tool_call>"
        return {"role": "assistant", "content": text}
    if turn.role == "tool":
        return {"role": "input", "content": [{"type": "text", "text": turn.text}]}
    content: list[dict[str, Any]] = []
    if turn.audio is not None and len(turn.audio):
        content.append({"type": "audio", "audio": np.asarray(turn.audio, dtype=np.float32)})
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    return {"role": "human", "content": content}


def prompt_pieces(msgs: list[dict[str, Any]], utils: Any) -> tuple[list[Any], list[Any]]:
    """``StepAudio2.apply_chat_template`` with audio arrays instead of file paths: the prompt pieces (strings) and
    the mel spectrograms of the audio, in order."""
    import torch

    results: list[Any] = []
    mels: list[Any] = []
    for msg in msgs:
        role = "human" if msg["role"] == "user" else msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            results.append(f"<|BOT|>{role}\n{content}" + ("<|EOT|>" if msg.get("eot", True) else ""))
        elif isinstance(content, list):
            results.append(f"<|BOT|>{role}\n")
            for item in content:
                if item["type"] == "text":
                    results.append(item["text"])
                elif item["type"] == "audio":
                    audio = torch.from_numpy(np.asarray(item["audio"], dtype=np.float32))
                    for i in range(0, audio.shape[0], 16000 * PIECE_S):
                        mel = utils.log_mel_spectrogram(audio[i : i + 16000 * PIECE_S], n_mels=128, padding=479)
                        mels.append(mel)
                        results.append(
                            "<audio_start>" + "<audio_patch>" * utils.compute_token_num(mel.shape[1]) + "<audio_end>"
                        )
            if msg.get("eot", True):
                results.append("<|EOT|>")
        elif content is None:
            results.append(f"<|BOT|>{role}\n")
    return results, mels


class StepAudio2S2S(SpeechToSpeechModel):
    sample_rate = 24000
    supports_tools = True
    placement_report: dict[str, Any] | None = None

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        utils: Any,
        token2wav: Any,
        voices: dict[str, str],
        *,
        generation_config: Any,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        sampling: dict[str, Any] | None = None,
    ) -> None:
        self.llm, self.tokenizer, self.utils, self.token2wav = llm, tokenizer, utils, token2wav
        self.voice_files = voices
        self.voices = to_voices({"id": v, "name": v.capitalize(), "gender": v[0]} for v in voices)
        self.generation_config = generation_config
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sampling = dict(sampling or {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.05})
        self.eot = int(tokenizer.convert_tokens_to_ids("<|EOT|>"))
        self._lock = threading.Lock()

    def voice(self, name: str | None) -> str:
        if not name:
            return next(iter(self.voice_files))
        key = name.lower()
        if key not in self.voice_files:
            raise InvalidRequest(f"unknown voice {name!r}; this model has {', '.join(self.voice_files)}")
        return key

    def inputs(self, request: ConversationRequest) -> dict[str, Any]:
        import torch

        pieces, mels = prompt_pieces(messages(request, self.supports_tools), self.utils)
        ids = []
        for piece in pieces:
            ids.append(self.tokenizer(text=piece, return_tensors="pt", padding=True)["input_ids"])
        prompt = torch.cat(ids, dim=-1).to(self.device)
        out: dict[str, Any] = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt), "wavs": None,
                               "wav_lens": None}  # fmt: skip
        if mels:
            wavs, lens = self.utils.padding_mels(mels)
            out["wavs"], out["wav_lens"] = wavs.to(self.device), lens.to(self.device)
        return out

    def respond(
        self, request: ConversationRequest, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[ConversationChunk]:
        if cancelled is not None and cancelled():
            return
        speak = request.audio_out and self.token2wav is not None
        voice = self.voice(request.voice) if request.audio_out else None
        limit = min(request.max_output_tokens or self.max_new_tokens, self.max_new_tokens)
        with self._lock, no_grad():
            inputs = self.inputs(request)
            config = self.generation_config(
                max_new_tokens=limit, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.eot, do_sample=True,
                **self.sampling,
            )  # fmt: skip
            stream = TokenQueue(skip_prompt=True)
            stop = threading.Event()  # set when the caller stops reading (a closed generator)

            def halted() -> bool:
                return stop.is_set() or (cancelled is not None and cancelled())

            kwargs = {**inputs, "generation_config": config, "stopping_criteria": stopping_criteria(halted)}
            thread = generate_in_thread(self.llm.generate, kwargs, stream)
            text = GrowingText(self._decode)
            tools = ToolCallFilter()
            speech: list[int] = []
            prompt_wav = self.voice_files[voice] if voice else ""
            lookahead = int(getattr(getattr(self.token2wav, "flow", None), "pre_lookahead_len", 3))
            if speak:
                self.token2wav.set_stream_cache(prompt_wav)
            try:
                for token in stream:
                    if cancelled is not None and cancelled():
                        break
                    if token < TEXT_LIMIT:
                        said, calls = tools.feed(text.add(token))
                        if said:
                            yield ConversationChunk(text=said)
                        for call in calls:
                            yield ConversationChunk(tool_call=call)
                    elif token >= AUDIO_OFFSET and speak:
                        code = token - AUDIO_OFFSET
                        if code < AUDIO_CODES:
                            speech.append(code)
                        if len(speech) >= CHUNK + lookahead:
                            pcm = self.token2wav.stream(speech[: CHUNK + lookahead], prompt_wav=prompt_wav)
                            speech = speech[CHUNK:]
                            yield ConversationChunk(audio=_pcm(pcm))
                if cancelled is not None and cancelled():
                    return
                said, calls = tools.feed(text.flush(), final=True)
                if said:
                    yield ConversationChunk(text=said)
                for call in calls:
                    yield ConversationChunk(tool_call=call)
                if speak and speech:
                    pcm = self.token2wav.stream(speech, prompt_wav=prompt_wav, last_chunk=True)
                    yield ConversationChunk(audio=_pcm(pcm))
            finally:
                stop.set()
                thread.join()

    def _decode(self, ids: list[int]) -> str:
        text = str(self.tokenizer.decode(ids))
        for mark in MARKS:
            text = text.replace(mark, "")
        return text


def _pcm(data: bytes) -> FloatArray:
    """token2wav's 16-bit little-endian PCM as float32."""
    out: FloatArray = np.frombuffer(data, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    return out


def _module(name: str, path: Path) -> Any:
    """A module of the bundle under a name of its own (``utils`` is too common a name to put on ``sys.path``)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(ctx: LoadContext) -> StepAudio2S2S:
    path = model_dir(ctx.path, ctx.options)
    trust = remote_code.trust_remote_code(ctx, path)
    remote_code.check_pinned_files(path, ctx.options, ["token2wav/flow.yaml"], ctx.model_id)
    transformers = require("transformers", "step-audio-2")
    code = upstream.ensure("step-audio-2")
    utils = _module("bolo_torch_step_audio2_utils", code / "utils.py")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust, padding_side="right")
    split = split_kwargs(ctx)
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=trust, torch_dtype=ctx.dtype, **split
    )
    if not split:
        llm = llm.to(ctx.device)
    llm.eval()
    eot = tokenizer.convert_tokens_to_ids("<|EOT|>")
    tokenizer.eos_token = "<|EOT|>"
    llm.config.eos_token_id = eot
    token2wav = None
    if ctx.device.startswith("cuda"):
        token2wav_mod = importlib.import_module("token2wav")
        token2wav = token2wav_mod.Token2wav(
            str(path / "token2wav"), float16=bool(ctx.option("token2wav_float16", False))
        )
    else:
        log.warning(
            "%s: Step-Audio 2's token2wav decoder runs only on an NVIDIA GPU (its code calls .cuda()); on %s the "
            "model answers in text only",
            ctx.model_id, ctx.device,
        )  # fmt: skip
    voices = {name: str(code / rel) for name, rel in VOICES.items()}
    model = StepAudio2S2S(
        llm, tokenizer, utils, token2wav, voices, generation_config=transformers.GenerationConfig, device=ctx.device,
        max_new_tokens=int(ctx.option("max_new_tokens", 2048)),
    )  # fmt: skip
    if split:
        from bolo_torch.runtime import device_map_report

        model.placement_report = device_map_report(llm)
    return model
