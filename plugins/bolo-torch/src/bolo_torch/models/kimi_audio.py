"""Moonshot Kimi-Audio 7B Instruct as a turn-based speech-to-speech model (text and speech streamed).

The authors' code (github.com/MoonshotAI/Kimi-Audio at the pinned commit, ``kimia_infer``) builds everything:
``KimiAudio`` loads the language model (the model repository's remote code, opt in with ``allow_remote_code``), the
prompt manager (GLM-4-Voice speech tokens and Whisper large-v3 features of each user turn, Kimi's two parallel
token streams) and the streaming flow-matching detokenizer with its BigVGAN vocoder (24 kHz). The plugin then runs
``KimiAudio._generate_loop`` step by step instead of all at once, so the answer streams:

* each step samples a text token and a speech token (``KimiASampler``, the sampling of the authors' ``infer.py``);
  the speech stream starts ``kimia_mimo_audiodelaytokens`` steps after the text stream;
* text goes out as it is generated, until the text stream's end token;
* speech tokens (from ``kimia_token_offset`` up) go to ``detokenize_streaming`` 30 at a time, as
  ``KimiAudio.detokenize_audio`` does (12.5 tokens a second, upsampled 4 times): the first audio after the delay and
  30 speech tokens (about 2.4 s of speech);
* the answer ends when the speech stream ends (``<|im_msg_end|>`` or ``<|im_media_end|>``), or at the step limit of
  ``KimiAudio.generate`` (1500 positions in all for speech output).

The authors' prompt manager loads the speech tokenizer by its Hugging Face name (``THUDM/glm-4-voice-tokenizer``,
whatever its newest revision is); the plugin hands it the pinned copy bolo downloaded with the model instead.
User audio goes to ``kimia_infer`` as temporary WAV files, as its API takes paths. Session instructions go first, as
a user message (the model has no system role). Kimi-Audio's code calls ``torch.cuda`` throughout and imports
flash-attn: it runs on an NVIDIA GPU only.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from bolo.engines.base import ConversationChunk, ConversationRequest, SpeechToSpeechModel

from bolo_torch import remote_code, upstream
from bolo_torch.models import peak_safe, to_voices
from bolo_torch.models.s2s_util import GrowingText, WavFiles, no_grad, to_numpy
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency, model_dir, require

log = logging.getLogger("bolo_torch.kimi_audio")

TOKENIZER_DIR = "glm-4-voice-tokenizer"
#: KimiAudio.generate: speech output stops at 120 s of 12.5 Hz positions, prompt included
MAX_POSITIONS = int(12.5 * 120)
CHUNK = 30
#: infer.py's sampling
SAMPLING = {
    "audio_temperature": 0.8, "audio_top_k": 10, "text_temperature": 0.0, "text_top_k": 5,
    "audio_repetition_penalty": 1.0, "audio_repetition_window_size": 64, "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}  # fmt: skip


def chats(request: ConversationRequest, wavs: WavFiles) -> list[dict[str, Any]]:
    """The conversation as ``KimiAPromptManager.get_prompt`` takes it."""
    out: list[dict[str, Any]] = []
    if request.instructions and request.instructions.strip():
        out.append({"role": "user", "message_type": "text", "content": request.instructions.strip()})
    for turn in request.turns:
        if turn.role == "assistant":
            if turn.text:
                out.append({"role": "assistant", "message_type": "text", "content": turn.text})
            continue
        if turn.role == "tool":
            out.append({"role": "user", "message_type": "text", "content": f"(result of a tool call: {turn.text})"})
            continue
        if turn.audio is not None and len(turn.audio):
            out.append({"role": "user", "message_type": "audio", "content": wavs.write(turn.audio)})
        if turn.text:
            out.append({"role": "user", "message_type": "text", "content": turn.text})
    return out


class KimiAudioS2S(SpeechToSpeechModel):
    sample_rate = 24000
    supports_tools = False

    def __init__(
        self, kimi: Any, sampler: Any, *, device: str = "cuda", sampling: dict[str, Any] | None = None
    ) -> None:
        """``kimi`` is a ``kimia_infer.api.kimia.KimiAudio``; ``sampler`` its ``KimiASampler`` class."""
        self.kimi = kimi
        self.sampler = sampler
        self.device = device
        self.sampling = dict(SAMPLING, **(sampling or {}))
        self.voices = to_voices([{"id": "default", "name": "Kimi"}])
        self._lock = threading.Lock()

    def voice(self, name: str | None) -> str:
        if name and name.lower() not in ("default", "kimi"):
            raise InvalidRequest(f"unknown voice {name!r}; Kimi-Audio has one voice (default)")
        return "default"

    def respond(
        self, request: ConversationRequest, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[ConversationChunk]:
        if cancelled is not None and cancelled():
            return
        if request.audio_out:
            self.voice(request.voice)
        if request.audio_out and self.kimi.detokenizer is None:
            raise InvalidRequest("this Kimi-Audio was loaded without its detokenizer (text only)")
        output = "both" if request.audio_out else "text"
        with self._lock, no_grad(), WavFiles() as wavs:
            history = self.kimi.prompt_manager.get_prompt(chats(request, wavs), output_type=output)
            yield from self._stream(history, output, request.max_output_tokens, cancelled)

    def _stream(
        self, history: Any, output: str, limit: int | None, cancelled: Callable[[], bool] | None
    ) -> Iterator[ConversationChunk]:
        """``KimiAudio.generate`` and ``_generate_loop``, one step at a time, with the answer handed out as it grows."""
        import torch

        kimi, extra, device = self.kimi, self.kimi.extra_tokens, self.device
        audio_ids, text_ids, continuous_mask, _, _ = history.to_tensor()
        features = [f.to(device) for f in history.continuous_feature]
        prompt = int(audio_ids.shape[1])
        steps = MAX_POSITIONS - prompt if output == "both" else 7500 - prompt
        if limit:
            steps = min(steps, int(limit))
        if steps <= 0:
            raise InvalidRequest("the conversation is too long for Kimi-Audio (120 s of positions in all)")
        delay, offset = int(kimi.kimia_text_audiodelaytokens), int(kimi.kimia_token_offset)
        eod = (extra.msg_end, extra.media_end)
        s = self.sampling
        sampler = self.sampler(
            audio_top_k=s["audio_top_k"], audio_temperature=s["audio_temperature"],
            audio_repetition_penalty=s["audio_repetition_penalty"],
            audio_repetition_window_size=s["audio_repetition_window_size"], text_top_k=s["text_top_k"],
            text_temperature=s["text_temperature"], text_repetition_penalty=s["text_repetition_penalty"],
            text_repetition_window_size=s["text_repetition_window_size"],
        )  # fmt: skip
        previous_audio = torch.zeros((4096,), dtype=torch.int, device=device)
        previous_text = torch.zeros((4096,), dtype=torch.int, device=device)
        audio_in, text_in = audio_ids.to(device), text_ids.to(device)
        positions = torch.arange(0, prompt, device=device).unsqueeze(0).long()
        whisper: list[Any] | None = features
        mask: Any = continuous_mask.to(device)
        past = None
        last = prompt - 1
        text_done = False
        text = GrowingText(lambda ids: str(kimi.prompt_manager.text_tokenizer.decode(ids)))
        speech: list[int] = []
        if output == "both":
            kimi.detokenizer.clear_states()
        first_chunk = True
        for i in range(steps):
            if cancelled is not None and cancelled():
                return
            audio_logits, text_logits, past = kimi.alm.forward(
                input_ids=audio_in, text_input_ids=text_in, whisper_input_feature=whisper,
                is_continuous_mask=mask, position_ids=positions, past_key_values=past, return_dict=False,
            )  # fmt: skip
            next_text = sampler.sample_text_logits(text_logits, recent_tokens=previous_text[:i] if i > 0 else None)
            next_audio = sampler.sample_audio_logits(audio_logits, recent_tokens=previous_audio[:i] if i > 0 else None)
            if text_done:
                next_text.fill_(extra.kimia_text_blank)
            elif next_text.item() == extra.kimia_text_eos:
                text_done = True
                rest = text.flush()
                if rest:
                    yield ConversationChunk(text=rest)
            else:
                token = int(next_text.item())
                if token < offset:
                    piece = text.add(token)
                    if piece:
                        yield ConversationChunk(text=piece)
            previous_text[i : i + 1] = next_text
            if i < delay or output == "text":
                next_audio.fill_(extra.kimia_text_blank)
            else:
                code = int(next_audio.item())
                if code >= offset:
                    speech.append(code - offset)
                if len(speech) > CHUNK:  # one token is always kept back, so the last chunk can say is_final
                    yield self._speech(speech[:CHUNK], final=False, first=first_chunk)
                    first_chunk = False
                    speech = speech[CHUNK:]
            previous_audio[i : i + 1] = next_audio
            if (output == "text" and text_done) or (output == "both" and next_audio.item() in eod):
                break
            audio_in, text_in = next_audio.unsqueeze(1), next_text.unsqueeze(1)
            last += 1
            positions = torch.zeros(1, 1, device=device).fill_(last).long().view(1, 1)
            whisper, mask = None, None
        if cancelled is not None and cancelled():
            return
        if not text_done:
            rest = text.flush()
            if rest:
                yield ConversationChunk(text=rest)
        if output == "both" and speech:
            yield self._speech(speech, final=True, first=first_chunk)

    def _speech(self, codes: list[int], *, final: bool, first: bool) -> ConversationChunk:
        import torch

        tokens = torch.tensor(codes, device=self.device).long().unsqueeze(0)
        wav = self.kimi.detokenizer.detokenize_streaming(tokens, is_final=final, upsample_factor=4)
        return ConversationChunk(audio=peak_safe(to_numpy(wav)))


def _pinned_tokenizer(prompt_manager: Any, folder: Path) -> Callable[[str], Any]:
    """``Glm4Tokenizer`` for ``KimiAPromptManager``, which asks for ``THUDM/glm-4-voice-tokenizer`` by name: the
    pinned copy bolo downloaded with the model, whatever name it is asked for."""
    original = getattr(prompt_manager, "_bolo_torch_glm4_tokenizer", prompt_manager.Glm4Tokenizer)
    prompt_manager._bolo_torch_glm4_tokenizer = original

    def make(_name: str) -> Any:
        return original(str(folder))

    return make


def load(ctx: LoadContext) -> KimiAudioS2S:
    if not ctx.device.startswith("cuda"):
        raise InvalidRequest(
            f"Kimi-Audio runs only on an NVIDIA GPU (its code calls torch.cuda throughout and imports flash-attn); this "
            f"load would run on {ctx.device}. Pin device = cuda for the model, or use another speech-to-speech model."
        )
    path = model_dir(ctx.path, ctx.options)
    remote_code.trust_remote_code(ctx, path)  # KimiAudio passes trust_remote_code=True itself; only after this
    tokenizer_dir = path / TOKENIZER_DIR
    if not (tokenizer_dir / "config.json").is_file():
        raise InvalidRequest(f"{tokenizer_dir} is missing: pull the model again (it brings the speech tokenizer)")
    require("transformers", "kimi-audio")
    if importlib.util.find_spec("flash_attn") is None:
        raise MissingDependency(
            "Kimi-Audio's Whisper encoder imports flash-attn, which is not installed in the plugin's environment. On "
            "Linux: pip install flash-attn --no-build-isolation (it builds against your CUDA toolkit). On Windows "
            "there is no official build: use a flash-attn wheel built for your Python, torch and CUDA versions, or "
            "run the plugin under WSL2."
        )
    upstream.ensure("kimi-audio")
    import torch

    if ":" in ctx.device:
        torch.cuda.set_device(int(ctx.device.split(":", 1)[1]))
    manager_mod = importlib.import_module("kimia_infer.api.prompt_manager")
    setattr(manager_mod, "Glm4Tokenizer", _pinned_tokenizer(manager_mod, tokenizer_dir))  # noqa: B010
    kimia = importlib.import_module("kimia_infer.api.kimia")
    sampler = importlib.import_module("kimia_infer.utils.sampler").KimiASampler
    kimi = kimia.KimiAudio(model_path=str(path), load_detokenizer=bool(ctx.option("detokenizer", True)))
    return KimiAudioS2S(kimi, sampler, device=ctx.device)
