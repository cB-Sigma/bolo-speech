"""Orpheus-style text-to-speech: a Llama language model writes SNAC codec tokens (Veena, Svara).

Token layout (both models): audio tokens start at 128266; each 85 ms frame is 7 tokens, the i-th token of a frame
offset by ``i * 4096``; they map onto SNAC's three code levels as 1 + 2 + 4 codes. The prompts differ:

* Veena (Maya Research model card): ``[START_OF_HUMAN] "<spk_{voice}> {text}" [END_OF_HUMAN] [START_OF_AI]
  [START_OF_SPEECH]``, sampling with temperature 0.4, top-p 0.9, repetition penalty 1.05.
* Svara (Kenpath svara-tts-inference, tts_engine/encoder.py): ``[BOS] [START_OF_HUMAN] [AUDIO]
  "{speaker name}: {text}" [END_OF_HUMAN] [END_OF_TURN] [START_OF_AI] [START_OF_SPEECH]``, speaker names such
  as "Hindi (Male)", temperature 0.75, top-p 0.9, repetition penalty 1.1.

Audio is streamed while the model generates: every few frames the new frames are decoded with some frames of
left context, holding back one frame of look-ahead until the end.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from bolo.engines.base import FloatArray, TTSModel

from bolo_torch.models import to_voices, voices_from_options
from bolo_torch.runtime import (
    InvalidRequest,
    LoadContext,
    device_map_report,
    hub_snapshot,
    model_dir,
    require,
    split_kwargs,
)
from bolo_torch.text import split_sentences

BOS = 128000
END_OF_TURN = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
SVARA_AUDIO = 156939
AUDIO_OFFSET = 128266
CODEBOOK = 4096
FRAME = 7

SNAC_REPO = "hubertsiuzdak/snac_24khz"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"

FORMATS: dict[str, dict[str, Any]] = {
    "veena": {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.05, "eos": [END_OF_SPEECH, END_OF_AI]},
    "svara": {"temperature": 0.75, "top_p": 0.9, "repetition_penalty": 1.1, "eos": [END_OF_SPEECH]},
}


def build_prompt(fmt: str, tokenizer: Any, text: str, speaker: str) -> list[int]:
    if fmt == "veena":
        ids = tokenizer.encode(f"<spk_{speaker}> {text}", add_special_tokens=False)
        return [START_OF_HUMAN, *ids, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    if fmt == "svara":
        ids = tokenizer.encode(f"{speaker}: {text}", add_special_tokens=False)
        return [BOS, START_OF_HUMAN, SVARA_AUDIO, *ids, END_OF_HUMAN, END_OF_TURN, START_OF_AI, START_OF_SPEECH]
    raise InvalidRequest(f"unknown prompt_format {fmt!r} (veena or svara)")


def is_audio_token(token: int) -> bool:
    return AUDIO_OFFSET <= token < AUDIO_OFFSET + FRAME * CODEBOOK


def frames_from_tokens(tokens: list[int]) -> list[list[int]]:
    """Complete 7-code frames (codes 0..4095) from the audio tokens of a generation, in order."""
    codes = [t - AUDIO_OFFSET - (i % FRAME) * CODEBOOK for i, t in enumerate(t for t in tokens if is_audio_token(t))]
    frames = [codes[i : i + FRAME] for i in range(0, len(codes) - len(codes) % FRAME, FRAME)]
    return [f for f in frames if all(0 <= c < CODEBOOK for c in f)]


def snac_levels(frames: list[list[int]]) -> list[list[int]]:
    """De-interleave frames into SNAC's three levels (1, 2 and 4 codes per frame)."""
    levels: list[list[int]] = [[], [], []]
    for f in frames:
        levels[0].append(f[0])
        levels[1].extend((f[1], f[4]))
        levels[2].extend((f[2], f[3], f[5], f[6]))
    return levels


class _TokenQueue:
    """A transformers streamer: receives generated token ids (the prompt first, which is skipped)."""

    def __init__(self) -> None:
        self.items: queue.Queue[int | None] = queue.Queue()
        self._prompt_seen = False

    def put(self, value: Any) -> None:
        ids = value.reshape(-1).tolist() if hasattr(value, "reshape") else list(value)
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        for token in ids:
            self.items.put(int(token))

    def end(self) -> None:
        self.items.put(None)

    def __iter__(self) -> Iterator[int]:
        while True:
            token = self.items.get()
            if token is None:
                return
            yield token


class OrpheusSNAC(TTSModel):
    #: how the weights were split across the GPU and the CPU (sent to bolo in the load reply), when known
    placement_report: dict[str, Any] | None = None

    def __init__(
        self,
        generate: Callable[[list[int], Any, Callable[[], bool], int], None],
        tokenizer: Any,
        decode: Callable[[list[list[int]]], FloatArray],
        *,
        prompt_format: str,
        voices: list[dict[str, Any]],
        sample_rate: int = 24000,
        max_chars: int = 200,
        max_new_tokens: int = 2048,
        stream_frames: int = 4,
        context_frames: int = 3,
    ) -> None:
        if prompt_format not in FORMATS:
            raise InvalidRequest(f"unknown prompt_format {prompt_format!r} (veena or svara)")
        self.generate = generate
        self.tokenizer = tokenizer
        self.decode = decode
        self.prompt_format = prompt_format
        self._voices = {str(v["id"]).lower(): v for v in voices}
        self.voices = to_voices(voices)
        self.sample_rate = sample_rate
        self.max_chars = max_chars
        self.max_new_tokens = max_new_tokens
        self.stream_frames = max(1, stream_frames)
        self.context_frames = max(0, context_frames)
        self._lock = threading.Lock()

    def speaker(self, voice: str | None) -> str:
        if not self._voices:
            if not voice:
                raise InvalidRequest("this model needs a voice")
            return voice
        entry = self._voices.get((voice or "").strip().lower()) if voice else next(iter(self._voices.values()))
        if entry is None:
            raise InvalidRequest(f"unknown voice {voice!r}")
        # Veena's prompt names the voice by id (<spk_kavya>), Svara's by its display name ("Hindi (Male)")
        return str(entry["id"]) if self.prompt_format == "veena" else str(entry.get("name") or entry["id"])

    def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[FloatArray]:
        speaker = self.speaker(voice)
        stop = cancelled or (lambda: False)
        for sentence in split_sentences(text, self.max_chars):
            if stop():
                return
            prompt = build_prompt(self.prompt_format, self.tokenizer, sentence, speaker)
            budget = min(self.max_new_tokens, int(len(sentence) * 1.3) * FRAME + 3 * FRAME)
            with self._lock:
                yield from self._stream(prompt, stop, budget)

    def _stream(self, prompt: list[int], stop: Callable[[], bool], budget: int) -> Iterator[FloatArray]:
        tokens = _TokenQueue()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                self.generate(prompt, tokens, stop, budget)
            except BaseException as exc:
                errors.append(exc)
            finally:
                tokens.end()

        worker = threading.Thread(target=run, daemon=True, name="orpheus-generate")
        worker.start()
        audio_tokens: list[int] = []
        emitted = 0  # frames already sent
        for token in tokens:
            if is_audio_token(token):
                audio_tokens.append(token)
                ready = len(audio_tokens) // FRAME - 1  # keep one frame of look-ahead
                if ready - emitted >= self.stream_frames:
                    chunk = self._decode_range(frames_from_tokens(audio_tokens), emitted, ready)
                    emitted = ready
                    if chunk.size:
                        yield chunk
            if stop():
                break
        worker.join()
        if errors:
            raise errors[0]
        frames = frames_from_tokens(audio_tokens)
        if len(frames) > emitted and not stop():
            chunk = self._decode_range(frames, emitted, len(frames))
            if chunk.size:
                yield chunk

    def _decode_range(self, frames: list[list[int]], start: int, end: int) -> FloatArray:
        """Audio of ``frames[start:end]``, decoded with up to ``context_frames`` frames of left context."""
        end = min(end, len(frames))
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        left = max(0, start - self.context_frames)
        window = frames[left:end]
        audio = np.asarray(self.decode(snac_levels(window)), dtype=np.float32).reshape(-1)
        per_frame = len(audio) // len(window)
        return np.clip(audio[(start - left) * per_frame :], -1.0, 1.0)


def load(ctx: LoadContext) -> OrpheusSNAC:
    import torch

    transformers = require("transformers", "orpheus")
    snac = require("snac", "orpheus")
    root = model_dir(ctx.path, ctx.options)
    fmt = str(ctx.option("prompt_format") or ("svara" if "svara" in (ctx.model_id + str(root)).lower() else "veena"))
    params = FORMATS.get(fmt)
    if params is None:
        raise InvalidRequest(f"unknown prompt_format {fmt!r} (veena or svara)")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root))
    split = split_kwargs(ctx)  # bolo found too little GPU memory for all of it: layers on the GPU while they fit
    model = transformers.AutoModelForCausalLM.from_pretrained(str(root), torch_dtype=ctx.dtype, **split)
    model = (model if split else model.to(ctx.device)).eval()
    report = device_map_report(model)
    codec_dir = hub_snapshot(str(ctx.option("codec_repo", SNAC_REPO)), str(ctx.option("codec_revision", SNAC_REVISION)))
    codec = snac.SNAC.from_config(str(codec_dir / "config.json"))
    codec.load_state_dict(torch.load(str(codec_dir / "pytorch_model.bin"), map_location="cpu", weights_only=True))
    codec = codec.to(ctx.device).eval()

    class Stop(transformers.StoppingCriteria):  # type: ignore[misc,name-defined]
        def __init__(self, cancelled: Callable[[], bool]) -> None:
            self.cancelled = cancelled

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return bool(self.cancelled())

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params["eos"][0]

    def generate(prompt: list[int], streamer: Any, cancelled: Callable[[], bool], budget: int) -> None:
        ids = torch.tensor([prompt], device=ctx.device)
        with torch.inference_mode():
            model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=budget, do_sample=True,
                temperature=float(ctx.option("temperature", params["temperature"])),
                top_p=float(ctx.option("top_p", params["top_p"])),
                repetition_penalty=float(ctx.option("repetition_penalty", params["repetition_penalty"])),
                eos_token_id=params["eos"], pad_token_id=pad, streamer=streamer,
                stopping_criteria=transformers.StoppingCriteriaList([Stop(cancelled)]),
            )  # fmt: skip

    def decode(levels: list[list[int]]) -> FloatArray:
        codes = [torch.tensor(level, dtype=torch.int32, device=ctx.device).unsqueeze(0) for level in levels]
        with torch.inference_mode():
            audio = codec.decode(codes)
        return audio.reshape(-1).to(torch.float32).cpu().numpy()  # type: ignore[no-any-return]

    tts = OrpheusSNAC(
        generate, tokenizer, decode, prompt_format=fmt, voices=voices_from_options(ctx.options),
        max_chars=int(ctx.option("max_chars", 200)), max_new_tokens=int(ctx.option("max_new_tokens", 2048)),
        stream_frames=int(ctx.option("stream_frames", 4)), context_frames=int(ctx.option("context_frames", 3)),
    )  # fmt: skip
    tts.placement_report = report  # how the weights were split (info.placement in the load reply)
    return tts
