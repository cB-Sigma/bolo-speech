"""Qwen2.5-Omni and Qwen3-Omni as turn-based speech-to-speech models (transformers' own classes, no remote code).

The model card's recipe: the conversation goes through the processor's chat template (each user turn's audio as an
``audio`` part, 16 kHz), ``generate`` runs the Thinker (text) and the Talker with its code-to-wave decoder (24 kHz
speech in one of the model's voices), and the new tokens are the answer's text. bolo sends the whole conversation
with every turn (the plugin protocol's ``respond``), so nothing is kept here between turns.

* Qwen2.5-Omni (``Qwen2_5OmniForConditionalGeneration``, transformers 4.52 and later): speech comes out only with
  the system prompt the model was trained with ("You are Qwen, a virtual human ..."), so the session's instructions
  are added after it. Voices Chelsie (default) and Ethan.
* Qwen3-Omni (``Qwen3OmniMoeForConditionalGeneration``, transformers 4.57 and later): any system prompt; voices
  Ethan (default), Chelsie and Aiden.

The answer comes back in one piece when generation ends (``generate`` gives the audio only at the end), so the time
to first audio is the time of the whole answer. Tools are not passed: the models were not trained for function
calling in this format. A cancel takes effect before generation starts or once it ends.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from bolo.engines.base import ConversationChunk, ConversationRequest, ConversationTurn, SpeechToSpeechModel

from bolo_torch.models import peak_safe, to_voices
from bolo_torch.runtime import InvalidRequest, LoadContext, MissingDependency, model_dir, require, split_kwargs

QWEN25_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and "
    "visual inputs, as well as generating text and speech."
)


@dataclass(frozen=True)
class Family:
    model_class: str
    processor_class: str
    transformers: str
    voices: tuple[str, ...]
    #: a system prompt the model needs for speech output (the session's instructions follow it)
    system: str | None = None
    #: generate() returns the Thinker's output as an object with ``sequences`` (Qwen3-Omni)
    returns_dict: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


FAMILIES: dict[str, Family] = {
    "qwen2.5-omni": Family(
        "Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor", "4.52", ("Chelsie", "Ethan"), QWEN25_SYSTEM
    ),
    "qwen3-omni": Family(
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeProcessor",
        "4.57",
        ("Ethan", "Chelsie", "Aiden"),
        returns_dict=True,
        extra={"thinker_return_dict_in_generate": True},
    ),
}


def conversation(family: Family, request: ConversationRequest) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """The chat-template messages of a request and the audio of its user turns, in order."""
    system = "\n\n".join(x for x in (family.system, (request.instructions or "").strip()) if x)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    audios: list[np.ndarray] = []
    for turn in request.turns:
        messages.append(_message(turn, audios))
    # the chat template wants turns to alternate; merge neighbours with the same role
    merged: list[dict[str, Any]] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append({"role": m["role"], "content": list(m["content"])})
    return merged, audios


def _message(turn: ConversationTurn, audios: list[np.ndarray]) -> dict[str, Any]:
    if turn.role == "assistant":
        return {"role": "assistant", "content": [{"type": "text", "text": turn.text}]}
    if turn.role == "tool":
        return {"role": "user", "content": [{"type": "text", "text": f"(result of a tool call: {turn.text})"}]}
    content: list[dict[str, Any]] = []
    if turn.audio is not None and len(turn.audio):
        audios.append(np.asarray(turn.audio, dtype=np.float32))
        content.append({"type": "audio", "audio": f"turn-{len(audios)}"})
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    return {"role": "user", "content": content}


class OmniS2S(SpeechToSpeechModel):
    sample_rate = 24000
    supports_tools = False
    #: how the weights were split across the GPU and the CPU (sent to bolo in the load reply), when known
    placement_report: dict[str, Any] | None = None

    def __init__(
        self, model: Any, processor: Any, family: Family, *, device: str = "cpu", max_new_tokens: int = 512
    ) -> None:
        self.model = model
        self.processor = processor
        self.family = family
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.voices = to_voices({"id": v, "name": v} for v in family.voices)
        self._lock = threading.Lock()

    def voice(self, name: str | None) -> str:
        if not name:
            return self.family.voices[0]
        for v in self.family.voices:
            if v.lower() == name.lower():
                return v
        raise InvalidRequest(f"unknown voice {name!r}; this model has {', '.join(self.family.voices)}")

    def respond(
        self, request: ConversationRequest, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[ConversationChunk]:
        if cancelled is not None and cancelled():
            return
        messages, audios = conversation(self.family, request)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=text, audio=audios or None, return_tensors="pt", padding=True, use_audio_in_video=False
        )
        inputs = inputs.to(self.device)
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None and "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype)
        limit = min(request.max_output_tokens or self.max_new_tokens, self.max_new_tokens)
        kwargs: dict[str, Any] = {
            **self.family.extra, "use_audio_in_video": False, "return_audio": request.audio_out,
            "thinker_max_new_tokens": limit,
        }  # fmt: skip
        if request.audio_out:
            kwargs["speaker"] = self.voice(request.voice)
        with self._lock, _no_grad():
            out = self.model.generate(**inputs, **kwargs)
        ids, audio = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        sequences: Any = getattr(ids, "sequences", ids) if self.family.returns_dict else ids
        prompt = int(inputs["input_ids"].shape[1])
        answer = self.processor.batch_decode(
            sequences[:, prompt:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if cancelled is not None and cancelled():
            return
        samples = None
        if request.audio_out and audio is not None:
            samples = peak_safe(_numpy(audio))
        yield ConversationChunk(text=answer, audio=samples if samples is not None and samples.size else None)


def _numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _no_grad() -> Any:
    try:
        import torch
    except ImportError:  # the fakes of the tests
        import contextlib

        return contextlib.nullcontext()
    return torch.inference_mode()


def load(ctx: LoadContext) -> OmniS2S:
    name = str(ctx.option("family", "qwen2.5-omni"))
    family = FAMILIES.get(name)
    if family is None:
        raise InvalidRequest(f"unknown omni family {name!r} ({', '.join(FAMILIES)})")
    transformers = require("transformers", "omni")
    model_cls = getattr(transformers, family.model_class, None)
    processor_cls = getattr(transformers, family.processor_class, None)
    if model_cls is None or processor_cls is None:
        raise MissingDependency(
            f"transformers {getattr(transformers, '__version__', '?')} has no {family.model_class}; "
            f'{name} needs transformers {family.transformers} or later: pip install "bolo-torch[omni]"'
        )
    path = model_dir(ctx.path, ctx.options)
    split = split_kwargs(ctx)
    model = model_cls.from_pretrained(str(path), torch_dtype=ctx.dtype, **split)
    if not split:
        model = model.to(ctx.device)
    model.eval()
    processor = processor_cls.from_pretrained(str(path))
    out = OmniS2S(model, processor, family, device=ctx.device, max_new_tokens=int(ctx.option("max_new_tokens", 512)))
    if split:
        from bolo_torch.runtime import device_map_report

        out.placement_report = device_map_report(model)
    return out
