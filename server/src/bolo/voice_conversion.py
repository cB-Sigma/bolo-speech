"""Voice conversion: a recording of speech in, the same words out in another voice.

The target voice is, in order of precedence: a recording sent with the request (``reference``), a saved voice of
the library (its consented reference recording), one of bolo's reference clips (``en-short``, ``en-long``,
``hi-short``), or, for a model with voices of its own (RVC), one of those. The models run on audio.cpp
(:class:`bolo.engines.tts_server.ServerVC`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.encode import encode_audio
from bolo.engines.base import FloatArray
from bolo.errors import InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo
    from bolo.obs.trace import Trace

#: the longest source recording one request converts
MAX_SOURCE_S = 300.0


@dataclass
class Conversion:
    samples: FloatArray
    sample_rate: int
    model: str
    #: what the target was: {"saved": id} | {"clip": id} | {"voice": id} | {"reference": "upload"}
    target: dict[str, str]


def _target(core: Bolo, voice: str | None, reference: bytes | None) -> tuple[bytes | None, str | None, dict[str, str]]:
    """(recording, model voice, description) for the request's target."""
    from bolo import samples
    from bolo.core import reference_wav

    if reference is not None:
        return reference_wav(reference), None, {"reference": "upload"}
    if not voice:
        return None, None, {}
    saved = core.voices.find(voice)
    if saved is not None:
        return reference_wav(saved.reference_bytes()), None, {"saved": saved.id}
    clips = {str(e["id"]): e for e in samples.index() if e.get("kind") == "audio"}
    found = clips.get(voice)
    if found is not None:
        return reference_wav(samples.read_file(str(found["file"]))), None, {"clip": voice}
    return None, voice, {"voice": voice}


async def convert_voice(
    core: Bolo,
    samples: FloatArray,
    rate: int,
    *,
    model: str | None = None,
    voice: str | None = None,
    reference: bytes | None = None,
    options: dict[str, Any] | None = None,
    trace: Trace,
    watermark: Any = None,
) -> Conversion:
    """Convert ``samples`` (mono float32 at ``rate``) to the target voice; see the module docstring."""
    from bolo.engines.base import VoiceConversionModel

    if len(samples) < rate * 0.3:
        raise InvalidRequest("the recording is shorter than 0.3 s", param="file")
    if len(samples) > rate * MAX_SOURCE_S:
        raise InvalidRequest(f"the recording is longer than {MAX_SOURCE_S:.0f} s", param="file")
    spec = core.resolve(model, "voice-conversion")
    trace.model = spec.id
    trace.audio_seconds = len(samples) / rate
    target_audio, model_voice, described = _target(core, voice, reference)
    source = encode_audio(np.asarray(samples, dtype=np.float32), rate, "wav")
    core.require_components(spec)
    async with core.manager.use(spec.id) as entry:
        vc = entry.model
        if not isinstance(vc, VoiceConversionModel):
            raise InvalidRequest(f"model '{spec.id}' does not convert voices", param="model")
        if vc.needs_reference and target_audio is None:
            raise InvalidRequest(
                f"model '{spec.id}' imitates a voice from a recording: send reference_audio, or name a saved voice "
                "or a reference clip (en-short, en-long, hi-short) as voice"
                + (f"; {voice!r} is none of these" if voice else ""),
                param="voice",
            )
        if not vc.needs_reference and target_audio is not None:
            names = ", ".join(v.id for v in vc.voices)
            raise InvalidRequest(
                f"model '{spec.id}' converts to its own voices ({names}), not to a recording", param="voice"
            )
        with trace.phase("convert"):
            out, _info = await entry.queue.submit("convert", (source, target_audio, model_voice, dict(options or {})))
        out_rate = vc.sample_rate
    trace.extra["voice_conversion"] = {"model": spec.id, **described}
    samples_out = np.asarray(out, dtype=np.float32)
    # converted speech imitates a voice: under the "cloned" policy it is marked like a cloned voice
    from bolo.watermark import choose, marker

    async with marker(core, choose(core, watermark, cloned=True)) as mark:
        if mark is not None:
            samples_out = await mark.mark(samples_out, out_rate)
            trace.extra["watermark"] = mark.record()
    return Conversion(samples_out, out_rate, spec.id, described)
