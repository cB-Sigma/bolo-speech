"""Speaker similarity: how much a synthesised voice sounds like its reference.

The score is the cosine similarity of two speaker embeddings, computed by the speaker-embedding model of a
diarization pipeline (setting ``voice_similarity_model``; default: the default diarization model, whose ERes2Net
embedder from 3D-Speaker was trained on VoxCeleb). It reuses what diarization uses to name known speakers.

What the numbers mean was measured, not assumed (docs/measurements/speaker-similarity-calibration.json): with
the ERes2Net embedder on LibriSpeech dev-clean (CC BY 4.0, real read speech, 16 speakers, 5 utterances each),
160 pairs of different utterances by the same person scored 0.45 to 0.93 (median 0.80) and 3000 pairs by
different people -0.30 to 0.53 (median 0.11, 95th percentile 0.36). At 0.50, 99.4 % of same-speaker pairs and
0.1 % of different-speaker pairs are at or above; at 0.30, 10 % of different-speaker pairs still are. The verdicts
below follow those numbers. They are a rule of thumb for "does the clone sound like the person", not proof of
identity; synthesised speech was not part of the calibration set, and another embedding model has another scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from bolo.audio.resample import resample
from bolo.engines.base import FloatArray
from bolo.errors import InvalidRequest

if TYPE_CHECKING:
    from bolo.core import Bolo

EMBEDDING_RATE = 16000
#: at or above: the same speaker, as far as the embedding model can tell (calibrated, see the module docs)
SAME_SPEAKER = 0.50
#: at or above (and below SAME_SPEAKER): a resemblance, but real different-speaker pairs reach this too
SIMILAR = 0.30
#: measured with pyannote-3.0-eres2net (ERes2Net, 3D-Speaker) on LibriSpeech dev-clean; see the module docs
CALIBRATION: dict[str, Any] = {
    "embedding_model": "pyannote-3.0-eres2net",
    "data": "LibriSpeech dev-clean (CC BY 4.0): 16 speakers, 5 utterances each, 4 to 15 s",
    "same_speaker": {"pairs": 160, "min": 0.451, "median": 0.797, "max": 0.929},
    "different_speaker": {"pairs": 3000, "min": -0.296, "median": 0.114, "p95": 0.356, "max": 0.53},
    "file": "docs/measurements/speaker-similarity-calibration.json",
}
#: the longest stretch of audio embedded (seconds); longer clips use their first half minute
MAX_EMBED_S = 30.0


def cosine(a: FloatArray, b: FloatArray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def verdict(score: float | None) -> str | None:
    """``same speaker``, ``similar`` or ``different`` for a score from the default embedding model."""
    if score is None:
        return None
    if score >= SAME_SPEAKER:
        return "same speaker"
    if score >= SIMILAR:
        return "similar"
    return "different"


def embedding_model(core: Bolo) -> str:
    return str(core.settings.voice_similarity_model or core.settings.default_diarization)


def available(core: Bolo) -> bool:
    """True when the embedding model is installed (or can be pulled on first use)."""
    name = embedding_model(core)
    try:
        spec = core.resolve(name, "diarization")
    except Exception:
        return False
    return bool(core.manager.is_installed(spec) or core.settings.auto_pull)


async def embed(core: Bolo, samples: FloatArray, sample_rate: int, model: str | None = None) -> FloatArray:
    """The speaker embedding of a recording (resampled to 16 kHz, at most ``MAX_EMBED_S`` seconds)."""
    name = model or embedding_model(core)
    spec = core.resolve(name, "diarization")
    x = resample(np.asarray(samples, dtype=np.float32), sample_rate, EMBEDDING_RATE)[
        : int(MAX_EMBED_S * EMBEDDING_RATE)
    ]
    if len(x) < EMBEDDING_RATE // 2:
        raise InvalidRequest("speaker similarity needs at least half a second of audio")
    async with core.manager.use(spec.id) as entry:
        if not entry.queue.handles("embed"):
            raise InvalidRequest(f"model '{spec.id}' has no speaker-embedding model", param="model")
        vector, _ = await entry.queue.submit("embed", x)
    return np.asarray(vector, dtype=np.float32)


async def score(
    core: Bolo, a: FloatArray, rate_a: int, b: FloatArray, rate_b: int, model: str | None = None
) -> dict[str, Any]:
    """Cosine similarity of the speakers of two recordings, with the verdict and the embedding model."""
    name = model or embedding_model(core)
    ea = await embed(core, a, rate_a, name)
    eb = await embed(core, b, rate_b, name)
    value = round(cosine(ea, eb), 4)
    used = core.resolve(name, "diarization").id
    return {
        "similarity": value,
        "verdict": verdict(value),
        "embedding_model": used,
        # the verdict's thresholds were measured with this model only; others have their own scale
        "calibrated": used == CALIBRATION["embedding_model"],
    }


#: what a voice is scored with when no text is given: one plain sentence per language
TEST_SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain starts.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund und läuft nach Hause, bevor der Regen beginnt.",
    "es": "El rápido zorro marrón salta sobre el perro perezoso y vuelve a casa antes de que empiece a llover.",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux, puis il rentre avant que la pluie commence.",
    "it": "La veloce volpe marrone salta sopra il cane pigro e torna a casa prima che cominci a piovere.",
    "pt": "A rápida raposa marrom pula sobre o cão preguiçoso e volta para casa antes que comece a chover.",
    "nl": "De snelle bruine vos springt over de luie hond en rent naar huis voordat het gaat regenen.",
    "hi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है और बारिश शुरू होने से पहले घर लौट जाती है।",
}


def test_sentence(language: str | None) -> str:
    code = (language or "en").replace("_", "-").split("-")[0].lower()
    return TEST_SENTENCES.get(code, TEST_SENTENCES["en"])


async def score_voice(
    core: Bolo, voice_id: str, model: str | None = None, text: str | None = None, keep_audio: bool = False
) -> dict[str, Any]:
    """Speak a test sentence in a saved voice with ``model`` and score it against the voice's reference.

    Returns similarity, verdict, the model, the speed of that synthesis (rtf, ttfa_ms, audio_s) and, with
    ``keep_audio``, the synthesised samples (``samples``, ``sample_rate``; not JSON)."""
    import time

    from bolo.audio.decode import decode_audio
    from bolo.obs.trace import Trace
    from bolo.voices import now_iso

    voice = core.voices.get(voice_id)
    if voice.is_blend:
        raise InvalidRequest(
            f"'{voice.id}' is a saved blend: it has no recording to compare a clone with", param="voice"
        )
    spec = core.resolve(model or voice.default_model, "tts")
    sentence = (text or "").strip() or test_sentence(voice.language)
    trace = Trace("tts", "score", spec.id)
    parts: list[FloatArray] = []
    rate = spec.sample_rate
    started = time.perf_counter()
    try:
        async for chunk in core.synthesize(
            sentence, model=spec.id, voice=voice.id, custom_voice=True, trace=trace, language=voice.language
        ):
            trace.mark_ttfa()
            parts.append(chunk.samples)
            rate = chunk.sample_rate
    finally:
        core.complete_tts(trace)
    wall = time.perf_counter() - started
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    reference, ref_rate = decode_audio(voice.reference_bytes())
    result = await score(core, reference, ref_rate, audio, rate)
    audio_s = len(audio) / rate if rate else 0.0
    out: dict[str, Any] = {
        "voice": voice.id,
        "model": spec.id,
        **result,
        "text": sentence,
        "audio_s": round(audio_s, 2),
        "rtf": round(wall / audio_s, 3) if audio_s else None,
        "ttfa_ms": round(trace.ttfa_ms) if trace.ttfa_ms is not None else None,
        "at": now_iso(),
    }
    if keep_audio:
        out["samples"], out["sample_rate"] = audio, rate
    return out
