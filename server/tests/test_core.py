from __future__ import annotations

import asyncio

import numpy as np
import pytest

from bolo.core import Bolo
from bolo.errors import InvalidRequest, ModelNotFound
from bolo.obs.trace import Trace
from conftest import speechlike


async def test_transcribe_resamples_and_reports_words(core: Bolo) -> None:
    audio8k = speechlike(2.0, 8000)
    result = await core.transcribe(audio8k, 8000, word_timestamps=True)
    assert result.model == "fake-stt" and result.duration == 2.0
    assert result.text == "the quick brown fox jumps"
    assert [w.w for w in result.words] == result.text.split()
    assert result.words[-1].t1 <= 2.01
    trace = core.traces.recent()[0]
    assert trace.kind == "stt" and trace.text_preview == result.text and trace.chars == len(result.text)


async def test_transcribe_empty_and_silent_audio(core: Bolo) -> None:
    result = await core.transcribe(np.zeros(0, dtype=np.float32), 16000)
    assert result.text == "" and result.segments == []
    result = await core.transcribe(np.zeros(16000 * 3, dtype=np.float32), 16000, vad_filter=True)
    assert result.text == ""  # the VAD found nothing to decode


async def test_translate_requires_capable_model_and_task_must_match(core: Bolo) -> None:
    assert (await core.transcribe(speechlike(1.0), 16000, task="translate")).text
    with pytest.raises(InvalidRequest, match="cannot be used for stt"):
        await core.transcribe(speechlike(1.0), 16000, model="fake-tts")
    with pytest.raises(ModelNotFound):
        await core.transcribe(speechlike(1.0), 16000, model="missing")
    assert core.traces.total_errors == 0  # nothing ran, nothing recorded as a model error


async def test_engine_failure_is_recorded_as_error(core: Bolo) -> None:
    entry = await core.manager.load("fake-stt")
    entry.model.fail_on = "boom"  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        await core.transcribe(speechlike(1.0), 16000, prompt="boom")
    trace = core.traces.recent()[0]
    assert trace.status == "error" and "fake engine failure" in (trace.error or "")
    assert core.traces.models["fake-stt"].errors == 1
    assert core.traces.models["fake-stt"].final.summary() is None


async def test_cancelled_request_is_recorded_and_frees_the_queue(core: Bolo) -> None:
    entry = await core.manager.load("fake-stt")
    entry.model.base_delay_s = 0.3  # type: ignore[attr-defined]
    blocker = asyncio.ensure_future(core.transcribe(speechlike(1.0), 16000))
    await asyncio.sleep(0.05)
    doomed = asyncio.ensure_future(core.transcribe(speechlike(1.0), 16000))
    await asyncio.sleep(0.05)
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed
    await blocker
    assert entry.model.calls == [1]  # type: ignore[attr-defined]  the cancelled job never ran
    statuses = sorted(t.status for t in core.traces.recent())
    assert statuses == ["cancelled", "ok"]
    assert entry.active == 0 and entry.queue.depth == 0


async def test_synthesize_voices_speed_and_validation(core: Bolo) -> None:
    audio, rate = await core.synthesize_all("Hello there.", voice="beta")
    assert rate == 24000 and len(audio) == pytest.approx(len("Hello there.") * 0.06 * 24000, rel=0.05)
    fast, _ = await core.synthesize_all("Hello there.", speed=2.0)
    assert len(fast) == pytest.approx(len(audio) / 2, rel=0.05)
    mapped, _ = await core.synthesize_all("Hello there.", voice="alloy")  # OpenAI voice names map onto model voices
    assert len(mapped) == len(audio)
    with pytest.raises(InvalidRequest, match="Voices: alpha, beta, gamma"):
        await core.synthesize_all("Hello.", voice="nobody")
    with pytest.raises(InvalidRequest, match="empty"):
        await core.synthesize_all("   ")
    with pytest.raises(InvalidRequest, match="speed"):
        await core.synthesize_all("Hi.", speed=9.0)
    statuses = [t.status for t in core.traces.recent(kind="tts")]
    assert statuses.count("ok") == 3 and statuses.count("error") == 3


async def test_synthesize_is_sentence_by_sentence_and_normalised(core: Bolo) -> None:
    entry = await core.manager.load("fake-tts")
    texts: list[str] = []
    async for chunk in core.synthesize("I have 2 cats. That is a **bold** claim! नमस्ते दुनिया।"):
        if chunk.first_of_text:
            texts.append(chunk.text)
    assert texts == ["I have two cats.", "That is a bold claim!", "नमस्ते दुनिया।"]
    assert entry.model.calls == texts  # type: ignore[attr-defined]
    entry.model.calls.clear()  # type: ignore[attr-defined]
    async for _ in core.synthesize("I have 2 cats.", normalize=False):
        pass
    assert entry.model.calls == ["I have 2 cats."]  # type: ignore[attr-defined]


async def test_abandoned_synthesis_stops_the_engine(core: Bolo) -> None:
    entry = await core.manager.load("fake-tts")
    trace = Trace("tts", "http", "")
    gen = core.synthesize("One sentence here. Two sentences here. Three sentences here.", trace=trace)
    async for _chunk in gen:
        break  # the client stopped listening after the first audio
    await gen.aclose()
    core.complete_tts(trace)
    await asyncio.sleep(0.1)
    assert trace.status == "cancelled"
    assert entry.model.calls == ["One sentence here."]  # type: ignore[attr-defined]  later sentences were never synthesised
    assert entry.active == 0


async def test_stats_shape(core: Bolo) -> None:
    await core.transcribe(speechlike(1.0), 16000)
    await core.synthesize_all("Hello.")
    stats = core.stats()
    assert set(stats) == {
        "uptime_s", "active_streams", "queue_depth", "system", "totals", "models", "separation", "providers",
        "fallbacks",
    }  # fmt: skip
    assert set(stats["system"]) == {"cpu_percent", "rss_bytes", "threads"}
    assert stats["totals"]["requests"] == 2 and stats["totals"]["audio_seconds_in"] == 1.0
    by_id = {m["id"]: m for m in stats["models"]}
    stt = by_id["fake-stt"]
    assert stt["loaded"] and stt["requests"] == 1 and stt["ttl_s"] == 300.0
    assert set(stt["latency_ms"]) == {"first_partial", "final", "ttfa"}
    assert set(stt["latency_ms"]["final"]) == {"p50", "p95", "p99"}
    assert set(stt["rtf"]) == {"p50", "p95"}
    assert by_id["fake-tts"]["latency_ms"]["final"] is None
    assert stats["providers"] == [] and stats["fallbacks"] == []
    info = core.info()
    assert info["name"] == "bolo" and info["device"] == "cpu" and info["auth_required"] is False
    assert info["auto_pull"] is False and info["providers"] == []
    assert {"id": "fake", "version": info["version"], "available": True, "reason": None} in info["engines"]


async def test_fallback_vad_rescues_speech_the_neural_detector_missed() -> None:
    from bolo.engines.base import VadDetector, VadEvent
    from bolo.streaming.vad import FallbackVad

    class Deaf(VadDetector):
        in_speech = False

        def accept(self, samples):  # type: ignore[no-untyped-def]
            return []

        def flush(self):  # type: ignore[no-untyped-def]
            return []

    class Working(Deaf):
        def __init__(self) -> None:
            self.n = 0

        def accept(self, samples):  # type: ignore[no-untyped-def]
            self.n += 1
            return (
                [VadEvent("speech_start", 8000)]
                if self.n == 6
                else [VadEvent("speech_end", 40000)]
                if self.n == 30
                else []
            )

    audio = np.concatenate([np.zeros(8000, dtype=np.float32), speechlike(2.0), np.zeros(16000, dtype=np.float32)])

    def run(det: FallbackVad) -> list[tuple[str, float]]:
        events = []
        for off in range(0, len(audio), 1600):
            events += det.accept(audio[off : off + 1600])
        events += det.flush()
        return [(e.kind, round(e.sample / 16000, 1)) for e in events]

    deaf = FallbackVad(Deaf())
    events = run(deaf)
    assert [k for k, _ in events] == ["speech_start", "speech_end"] and deaf.fallbacks == 1
    assert events[0][1] == pytest.approx(0.5, abs=0.15) and events[1][1] == pytest.approx(2.6, abs=0.2)
    working = FallbackVad(Working())
    assert run(working) == [("speech_start", 0.5), ("speech_end", 2.5)] and working.fallbacks == 0  # primary wins


async def test_voice_that_returns_nothing_is_an_error_not_silence(core: Bolo) -> None:
    from bolo.errors import BoloError

    entry = await core.manager.load("fake-tts")
    entry.model.seconds_per_char = 0.0  # type: ignore[attr-defined]
    entry.model.synthesize = lambda *a, **k: iter(())  # type: ignore[method-assign]
    with pytest.raises(BoloError, match="produced no audio"):
        await core.synthesize_all("Hello there.")
    assert core.traces.recent(kind="tts")[0].status == "error"


def test_realtime_and_quality_tts_defaults(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from bolo.config import Settings as S

    core = Bolo(S(data_dir=tmp_path))
    assert core.settings.default_realtime_tts == "piper-en-us-amy-low"
    assert core.resolve("tts-1", "tts").id == "piper-en-us-amy-low"
    assert core.resolve("tts-1-hd", "tts").id == core.resolve(None, "tts").id == "kokoro-v1.0-int8"
    assert core.resolve("realtime", "tts").id == "piper-en-us-amy-low"
    assert core.resolve("realtime", "stt").id == core.settings.default_realtime_stt
    assert core.realtime_tts(None).id == "piper-en-us-amy-low"
    assert core.realtime_tts("alloy").id == "piper-en-us-amy-low"
    assert core.realtime_tts("af_heart").id == "kokoro-v1.0-int8"  # only the quality model has that voice
