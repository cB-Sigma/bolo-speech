"""Wyoming protocol server (Home Assistant voice pipelines), speech-to-text and text-to-speech.

Built on the MIT-licensed ``wyoming`` package. One TCP port serves both
services, like a combined faster-whisper + piper add-on: Home Assistant's
Wyoming integration discovers the ASR models and TTS voices from ``describe``.

Supported events: ``describe`` → ``info``; ``transcribe``, ``audio-start``,
``audio-chunk``, ``audio-stop`` → ``transcript``; ``synthesize`` →
``audio-start``, ``audio-chunk``…, ``audio-stop``; and streaming synthesis
(``synthesize-start``, ``synthesize-chunk``, ``synthesize-stop`` →
audio…, ``synthesize-stopped``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from bolo import __version__
from bolo.audio.codecs import float_to_pcm16, pcm16_to_float
from bolo.errors import BoloError
from bolo.obs.trace import Trace
from bolo.text.segment import IncrementalSegmenter

if TYPE_CHECKING:
    from bolo.core import Bolo

log = logging.getLogger("bolo.wyoming")

CHUNK_SAMPLES = 2048


def _require() -> Any:
    try:
        import wyoming
    except ImportError as exc:
        raise RuntimeError(
            "the Wyoming protocol needs the 'wyoming' package: pip install 'bolo-speech[wyoming]'"
        ) from exc
    return wyoming


def build_info(core: Bolo) -> Any:
    from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker, WakeProgram
    from wyoming.info import WakeModel as WyomingWakeModel

    attribution = Attribution(name="bolo", url="https://github.com/cB-Sigma/bolo-speech")
    asr_models = []
    voices = []
    wake_models = []
    for m in core.catalog_api(installed=True):
        origin = Attribution(name=m["family"], url=m["source_url"] or "https://github.com/cB-Sigma/bolo-speech")
        if m["task"] == "stt":
            asr_models.append(
                AsrModel(
                    name=m["id"],
                    attribution=origin,
                    installed=True,
                    description=m["description"],
                    version=None,
                    languages=m["languages"],
                )
            )
        elif m["task"] == "wake-word":
            phrases = list(m.get("wake_phrases") or [])
            wake_models.append(
                WyomingWakeModel(
                    name=m["id"],
                    attribution=origin,
                    installed=True,
                    description=f"{m['description']} Licence: {m['license']}.",
                    version=None,
                    languages=m["languages"],
                    phrase=", ".join(phrases).lower() or None,
                )
            )
        elif m["task"] == "tts":
            model_voices = m["voices"] or [
                {"id": "default", "name": m["id"], "language": (m["languages"] or ["en"])[0]}
            ]
            if len(model_voices) == 1:
                v = model_voices[0]
                voices.append(
                    TtsVoice(
                        name=m["id"],
                        attribution=origin,
                        installed=True,
                        description=f"{v.get('name') or v['id']} ({m['id']})",
                        version=None,
                        languages=[v.get("language") or (m["languages"] or ["en"])[0]],
                    )
                )
            else:
                by_lang: dict[str, list[str]] = {}
                for v in model_voices:
                    by_lang.setdefault(v.get("language") or (m["languages"] or ["en"])[0], []).append(v["id"])
                for lang, ids in by_lang.items():
                    voices.append(
                        TtsVoice(
                            name=f"{m['id']}:{lang}",
                            attribution=origin,
                            installed=True,
                            description=f"{m['id']} ({lang})",
                            version=None,
                            languages=[lang],
                            speakers=[TtsVoiceSpeaker(name=i) for i in ids],
                        )
                    )
    asr_extra: dict[str, Any] = {}
    if "supports_transcript_streaming" in getattr(AsrProgram, "__dataclass_fields__", {}):
        asr_extra["supports_transcript_streaming"] = True  # wyoming >= 1.7: transcript-start/-chunk/-stop
    asr = [
        AsrProgram(
            name="bolo",
            attribution=attribution,
            installed=True,
            description="bolo speech-to-text",
            version=__version__,
            models=asr_models,
            **asr_extra,
        )
    ]
    tts = [
        TtsProgram(
            name="bolo",
            attribution=attribution,
            installed=True,
            description="bolo text-to-speech",
            version=__version__,
            voices=voices,
            supports_synthesize_streaming=True,
        )
    ]
    wake = [
        WakeProgram(
            name="bolo",
            attribution=attribution,
            installed=True,
            description="bolo wake words (a keyword spotter takes '<model>:<phrase>' names too)",
            version=__version__,
            models=wake_models,
        )
    ]
    return Info(asr=asr if asr_models else [], tts=tts if voices else [], wake=wake if wake_models else [])


def parse_voice(core: Bolo, voice: Any) -> tuple[str | None, str | None]:
    """Map a Wyoming voice (name + speaker) to ``(model, voice id)``."""
    if voice is None:
        return None, None
    name = getattr(voice, "name", None)
    speaker = getattr(voice, "speaker", None)
    if not name:
        return None, speaker
    model, _, _lang = name.partition(":")
    if core.catalog.resolve(model) is not None:
        return model, speaker
    return None, speaker or name


def make_handler(core: Bolo) -> Any:
    _require()
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.error import Error
    from wyoming.event import Event
    from wyoming.info import Describe
    from wyoming.server import AsyncEventHandler
    from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped
    from wyoming.wake import Detect, Detection, NotDetected

    try:  # wyoming >= 1.7
        from wyoming.asr import TranscriptChunk, TranscriptStart, TranscriptStop

        _TRANSCRIPT_STREAMING: tuple[Any, Any, Any] | None = (TranscriptStart, TranscriptChunk, TranscriptStop)
    except ImportError:  # pragma: no cover - older wyoming
        _TRANSCRIPT_STREAMING = None

    class Handler(AsyncEventHandler):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model: str | None = None
            self.language: str | None = None
            self.audio: list[np.ndarray[Any, Any]] = []
            self.rate = 16000
            self.stream_voice: Any = None
            self.segmenter: IncrementalSegmenter | None = None
            self.streaming_tts = False
            self.started_audio = False
            #: wake-word detection (a Detect event came first): name -> detector, and the models kept in use
            self.wake: dict[str, Any] | None = None
            self.wake_stack: contextlib.AsyncExitStack | None = None
            self.wake_samples = 0
            self.wake_detected = False
            self.wake_resampler: Any = None

        async def disconnect(self) -> None:
            await self._end_wake()

        async def _end_wake(self) -> None:
            for detector in (self.wake or {}).values():
                with contextlib.suppress(Exception):
                    detector.close()
            self.wake = None
            stack, self.wake_stack = self.wake_stack, None
            if stack is not None:
                with contextlib.suppress(Exception):
                    await stack.aclose()

        async def _start_wake(self, names: list[str]) -> None:
            """Detectors for the names a Detect event gives: model ids, or '<model>:<phrase>' for a keyword
            spotter; none: the default wake-word model."""
            from bolo.wake import WakeModel, parse_keywords

            await self._end_wake()
            stack = contextlib.AsyncExitStack()
            detectors: dict[str, Any] = {}
            try:
                for name in names or [core.settings.default_wake_word]:
                    model_name, _, phrase = name.partition(":")
                    spec = core.resolve(model_name or None, "wake-word")
                    entry = await stack.enter_async_context(core.manager.use(spec.id))
                    model = entry.model
                    if not isinstance(model, WakeModel):
                        raise BoloError(f"'{spec.id}' is not a wake-word model")
                    keywords = parse_keywords(phrase) if phrase else None
                    if keywords and not model.custom_keywords:
                        raise BoloError(f"'{spec.id}' takes no phrases of its own")
                    detectors[name] = await asyncio.to_thread(model.create_detector, keywords, None)
            except BaseException:
                await stack.aclose()
                raise
            self.wake, self.wake_stack = detectors, stack
            self.wake_samples, self.wake_detected, self.wake_resampler = 0, False, None

        async def handle_event(self, event: Event) -> bool:
            try:
                return await self._handle(event)
            except BoloError as exc:
                await self.write_event(Error(text=exc.message, code=exc.code).event())
                return True
            except Exception as exc:
                log.exception("wyoming handler failed")
                await self.write_event(Error(text=f"{type(exc).__name__}: {exc}", code="internal_error").event())
                return True

        async def _handle(self, event: Event) -> bool:
            if Describe.is_type(event.type):
                await self.write_event(build_info(core).event())
            elif Detect.is_type(event.type):
                await self._start_wake(list(Detect.from_event(event).names or []))
            elif Transcribe.is_type(event.type):
                await self._end_wake()
                req = Transcribe.from_event(event)
                self.model = req.name if req.name and core.catalog.resolve(req.name) else None
                self.language = req.language
            elif AudioStart.is_type(event.type):
                self.audio = []
                self.rate = AudioStart.from_event(event).rate
            elif AudioChunk.is_type(event.type) and self.wake is not None:
                await self._wake_audio(AudioChunk.from_event(event))
            elif AudioStop.is_type(event.type) and self.wake is not None:
                if not self.wake_detected:
                    await self.write_event(NotDetected().event())
                await self._end_wake()
            elif AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                self.rate = chunk.rate
                samples = np.frombuffer(chunk.audio, dtype="<i2") if chunk.width == 2 else None
                if samples is None:
                    raise BoloError("only 16-bit PCM audio is supported over Wyoming")
                if chunk.channels > 1:
                    samples = samples.reshape(-1, chunk.channels).mean(axis=1).astype("<i2")
                self.audio.append(pcm16_to_float(samples))
            elif AudioStop.is_type(event.type):
                samples_f = np.concatenate(self.audio) if self.audio else np.zeros(0, dtype=np.float32)
                self.audio = []
                trace = Trace("stt", "wyoming", "")
                language = (self.language or "").split("-")[0].split("_")[0] or None
                streaming = _TRANSCRIPT_STREAMING is not None
                done_text: list[str] = []
                if streaming:
                    start, chunk_cls, _stop = _TRANSCRIPT_STREAMING  # type: ignore[misc]
                    await self.write_event(start(language=self.language).event())

                    async def on_segment(segment: Any) -> None:
                        # each chunk carries the transcript so far (it replaces the previous one)
                        done_text.append(segment.text.strip())
                        await self.write_event(chunk_cls(text=" ".join(t for t in done_text if t)).event())

                result = await core.transcribe(
                    samples_f, self.rate, model=self.model, language=language, trace=trace,
                    on_segment=on_segment if streaming else None,
                )  # fmt: skip
                await self.write_event(Transcript(text=result.text, language=result.language or self.language).event())
                if streaming:
                    await self.write_event(_TRANSCRIPT_STREAMING[2]().event())  # type: ignore[index]
            elif Synthesize.is_type(event.type):
                if self.streaming_tts:
                    return True  # compatibility event sent alongside the streaming ones: ignore
                req = Synthesize.from_event(event)
                await self._speak(req.text, req.voice, first=True, last=True)
            elif SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self.stream_voice = start.voice
                self.segmenter = IncrementalSegmenter()
                self.streaming_tts = True
                self.started_audio = False
            elif SynthesizeChunk.is_type(event.type):
                if self.segmenter is not None:
                    for sentence in self.segmenter.push(SynthesizeChunk.from_event(event).text):
                        await self._speak(sentence, self.stream_voice, first=not self.started_audio, last=False)
            elif SynthesizeStop.is_type(event.type):
                if self.segmenter is not None:
                    for sentence in self.segmenter.flush():
                        await self._speak(sentence, self.stream_voice, first=not self.started_audio, last=False)
                if self.started_audio:
                    await self.write_event(AudioStop().event())
                await self.write_event(SynthesizeStopped().event())
                self.segmenter = None
                self.streaming_tts = False
            return True

        async def _wake_audio(self, chunk: Any) -> None:
            from bolo.audio.resample import StreamResampler
            from bolo.wake import SR

            if chunk.width != 2:
                raise BoloError("only 16-bit PCM audio is supported over Wyoming")
            pcm = np.frombuffer(chunk.audio, dtype="<i2")
            if chunk.channels > 1:
                pcm = pcm.reshape(-1, chunk.channels).mean(axis=1).astype("<i2")
            samples = pcm16_to_float(pcm)
            if chunk.rate != SR:
                if self.wake_resampler is None:
                    self.wake_resampler = StreamResampler(chunk.rate, SR)
                samples = self.wake_resampler.process(samples)
            self.wake_samples += len(samples)
            for name, detector in (self.wake or {}).items():
                for hit in await asyncio.to_thread(detector.accept, samples):
                    self.wake_detected = True
                    await self.write_event(
                        Detection(name=name, timestamp=int(hit.time * 1000), context={"keyword": hit.keyword}).event()
                    )

        async def _speak(self, text: str, voice: Any, *, first: bool, last: bool) -> None:
            model, voice_id = parse_voice(core, voice)
            if model is None:
                model = core.realtime_tts(voice_id).id  # a voice assistant wants the first sound quickly
            trace = Trace("tts", "wyoming", "")
            sent_start = not first
            try:
                async for chunk in core.synthesize(text, model=model, voice=voice_id, trace=trace):
                    pcm = float_to_pcm16(chunk.samples)
                    if not sent_start:
                        await self.write_event(AudioStart(rate=chunk.sample_rate, width=2, channels=1).event())
                        sent_start = True
                        self.started_audio = True
                    trace.mark_ttfa()
                    for off in range(0, len(pcm), CHUNK_SAMPLES):
                        part = pcm[off : off + CHUNK_SAMPLES].tobytes()
                        await self.write_event(
                            AudioChunk(audio=part, rate=chunk.sample_rate, width=2, channels=1).event()
                        )
            except BoloError as exc:
                trace.finish("error", exc.message)
                raise
            except asyncio.CancelledError:
                trace.finish("cancelled", "client disconnected")
                raise
            finally:
                core.complete_tts(trace)
            if last and sent_start:
                await self.write_event(AudioStop().event())

    return Handler


async def start_wyoming(core: Bolo, host: str, port: int) -> Any:
    """Start the Wyoming TCP server on the running loop and return it (call ``.stop()`` to end)."""
    _require()
    from wyoming.server import AsyncTcpServer

    server = AsyncTcpServer(host, port)
    await server.start(make_handler(core))
    return server
