"""Wake words and keyword spotting: "hey jarvis", "alexa" or your own phrase, heard in a stream of audio.

Catalog task ``wake-word`` (docs/api.md, "Wake words"). Three families:

* sherpa-onnx keyword spotters (``kind: sherpa_kws``; Apache-2.0): small streaming Zipformer transducers that
  look for any phrase given as text. English phrases are written as BPE pieces (the GigaSpeech model; needs
  sentencepiece) or as phones from the model's own lexicon (the Chinese-English model; Chinese through pypinyin).
* openWakeWord (``kind: openwakeword``): a shared front-end (log-mel spectrogram and speech embedding models,
  Apache-2.0) and one small head per phrase. The pretrained heads are CC BY-NC-SA 4.0: not for commercial use.
* microWakeWord (``kind: micro_wakeword``): streaming TensorFlow Lite models made for microcontrollers (ESPHome),
  through the pymicro-wakeword package (Apache-2.0), which carries the models and TensorFlow Lite's runtime.

Every model gives a :class:`WakeDetector` that takes 16 kHz mono float32 in pieces of any length and returns what
it heard, with the time (seconds since the detector started) at which it heard it.
"""

from __future__ import annotations

import abc
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import Engine, EngineStatus, FloatArray, LoadOptions, Model
from bolo.errors import InvalidRequest

log = logging.getLogger("bolo.wake")

SR = 16000
#: after a detection, the phrase-model detectors stay quiet this long (the tail of the phrase is the same phrase)
QUIET_S = 1.5


@dataclass
class WakeDetection:
    keyword: str
    #: seconds from the start of the stream to the moment the phrase was recognised (its end, give or take)
    time: float
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"keyword": self.keyword, "time": round(self.time, 3)}
        if self.score is not None:
            out["score"] = round(self.score, 4)
        return out


class WakeDetector(abc.ABC):
    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords
        self.samples = 0  # fed so far

    @abc.abstractmethod
    def accept(self, samples: FloatArray) -> list[WakeDetection]: ...

    def finish(self) -> list[WakeDetection]:
        """The end of the stream: anything the detector still holds back."""
        return self.accept(np.zeros(int(0.5 * SR), dtype=np.float32))

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release what the detector holds (an interpreter)."""


class WakeModel(Model):
    task = "wake-word"
    sample_rate = SR
    id: str = ""
    #: takes phrases as text (``keywords``); False: it hears the phrases it was trained on only
    custom_keywords: bool = False
    #: the phrases it listens for when a request names none
    default_keywords: list[str] = []
    default_threshold: float = 0.5

    @abc.abstractmethod
    def create_detector(self, keywords: list[str] | None = None, threshold: float | None = None) -> WakeDetector: ...

    def describe(self) -> dict[str, Any]:
        return {
            "custom_keywords": self.custom_keywords,
            "keywords": list(self.default_keywords),
            "threshold": self.default_threshold,
        }


def parse_keywords(raw: Any) -> list[str] | None:
    """``keywords`` of a request: a list, or one string with commas, semicolons or new lines between phrases."""
    if raw in (None, "", []):
        return None
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise InvalidRequest("keywords must be phrases (strings)", param="keywords")
        out += [p.strip() for p in re.split(r"[,;\n]", item) if p.strip()]
    if len(out) > 50:
        raise InvalidRequest("at most 50 keywords", param="keywords")
    for phrase in out:
        if len(phrase) > 80:
            raise InvalidRequest(f"keyword too long: {phrase[:40]!r}...", param="keywords")
    return out or None


def _label(phrase: str) -> str:
    return re.sub(r"\s+", "_", phrase.strip())


# -- sherpa-onnx keyword spotting ---------------------------------------------------------------------------------
class SherpaKeywordSpotter(WakeModel):
    custom_keywords = True
    default_threshold = 0.25

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        from bolo.engines.sherpa import find_file

        self.id = str(spec.get("id") or spec.get("model_id") or "")
        o = {**(spec.get("options") or {}), **spec, **opts.options}
        chunk = str(o.get("chunk", "chunk-16"))
        self.tokens_type = str(o.get("tokens_type", "bpe"))
        self.path = path
        self._lock = threading.Lock()
        tokens = find_file(path, "**/tokens.txt")
        # the phrases it listens for by default, spelled in the catalog (phrase -> tokens): no tokenizer needed
        defaults: dict[str, str] = {str(k): str(v) for k, v in (o.get("keywords") or {}).items()}
        self.default_keywords = list(defaults)
        self._kws_file = str(path / "bolo-keywords.txt")
        Path(self._kws_file).write_text(
            "".join(f"{tok} @{_label(phrase)}\n" for phrase, tok in defaults.items()), encoding="utf-8"
        )
        self._default_names = {_label(p): p for p in defaults}
        self._bpe: Any = None
        self._lexicon: dict[str, list[str]] = {}
        if self.tokens_type == "bpe":
            self._bpe_path = find_file(path, "**/bpe.model")
        else:
            lexicon = find_file(path, "**/en.phone", required=False)
            if lexicon:
                for line in Path(lexicon).read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts and parts[0] not in self._lexicon:
                        self._lexicon[parts[0]] = parts[1:]
        self._symbols = {line.split()[0] for line in Path(tokens).read_text(encoding="utf-8").splitlines() if line}
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=find_file(path, f"**/encoder-*{chunk}*.onnx"),
            decoder=find_file(path, f"**/decoder-*{chunk}*.onnx"),
            joiner=find_file(path, f"**/joiner-*{chunk}*.onnx"),
            keywords_file=self._kws_file,
            num_threads=max(1, opts.threads),
            keywords_threshold=float(o.get("threshold", self.default_threshold)),
            keywords_score=float(o.get("boost", 1.0)),
            provider="cpu",
        )

    def tokenize(self, phrase: str) -> list[str]:
        """A phrase as the model's tokens (BPE pieces, or phones and pinyin)."""
        if self.tokens_type == "bpe":
            if self._bpe is None:
                try:
                    import sentencepiece
                except ImportError as exc:
                    raise InvalidRequest(
                        f"keywords of your own for '{self.id}' need sentencepiece: pip install 'bolo-speech[wake]'",
                        param="keywords",
                    ) from exc
                self._bpe = sentencepiece.SentencePieceProcessor(model_file=self._bpe_path)
            pieces: list[str] = list(self._bpe.encode(phrase.upper(), out_type=str))
        else:
            pieces = []
            for word in re.findall(r"[A-Za-z']+|[^\sA-Za-z']", phrase):
                if re.match(r"[A-Za-z']", word):
                    phones = self._lexicon.get(word.upper())
                    if phones is None:
                        raise InvalidRequest(
                            f"'{self.id}' has no pronunciation for {word!r}; try another spelling, or the "
                            "GigaSpeech spotter (kws-zipformer-gigaspeech), which spells any English word",
                            param="keywords",
                        )
                    pieces += phones
                else:
                    pieces += _pinyin(word, self.id)
        unknown = [p for p in pieces if p not in self._symbols]
        if unknown:
            raise InvalidRequest(
                f"'{self.id}' cannot spell {phrase!r} (unknown tokens {unknown[:5]})", param="keywords"
            )
        return pieces

    def create_detector(self, keywords: list[str] | None = None, threshold: float | None = None) -> WakeDetector:
        return SherpaKwsDetector(self, keywords, threshold)


def _pinyin(text: str, model_id: str) -> list[str]:
    try:
        from pypinyin import Style, pinyin
        from pypinyin.contrib.tone_convert import to_finals_tone, to_initials
    except ImportError as exc:
        raise InvalidRequest(
            f"Chinese keywords for '{model_id}' need pypinyin: pip install pypinyin", param="keywords"
        ) from exc
    out: list[str] = []
    for (syllable,) in pinyin(text, style=Style.TONE, heteronym=False):
        initial = to_initials(syllable, strict=False)
        final = to_finals_tone(syllable, strict=False)
        out += [initial, final] if initial else [final]
    return out


class SherpaKwsDetector(WakeDetector):
    def __init__(self, model: SherpaKeywordSpotter, keywords: list[str] | None, threshold: float | None) -> None:
        self.model = model
        spec = None
        if keywords:
            parts = []
            for phrase in keywords:
                entry = " ".join(model.tokenize(phrase)) + f" @{_label(phrase)}"
                if threshold is not None:
                    entry = entry.replace(" @", f" #{float(threshold):g} @", 1)
                parts.append(entry)
            spec = "/".join(parts)
            self._names = {_label(p).lower(): p for p in keywords}
        else:
            self._names = {k.lower(): v for k, v in model._default_names.items()}
        self._custom = bool(keywords)
        super().__init__(keywords or model.default_keywords)
        with model._lock:
            self.stream = model.kws.create_stream(spec) if spec else model.kws.create_stream()

    def accept(self, samples: FloatArray) -> list[WakeDetection]:
        out: list[WakeDetection] = []
        kws = self.model.kws
        self.samples += len(samples)
        with self.model._lock:
            self.stream.accept_waveform(SR, np.asarray(samples, dtype=np.float32))
            while kws.is_ready(self.stream):
                kws.decode_stream(self.stream)
                hit = kws.get_result(self.stream)
                if hit:
                    kws.reset_stream(self.stream)
                    # a stream given its own phrases still hears the model's default ones: those are left out
                    # (matched without case: a phrase that is also a default one may come back under that name)
                    name = self._names.get(hit.lower())
                    if name is not None or not self._custom:
                        out.append(WakeDetection(name or hit.replace("_", " "), self.samples / SR))
        return out


# -- openWakeWord -------------------------------------------------------------------------------------------------
OWW_CHUNK = 1280  # 80 ms


class OpenWakeWord(WakeModel):
    """openWakeWord's pipeline in ONNX Runtime: a log-mel spectrogram model and a speech embedding model shared by
    every phrase, then the phrase's own head on the last 16 embeddings (1.28 s)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import onnxruntime as ort

        self.id = str(spec.get("id") or spec.get("model_id") or "")
        o = {**(spec.get("options") or {}), **spec, **opts.options}
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, opts.threads)
        so.inter_op_num_threads = 1
        cpu = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(str(path / o.get("melspectrogram", "melspectrogram.onnx")), so, cpu)
        self.emb = ort.InferenceSession(str(path / o.get("embedding", "embedding_model.onnx")), so, cpu)
        self.head = ort.InferenceSession(str(path / o["model"]), so, cpu)
        self.head_input = self.head.get_inputs()[0].name
        self.default_keywords = [str(o.get("phrase") or spec.get("id", ""))]
        self.default_threshold = float(o.get("threshold", 0.5))
        self._lock = threading.Lock()
        #: the buffers after the warm-up every detector starts from, computed once
        self.warm_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def create_detector(self, keywords: list[str] | None = None, threshold: float | None = None) -> WakeDetector:
        return OpenWakeWordDetector(self, threshold if threshold is not None else self.default_threshold)


class OpenWakeWordDetector(WakeDetector):
    def __init__(self, model: OpenWakeWord, threshold: float) -> None:
        super().__init__(model.default_keywords)
        self.model = model
        self.threshold = threshold
        self._raw = np.zeros(0, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._mel = np.ones((76, 32), dtype=np.float32)
        self._feat = np.zeros((0, 96), dtype=np.float32)
        self._armed = True
        self._quiet_until = 0.0
        warm = model.warm_state
        if warm is None:
            # as openWakeWord does: the buffers start with what four seconds of faint noise leave in them
            noise = np.random.default_rng(0).integers(-1000, 1000, 4 * SR).astype(np.float32)
            for i in range(0, len(noise), OWW_CHUNK):
                self._step(noise[i : i + OWW_CHUNK])
            model.warm_state = (self._raw.copy(), self._mel.copy(), self._feat.copy())
        else:
            self._raw, self._mel, self._feat = (a.copy() for a in warm)

    def _step(self, chunk: np.ndarray) -> float:
        m = self.model
        self._raw = np.concatenate([self._raw, chunk])[-(OWW_CHUNK * 10) :]
        with m._lock:
            mel = m.mel.run(None, {"input": self._raw[-(OWW_CHUNK + 160 * 3) :][None, :]})[0]
            self._mel = np.concatenate([self._mel, np.squeeze(mel) / 10.0 + 2.0])[-970:].astype(np.float32)
            emb = m.emb.run(None, {"input_1": self._mel[-76:][None, :, :, None]})[0].reshape(1, 96)
            self._feat = np.concatenate([self._feat, emb])[-120:].astype(np.float32)
            if len(self._feat) < 16:
                return 0.0
            out = m.head.run(None, {m.head_input: self._feat[-16:][None]})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def accept(self, samples: FloatArray) -> list[WakeDetection]:
        out: list[WakeDetection] = []
        self._pending = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32) * 32767.0])
        while len(self._pending) >= OWW_CHUNK:
            chunk, self._pending = self._pending[:OWW_CHUNK], self._pending[OWW_CHUNK:]
            self.samples += OWW_CHUNK
            score = self._step(chunk)
            now = self.samples / SR
            if score >= self.threshold and self._armed and now >= self._quiet_until:
                out.append(WakeDetection(self.keywords[0], now, score))
                self._armed = False
                self._quiet_until = now + QUIET_S
            elif score < self.threshold / 2:  # one utterance of the phrase is one detection
                self._armed = True
        return out


# -- microWakeWord ------------------------------------------------------------------------------------------------
class MicroWakeWordModel(WakeModel):
    def __init__(self, spec: dict[str, Any], opts: LoadOptions) -> None:
        from pymicro_wakeword import MicroWakeWord, Model

        self.id = str(spec.get("id") or spec.get("model_id") or "")
        o = {**(spec.get("options") or {}), **spec, **opts.options}
        self.name = str(o["model"])
        self._cls, self._model_enum = MicroWakeWord, Model
        probe = self.make()
        self.default_threshold = float(o.get("threshold", probe.probability_cutoff))
        self.default_keywords = [str(o.get("phrase") or getattr(probe, "wake_word", self.name))]
        probe.close()

    def make(self) -> Any:
        """A fresh interpreter (its state is the stream's)."""
        return self._cls.from_builtin(self._model_enum(self.name))

    def create_detector(self, keywords: list[str] | None = None, threshold: float | None = None) -> WakeDetector:
        return MicroWakeWordDetector(self, threshold if threshold is not None else self.default_threshold)


class MicroWakeWordDetector(WakeDetector):
    def __init__(self, model: MicroWakeWordModel, threshold: float) -> None:
        from pymicro_wakeword import MicroWakeWordFeatures

        super().__init__(model.default_keywords)
        self.ww = model.make()
        self.ww.probability_cutoff = threshold
        self.features = MicroWakeWordFeatures()
        self._armed = True

    def accept(self, samples: FloatArray) -> list[WakeDetection]:
        out: list[WakeDetection] = []
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        self.samples += len(pcm)
        now = self.samples / SR
        for feature in self.features.process_streaming(pcm.tobytes()):
            prob = self.ww.process_streaming_prob(feature)
            if prob is None or getattr(self.ww, "_features", None):  # still gathering the next stride: no new score
                continue
            if prob > self.ww.probability_cutoff and self._armed:
                out.append(WakeDetection(self.keywords[0], now, float(prob)))
                self._armed = False
            elif prob < self.ww.probability_cutoff / 2:
                self._armed = True
        return out

    def close(self) -> None:
        self.ww.close()


class MicroWakeWordEngine(Engine):
    id = "pymicro-wakeword"
    name = "microWakeWord (pymicro-wakeword)"
    license = "Apache-2.0"
    url = "https://github.com/OHF-Voice/pymicro-wakeword"
    kinds = ("micro_wakeword",)

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import pymicro_wakeword  # noqa: F401

            return EngineStatus(self.id, True, version("pymicro-wakeword"), None)
        except Exception as exc:
            return EngineStatus(
                self.id, False, None, f"{type(exc).__name__}: {exc}. Optional: pip install 'bolo-speech[wake]'"
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind != "micro_wakeword":
            raise ValueError(f"the pymicro-wakeword engine cannot load kind {kind!r}")
        return MicroWakeWordModel(spec, opts)


# -- tests ---------------------------------------------------------------------------------------------------------
class FakeWake(WakeModel):
    """Tests: "hears" its phrase whenever 0.1 s of audio is louder than -20 dBFS after a quieter stretch."""

    custom_keywords = True
    default_keywords = ["hey fake"]

    def __init__(self, spec: dict[str, Any]) -> None:
        self.id = str(spec.get("id") or "")

    def create_detector(self, keywords: list[str] | None = None, threshold: float | None = None) -> WakeDetector:
        return _LoudDetector(keywords or self.default_keywords)


class _LoudDetector(WakeDetector):
    def __init__(self, keywords: list[str]) -> None:
        super().__init__(keywords)
        self._buf = np.zeros(0, dtype=np.float32)
        self._armed = True

    def accept(self, samples: FloatArray) -> list[WakeDetection]:
        out: list[WakeDetection] = []
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        step = SR // 10
        while len(self._buf) >= step:
            piece, self._buf = self._buf[:step], self._buf[step:]
            self.samples += step
            loud = float(np.sqrt(np.mean(piece**2))) > 0.1
            if loud and self._armed:
                out.append(WakeDetection(self.keywords[0], self.samples / SR, 1.0))
            self._armed = not loud
        return out


# -- running a model over a recording -----------------------------------------------------------------------------
def detect(
    model: WakeModel, samples: FloatArray, keywords: list[str] | None, threshold: float | None
) -> list[WakeDetection]:
    """Every detection in a whole recording (16 kHz), fed as a live stream would be (0.1 s at a time)."""
    detector = model.create_detector(keywords, threshold)
    out: list[WakeDetection] = []
    step = SR // 10
    for i in range(0, len(samples), step):
        out += detector.accept(samples[i : i + step])
    out += detector.finish()
    return out


def wake_body(
    model_id: str, model: WakeModel, detector_keywords: list[str], detections: list[WakeDetection], duration: float
) -> dict[str, Any]:
    return {
        "model": model_id,
        "keywords": detector_keywords,
        "detected": bool(detections),
        "detections": [d.to_dict() for d in detections],
        "duration": round(duration, 3),
    }


def to_json(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)
