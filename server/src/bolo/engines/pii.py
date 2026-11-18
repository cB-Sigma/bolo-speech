"""Personal-information detection (task ``pii``), used to redact transcripts (bolo.redact).

* ``presidio`` engine (Microsoft Presidio's analyzer, MIT; ``pip install "bolo-speech[pii]"``):
  * kind ``presidio_patterns``: its pattern recognisers only (regular expressions with checksums and context words:
    e-mail addresses, telephone numbers, credit cards, IBANs, IP addresses, URLs, US, UK, Indian, Australian and
    other national ID numbers). No model, nothing downloaded; it does not find names.
  * kind ``presidio_spacy``: the same with a spaCy pipeline's named-entity recogniser (from the model directory) for
    names, places, organisations and dates.
* ``onnxruntime`` engine, kind ``gliner_pii`` (:class:`GlinerPii`): GLiNER, a span classifier that is told the
  entity types in words, from an ONNX export, with ``tokenizers``; better for names in many languages.

Entity types are given in Presidio's names (``PERSON``, ``PHONE_NUMBER``, ``EMAIL_ADDRESS``, ``CREDIT_CARD``,
``LOCATION``, ...), whatever the model calls them.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from bolo.engines.base import Engine, EngineStatus, LoadOptions, Model, PiiModel, PiiSpan

INSTALL = 'pip install "bolo-speech[pii]"'

#: Presidio's pattern recognisers that need no NLP model (class names in presidio_analyzer.predefined_recognizers)
PATTERN_RECOGNIZERS = (
    "EmailRecognizer", "PhoneRecognizer", "CreditCardRecognizer", "IbanRecognizer", "IpRecognizer", "UrlRecognizer",
    "CryptoRecognizer", "MedicalLicenseRecognizer", "UsSsnRecognizer", "UsBankRecognizer", "UsLicenseRecognizer",
    "UsItinRecognizer", "UsPassportRecognizer", "UkNinoRecognizer", "UkNhsRecognizer", "InAadhaarRecognizer",
    "InPanRecognizer", "InPassportRecognizer", "InVehicleRegistrationRecognizer", "InVoterRecognizer",
    "AuAbnRecognizer", "AuAcnRecognizer", "AuTfnRecognizer", "AuMedicareRecognizer", "SgFinRecognizer",
    "EsNifRecognizer", "ItFiscalCodeRecognizer", "PlPeselRecognizer",
)  # fmt: skip


def _resolve_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    """The highest-scoring spans that do not overlap, in text order (longer first on a tie)."""
    chosen: list[PiiSpan] = []
    for span in sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start)):
        if all(span.end <= c.start or span.start >= c.end for c in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: s.start)


class PresidioPii(PiiModel):
    """Presidio's analyzer: pattern recognisers, and with ``spacy`` a spaCy pipeline's named entities."""

    def __init__(self, kind: str, path: Path | None, spec: dict[str, Any]) -> None:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, predefined_recognizers

        self.threshold = float(spec.get("threshold", 0.4))
        self.languages = tuple(str(x) for x in spec.get("languages") or ("en",))
        registry = RecognizerRegistry(supported_languages=list(self.languages))
        for name in PATTERN_RECOGNIZERS:
            cls = getattr(predefined_recognizers, name, None)
            if cls is None:
                continue  # an older or newer Presidio without it
            for language in self.languages:
                try:
                    registry.add_recognizer(cls(supported_language=language))
                except Exception:  # a recogniser that needs a library or a language it does not have
                    continue
        nlp_engine: Any = None
        if kind == "presidio_spacy":
            from presidio_analyzer.nlp_engine import SpacyNlpEngine
            from presidio_analyzer.predefined_recognizers import SpacyRecognizer

            assert path is not None
            pipeline = _spacy_pipeline(path)
            nlp_engine = SpacyNlpEngine(models=[{"lang_code": self.languages[0], "model_name": str(pipeline)}])
            nlp_engine.load()
            registry.add_recognizer(SpacyRecognizer(supported_language=self.languages[0]))
        else:
            nlp_engine = _NoNlpEngine(self.languages)
        self._analyzer = AnalyzerEngine(
            registry=registry, nlp_engine=nlp_engine, supported_languages=list(self.languages)
        )
        self.entities = tuple(sorted(self._analyzer.get_supported_entities(language=self.languages[0])))
        self._lock = threading.Lock()

    def find(self, text: str, language: str | None, entities: tuple[str, ...]) -> list[PiiSpan]:
        language = language if language in self.languages else self.languages[0]
        asked = [e for e in entities if e in self.entities] if entities else None
        if entities and not asked:
            return []
        with self._lock:
            results = self._analyzer.analyze(
                text=text, language=language, entities=asked, score_threshold=self.threshold
            )
        return _resolve_overlaps([PiiSpan(r.entity_type, r.start, r.end, float(r.score)) for r in results])


class _NoNlpEngine:
    """An NLP engine that does nothing, for Presidio's pattern recognisers alone (no spaCy model to load)."""

    def __init__(self, languages: tuple[str, ...]) -> None:
        self.languages = languages

    def load(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def process_text(self, text: str, language: str) -> Any:
        from presidio_analyzer.nlp_engine import NlpArtifacts

        return NlpArtifacts(entities=[], tokens=[], tokens_indices=[], lemmas=[], nlp_engine=self, language=language)

    def process_batch(self, texts: Any, language: str, **kwargs: Any) -> Any:
        for text in texts:
            yield text, self.process_text(str(text), language)

    def is_stopword(self, word: str, language: str) -> bool:
        return False

    def is_punct(self, word: str, language: str) -> bool:
        return False

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return list(self.languages)


def _spacy_pipeline(path: Path) -> Path:
    """The spaCy pipeline directory inside a pulled model package (the folder holding ``config.cfg``)."""
    found = sorted(p.parent for p in path.rglob("config.cfg") if (p.parent / "meta.json").is_file())
    if not found:
        raise FileNotFoundError(f"no spaCy pipeline (config.cfg and meta.json) under {path}")
    return found[0]


#: what bolo calls GLiNER-PII's labels (the words the model is told), and the reverse
GLINER_LABELS = {
    "PERSON": "person",
    "PHONE_NUMBER": "phone number",
    "EMAIL_ADDRESS": "email",
    "LOCATION": "address",
    "ORGANIZATION": "organization",
    "DATE_OF_BIRTH": "date of birth",
    "CREDIT_CARD": "credit card number",
    "BANK_ACCOUNT": "bank account number",
    "IBAN_CODE": "iban",
    "PASSPORT": "passport number",
    "ID_NUMBER": "identity card number",
    "US_SSN": "social security number",
    "IP_ADDRESS": "ip address",
}
_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")


class GlinerPii(PiiModel):
    """GLiNER (a span classifier told the entity types in words) from an ONNX export, with its tokenizer.

    The text is split into words; the prompt ``<<ENT>> person <<ENT>> phone number ... <<SEP>>`` goes before them;
    the model scores every span of up to ``max_width`` words against every type, and the spans scoring over the
    threshold are kept, best first, without overlaps."""

    entities = tuple(GLINER_LABELS)

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import json

        from tokenizers import Tokenizer

        from bolo.ort_providers import CPU, create_session, session_options

        config = json.loads(_one(path, "gliner_config.json").read_text("utf-8"))
        self.max_width = int(config.get("max_width", 12))
        self.max_words = int(spec.get("max_words", config.get("max_len", 384)))
        self.ent_token = str(config.get("ent_token", "<<ENT>>"))
        self.sep_token = str(config.get("sep_token", "<<SEP>>"))
        self.threshold = float(spec.get("threshold", 0.5))
        self.tokenizer = Tokenizer.from_file(str(_one(path, "tokenizer.json")))
        model = _one(path, str(spec.get("file", "model*.onnx")))
        options = session_options()
        options.intra_op_num_threads = max(1, opts.threads)
        options.inter_op_num_threads = 1
        self.session = create_session(str(model), options, [CPU])
        self._lock = threading.Lock()

    def find(self, text: str, language: str | None, entities: tuple[str, ...]) -> list[PiiSpan]:
        types = [e for e in (entities or self.entities) if e in GLINER_LABELS]
        words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        if not types or not words:
            return []
        spans: list[PiiSpan] = []
        for first in range(0, len(words), self.max_words):  # long texts a window of words at a time
            window = words[first : first + self.max_words]
            for type_, a, b, score in self._spans(text, window, types):
                spans.append(PiiSpan(type_, window[a][0], window[b][1], score))
        return _resolve_overlaps(spans)

    def _spans(self, text: str, words: list[tuple[int, int]], types: list[str]) -> list[tuple[str, int, int, float]]:
        prompt: list[str] = []
        for type_ in types:
            prompt += [self.ent_token, GLINER_LABELS[type_]]
        prompt.append(self.sep_token)
        encoding = self.tokenizer.encode([*prompt, *(text[a:b] for a, b in words)], is_pretokenized=True)
        words_mask: list[int] = []
        previous = None
        for word_id in encoding.word_ids:
            first = word_id is not None and word_id >= len(prompt) and word_id != previous
            words_mask.append(word_id - len(prompt) + 1 if first and word_id is not None else 0)
            previous = word_id
        n, width = len(words), self.max_width
        span_idx = np.array([[(i, min(i + w, n - 1)) for i in range(n) for w in range(width)]], dtype=np.int64)
        span_mask = np.array([[i + w < n for i in range(n) for w in range(width)]], dtype=bool)
        feed = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            "words_mask": np.array([words_mask], dtype=np.int64),
            "text_lengths": np.array([[n]], dtype=np.int64),
            "span_idx": span_idx,
            "span_mask": span_mask,
        }
        with self._lock:
            logits = self.session.run(["logits"], feed)[0][0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        out: list[tuple[str, int, int, float]] = []
        for start, w, c in zip(*np.nonzero(probs > self.threshold)):
            end = int(start) + int(w)
            if end < n:
                out.append((types[int(c)], int(start), end, float(probs[start, w, c])))
        return out


def _one(path: Path, pattern: str) -> Path:
    found = sorted(path.rglob(pattern))
    if not found:
        raise FileNotFoundError(f"no {pattern} under {path}")
    return found[0]


class PresidioEngine(Engine):
    id = "presidio"
    name = "Presidio analyzer"
    license = "MIT"
    url = "https://github.com/microsoft/presidio"
    kinds = ("presidio_patterns", "presidio_spacy")

    def status(self) -> EngineStatus:
        try:
            from importlib.metadata import version

            import presidio_analyzer  # noqa: F401

            return EngineStatus(self.id, True, version("presidio-analyzer"), None)
        except Exception as exc:
            return EngineStatus(self.id, False, None, f"{type(exc).__name__}: {exc}. Install with: {INSTALL}")

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind not in self.kinds:
            raise ValueError(f"presidio engine cannot load kind {kind!r}")
        return PresidioPii(kind, path if kind == "presidio_spacy" else None, spec)
