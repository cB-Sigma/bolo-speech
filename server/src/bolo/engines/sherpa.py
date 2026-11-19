"""sherpa-onnx engine (Apache-2.0, https://github.com/k2-fsa/sherpa-onnx).

The broadest engine: non-streaming and streaming speech recognition, several
text-to-speech families and neural VAD, all through ONNX Runtime bundled in
the ``sherpa-onnx`` wheel (no PyTorch).

Model directories are matched by file-name patterns rather than exact names,
so the same loader handles the float and int8 exports of a model.

sherpa-onnx 2.x contains no phonemiser. Piper-format VITS, Kokoro, Matcha-en
and KittenTTS voices get their phonemes from :mod:`bolo.text.phonemize`
(uccharan, Apache-2.0, where it covers the language; otherwise an espeak-ng
the user installs). See docs/models.md, "Licences".
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import tempfile
import threading
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bolo import ort_providers
from bolo.engines.base import (
    DenoiseModel,
    DenoiseStream,
    DiarizationModel,
    Engine,
    EngineStatus,
    FloatArray,
    LoadOptions,
    Model,
    PunctuationModel,
    Segment,
    SeparationModel,
    SpeakerEmbeddingModel,
    SpeakerTurn,
    StreamResult,
    STTModel,
    STTRequest,
    STTResult,
    STTStream,
    TTSModel,
    VadDetector,
    VadEvent,
    VADModel,
    VadParams,
    Voice,
    Word,
)
from bolo.engines.blend import Blend, looks_like_blend, mix_styles, resolve_blend
from bolo.ort_providers import sherpa_provider

OFFLINE_KINDS = (
    "whisper",
    "moonshine",
    "moonshine_v2",
    "transducer",
    "nemo_transducer",
    "nemo_ctc",
    "zipformer_ctc",
    "sense_voice",
    "paraformer",
    "dolphin_ctc",
    "omnilingual_ctc",
    "telespeech_ctc",
    "wenet_ctc",
    "canary",
)
ONLINE_KINDS = (
    "online_transducer",
    "online_nemo_ctc",
    "online_zipformer2_ctc",
    "online_paraformer",
    "online_t_one_ctc",
)

log = logging.getLogger(__name__)
TTS_KINDS = ("vits", "kokoro", "matcha", "kitten")
VAD_KINDS = ("silero_vad", "ten_vad")
DIARIZATION_KINDS = ("pyannote_diarization",)
SEPARATION_KINDS = ("voxlock",)
DENOISE_KINDS = ("gtcrn", "dpdfnet")
SPEAKER_EMBEDDING_KINDS = ("speaker_embedding",)
AUDIO_TAGGING_KINDS = ("audio_tagging",)
#: punctuation restoration: CT-Transformer (Chinese and English, punctuation) and a CNN-BiLSTM (English, punctuation
#: and capital letters)
PUNCTUATION_KINDS = ("punct_ct_transformer", "punct_cnn_bilstm")
KWS_KINDS = ("sherpa_kws",)


def find_file(path: Path, *patterns: str, required: bool = True, prefer_int8: bool = True) -> str:
    """First file under ``path`` matching any pattern; int8 exports win when both exist."""
    for pattern in patterns:
        matches = sorted(p for p in path.glob(pattern) if p.is_file())
        if not matches:
            continue
        if prefer_int8:
            int8 = [m for m in matches if ".int8." in m.name or "quant" in m.name]
            if int8:
                return str(int8[0])
        return str(matches[0])
    if required:
        raise FileNotFoundError(f"no file matching {' or '.join(patterns)} in {path}")
    return ""


def find_dir(path: Path, name: str) -> str:
    d = path / name
    return str(d) if d.is_dir() else ""


def tokens_to_words(
    tokens: list[str], starts: list[float], end: float, log_probs: list[float] | None = None
) -> list[Word]:
    """Merge sub-word tokens with start times into words.

    Handles SentencePiece (``▁`` marks a word start), Whisper/BPE tokens
    (leading space) and unsegmented scripts (each CJK character is a word).
    With ``log_probs`` (one per token, as sherpa-onnx's ``ys_log_probs``), a word's ``p`` is the mean
    probability of its tokens.
    """
    words: list[Word] = []
    current = ""
    t0 = 0.0
    n = min(len(tokens), len(starts))
    if log_probs is not None and len(log_probs) < n:
        log_probs = None
    probs: list[float] = []

    def is_cjk(ch: str) -> bool:
        return "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯"

    def flush(until: float) -> None:
        nonlocal current
        text = current.strip()
        if text:
            p = round(sum(probs) / len(probs), 4) if probs else None
            words.append(Word(text, t0, max(until, t0), p))
        current = ""
        probs.clear()

    for i in range(n):
        tok = tokens[i]
        if not tok or (tok.startswith("<") and tok.endswith(">")):
            continue
        starts_word = tok.startswith(("▁", " ")) or (len(tok.strip()) == 1 and is_cjk(tok.strip()))
        if (starts_word or (current and is_cjk(current.strip()[-1:]))) and current.strip():
            flush(starts[i])
        if not current.strip():
            t0 = starts[i]
        current += tok.replace("▁", " ")
        if log_probs is not None and tok.replace("▁", "").strip():
            probs.append(math.exp(min(0.0, float(log_probs[i]))))
    flush(end)
    for i in range(len(words) - 1):  # close each word where the next begins
        words[i].t1 = max(words[i].t0, min(words[i].t1, words[i + 1].t0))
    return words


#: SenseVoice's emotion tags, as bolo names them (EMO_UNKNOWN: none)
SENSE_VOICE_EMOTIONS = {
    "HAPPY": "happy", "SAD": "sad", "ANGRY": "angry", "NEUTRAL": "neutral", "FEARFUL": "fearful",
    "DISGUSTED": "disgusted", "SURPRISED": "surprised",
}  # fmt: skip
#: SenseVoice's audio-event tags, as bolo names them (Event_UNK: none)
SENSE_VOICE_EVENTS = {
    "Speech": "speech", "BGM": "music", "Applause": "applause", "Laughter": "laughter", "Cry": "crying",
    "Sneeze": "sneeze", "Breath": "breath", "Cough": "cough",
}  # fmt: skip


def sense_voice_tags(emotion: str, event: str) -> tuple[str | None, list[str]]:
    """SenseVoice's ``<|HAPPY|>`` and ``<|Laughter|>`` tags as (emotion, events); unknown tags are left out."""
    emo = SENSE_VOICE_EMOTIONS.get(emotion.strip().strip("<|>"))
    events = [
        SENSE_VOICE_EVENTS[t] for t in event.replace("><", ">|<").split("|") if t.strip("<>") in SENSE_VOICE_EVENTS
    ]
    return emo, events


def _log_probs(res: Any, n: int) -> list[float] | None:
    """The log probability of each token of a result (``ys_log_probs``), when the decoder gave one per token."""
    try:
        values = [float(x) for x in getattr(res, "ys_log_probs", None) or []]
    except (TypeError, ValueError):
        return None
    return values if len(values) >= n else None


_SENSE_VOICE_LANGS = {"<|zh|>": "zh", "<|en|>": "en", "<|ja|>": "ja", "<|ko|>": "ko", "<|yue|>": "yue"}


def line_is_blank_only(symbols: set[str]) -> bool:
    """True when a token table holds no usable characters (then it is not a character table)."""
    return not any(s.strip() for s in symbols)


#: Whisper hears 30 s at a time; sherpa-onnx counts the audio it is given to set the token limit (6 a second), so
#: pieces are padded to just under that (from 29.5 s on it warns that it drops the rest)
WHISPER_PAD_S = 29.0
#: the token limit of a padded piece (6 per second of padded audio, in whole seconds)
WHISPER_TOKEN_CAP = int(WHISPER_PAD_S) * 6
#: a piece whose text comes this close to the limit is taken to have reached it (the count of
#: :func:`whisper_token_count` can be one or two off Whisper's own)
WHISPER_CAP_MARGIN = 4
#: a piece shorter than twice this is not split again, even when its text fills the limit
WHISPER_MIN_PIECE_S = 2.0


def _char_class(ch: str) -> str:
    cat = unicodedata.category(ch)
    if ch.isspace():
        return "S"
    return "L" if cat[0] == "L" else "N" if cat[0] == "N" else "O"


def whisper_pieces(text: str) -> list[str]:
    """``text`` cut the way Whisper's (GPT-2) pre-tokeniser cuts it before BPE: an optional space, then a run of
    letters, of digits or of other marks. Vowel signs and viramas of Indic scripts are not letters there, so a
    Telugu or Bengali word falls into several pieces, which is part of why it costs so many tokens."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1 if text[i] == " " and i + 1 < n and not text[i + 1].isspace() else i
        kind = _char_class(text[j])
        end = j + 1
        while end < n and _char_class(text[end]) == kind:
            end += 1
        out.append(text[i:end])
        i = end
    return out


def whisper_token_count(text: str, ranks: dict[bytes, int]) -> int:
    """How many tokens Whisper's byte-level BPE (``ranks``: token bytes -> id, which is its merge rank) spends on
    ``text``: the pieces' bytes merged pairwise, the lowest-ranked pair first."""
    total = 0
    for piece in whisper_pieces(text):
        parts = [bytes([b]) for b in piece.encode("utf-8")]
        while len(parts) > 1:
            best: tuple[int, int] | None = None
            for i in range(len(parts) - 1):
                rank = ranks.get(parts[i] + parts[i + 1])
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, i)
            if best is None:
                break
            i = best[1]
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        total += len(parts)
    return total


def whisper_ranks(tokens: str) -> dict[bytes, int]:
    """A sherpa-onnx Whisper token table (``<base64 bytes> <id>`` a line) as token bytes -> id."""
    import base64
    import binascii

    ranks: dict[bytes, int] = {}
    for line in Path(tokens).read_text("utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        try:
            ranks[base64.b64decode(parts[0], validate=True)] = int(parts[1])
        except (binascii.Error, ValueError):
            continue
    return ranks


#: a run of 1 to 12 characters said five times or more back to back ("বে বে বে ...", "ల్ల్ల్ల్..."): greedy decoding
#: that has fallen into a loop, which sherpa-onnx cannot retry at a higher temperature (bolo's repetition guard
#: allows four copies of a phrase, too)
_LOOP = re.compile(r"(\S.{0,11}?)(?:\s*\1){4,}")


def cut_loops(text: str) -> str:
    """``text`` with each loop cut to one copy of what repeats ("న్న్న్న్..." -> "న్"): what is left when a piece too
    short to split again still loops (bolo's repetition guard catches repeated words, not syllables)."""
    return _LOOP.sub(lambda m: m.group(1) if any(ch.isalpha() for ch in m.group(1)) else m.group(0), text)


def looping(text: str) -> bool:
    """Whether ``text`` has such a loop (of something with a letter in it: "2000000" is a number)."""
    return any(any(ch.isalpha() for ch in m.group(1)) for m in _LOOP.finditer(text))


def quietest_point(samples: FloatArray, a: int, b: int, sr: int) -> int:
    """The sample in the middle half of ``samples[a:b]`` where 20 ms of audio is quietest: where to cut it."""
    frame = max(1, int(0.02 * sr))
    lo, hi = a + (b - a) // 4, a + 3 * (b - a) // 4
    n = (hi - lo) // frame
    if n < 2:
        return (a + b) // 2
    frames = np.asarray(samples[lo : lo + n * frame], dtype=np.float32).reshape(n, frame)
    return int(lo + int(np.argmin((frames**2).mean(axis=1))) * frame + frame // 2)


#: recognisers whose sherpa-onnx decoder takes hotwords (modified beam search with a context graph)
HOTWORD_KINDS = ("transducer", "nemo_transducer")
#: how much a hotword's tokens are favoured (sherpa-onnx's hotwords_score); a phrase can set its own ("Zintrava :3")
HOTWORDS_SCORE = 4.0


def hotword_phrases(text: str | None) -> list[str]:
    """The phrases of a request's ``hotwords``: separated by commas, semicolons, slashes or new lines."""
    return [p.strip() for p in re.split(r"[,;/\n]+", text or "") if p.strip()]


def token_case(tokens: str) -> str:
    """``upper`` when a model's token table spells words in capitals only (most English Zipformers), ``lower`` when
    in small letters only, else ``mixed``: hotwords are written the way the model writes."""
    pieces = (line.rpartition(" ")[0] for line in Path(tokens).read_text("utf-8", errors="replace").splitlines())
    letters = "".join(p for p in pieces if not (p.startswith("<") and p.endswith(">")))  # not <blk>, <unk>
    upper = sum(ch.isupper() for ch in letters)
    lower = sum(ch.islower() for ch in letters)
    if upper and not lower:
        return "upper"
    if lower and not upper:
        return "lower"
    return "mixed"


def bpe_vocab_from_tokens(tokens: str, dest: Path, spm_model: str = "") -> str:
    """A SentencePiece vocabulary (``piece<TAB>score``) for sherpa-onnx to split hotwords into the model's tokens.

    From the model's own ``bpe.model`` when it has one (the scores of a unigram model are what makes the split
    right: with made-up ones the hotwords of the GigaSpeech Zipformer were never matched); otherwise from
    ``tokens.txt``, a token's id standing for its merge rank, which is right for SentencePiece BPE models."""
    pieces = sentencepiece_pieces(Path(spm_model)) if spm_model else []
    if not pieces:
        for line in Path(tokens).read_text("utf-8").splitlines():
            piece, _, index = line.rpartition(" ")
            if piece and index.strip().lstrip("-").isdigit():
                pieces.append((piece, -float(index)))
    dest.write_text("".join(f"{piece}\t{score:.6g}\n" for piece, score in pieces), "utf-8")
    return str(dest)


def _varint(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _proto_fields(data: bytes) -> Iterator[tuple[int, int, Any]]:
    """The (field, wire type, value) of a protobuf message: enough to read a SentencePiece model."""
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        value: Any
        if wire == 0:
            value, i = _varint(data, i)
        elif wire == 1:
            value, i = data[i : i + 8], i + 8
        elif wire == 2:
            size, i = _varint(data, i)
            value, i = data[i : i + size], i + size
        elif wire == 5:
            value, i = data[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def sentencepiece_pieces(path: Path) -> list[tuple[str, float]]:
    """The pieces and scores of a SentencePiece model file (``ModelProto.pieces``), read without sentencepiece."""
    import struct

    out: list[tuple[str, float]] = []
    try:
        for number, wire, value in _proto_fields(path.read_bytes()):
            if number != 1 or wire != 2:
                continue
            piece, score = "", 0.0
            for n2, w2, v2 in _proto_fields(value):
                if n2 == 1 and w2 == 2:
                    piece = v2.decode("utf-8")
                elif n2 == 2 and w2 == 5:
                    score = struct.unpack("<f", v2)[0]
            if piece:
                out.append((piece, score))
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        return []
    return out


def hotwords_for(text: str | None, case: str) -> str:
    """A request's ``hotwords`` as sherpa-onnx takes them for a stream: phrases joined by ``/``, in the model's case."""
    phrases = hotword_phrases(text)
    if case == "upper":
        phrases = [_case_phrase(p, str.upper) for p in phrases]
    elif case == "lower":
        phrases = [_case_phrase(p, str.lower) for p in phrases]
    return "/".join(phrases)


def _case_phrase(phrase: str, change: Callable[[str], str]) -> str:
    words, _, score = phrase.partition(" :")  # "Zintrava Labs :3" keeps its score as it is
    return change(words) + (f" :{score}" if score else "")


def whisper_decoder(path: Path, options: dict[str, Any]) -> str:
    """The Whisper decoder to load: the int8 one, unless the catalog entry names a float one for CPUs where int8
    matrix products may saturate (``options.float_decoder_without_vnni``, a file pattern) and this is such a CPU."""
    from bolo import hardware

    int8 = find_file(path, "*decoder*.onnx")
    pattern = options.get("float_decoder_without_vnni")
    if not pattern or not hardware.int8_may_saturate(hardware.this_cpu()):
        return int8
    plain = sorted(f for f in path.glob(str(pattern)) if f.is_file() and ".int8." not in f.name)
    if plain:
        return str(plain[0])
    log.warning(
        "%s: this CPU has no AVX-512 VNNI, where the int8 decoder can give wrong text in some languages; pull the "
        "model again (`bolo pull`) to fetch its float32 decoder",
        path.name,
    )
    return int8


class SherpaOfflineSTT(STTModel):
    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        self._so = sherpa_onnx
        self.kind = kind
        self.path = path
        self.spec = spec
        self.threads = opts.threads
        self.provider = sherpa_provider(opts.device)
        self.max_batch = int(spec.get("max_batch", 8))
        self.max_chunk_s = float(spec.get("max_chunk_s", 30.0 if kind in ("whisper", "canary") else 60.0))
        self.supports_translate = (kind == "whisper" and not spec.get("english_only", False)) or kind == "canary"
        if kind == "canary":
            # Canary translates from English into each of its languages and from each of them into English
            self.translate_targets = tuple(str(x) for x in spec.get("translate_into") or ("en",))
        # The stock Whisper exports have no cross-attention outputs, so no token times.
        # Whisper exports give token times only when they were exported with cross-attention outputs
        # (sherpa-onnx scripts/whisper/export-onnx-with-attention.py); such a model says so with the catalog
        # option `token_timestamps: true`. Moonshine has no token times in this runtime.
        self.token_timestamps = kind == "whisper" and bool(spec.get("token_timestamps", False))
        self.supports_word_timestamps = self.token_timestamps or kind not in (
            "whisper", "moonshine", "moonshine_v2", "canary", "paraformer",
        )  # fmt: skip
        self.default_language = str(spec.get("language", ""))
        self._case = "mixed"
        self._ranks: dict[bytes, int] | None = None
        # hotwords: a transducer decodes them with modified beam search over a context graph of their tokens
        # (a second recogniser, built the first time a request has hotwords; greedy search stays the default)
        self.hotwords = kind in HOTWORD_KINDS and spec.get("hotwords", True) is not False
        self.request_fields = ("hotwords",) if self.hotwords else ()
        self._vocab_dir: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.Lock()
        self._recognizers: dict[tuple[str, str, bool], Any] = {}
        self._get(self.default_language, "transcribe")

    def _hotword_config(self) -> dict[str, Any]:
        """What a transducer needs to take hotwords: beam search, and how to split them into its tokens."""
        tokens = find_file(self.path, "tokens.txt")
        unit = str(self.spec.get("modeling_unit", "bpe"))
        config: dict[str, Any] = {
            "decoding_method": "modified_beam_search",
            "max_active_paths": int(self.spec.get("max_active_paths", 4)),
            "hotwords_score": float(self.spec.get("hotwords_score", HOTWORDS_SCORE)),
            "modeling_unit": unit,
        }
        if "bpe" in unit:
            if self._vocab_dir is None:
                self._vocab_dir = tempfile.TemporaryDirectory(prefix="bolo-hotwords-")
            spm = find_file(self.path, "bpe.model", required=False)
            config["bpe_vocab"] = bpe_vocab_from_tokens(tokens, Path(self._vocab_dir.name) / "bpe.vocab", spm)
        self._case = token_case(tokens)
        return config

    def _build(self, language: str, task: str, hot: bool = False) -> Any:
        so, p, t = self._so, self.path, self.threads
        common: dict[str, Any] = {"num_threads": t, "provider": self.provider}
        k = self.kind
        if k == "whisper":
            return so.OfflineRecognizer.from_whisper(
                encoder=find_file(p, "*encoder*.onnx"),
                decoder=whisper_decoder(p, self.spec),
                tokens=find_file(p, "*tokens.txt"),
                language=language,
                task=task,
                enable_token_timestamps=self.token_timestamps,
                **common,
            )
        if k == "moonshine":
            return so.OfflineRecognizer.from_moonshine(
                preprocessor=find_file(p, "preprocess*.onnx"),
                encoder=find_file(p, "encode*.onnx"),
                uncached_decoder=find_file(p, "uncached_decode*.onnx"),
                cached_decoder=find_file(p, "cached_decode*.onnx"),
                tokens=find_file(p, "tokens.txt"),
                **common,
            )
        if k == "moonshine_v2":
            return so.OfflineRecognizer.from_moonshine_v2(
                encoder=find_file(p, "encoder_model*.ort", "encoder_model*.onnx"),
                decoder=find_file(p, "decoder_model_merged*.ort", "decoder_model_merged*.onnx"),
                tokens=find_file(p, "tokens.txt"),
                **common,
            )
        if k in ("transducer", "nemo_transducer"):
            return so.OfflineRecognizer.from_transducer(
                encoder=find_file(p, "encoder*.onnx"),
                decoder=find_file(p, "decoder*.onnx"),
                joiner=find_file(p, "joiner*.onnx"),
                tokens=find_file(p, "tokens.txt"),
                model_type="nemo_transducer" if k == "nemo_transducer" else "transducer",
                **(self._hotword_config() if hot else {}),
                **common,
            )
        if k == "canary":
            return so.OfflineRecognizer.from_nemo_canary(
                encoder=find_file(p, "encoder*.onnx"),
                decoder=find_file(p, "decoder*.onnx"),
                tokens=find_file(p, "tokens.txt"),
                src_lang=language or "en",
                # "translate:<target>" for a translation, else the transcript is in the language spoken
                tgt_lang=task.partition(":")[2] if task.startswith("translate:") else language or "en",
                **common,
            )
        model = find_file(p, "model*.onnx")
        tokens = find_file(p, "tokens.txt")
        if k == "nemo_ctc":
            return so.OfflineRecognizer.from_nemo_ctc(model=model, tokens=tokens, **common)
        if k == "zipformer_ctc":
            return so.OfflineRecognizer.from_zipformer_ctc(model=model, tokens=tokens, **common)
        if k == "sense_voice":
            return so.OfflineRecognizer.from_sense_voice(
                model=model, tokens=tokens, language=language or "auto", use_itn=True, **common
            )
        if k == "paraformer":
            return so.OfflineRecognizer.from_paraformer(paraformer=model, tokens=tokens, **common)
        if k == "dolphin_ctc":
            return so.OfflineRecognizer.from_dolphin_ctc(model=model, tokens=tokens, **common)
        if k == "omnilingual_ctc":
            return so.OfflineRecognizer.from_omnilingual_asr_ctc(model=model, tokens=tokens, **common)
        if k == "telespeech_ctc":
            return so.OfflineRecognizer.from_telespeech_ctc(model=model, tokens=tokens, **common)
        if k == "wenet_ctc":
            return so.OfflineRecognizer.from_wenet_ctc(model=model, tokens=tokens, **common)
        raise ValueError(f"sherpa-onnx engine cannot load kind {k!r}")

    def _get(self, language: str, task: str, hot: bool = False) -> Any:
        """Recognisers are built per (language, task) for models that fix them at construction, and once more for
        hotwords (``hot``)."""
        if self.kind not in ("whisper", "sense_voice", "canary"):
            language, task = "", "transcribe"
        key = (language, task, hot)
        with self._lock:
            rec = self._recognizers.get(key)
            if rec is None:
                if len(self._recognizers) >= 3:  # bound memory: drop the oldest non-default variant
                    for old in list(self._recognizers):
                        if old != (self.default_language, "transcribe", False):
                            del self._recognizers[old]
                            break
                rec = self._recognizers[key] = self._build(language, task, hot)
            return rec

    def translation_problem(self, source: str | None, target: str) -> str | None:
        if self.kind != "canary":
            return None
        source = (source or "").split("-")[0].lower()
        if not source:
            return "Canary does not detect the spoken language: give `language` to translate"
        if source == target:
            return f"the audio is already in {target}: nothing to translate"
        if "en" not in (source, target):
            return f"Canary translates into English or from English, not from {source} into {target}"
        return None

    def _canary_target(self, source: str, target: str | None) -> str:
        """The language Canary translates ``source`` into (checked by :meth:`translation_problem` first)."""
        target = (target or "en").split("-")[0].lower()
        problem = self.translation_problem(source, target)
        if problem:
            raise ValueError(problem)
        return target

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        results: list[STTResult | None] = [None] * len(requests)
        groups: dict[tuple[str, str, bool], list[int]] = {}
        for i, r in enumerate(requests):
            lang = (r.language or self.default_language or "").split("-")[0].lower()
            hot = self.hotwords and bool(hotword_phrases(r.hotwords))
            task = r.task
            if self.kind == "canary" and task == "translate":
                task = f"translate:{self._canary_target(lang, r.target_language)}"
            groups.setdefault((lang, task, hot), []).append(i)
        for (lang, task, hot), idxs in groups.items():
            rec = self._get(lang, task, hot)
            if self.kind == "whisper":
                for i in idxs:
                    results[i] = self._whisper(rec, requests[i], lang)
                continue
            streams = []
            for i in idxs:
                s = rec.create_stream(hotwords_for(requests[i].hotwords, self._case)) if hot else rec.create_stream()
                s.accept_waveform(self.sample_rate, requests[i].samples)
                streams.append(s)
            rec.decode_streams(streams)
            for i, s in zip(idxs, streams):
                results[i] = self._result(s.result, requests[i], lang)
        return [r for r in results if r is not None]

    def _whisper_ranks(self) -> dict[bytes, int]:
        """Whisper's token table, read the first time a token count is needed (see :meth:`_whisper`)."""
        if self._ranks is None:
            self._ranks = whisper_ranks(find_file(self.path, "*tokens.txt"))
        return self._ranks

    def _whisper(self, rec: Any, req: STTRequest, lang: str) -> STTResult:
        """Whisper through sherpa-onnx, given room for every token the speech needs.

        sherpa-onnx's Whisper decoder stops after 6 tokens per second of audio it was given (and 224 in all).
        Whisper spends far more tokens than that on scripts its byte-level vocabulary splits finely: Tamil,
        Telugu, Bengali and others take 15 to 20 a second, so their text was cut off part-way through the
        sentence. Each piece is padded to just under 30 s of audio, which raises the limit to
        :data:`WHISPER_TOKEN_CAP` tokens; a piece whose text still reaches the limit is split at its quietest
        point and each half decoded again (down to :data:`WHISPER_MIN_PIECE_S`). When sherpa-onnx cannot hand the
        tokens over (one of them holds part of a character), they are counted by encoding the text again with
        Whisper's BPE (:func:`whisper_token_count`): Telugu and Bengali spend 1.4 to 2 bytes of UTF-8 a token, so
        an estimate from the length of the text missed that the limit was reached.
        """
        sr = self.sample_rate
        pieces: list[tuple[str, list[str], list[float], float]] = []  # text, tokens, token starts (s), offset
        detected = ""

        def decode(a: int, b: int) -> None:
            nonlocal detected
            if req.abort.is_set():
                return
            audio = np.asarray(req.samples[a:b], dtype=np.float32)
            padded = np.zeros(max(len(audio), int(WHISPER_PAD_S * sr)), dtype=np.float32)
            padded[: len(audio)] = audio
            stream = rec.create_stream()
            stream.accept_waveform(sr, padded)
            rec.decode_stream(stream)
            res = stream.result
            try:
                tokens = [str(t) for t in res.tokens]
                count = len(tokens)
            except UnicodeDecodeError:  # a token holding part of a character (Whisper's byte-level vocabulary)
                tokens = []
                count = whisper_token_count(str(res.text), self._whisper_ranks())
            full = count >= WHISPER_TOKEN_CAP - WHISPER_CAP_MARGIN
            if (full or looping(str(res.text))) and (b - a) >= 2 * WHISPER_MIN_PIECE_S * sr:
                cut = quietest_point(req.samples, a, b, sr)
                decode(a, cut)
                decode(cut, b)
                return
            detected = detected or str(getattr(res, "lang", "") or "")
            starts = [float(t) for t in getattr(res, "timestamps", [])]
            pieces.append((cut_loops(str(res.text)).strip(), tokens, starts, a / sr))

        decode(0, len(req.samples))
        text = " ".join(p[0] for p in pieces if p[0]).strip()
        duration = len(req.samples) / sr
        words: list[Word] = []
        if req.word_timestamps and self.supports_word_timestamps:
            for i, (_, tokens, starts, offset) in enumerate(pieces):
                end = pieces[i + 1][3] if i + 1 < len(pieces) else duration
                if tokens and len(starts) >= len(tokens):
                    words += tokens_to_words(tokens, [offset + t for t in starts], end)
        language = (detected.strip("<|>") or lang) or None
        return STTResult(text, language, [Segment(0.0, duration, text, words)] if text else [], words)

    def _result(self, res: Any, req: STTRequest, lang: str) -> STTResult:
        text = str(res.text).strip()
        duration = len(req.samples) / self.sample_rate
        detected = lang or None
        raw_lang = str(getattr(res, "lang", "") or "")
        if raw_lang:
            detected = _SENSE_VOICE_LANGS.get(raw_lang, raw_lang.strip("<|>")) or detected
        words: list[Word] = []
        if req.word_timestamps and self.supports_word_timestamps:
            tokens = [str(t) for t in res.tokens]
            starts = [float(t) for t in res.timestamps]
            if tokens and len(starts) >= len(tokens):
                words = tokens_to_words(tokens, starts, duration, _log_probs(res, len(tokens)))
        seg = Segment(0.0, duration, text, words)
        if self.kind == "sense_voice":
            seg.emotion, seg.events = sense_voice_tags(
                str(getattr(res, "emotion", "") or ""), str(getattr(res, "event", "") or "")
            )
        return STTResult(text, detected, [seg] if text else [], words)

    def close(self) -> None:
        self._recognizers.clear()
        if self._vocab_dir is not None:
            self._vocab_dir.cleanup()
            self._vocab_dir = None


PRIME_S = 1.0
_PRIME_NOISE = (np.random.default_rng(0).standard_normal(int(PRIME_S * 16000)) * 1e-4).astype(np.float32)
_PRIME_SILENCE = np.zeros(int(PRIME_S * 16000), dtype=np.float32)


class _SherpaStream(STTStream):
    def __init__(self, model: SherpaOnlineSTT, language: str | None = None) -> None:
        self.model = model
        self.language = language
        # Streaming encoders need some left context: without it the first words of audio that
        # starts abruptly (a file, a push-to-talk clip) are often dropped. A second of very
        # faint noise works where digital silence does not (measured on the 20M Zipformer).
        self._open(_PRIME_NOISE)

    def restart(self) -> None:
        """A new native stream (sherpa-onnx's ``reset`` keeps the encoder's state and its view of the audio).

        A live session restarts the stream for each utterance and gives it real audio from before the speech (the
        pre-roll); with that, a second of digital silence ahead of it keeps more first words than faint noise
        (simulated microphone takes on the 20M Zipformer: first word right in 13 of 24 rather than 8, word error
        rate 0.10 rather than 0.14)."""
        self._open(_PRIME_SILENCE)

    def _open(self, prime: FloatArray) -> None:
        self.stream = self.model.recognizer.create_stream()
        if self.language:  # must be set before any audio is accepted
            try:
                self.stream.set_option("language", self.language)
            except (AttributeError, RuntimeError):  # an older sherpa-onnx: the model then detects the language
                pass
        self.offset = PRIME_S
        self.stream.accept_waveform(self.model.sample_rate, prime)

    def accept(self, samples: FloatArray) -> None:
        self.stream.accept_waveform(self.model.sample_rate, samples)

    def finish(self) -> None:
        self.stream.input_finished()

    def result(self) -> StreamResult:
        rec = self.model.recognizer
        text = str(rec.get_result(self.stream)).strip()
        if self.model.lowercase and text.isupper():
            text = text.lower()
        words: list[Word] = []
        try:
            tokens = [str(t) for t in rec.tokens(self.stream)]
            starts = [float(t) for t in rec.timestamps(self.stream)]
            if tokens and len(starts) >= len(tokens):
                starts = [max(0.0, t - self.offset) for t in starts]
                words = tokens_to_words(tokens, starts, starts[-1] + 0.2)
                if self.model.lowercase:
                    for w in words:
                        if w.w.isupper():
                            w.w = w.w.lower()
        except Exception:
            words = []
        return StreamResult(text, words)

    def is_endpoint(self) -> bool:
        return bool(self.model.recognizer.is_endpoint(self.stream))

    def reset(self) -> None:
        self.model.recognizer.reset(self.stream)
        self.offset = 0.0


class SherpaOnlineSTT(STTModel):
    streaming = True
    supports_word_timestamps = True

    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        so, p = sherpa_onnx, path
        self.kind = kind
        self.max_batch = int(spec.get("max_batch", 8))
        self.max_chunk_s = float(spec.get("max_chunk_s", 60.0))
        self.lowercase = bool(spec.get("lowercase", True))
        #: prompt-conditioned models (Nemotron) take the language per stream; without it they detect it
        self.stream_language = bool(spec.get("stream_language", False))
        # Live sessions with a VAD give the recogniser speech only, each utterance a fresh stream with this much of
        # what came before; its own endpointing reset the stream mid-word when it had not decoded anything for 2.4 s.
        self.stream_preroll_s = float(opts.options.get("stream_preroll_s", spec.get("stream_preroll_s", 0.5)))
        # Browsers' automatic gain control leaves speech around -30 dBFS; these models were trained on louder audio.
        level = opts.options.get("stream_speech_dbfs", spec.get("stream_speech_dbfs", -20.0))
        self.stream_speech_dbfs = None if level is None else float(level)
        # the encoder's chunk (decode_chunk_len in a Zipformer's metadata, at 10 ms a frame) for the catalog to give
        self.stream_chunk_s = float(opts.options.get("stream_chunk_s", spec.get("stream_chunk_s", 0.0)))
        self.stream_primed_s = PRIME_S  # restart() gives a fresh stream this much digital silence first
        endpoint: dict[str, Any] = {
            "enable_endpoint_detection": True,
            "rule1_min_trailing_silence": 2.4,
            "rule2_min_trailing_silence": float(
                opts.options.get("endpoint_silence_s", spec.get("endpoint_silence_s", 0.8))
            ),
            "rule3_min_utterance_length": float(opts.options.get("max_utterance_s", 20.0)),
        }
        common: dict[str, Any] = {
            "num_threads": opts.threads,
            "provider": sherpa_provider(opts.device),
            **endpoint,
        }
        tokens = find_file(p, "tokens.txt")
        if kind == "online_transducer":
            # Beam search keeps a transducer from losing whole phrases where greedy search commits to blanks early
            # (the 20M Zipformer: "Speech servers" came out as "ers", "Good morning" as "in").
            decoding = str(opts.options.get("decoding_method", spec.get("decoding_method", "greedy_search")))
            self.recognizer = so.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=find_file(p, "encoder*.onnx"),
                decoder=find_file(p, "decoder*.onnx"),
                joiner=find_file(p, "joiner*.onnx"),
                decoding_method=decoding,
                max_active_paths=int(opts.options.get("max_active_paths", spec.get("max_active_paths", 4))),
                **common,
            )
        elif kind == "online_nemo_ctc":
            self.recognizer = so.OnlineRecognizer.from_nemo_ctc(
                tokens=tokens, model=find_file(p, "model*.onnx"), **common
            )
        elif kind == "online_zipformer2_ctc":
            self.recognizer = so.OnlineRecognizer.from_zipformer2_ctc(
                tokens=tokens, model=find_file(p, "*model*.onnx", "*.onnx"), **common
            )
        elif kind == "online_paraformer":
            self.recognizer = so.OnlineRecognizer.from_paraformer(
                tokens=tokens, encoder=find_file(p, "encoder*.onnx"), decoder=find_file(p, "decoder*.onnx"), **common
            )
        elif kind == "online_t_one_ctc":
            self.recognizer = so.OnlineRecognizer.from_t_one_ctc(
                tokens=tokens, model=find_file(p, "model*.onnx"), **common
            )
        else:
            raise ValueError(f"sherpa-onnx engine cannot load kind {kind!r}")

    def create_stream(self, language: str | None = None) -> STTStream:
        if self.stream_language and language and language != "auto":
            return _SherpaStream(self, language.split("-")[0].lower())
        return _SherpaStream(self)

    def decode_streams(self, streams: list[STTStream]) -> None:
        rec = self.recognizer
        native = [s.stream for s in streams if isinstance(s, _SherpaStream)]
        while True:
            ready = [s for s in native if rec.is_ready(s)]
            if not ready:
                return
            rec.decode_streams(ready)

    def transcribe(self, requests: list[STTRequest]) -> list[STTResult]:
        """Whole-utterance decoding with the streaming model (used by the HTTP endpoints)."""
        streams: list[STTStream] = []
        for r in requests:
            s = self.create_stream(r.language)
            s.accept(r.samples)
            s.accept(np.zeros(int(0.6 * self.sample_rate), dtype=np.float32))
            s.finish()
            streams.append(s)
        self.decode_streams(streams)
        out: list[STTResult] = []
        for r, s in zip(requests, streams):
            res = s.result()
            duration = len(r.samples) / self.sample_rate
            words = res.words if r.word_timestamps else []
            out.append(
                STTResult(res.text, r.language, [Segment(0.0, duration, res.text, words)] if res.text else [], words)
            )
        return out


# Kokoro voice-name prefix -> phonemiser language. Mandarin ("z") always uses the bundled lexicon.
_KOKORO_ESPEAK = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br", "j": "ja"}
_KOKORO_LEXICON_OK = ("a", "b", "z")


class SherpaTTS(TTSModel):
    """Text-to-speech through ``sherpa_onnx.OfflineTts``.

    sherpa-onnx 2.x has no built-in phonemiser. Voices trained on phonemes get
    them from :mod:`bolo.text.phonemize` (uccharan or espeak-ng); lexicon- and
    character-based voices need nothing.
    """

    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        self._so = sherpa_onnx
        self.kind = kind
        self.path = path
        self.spec = spec
        so, p = sherpa_onnx, path
        cfg: dict[str, Any] = {}
        lexicon = find_file(p, "lexicon.txt", required=False)
        dict_dir = find_dir(p, "dict")
        self.has_lexicon = bool(lexicon)
        if kind == "vits":
            cfg["vits"] = so.OfflineTtsVitsModelConfig(
                model=find_file(p, "*.onnx", prefer_int8=False),
                lexicon=lexicon,
                tokens=find_file(p, "tokens.txt"),
                dict_dir=dict_dir,
            )
        elif kind == "kokoro":
            names = spec.get("lexicons") or [f.name for f in sorted(p.glob("lexicon*.txt"))]
            lexicons = ",".join(str(p / n) for n in names if (p / n).is_file())
            self.has_lexicon = bool(lexicons)
            cfg["kokoro"] = so.OfflineTtsKokoroModelConfig(
                model=find_file(p, "model*.onnx"),
                voices=find_file(p, "voices.bin"),
                tokens=find_file(p, "tokens.txt"),
                lexicon=lexicons,
            )
        elif kind == "kitten":
            cfg["kitten"] = so.OfflineTtsKittenModelConfig(
                model=find_file(p, "model*.onnx"), voices=find_file(p, "voices.bin"), tokens=find_file(p, "tokens.txt")
            )
        elif kind == "matcha":
            cfg["matcha"] = so.OfflineTtsMatchaModelConfig(
                acoustic_model=find_file(p, "model-steps-3.onnx", "model*.onnx"),
                vocoder=find_file(p, "vocos*.onnx", "hifigan*.onnx", "*voc*.onnx"),
                lexicon=lexicon,
                tokens=find_file(p, "tokens.txt"),
                dict_dir=dict_dir,
            )
        else:
            raise ValueError(f"sherpa-onnx engine cannot load TTS kind {kind!r}")
        rule_fsts = ",".join(str(f) for f in sorted(p.glob("*.fst")))
        provider = sherpa_provider(opts.device)

        def build(parts: dict[str, Any]) -> Any:
            model_cfg = so.OfflineTtsModelConfig(num_threads=opts.threads, provider=provider, **parts)
            config = so.OfflineTtsConfig(model=model_cfg, rule_fsts=rule_fsts, max_num_sentences=1)
            if not config.validate():
                raise ValueError(f"sherpa-onnx rejected the TTS configuration for {p.name}")
            return so.OfflineTts(config)

        self._tts = build(cfg)
        # Kokoro voice blends (bolo.engines.blend): sherpa-onnx takes a voice only as a row of its voices file, and
        # the file must hold exactly the model's number of voices. Blends therefore get a second instance whose
        # voices file holds the blends in use (up to that number, least recently used first out), rebuilt when a
        # new blend arrives. It exists only once a blend was asked for.
        self.supports_blending = kind == "kokoro"
        self._build = build
        self._cfg = cfg
        self._blend_lock = threading.Lock()
        self._blend_tts: Any = None
        self._blends: OrderedDict[str, FloatArray] = OrderedDict()
        self._blend_slots: dict[str, int] = {}
        self.accepts_phonemizer = True
        self.sample_rate = int(self._tts.sample_rate)
        self.num_speakers = int(self._tts.num_speakers)
        self.model_lang = str(getattr(self._tts, "lang", "") or spec.get("espeak_voice", ""))
        # Piper-format voices carry the espeak voice in their metadata and have no lexicon.
        self.needs_phonemes = bool(spec.get("phonemes", False)) or (
            (kind == "vits" and bool(self.model_lang) and not self.has_lexicon)
            or kind == "kitten"
            or (kind == "matcha" and not self.has_lexicon)
        )
        # A VITS voice with neither lexicon nor phonemes reads characters from a closed set (Coqui-style):
        # whatever is not in tokens.txt would be skipped silently, so the text is checked first (fit_text).
        self.charset: frozenset[str] | None = None
        if kind == "vits" and not self.has_lexicon and not self.needs_phonemes:
            symbols: set[str] = set()
            for line in Path(find_file(p, "tokens.txt")).read_text(encoding="utf-8").splitlines():
                symbol = line.rsplit(" ", 1)[0] if " " in line else line
                if len(symbol) == 1:
                    symbols.add(symbol)
            if not line_is_blank_only(symbols):
                self.charset = frozenset(symbols)
        self._numbers = str(spec["numbers"]) if spec.get("numbers") else None
        # Training-time text cleaning of a character voice: catalog option, else the voice.json bolo's exporters write.
        self.text_cleaner: str | None = spec.get("text_cleaner")
        voice_json = p / "voice.json"
        if self.text_cleaner is None and voice_json.is_file():
            with contextlib.suppress(ValueError, OSError):
                self.text_cleaner = json.loads(voice_json.read_text(encoding="utf-8")).get("cleaner")
        declared = [
            Voice(
                str(v["id"]),
                str(v.get("name", "")),
                str(v.get("language", "")),
                str(v.get("gender", "")),
                int(v.get("sid", i)),
            )
            for i, v in enumerate(spec.get("voices", []))
        ]
        if declared:
            self.voices = tuple(declared)
        elif self.num_speakers <= 1:
            self.voices = (Voice("default", "Default", str(spec.get("language", "")), "", 0),)
        else:
            self.voices = tuple(
                Voice(str(i), f"Speaker {i}", str(spec.get("language", "")), "", i)
                for i in range(min(self.num_speakers, 64))
            )

    def fit_text(self, text: str, language: str | None) -> str:
        from bolo.text.normalize import clean_for_voice, fit_to_charset

        if self.text_cleaner:
            text = clean_for_voice(text, self.text_cleaner, self.charset)
        if self.charset is None:
            return text
        return fit_to_charset(text, self.charset, language, self._numbers)

    def warm_up(self) -> None:
        if not self.spec.get("warm_up", False):
            return
        try:
            for _ in self.synthesize(str(self.spec.get("warm_up_text", "Hello.")), None, 1.0, None):
                pass
        except Exception:  # a voice that cannot say the warm-up text is still usable
            pass

    def phoneme_target(self, voice: Voice | None, choice: str | None = None) -> tuple[str, str] | None:
        """``(profile, language)`` to phonemise with, or None when the text goes in as is (lexicon or characters)."""
        from bolo.text import phonemize

        if self.kind == "kokoro":
            prefix = voice.id[:1] if voice is not None and self.spec.get("multi_lang", False) else "a"
            language = _KOKORO_ESPEAK.get(prefix)
            if language is None:
                return None
            # English also works from the bundled lexicon, but words missing from it are skipped, so a
            # phonemiser is preferred whenever one is installed (uccharan's front end is a port of misaki,
            # the one Kokoro English was trained with).
            explicit = phonemize.parse_choice(choice)[0] != "auto"
            if (
                prefix in _KOKORO_LEXICON_OK
                and self.has_lexicon
                and not explicit
                and phonemize.select("kokoro", language) is None
            ):
                return None
            return "kokoro", language
        if self.needs_phonemes:
            return phonemize.PROFILE_BY_KIND.get(self.kind, "piper"), self.model_lang or "en-us"
        return None

    def espeak_voice_for(self, voice: Voice | None) -> str | None:
        """The phonemiser language for a voice (kept for callers of bolo 0.1)."""
        target = self.phoneme_target(voice)
        return target[1] if target else None

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        phonemizer: str | None = None,
    ) -> Iterator[FloatArray]:
        original = text
        v = self.find_voice(voice)
        tts = self._tts
        config = self._so.GenerationConfig()
        config.sid = v.sid if v else (int(voice) if voice and voice.isdigit() else 0)
        if v is None and voice and self.supports_blending and looks_like_blend(voice):
            blend = resolve_blend(voice, self.voices, self.path.name)
            v = blend.lead  # its language reads the text
            tts, config.sid = self._blend_voice(blend)
        config.speed = speed
        target = self.phoneme_target(v, phonemizer)
        if target:
            from bolo.text import phonemize

            profile, language = target
            ph, improved = phonemize.resolve(profile, language, phonemizer)
            codepoints = ph.voice_codepoints(text, language, profile, improved)
            if not codepoints:
                raise RuntimeError(f"the phonemiser ({ph.name}) produced nothing for {text[:60]!r} ({language})")
            config.phoneme_codepoints = codepoints
            text = ""

        def callback(_samples: Any, _progress: float) -> int:
            return 0 if cancelled is not None and cancelled() else 1

        audio = tts.generate(text, config, callback)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if len(samples) == 0 and not (cancelled is not None and cancelled()):
            # sherpa-onnx logs the reason (unknown tokens, missing lexicon, ...) and returns nothing.
            raise RuntimeError(
                f"the voice produced no audio for {original[:60]!r}; the text may contain only characters "
                "this voice cannot pronounce, or the model needs a lexicon or phonemiser that is missing"
            )
        if len(samples):
            yield samples

    def _blend_voice(self, blend: Blend) -> tuple[Any, int]:
        """The instance and speaker id that speak ``blend``; builds (or rebuilds) the blend instance when the
        blend is new. The first request with a new blend pays for loading the model once more."""
        key = blend.key
        with self._blend_lock:
            if key in self._blends and self._blend_tts is not None:
                self._blends.move_to_end(key)
                return self._blend_tts, self._blend_slots[key]
            voices_file = Path(self._cfg["kokoro"].voices)
            rows = max(1, self.num_speakers)
            table = np.fromfile(voices_file, dtype=np.float32).reshape(rows, -1)
            if key not in self._blends:
                if len(self._blends) >= rows:
                    old, _ = self._blends.popitem(last=False)
                    self._blend_slots.pop(old, None)
                self._blends[key] = mix_styles(table, blend)
            used = set(self._blend_slots.values())
            if key not in self._blend_slots:
                self._blend_slots[key] = next(i for i in range(rows) if i not in used)
            out = table.copy()  # unused rows keep the model's own voices
            for name, row in self._blends.items():
                out[self._blend_slots[name]] = row
            cfg = self._cfg["kokoro"]
            with tempfile.TemporaryDirectory(prefix="bolo-blend-") as tmp:
                path = Path(tmp) / "voices.bin"
                out.tofile(path)
                parts = dict(self._cfg)
                parts["kokoro"] = self._so.OfflineTtsKokoroModelConfig(
                    model=cfg.model, voices=str(path), tokens=cfg.tokens, lexicon=cfg.lexicon
                )
                self._blend_tts = None  # free the old one first: at most two copies of the model at a time
                self._blend_tts = self._build(parts)  # sherpa-onnx reads the voices file while loading
            return self._blend_tts, self._blend_slots[key]

    def close(self) -> None:
        self._blend_tts = None
        self._blends.clear()
        self._blend_slots.clear()


class _SherpaVadDetector(VadDetector):
    def __init__(self, so: Any, config: Any) -> None:
        self._vad = so.VoiceActivityDetector(config, buffer_size_in_seconds=120)
        self._open = False
        self._pos = 0

    @property
    def in_speech(self) -> bool:
        return self._open

    def _drain(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        vad = self._vad
        while not vad.empty():
            seg = vad.front
            start = int(seg.start)
            n = len(seg.samples)
            vad.pop()
            if not self._open:
                events.append(VadEvent("speech_start", start))
            events.append(VadEvent("speech_end", start + n))
            self._open = False
        return events

    def accept(self, samples: FloatArray) -> list[VadEvent]:
        # Feed the detector one model window at a time so that events do not depend on how
        # the caller happens to chunk the audio.
        events: list[VadEvent] = []
        step = 512
        for off in range(0, len(samples), step):
            piece = samples[off : off + step]
            self._vad.accept_waveform(piece)
            self._pos += len(piece)
            events.extend(self._drain())
            if not self._open and self._vad.is_speech_detected():
                self._open = True
                try:
                    start = int(self._vad.current_segment.start)
                except Exception:
                    start = max(0, self._pos - len(piece))
                events.append(VadEvent("speech_start", start))
        return events

    def flush(self) -> list[VadEvent]:
        self._vad.flush()
        events = self._drain()
        if self._open:  # no segment came out (too short): close it anyway
            self._open = False
            events.append(VadEvent("speech_end", self._pos))
        return events


class SherpaVAD(VADModel):
    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        self._so = sherpa_onnx
        self.kind = kind
        self.model_file = find_file(path, str(spec.get("file", "*.onnx")), "*.onnx", prefer_int8=False)
        self.create_detector()  # fail early if the file is unusable

    def create_detector(self, params: VadParams | None = None) -> VadDetector:
        so = self._so
        p = params or VadParams()
        kwargs = {
            "model": self.model_file,
            "threshold": p.threshold,
            "min_silence_duration": p.min_silence_s,
            "min_speech_duration": p.min_speech_s,
            "max_speech_duration": p.max_speech_s,
        }
        config = so.VadModelConfig(sample_rate=16000, num_threads=1)
        if self.kind == "ten_vad":
            config.ten_vad = so.TenVadModelConfig(window_size=256, **kwargs)
        else:
            config.silero_vad = so.SileroVadModelConfig(window_size=512, **kwargs)
        return _SherpaVadDetector(so, config)


class SherpaDiarization(DiarizationModel):
    """pyannote speaker segmentation + a speaker-embedding model + fast clustering, all in ONNX."""

    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        self._so = sherpa_onnx
        provider = sherpa_provider(opts.device)
        segmentation = find_file(path, str(spec.get("segmentation", "model*.onnx")), "model*.onnx")
        embedding = find_file(
            path, str(spec.get("embedding", "embedding/*.onnx")), "embedding/*.onnx", prefer_int8=False
        )
        self.default_threshold = float(spec.get("threshold", 0.5))
        self._embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding, num_threads=opts.threads, provider=provider
        )
        self._segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation),
            num_threads=opts.threads,
            provider=provider,
        )
        self._min_on = float(spec.get("min_duration_on", 0.3))
        self._min_off = float(spec.get("min_duration_off", 0.5))
        config = self._config(None, None)
        if not config.validate():
            raise ValueError(f"sherpa-onnx rejected the diarization configuration for {path.name}")
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self._embedding_config)
        self._lock = threading.Lock()
        self.sample_rate = int(self._sd.sample_rate)

    def _config(self, num_speakers: int | None, threshold: float | None) -> Any:
        so = self._so
        clustering = so.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers and num_speakers > 0 else -1,
            threshold=threshold if threshold is not None else self.default_threshold,
        )
        return so.OfflineSpeakerDiarizationConfig(
            segmentation=self._segmentation_config,
            embedding=self._embedding_config,
            clustering=clustering,
            min_duration_on=self._min_on,
            min_duration_off=self._min_off,
        )

    def diarize(
        self, samples: FloatArray, num_speakers: int | None = None, threshold: float | None = None
    ) -> list[SpeakerTurn]:
        with self._lock:  # the clustering settings live on the shared object
            self._sd.set_config(self._config(num_speakers, threshold))
            result = self._sd.process(samples).sort_by_start_time()
        order: dict[int, int] = {}
        turns: list[SpeakerTurn] = []
        for seg in result:
            speaker = order.setdefault(int(seg.speaker), len(order))
            turns.append(SpeakerTurn(float(seg.start), float(seg.end), speaker))
        return turns

    def embed(self, samples: FloatArray) -> FloatArray:
        return _embed(self._extractor, self.sample_rate, samples)


def _embed(extractor: Any, sample_rate: int, samples: FloatArray) -> FloatArray:
    """A unit-length embedding from a sherpa-onnx speaker-embedding extractor."""
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    vector = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    out: FloatArray = (vector / norm).astype(np.float32) if norm > 0 else vector
    return out


class SherpaSpeakerEmbedding(SpeakerEmbeddingModel):
    """A speaker-embedding model on its own (3D-Speaker, WeSpeaker and NeMo exports with sherpa-onnx's metadata)."""

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        model = find_file(path, str(spec.get("file", "*.onnx")), "*.onnx", prefer_int8=False)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model, num_threads=opts.threads, provider=sherpa_provider(opts.device)
        )
        if not config.validate():
            raise ValueError(f"sherpa-onnx rejected the speaker-embedding model {Path(model).name}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dimensions = int(self._extractor.dim)

    def embed(self, samples: FloatArray) -> FloatArray:
        return _embed(self._extractor, self.sample_rate, samples)


class SherpaPunctuation(PunctuationModel):
    """Punctuation restoration: CT-Transformer (``punct_ct_transformer``: Chinese and English, punctuation only) or
    the English CNN-BiLSTM (``punct_cnn_bilstm``: punctuation and capital letters)."""

    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx as so

        self.kind = kind
        common: dict[str, Any] = {"num_threads": opts.threads, "provider": sherpa_provider(opts.device)}
        self._lock = threading.Lock()  # one text at a time: the bindings keep no per-call state, but be safe
        if kind == "punct_ct_transformer":
            config = so.OfflinePunctuationConfig(
                model=so.OfflinePunctuationModelConfig(ct_transformer=find_file(path, "model*.onnx"), **common)
            )
            if not config.validate():
                raise ValueError(f"sherpa-onnx rejected the punctuation model in {path}")
            offline = so.OfflinePunctuation(config)
            self._run: Callable[[str], str] = offline.add_punctuation
        elif kind == "punct_cnn_bilstm":
            config2 = so.OnlinePunctuationConfig(
                model_config=so.OnlinePunctuationModelConfig(
                    cnn_bilstm=find_file(path, "model*.onnx"), bpe_vocab=find_file(path, "bpe.vocab"), **common
                )
            )
            if not config2.validate():
                raise ValueError(f"sherpa-onnx rejected the punctuation model in {path}")
            online = so.OnlinePunctuation(config2)
            self._run = online.add_punctuation_with_case
        else:
            raise ValueError(f"sherpa-onnx engine cannot load kind {kind!r}")

    def punctuate(self, text: str) -> str:
        if not text.strip():
            return text
        with self._lock:
            out = str(self._run(text)).strip()
        if self.kind == "punct_ct_transformer":
            out = latin_punctuation(out)
        return out


#: CT-Transformer writes Chinese (full-width) punctuation everywhere; after a Latin letter or digit it becomes the
#: ASCII mark and a space ("morning。could" -> "morning. could"), so English words stay apart
_FULL_WIDTH = {"，": ",", "。": ".", "？": "?", "！": "!", "；": ";", "：": ":", "、": ","}
_AFTER_LATIN = re.compile(r"(?<=[A-Za-z0-9'])([，。？！；：、])\s*")


def latin_punctuation(text: str) -> str:
    return _AFTER_LATIN.sub(lambda m: _FULL_WIDTH[m.group(1)] + " ", text).strip()


class SherpaVoxLock(SeparationModel):
    """VoxLock target-speaker extraction (sherpa-onnx 2.0.1+): one extractor per enrolled voice.

    sherpa-onnx's Python binding takes the voice sample only as a file path when the extractor is
    built (the C++ ``SetVoiceSample`` is not exposed), so every enrolment writes a temporary WAV and
    builds an extractor of its own. Building one loads both graphs (about 2 s on a 4-vCPU machine),
    which is why extractors are kept until released.
    """

    def __init__(self, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx

        if not hasattr(sherpa_onnx, "OfflineSpeechDenoiserVoxLockModelConfig"):
            raise RuntimeError("VoxLock needs sherpa-onnx 2.0.1 or newer: pip install -U 'sherpa-onnx>=2.0.1'")
        self._so = sherpa_onnx
        self.enroll_model = find_file(path, str(spec.get("enroll_model", "enroll.onnx")), "enroll*.onnx")
        self.chunk_model = find_file(path, str(spec.get("chunk_model", "chunk_int8.onnx")), "chunk*.onnx")
        self.threads = opts.threads
        self.provider = sherpa_provider(opts.device)
        self.max_cached = int(spec.get("max_cached", 16))
        self._extractors: dict[str, Any] = {}
        self._lock = threading.Lock()

    def enrol(self, key: str, sample: FloatArray) -> None:
        import tempfile

        import soundfile as sf

        so = self._so
        with tempfile.TemporaryDirectory(prefix="bolo-voice-") as tmp:
            voice = str(Path(tmp) / "voice.wav")
            sf.write(voice, np.clip(sample, -1.0, 1.0), 16000, subtype="PCM_16")
            config = so.OfflineSpeechDenoiserConfig(
                model=so.OfflineSpeechDenoiserModelConfig(
                    voxlock=so.OfflineSpeechDenoiserVoxLockModelConfig(
                        enroll_model=self.enroll_model, chunk_model=self.chunk_model, voice_sample=voice
                    ),
                    num_threads=self.threads,
                    provider=self.provider,
                )
            )
            extractor = so.OfflineSpeechDenoiser(config)  # reads the voice sample now
        with self._lock:
            self._extractors.pop(key, None)
            self._extractors[key] = extractor
            while len(self._extractors) > self.max_cached:  # callers release their keys; this is a backstop
                self._extractors.pop(next(iter(self._extractors)))

    def extract(self, key: str, mixture: FloatArray) -> FloatArray:
        with self._lock:
            extractor = self._extractors.get(key)
        if extractor is None:
            raise KeyError(f"no voice enrolled under {key!r}")
        out = extractor(np.ascontiguousarray(mixture, dtype=np.float32), 16000)
        audio = np.asarray(out.samples, dtype=np.float32)
        if int(out.sample_rate) != 16000:
            from bolo.audio.resample import resample

            audio = resample(audio, int(out.sample_rate), 16000)
        if len(audio) < len(mixture):
            audio = np.pad(audio, (0, len(mixture) - len(audio)))
        return np.ascontiguousarray(audio[: len(mixture)], dtype=np.float32)

    def release(self, key: str) -> None:
        with self._lock:
            self._extractors.pop(key, None)


class SherpaDenoiser(DenoiseModel):
    """Speech enhancement with sherpa-onnx's offline denoiser: GTCRN (one ONNX graph of about 0.5 MB) or DPDFNet
    (Ceva's dual-path DeepFilterNet, 9 to 15 MB, at 8, 16 or 48 kHz; the rate is in the graph's metadata).

    DPDFNet takes ``attenuation_limit_db`` (catalog option or pin): at most this many dB of noise are removed, which
    keeps some of the original under the speech; 0 (the default) is no limit."""

    def __init__(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> None:
        import sherpa_onnx as so

        model = find_file(path, str(spec.get("model", "*.onnx")), "*.onnx")
        common: dict[str, Any] = {"num_threads": opts.threads, "provider": sherpa_provider(opts.device)}
        if kind == "dpdfnet":
            if not hasattr(so, "OfflineSpeechDenoiserDpdfNetModelConfig"):
                raise RuntimeError("this sherpa-onnx has no DPDFNet denoiser; upgrade it (pip install -U sherpa-onnx)")
            limit = float(spec.get("attenuation_limit_db") or 0.0)
            if not 0.0 <= limit <= 100.0:
                raise ValueError(f"attenuation_limit_db must be between 0 and 100, got {limit:g}")
            dpdfnet = so.OfflineSpeechDenoiserDpdfNetModelConfig(model=model, attenuation_limit_db=limit)
            config = so.OfflineSpeechDenoiserModelConfig(dpdfnet=dpdfnet, **common)
        else:
            gtcrn = so.OfflineSpeechDenoiserGtcrnModelConfig(model=model)
            config = so.OfflineSpeechDenoiserModelConfig(gtcrn=gtcrn, **common)
        self._denoiser = so.OfflineSpeechDenoiser(so.OfflineSpeechDenoiserConfig(model=config))
        self.sample_rate = int(self._denoiser.sample_rate)
        self._config = config
        self._kind = kind
        self._stream_delay = spec.get("stream_delay")

    def denoise(self, samples: FloatArray) -> FloatArray:
        out = self._denoiser(np.ascontiguousarray(samples, dtype=np.float32), self.sample_rate)
        audio = np.asarray(out.samples, dtype=np.float32)
        if int(out.sample_rate) != self.sample_rate:
            from bolo.audio.resample import resample

            audio = resample(audio, int(out.sample_rate), self.sample_rate)
        if len(audio) < len(samples):  # the model works in whole frames: the last few milliseconds come back short
            audio = np.pad(audio, (0, len(samples) - len(audio)))
        return np.ascontiguousarray(audio[: len(samples)], dtype=np.float32)

    def create_stream(self) -> DenoiseStream:
        """sherpa-onnx's online denoiser (its own state per stream), when this sherpa-onnx has one."""
        import sherpa_onnx as so

        if not hasattr(so, "OnlineSpeechDenoiser"):
            return super().create_stream()
        online = so.OnlineSpeechDenoiser(so.OnlineSpeechDenoiserConfig(model=self._config))
        # measured against the offline denoiser (sherpa-onnx 2.0.1): GTCRN's online output is in step with its
        # input; DPDFNet's runs four frame shifts behind (40 ms)
        delay = self._stream_delay
        if delay is None:
            delay = 4 * int(online.frame_shift_in_samples) if self._kind == "dpdfnet" else 0
        return _SherpaDenoiseStream(online, self.sample_rate, int(delay))


class _SherpaDenoiseStream(DenoiseStream):
    def __init__(self, online: Any, rate: int, delay: int) -> None:
        self.online = online
        self.rate = rate
        self.delay = delay
        self.taken = self.made = self.given = 0

    def _keep(self, out: FloatArray) -> FloatArray:
        start = max(0, self.delay - self.made)  # the model's delay is cut off once, at the start
        self.made += len(out)
        out = out[start:][: max(0, self.taken - self.given)]
        self.given += len(out)
        return np.ascontiguousarray(out, dtype=np.float32)

    def process(self, samples: FloatArray) -> FloatArray:
        self.taken += len(samples)
        out = self.online(np.ascontiguousarray(samples, dtype=np.float32), self.rate)
        return self._keep(np.asarray(out.samples, dtype=np.float32))

    def finish(self) -> FloatArray:
        need = self.taken - self.given
        if need <= 0:
            return np.zeros(0, dtype=np.float32)
        tail = []
        if self.delay:  # push what the delay holds back out of the model
            pad = np.zeros(self.delay + 4 * int(self.online.frame_shift_in_samples), dtype=np.float32)
            tail.append(np.asarray(self.online(pad, self.rate).samples, dtype=np.float32))
        tail.append(np.asarray(self.online.flush().samples, dtype=np.float32))
        out = self._keep(np.concatenate(tail))
        if len(out) < need:
            self.given += need - len(out)
            out = np.pad(out, (0, need - len(out)))
        return out


#: what sherpa-onnx says on standard error when it was asked for an accelerator it was not built with
SHERPA_CPU_FALLBACK = re.compile(r"fallback to cpu|SHERPA_ONNX_ENABLE_GPU", re.IGNORECASE)


class SherpaOnnxEngine(Engine):
    id = "sherpa-onnx"
    name = "sherpa-onnx"
    license = "Apache-2.0"
    url = "https://github.com/k2-fsa/sherpa-onnx"
    kinds = (
        OFFLINE_KINDS
        + ONLINE_KINDS
        + TTS_KINDS
        + VAD_KINDS
        + DIARIZATION_KINDS
        + SEPARATION_KINDS
        + DENOISE_KINDS
        + SPEAKER_EMBEDDING_KINDS
        + AUDIO_TAGGING_KINDS
        + PUNCTUATION_KINDS
        + KWS_KINDS
    )

    def status(self) -> EngineStatus:
        try:
            import sherpa_onnx

            return EngineStatus(self.id, True, str(getattr(sherpa_onnx, "__version__", "unknown")), None)
        except Exception as exc:
            return EngineStatus(
                self.id, False, None, f"{type(exc).__name__}: {exc}. Install with: pip install 'bolo-speech[sherpa]'"
            )

    def load(self, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        """The model; asked for ``cuda``, also whether sherpa-onnx started it there: a build without CUDA says
        "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON ... Fallback to cpu!" on standard error and runs on the CPU,
        which is recorded as a provider fallback (bolo.cuda_libs), so that the model's placement says the CPU."""
        if sherpa_provider(opts.device) != "cuda":
            return self._load(kind, path, spec, opts)
        from bolo import cuda_libs

        with cuda_libs.native_stderr() as said:
            model = self._load(kind, path, spec, opts)
        text = "".join(said)
        if SHERPA_CPU_FALLBACK.search(text):
            lines = [ln for ln in text.splitlines() if SHERPA_CPU_FALLBACK.search(ln) or "GPU" in ln]
            cuda_libs.record(
                cuda_libs.ProviderFallback(
                    "sherpa-onnx", "cuda", "cpu", [], "\n".join(lines[-2:]), ort_providers.SHERPA_CUDA_RECIPE
                )
            )
        else:
            cuda_libs.record(accelerated=True)
        return model

    def _load(self, kind: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Model:
        if kind in OFFLINE_KINDS:
            return SherpaOfflineSTT(kind, path, spec, opts)
        if kind in ONLINE_KINDS:
            return SherpaOnlineSTT(kind, path, spec, opts)
        if kind in TTS_KINDS:
            return SherpaTTS(kind, path, spec, opts)
        if kind in VAD_KINDS:
            return SherpaVAD(kind, path, spec, opts)
        if kind in DIARIZATION_KINDS:
            return SherpaDiarization(kind, path, spec, opts)
        if kind in SEPARATION_KINDS:
            return SherpaVoxLock(path, spec, opts)
        if kind in DENOISE_KINDS:
            return SherpaDenoiser(kind, path, spec, opts)
        if kind in SPEAKER_EMBEDDING_KINDS:
            return SherpaSpeakerEmbedding(path, spec, opts)
        if kind in AUDIO_TAGGING_KINDS:
            from bolo.engines.audio_classify import SherpaAudioTagging

            return SherpaAudioTagging(path, spec, opts)
        if kind in PUNCTUATION_KINDS:
            return SherpaPunctuation(kind, path, spec, opts)
        if kind in KWS_KINDS:
            from bolo.wake import SherpaKeywordSpotter

            return SherpaKeywordSpotter(path, spec, opts)
        raise ValueError(f"sherpa-onnx engine cannot load kind {kind!r}")
