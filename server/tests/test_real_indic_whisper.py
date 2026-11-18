"""Whisper through sherpa-onnx on real Tamil, Telugu and Bengali speech. Opt-in: ``BOLO_TEST_REAL=1 pytest -m real``.

Regression test: on these languages ``whisper-large-v3-turbo`` gave text that stopped part-way through the sentence
or ran into repeated syllables. sherpa-onnx stops Whisper after 6 tokens per second of audio and Whisper's tokens of
these scripts come 15 to 25 a second (bolo.engines.sherpa, ``SherpaOfflineSTT._whisper``, which also counts them
when sherpa-onnx cannot hand them over); and it decodes greedily, so a loop it falls into is cut to one copy, and
the catalog marks the model not recommended for Telugu and Bengali, where that still happens.

The audio is the first two utterances of FLEURS ``ta_in``, ``te_in`` and ``bn_in`` dev (Google, CC-BY-4.0), read from
the start of the dataset's pinned archives (about 1.5 MB of each), with their reference transcripts. The model
(1 GB) is pulled into ``BOLO_DATA_DIR``.
"""

from __future__ import annotations

import csv
import io
import os
import re
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo.config import Settings
from bolo.text.normalize import normalize_for_wer
from conftest import LiveServer

pytestmark = [pytest.mark.real, pytest.mark.timeout(3600)]
pytest.importorskip("sherpa_onnx")

MODEL = "whisper-large-v3-turbo"
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data"
CLIPS = 2


def _data_dir() -> Path:
    return Path(os.environ.get("BOLO_DATA_DIR", Path.home() / ".cache" / "bolo"))


class _Stream:
    """A file-like view of an HTTP download, read only as far as the reader goes."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self.chunks, self.buf = chunks, b""

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self.buf) < size:
            chunk = next(self.chunks, None)
            if chunk is None:
                break
            self.buf += chunk
        if size < 0:
            size = len(self.buf)
        out, self.buf = self.buf[:size], self.buf[size:]
        return out


def fleurs(config: str, count: int = CLIPS) -> list[tuple[bytes, str]]:
    """The first ``count`` FLEURS dev utterances of ``config`` (``ta_in``): (WAV bytes, reference transcript),
    cached on disk."""
    base = f"{FLEURS}/{config}"
    cache = _data_dir() / "validation" / f"fleurs-{config}-dev"
    cache.mkdir(parents=True, exist_ok=True)
    tsv = cache / "dev.tsv"
    if not tsv.exists():
        tsv.write_bytes(httpx.get(f"{base}/dev.tsv", follow_redirects=True, timeout=120).raise_for_status().content)
    wavs = sorted(cache.glob("*.wav"))
    if len(wavs) < count:
        with httpx.stream("GET", f"{base}/audio/dev.tar.gz", follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with tarfile.open(fileobj=_Stream(r.iter_bytes()), mode="r|gz") as archive:  # type: ignore[call-overload]
                for member in archive:
                    if member.isfile() and member.name.endswith(".wav"):
                        data = archive.extractfile(member)
                        assert data is not None
                        (cache / Path(member.name).name).write_bytes(data.read())
                        if len(list(cache.glob("*.wav"))) >= count:
                            break
        wavs = sorted(cache.glob("*.wav"))
    rows = {row[1]: row[2] for row in csv.reader(io.StringIO(tsv.read_text("utf-8")), delimiter="\t") if len(row) > 2}
    return [(w.read_bytes(), rows[w.name]) for w in wavs[:count]]


def char_error_rate(reference: str, hypothesis: str, language: str) -> float:
    ref = normalize_for_wer(reference, language).replace(" ", "")
    hyp = normalize_for_wer(hypothesis, language).replace(" ", "")
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / max(1, len(ref))


def repeated_run(text: str) -> str | None:
    """A run of 1 to 6 characters said more than four times back to back ("த்துக்த்துக்த்துக்..."), if any."""
    found = re.search(r"(.{1,6}?)\1{4,}", text)
    return found.group(0) if found else None


@pytest.fixture(scope="module")
def live() -> Iterator[LiveServer]:
    srv = LiveServer(Settings(data_dir=_data_dir(), auto_pull=True, device="cpu"))
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


#: language -> (the highest character error rate allowed, or None to ask only for whole text without loops; whether
#: the catalog recommends the model for it). Measured on these clips: Tamil 0.09 and 0.11, Telugu 0.17 and 0.19,
#: Bengali 0.33 and 0.59; Telugu and Bengali loop on other FLEURS utterances, so they are not recommended
LANGUAGES = {"ta": (0.35, True), "te": (0.4, False), "bn": (None, False)}


@pytest.mark.parametrize("language", list(LANGUAGES))
def test_sherpa_whisper_turbo_gives_whole_sentences_without_loops(live: LiveServer, language: str) -> None:
    spec = live.core.catalog.resolve(MODEL)
    assert spec is not None
    max_cer, recommended = LANGUAGES[language]
    assert (spec.not_recommended(language) is None) == recommended
    results: list[dict[str, Any]] = []
    for audio, reference in fleurs(f"{language}_in"):
        r = httpx.post(
            live.url + "/v1/audio/transcriptions",
            data={"model": MODEL, "language": language},
            files={"file": ("clip.wav", audio, "audio/wav")},
            timeout=1800,
        )
        assert r.status_code == 200, r.text
        text = r.json()["text"]
        results.append({"text": text, "reference": reference, "cer": char_error_rate(reference, text, language)})
    for res in results:
        print(f"{language} CER {res['cer']:.3f}: {res['text']}")
        assert repeated_run(res["text"]) is None, res
        # the whole sentence, not its first half: as long as the reference, give or take
        assert len(res["text"]) >= 0.8 * len(res["reference"]), res
        if max_cer is not None:
            assert res["cer"] < max_cer, res
