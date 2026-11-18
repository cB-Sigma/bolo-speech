"""Whisper through sherpa-onnx on real Tamil speech. Opt-in: ``BOLO_TEST_REAL=1 pytest -m real``.

Regression test: on Tamil (and Telugu, Bengali) ``whisper-large-v3-turbo`` gave text that stopped part-way through the
sentence or ran into repeated syllables, because sherpa-onnx stops Whisper after 6 tokens per second of audio and
Whisper's tokens of these scripts come 15 to 20 a second (bolo.engines.sherpa, ``SherpaOfflineSTT._whisper``).

The audio is the first two utterances of FLEURS ``ta_in`` dev (Google, CC-BY-4.0), read from the start of the
dataset's pinned archive (about 1.5 MB of it), with their reference transcripts. The model (1 GB) is pulled into
``BOLO_DATA_DIR``.
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
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/ta_in"
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


def fleurs_tamil(count: int = CLIPS) -> list[tuple[bytes, str]]:
    """The first ``count`` FLEURS ta_in dev utterances: (WAV bytes, reference transcript), cached on disk."""
    cache = _data_dir() / "validation" / "fleurs-ta_in-dev"
    cache.mkdir(parents=True, exist_ok=True)
    tsv = cache / "dev.tsv"
    if not tsv.exists():
        tsv.write_bytes(httpx.get(f"{FLEURS}/dev.tsv", follow_redirects=True, timeout=120).raise_for_status().content)
    wavs = sorted(cache.glob("*.wav"))
    if len(wavs) < count:
        with httpx.stream("GET", f"{FLEURS}/audio/dev.tar.gz", follow_redirects=True, timeout=120) as r:
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


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference, "ta").replace(" ", "")
    hyp = normalize_for_wer(hypothesis, "ta").replace(" ", "")
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


def test_sherpa_whisper_turbo_gives_whole_tamil_sentences(live: LiveServer) -> None:
    results: list[dict[str, Any]] = []
    for audio, reference in fleurs_tamil():
        r = httpx.post(
            live.url + "/v1/audio/transcriptions",
            data={"model": MODEL, "language": "ta"},
            files={"file": ("clip.wav", audio, "audio/wav")},
            timeout=1800,
        )
        assert r.status_code == 200, r.text
        text = r.json()["text"]
        results.append({"text": text, "reference": reference, "cer": char_error_rate(reference, text)})
    for res in results:
        print(f"CER {res['cer']:.3f}: {res['text']}")
        assert repeated_run(res["text"]) is None, res
        # the whole sentence, not its first half: as long as the reference, give or take
        assert len(res["text"]) >= 0.8 * len(res["reference"]), res
        assert res["cer"] < 0.35, res
