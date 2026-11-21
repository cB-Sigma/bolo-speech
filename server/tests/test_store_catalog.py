from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from bolo.catalog import FAKE_MODELS, Catalog, CatalogError, ModelSpec, packaged_entries
from bolo.custom import choose_files, detect_kind
from bolo.errors import DownloadError, GatedModel, ModelNotFound
from bolo.store import HubClient, ModelStore, RemoteFile, Source, git_blob_sha1, match_patterns, safe_id

BIG = bytes(range(256)) * 4096  # 1 MiB of LFS content
SMALL = b"token 0\nother 1\n"
FILES = {"model.int8.onnx": BIG, "tokens.txt": SMALL, "README.md": b"# readme\n", "test_wavs/0.wav": b"RIFF"}


class Hub(http.server.BaseHTTPRequestHandler):
    """A tiny stand-in for the Hugging Face Hub: listing API, resolve URLs, Range support."""

    requests: list[tuple[str, str | None]] = []
    truncate_once = False
    corrupt = False
    gated = False
    slow = 0.0  # seconds a file takes to send

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        cls = type(self)
        cls.requests.append((self.path, self.headers.get("Range")))
        if cls.gated and self.headers.get("Authorization") != "Bearer good-token":
            self.send_response(401)
            self.end_headers()
            return
        if self.path.startswith("/api/models/org/model/tree/"):
            listing = []
            for path, data in FILES.items():
                item: dict[str, object] = {
                    "type": "file",
                    "path": path,
                    "size": len(data),
                    "oid": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
                }
                if path.endswith(".onnx"):
                    item["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
                listing.append(item)
            listing.append({"type": "directory", "path": "test_wavs"})
            body = json.dumps(listing).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/models/"):
            self.send_response(404)
            self.end_headers()
            return
        prefix = "/org/model/resolve/main/"
        if self.path.startswith(prefix) and self.path[len(prefix) :] in FILES:
            data = FILES[self.path[len(prefix) :]]
            if cls.corrupt and self.path.endswith(".onnx"):
                data = b"X" + data[1:]
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
            else:
                self.send_response(200)
            payload = data[start:]
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if cls.truncate_once and self.path.endswith(".onnx"):
                cls.truncate_once = False
                self.wfile.write(payload[: len(payload) // 3])  # connection drops mid-file
                self.wfile.flush()
                self.connection.close()
                return
            if cls.slow:
                step = max(1, len(payload) // 16)
                for i in range(0, len(payload), step):
                    self.wfile.write(payload[i : i + step])
                    self.wfile.flush()
                    time.sleep(cls.slow / 16)
                return
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def hub() -> Iterator[str]:
    Hub.requests, Hub.truncate_once, Hub.corrupt, Hub.gated, Hub.slow = [], False, False, False, 0.0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


SOURCE = Source(repo="org/model", include=["*.onnx", "tokens.txt"], exclude=["test_wavs/*"])


def test_pull_verifies_and_stores_by_content(hub: str, tmp_path: Path) -> None:
    store = ModelStore(tmp_path, HubClient(hub))
    progress: list[tuple[int, int]] = []
    manifest = store.pull("my-model", SOURCE, progress=lambda done, total: progress.append((done, total)))
    assert [f["path"] for f in manifest["files"]] == ["model.int8.onnx", "tokens.txt"]  # include/exclude applied
    assert manifest["size_bytes"] == len(BIG) + len(SMALL) and manifest["complete"] is True
    assert progress[0] == (0, len(BIG) + len(SMALL)) and progress[-1][0] == progress[-1][1]
    mdir = store.model_dir("my-model")
    assert (mdir / "model.int8.onnx").read_bytes() == BIG and (mdir / "tokens.txt").read_bytes() == SMALL
    digest = hashlib.sha256(BIG).hexdigest()
    blob = tmp_path / "blobs" / f"sha256-{digest}"
    assert blob.is_file() and blob.samefile(mdir / "model.int8.onnx")  # hard link into the content-addressed cache
    assert store.is_installed("my-model") and store.size_on_disk("my-model") == manifest["size_bytes"]
    assert [m["id"] for m in store.installed()] == ["my-model"]
    assert not list((tmp_path / "blobs" / "partial").glob("*.part"))


def test_second_model_with_same_content_downloads_nothing(hub: str, tmp_path: Path) -> None:
    store = ModelStore(tmp_path, HubClient(hub))
    store.pull("a", SOURCE)
    before = len([r for r in Hub.requests if "/resolve/" in r[0] and r[0].endswith(".onnx")])
    store.pull("b", SOURCE)
    after = len([r for r in Hub.requests if "/resolve/" in r[0] and r[0].endswith(".onnx")])
    assert after == before  # the LFS blob was already in the cache
    assert store.remove("a") is True
    assert (store.model_dir("b") / "model.int8.onnx").read_bytes() == BIG  # still referenced by b
    assert store.remove("b") is True and store.remove("b") is False
    assert not list((tmp_path / "blobs").glob("sha256-*"))  # garbage collected


def test_interrupted_download_resumes_with_range(hub: str, tmp_path: Path) -> None:
    Hub.truncate_once = True
    store = ModelStore(tmp_path, HubClient(hub))
    manifest = store.pull("resumed", SOURCE)
    assert (store.model_dir("resumed") / "model.int8.onnx").read_bytes() == BIG
    ranges = [r[1] for r in Hub.requests if r[0].endswith("model.int8.onnx")]
    assert ranges[0] is None and ranges[1] is not None and ranges[1].startswith("bytes=")
    resumed_at = int(ranges[1][6:-1])
    assert len(BIG) // 3 - 65536 < resumed_at <= len(BIG) // 3  # continued where the connection dropped
    assert manifest["files"][0]["sha256"] == hashlib.sha256(BIG).hexdigest()


def test_checksum_mismatch_is_rejected(hub: str, tmp_path: Path) -> None:
    Hub.corrupt = True
    store = ModelStore(tmp_path, HubClient(hub))
    with pytest.raises(DownloadError, match="checksum mismatch"):
        store.pull("bad", SOURCE)
    assert not store.is_installed("bad")
    assert not list((tmp_path / "blobs").glob("sha256-*")) and not list((tmp_path / "blobs" / "partial").glob("*"))


def test_small_files_are_verified_by_git_object_id(hub: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ModelStore(tmp_path, HubClient(hub))
    monkeypatch.setitem(FILES, "tokens.txt", SMALL)
    plan = store.plan(SOURCE)
    tokens = next(f for f in plan if f.path == "tokens.txt")
    assert tokens.sha256 is None and tokens.git_oid == hashlib.sha1(b"blob %d\0" % len(SMALL) + SMALL).hexdigest()
    tampered = RemoteFile(tokens.path, tokens.url, tokens.size, None, "0" * 40)
    with pytest.raises(DownloadError, match="git object id"):
        store.fetch_blob(tampered)
    blob, digest = store.fetch_blob(tokens)
    assert git_blob_sha1(blob) == tokens.git_oid and digest == hashlib.sha256(SMALL).hexdigest()


def test_gated_and_missing_repositories(hub: str, tmp_path: Path) -> None:
    Hub.gated = True
    store = ModelStore(tmp_path, HubClient(hub))
    with pytest.raises(GatedModel, match="BOLO_HF_TOKEN"):
        store.pull("locked", SOURCE)
    ok = ModelStore(tmp_path, HubClient(hub, token="good-token"))
    assert ok.pull("locked", SOURCE)["complete"]
    Hub.gated = False
    with pytest.raises(ModelNotFound):
        store.plan(Source(repo="org/absent"))
    with pytest.raises(DownloadError, match="matched no files"):
        store.plan(Source(repo="org/model", include=["*.gguf"]))


def test_url_sources_and_cancellation(hub: str, tmp_path: Path) -> None:
    store = ModelStore(tmp_path, HubClient(hub))
    src = Source(
        urls=[
            {
                "path": "sub/dir/m.onnx",
                "url": f"{hub}/org/model/resolve/main/model.int8.onnx",
                "sha256": hashlib.sha256(BIG).hexdigest(),
            }
        ]
    )
    store.pull("by-url", src)
    assert (store.model_dir("by-url") / "sub" / "dir" / "m.onnx").read_bytes() == BIG
    with pytest.raises(DownloadError, match="cancelled"):
        ModelStore(tmp_path / "other", HubClient(hub)).pull("x", SOURCE, cancelled=lambda: True)


def test_helpers() -> None:
    assert safe_id("hf:org/name") == "hf--org--name"
    assert match_patterns("a/b.onnx", ["*.onnx"], []) and not match_patterns("a/b.onnx", ["*.onnx"], ["a/*"])
    assert Source.from_dict(SOURCE.to_dict()) == SOURCE


# -- catalog --------------------------------------------------------------
def test_every_packaged_entry_is_well_formed() -> None:
    entries = packaged_entries()
    assert len(entries) >= 150
    every = [ModelSpec.from_dict(e) for e in entries]
    ids = [s.id for s in every]
    assert len(ids) == len(set(ids))
    # no catalog-only entries: everything listed can be pulled and run (docs/models-unsupported.md has the rest)
    assert [s.id for s in every if not s.runnable or s.status == "pending"] == []
    # entries whose files are not published yet: described, but without checksums and never offered
    pending = [s for s in every if s.status == "pending"]
    # (a release asset of this repository, or a repository of the project's Hugging Face organisation)
    assert all(
        s.validated == "untested"
        and (
            (
                s.source.urls
                and s.source.urls[0]["url"].startswith("https://github.com/cB-Sigma/bolo-speech/releases/download/")
            )
            or (s.source.repo or "").startswith("codebucketsolutions/")
            # external TTS servers: upstream has no file the runtime can load yet (conversion needed, fork-only)
            or s.engine == "tts-server"
        )
        for s in pending
    )
    specs = [s for s in every if s.status != "pending"]
    for s in specs:
        assert s.license and s.license != "unknown", s.id
        if s.cloud:  # nothing to download: the provider's terms stand where a licence would
            assert s.engine == "remote" and s.license_url and s.size_bytes == 0, s.id
            assert not (s.source.repo or s.source.urls), s.id
            continue
        if s.builtin:  # nothing to download: code that is part of bolo or of a Python package (WebRTC VAD, the AECs)
            assert (
                s.engine in ("builtin", "webrtcvad", "speexdsp", "webrtc-apm", "pymicro-wakeword", "presidio")
                and s.size_bytes == 0
            ), s.id
            assert not (s.source.repo or s.source.urls), s.id
            assert s.description and s.source_url.startswith("https://"), s.id
            continue
        if s.source.repo:
            assert len(s.source.revision) == 40, f"{s.id}: revisions are pinned to a commit"
            assert "espeak-ng-data/*" in s.source.exclude or s.engine != "sherpa-onnx" or s.source.include, s.id
        else:  # plain URLs: the revision is part of the URL, or the file's content is pinned by its checksum
            assert s.source.urls, s.id
            for u in s.source.urls:
                assert re.search(r"/resolve/[0-9a-f]{40}/", u["url"]) or u.get("sha256"), s.id
                assert u.get("sha256") or (u.get("size") or 0) < 1_000_000, f"{s.id}: {u['path']} needs a sha256"
        assert s.size_bytes > 100_000 or s.task == "itn", s.id  # a rule FST can be a few kilobytes
        assert s.description and s.source_url.startswith("https://"), s.id
        assert s.engine in (
            "sherpa-onnx", "faster-whisper", "onnxruntime", "plugin:torch", "tts-server", "rnnoise", "duplex",
            "whisper.cpp", "builtin", "mlx", "wetext", "kaldifst", "presidio", "cactus",
        ), s.id  # fmt: skip
        if s.task == "tts":
            assert s.sample_rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000), s.id
        if s.validated == "cpu":
            assert s.measured.get("rtf") is not None and s.measured.get("hardware"), s.id
        if "non-commercial" in s.license.lower() or "-nc" in s.license.lower():
            assert s.commercial_use is False, s.id
    tasks = {s.task for s in specs}
    assert tasks == {
        "stt",
        "tts",
        "vad",
        "diarization",
        "turn",
        "separation",
        "denoise",
        "lid",
        "audio-llm",
        "speech-to-speech",
        "speaker-embedding",
        "audio-tagging",
        "emotion",
        "voice-conversion",
        "align",
        "punctuation",
        "itn",
        "pii",
        "echo-cancellation",
        "wake-word",
        "watermark",
    }
    needs = [s for s in specs if s.needs_phonemes]
    assert len(needs) >= 80 and all(s.task == "tts" for s in needs)
    assert not [s.id for s in specs if s.family == "Kokoro" and s.requires]  # the default voice needs nothing extra
    assert sum(s.tier == "realtime" and s.task == "stt" for s in specs) >= 10
    assert sum(s.streaming for s in specs) >= 15
    hindi_tts = [s for s in specs if s.task == "tts" and "hi" in s.languages]
    assert len(hindi_tts) >= 4
    indic = {lang for s in specs for lang in s.languages} & {
        "hi",
        "bn",
        "ta",
        "te",
        "mr",
        "gu",
        "kn",
        "ml",
        "pa",
        "ur",
        "ne",
        "or",
        "as",
    }
    assert len(indic) >= 12
    # every scheduled language of India has a speech-to-text model of its own
    scheduled = [
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    ]
    stt_langs = {lang for s in specs if s.task == "stt" and s.family == "IndicConformer" for lang in s.languages}
    assert set(scheduled) <= stt_langs
    converted = [s for s in specs if s.source.convert == "piper"]
    assert len(converted) >= 10 and all(s.requires == ["phonemizer"] and s.kind == "vits" for s in converted)
    gated = [s for s in specs if s.gated]
    # a gated model may only be marked validated when it was really run, which leaves measurements
    assert gated and all(s.source_url and s.source.repo for s in gated)
    assert all(s.validated == "untested" or s.measured for s in gated)
    plugin = [s for s in specs if s.plugin]
    assert plugin and all(s.plugin == "torch" and s.options.get("plugin_install") for s in plugin)


def test_catalog_aliases_defaults_and_search() -> None:
    cat = Catalog.load(None, include_fake=True)
    assert len(cat) == len(packaged_entries()) + len(FAKE_MODELS)
    from bolo.config import Settings

    s = Settings()
    for name in (s.default_stt, s.default_realtime_stt, s.default_tts, s.default_vad):
        assert cat.resolve(name) is not None, name
    assert cat.resolve(s.default_realtime_stt).tier == "realtime"  # type: ignore[union-attr]
    assert cat.resolve("MOONSHINE-TINY").id == "moonshine-tiny-en"  # type: ignore[union-attr]
    assert cat.resolve("nope") is None
    assert {m.id for m in cat.search(q="kitten int8")} == {"kitten-nano-0.8-int8"}
    assert all(m.task == "vad" for m in cat.search(task="vad"))
    api = cat.resolve("kokoro-v1.0-int8").to_api(installed=True)  # type: ignore[union-attr]
    assert api["voices"][0]["id"] == "af_heart" and len(api["voices"]) == 54
    assert set(api["voices"][0]) == {"id", "name", "language", "gender"}
    assert "partials" in cat.resolve("whisper-tiny").to_api()["features"]  # type: ignore[union-attr]


def test_catalog_rejects_bad_entries(tmp_path: Path) -> None:
    good = {
        "id": "x",
        "task": "stt",
        "engine": "fake",
        "kind": "k",
        "family": "F",
        "languages": ["en"],
        "license": "MIT",
        "source": {"repo": "a/b"},
    }
    ModelSpec.from_dict(good)
    for patch, message in [
        ({"id": "Bad Id"}, "invalid model id"),
        ({"task": "asr"}, "unknown task"),
        ({"tier": "fast"}, "unknown tier"),
        ({"license": ""}, "licence"),
        ({"source": {}}, "no download source"),
        ({"validated": "gpu"}, "validated must be"),
        ({"validated": "cpu"}, "must record their measurements"),
        ({"surprise": 1}, "unknown fields"),
        ({"voices": [{"name": "no id"}]}, "voice without id"),
    ]:
        with pytest.raises(CatalogError, match=message):
            ModelSpec.from_dict({**good, **patch})
    with pytest.raises(CatalogError, match="missing"):
        ModelSpec.from_dict({"id": "x"})
    cat = Catalog([ModelSpec.from_dict(good)], tmp_path / "custom.yaml")
    with pytest.raises(CatalogError, match="duplicate"):
        cat.add(ModelSpec.from_dict(good))
    with pytest.raises(CatalogError, match="already taken"):
        cat.add(ModelSpec.from_dict({**good, "id": "y", "aliases": ["X"]}))
    cat.add_custom(ModelSpec.from_dict({**good, "id": "mine", "aliases": ["org/mine"]}))
    reloaded = Catalog.load(tmp_path)
    assert reloaded.resolve("org/mine").custom is True  # type: ignore[union-attr]
    cat.remove_custom("mine")
    assert Catalog.load(tmp_path).resolve("mine") is None


def test_detect_kind_for_hf_pulls() -> None:
    assert (
        detect_kind("Systran/faster-whisper-tiny", ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"])
        == "ct2-whisper"
    )
    assert (
        detect_kind(
            "x/sherpa-onnx-whisper-tiny", ["tiny-encoder.int8.onnx", "tiny-decoder.int8.onnx", "tiny-tokens.txt"]
        )
        == "whisper"
    )
    assert (
        detect_kind(
            "x/moonshine",
            [
                "preprocess.onnx",
                "encode.int8.onnx",
                "uncached_decode.int8.onnx",
                "cached_decode.int8.onnx",
                "tokens.txt",
            ],
        )
        == "moonshine"
    )
    transducer = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
    assert detect_kind("x/sherpa-onnx-streaming-zipformer-fr", transducer) == "online_transducer"
    assert detect_kind("x/sherpa-onnx-zipformer-ru", transducer) == "transducer"
    assert detect_kind("x/sherpa-onnx-nemo-parakeet-tdt", transducer) == "nemo_transducer"
    assert detect_kind("x/vits-piper-te_IN-maya", ["te_IN-maya.onnx", "te_IN-maya.onnx.json", "tokens.txt"]) == "vits"
    assert detect_kind("x/kokoro-en", ["model.onnx", "voices.bin", "tokens.txt"]) == "kokoro"
    assert detect_kind("x/sense-voice-small", ["model.int8.onnx", "tokens.txt"]) == "sense_voice"
    assert detect_kind("x/vad", ["silero_vad.onnx"]) == "silero_vad"
    assert detect_kind("x/mystery", ["model.onnx", "tokens.txt"]) is None  # ambiguous: the user must pass --kind
    files = [
        RemoteFile(p, "") for p in ["encoder.onnx", "encoder.int8.onnx", "decoder.onnx", "tokens.txt", "x.fp16.onnx"]
    ]
    exclude = choose_files("transducer", files)
    assert "encoder.onnx" in exclude and "decoder.onnx" not in exclude and "x.fp16.onnx" in exclude
    assert "encoder.int8.onnx" in choose_files("transducer", files, precision="float")


@pytest.mark.network
def test_catalog_sources_resolve_on_the_hub() -> None:
    """Every catalog entry's pinned revision exists and its patterns match files (HEAD-cheap listing calls)."""
    import os

    if os.environ.get("BOLO_TEST_NETWORK") != "1":
        pytest.skip("set BOLO_TEST_NETWORK=1 to check all catalog sources against huggingface.co")
    hub = HubClient()
    store = ModelStore(Path("/nonexistent"), hub)
    failures = []
    for entry in packaged_entries():
        spec = ModelSpec.from_dict(entry)
        if spec.status == "pending":  # its files are not published yet
            continue
        try:
            files = store.plan(spec.source)
            total = sum(f.size or 0 for f in files)
            if abs(total - spec.size_bytes) > 0.02 * spec.size_bytes:
                failures.append(f"{spec.id}: size {total} != catalog {spec.size_bytes}")
        except Exception as exc:
            failures.append(f"{spec.id}: {exc}")
    assert not failures, "\n".join(failures)


def _installed(tmp_path: Path, model_id: str = "m") -> tuple[ModelStore, Path]:
    store = ModelStore(tmp_path)
    mdir = store.model_dir(model_id)
    mdir.mkdir(parents=True)
    (mdir / "weights.bin").write_bytes(b"x" * 10)
    (mdir / "manifest.json").write_text(json.dumps({"id": model_id, "complete": True, "files": []}), encoding="utf-8")
    return store, mdir


def _in_use(path: Path) -> PermissionError:
    exc = PermissionError(13, "The process cannot access the file because it is being used by another process")
    exc.filename = str(path / "manifest.json")
    exc.winerror = 32  # type: ignore[attr-defined]  # what Windows sets; patched in so this runs everywhere
    return exc


def test_removal_waits_out_a_sharing_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows: an antivirus scanner holds a just-written manifest.json for a moment (WinError 32). bolo rm tries
    again with a growing pause, and deletes the folder once the file is let go."""
    import shutil

    from bolo import store as store_mod

    store, mdir = _installed(tmp_path)
    tries: list[Path] = []

    def remover(path: Path) -> None:
        tries.append(path)
        if len(tries) < 3:
            raise _in_use(path)
        shutil.rmtree(path)

    pauses: list[float] = []
    monkeypatch.setattr(store_mod.time, "sleep", pauses.append)
    store.remover = remover
    assert store.remove("m") is True
    assert len(tries) == 3 and pauses == list(store_mod.REMOVE_BACKOFF_S[:2]) and not mdir.exists()


def test_removal_that_stays_blocked_names_the_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo import store as store_mod
    from bolo.errors import FilesInUse

    store, mdir = _installed(tmp_path)
    pauses: list[float] = []
    monkeypatch.setattr(store_mod.time, "sleep", pauses.append)

    def remover(path: Path) -> None:
        raise _in_use(path)

    store.remover = remover
    with pytest.raises(FilesInUse) as info:
        store.remove("m")
    assert str(mdir) in info.value.message and "WinError 32" in info.value.message and info.value.status == 409
    assert pauses == list(store_mod.REMOVE_BACKOFF_S) and sum(pauses) < 5
    # any other error is not waited for
    pauses.clear()

    def denied(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    store.remover = denied
    with pytest.raises(PermissionError):
        store.remove("m")
    assert pauses == []


def test_default_removal_drops_the_manifest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A folder only partly deleted is not taken for an installed model any more."""
    import shutil

    from bolo import store as store_mod

    store, mdir = _installed(tmp_path)
    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)
    real = shutil.rmtree

    def stuck(path: object, *a: object, **k: object) -> None:
        raise _in_use(Path(str(path)))

    monkeypatch.setattr(store_mod.shutil, "rmtree", stuck)
    with pytest.raises(store_mod.FilesInUse):
        store.remove("m")
    assert not store.is_installed("m") and (mdir / "weights.bin").exists()
    monkeypatch.setattr(store_mod.shutil, "rmtree", real)
    assert store.remove("m") is True and not mdir.exists()


class _CountingFile:
    """A file whose open handles are counted (the manifest reads of a store)."""

    open_now = 0

    def __init__(self, fh: object) -> None:
        self._fh = fh
        _CountingFile.open_now += 1

    def read(self, *a: object) -> bytes:
        return self._fh.read(*a)  # type: ignore[attr-defined,no-any-return]

    def __enter__(self) -> _CountingFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self._fh.close()  # type: ignore[attr-defined]
        _CountingFile.open_now -= 1


def test_a_manifest_read_in_another_thread_does_not_block_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The laptop's WinError 32: right after a load failed with ComponentMissing, the hardware advisor's background
    refresh was reading every manifest while the model was removed. Removal and manifest reads now take turns, so a
    deletion never meets an open manifest (which Windows refuses)."""
    import time as _time

    from bolo import logs
    from bolo import store as store_mod

    store, mdir = _installed(tmp_path)
    real_open = logs.open_shared
    reading = threading.Event()

    def slow_open(path: Path, binary: bool = False) -> _CountingFile:
        handle = _CountingFile(real_open(path, binary))
        if "manifest" in str(path):
            reading.set()
            _time.sleep(0.3)  # a slow disk, or an antivirus scan of the read
        return handle

    def remover(path: Path) -> None:
        if _CountingFile.open_now:  # what Windows does when the file is open without FILE_SHARE_DELETE
            raise _in_use(path)
        shutil.rmtree(path)

    import shutil

    monkeypatch.setattr(logs, "open_shared", slow_open)
    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)  # tries again at once: no luck from waiting
    store.remover = remover
    reader = threading.Thread(target=store.manifest, args=("m",))
    reader.start()
    assert reading.wait(5)
    assert store.remove("m") is True and not mdir.exists()
    reader.join(5)
    assert _CountingFile.open_now == 0


async def test_no_manifest_stays_open_after_a_component_missing_load(
    settings: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that fails with ComponentMissing leaves no handle on its manifest.json, and can be removed at once
    (tools/validate_models.py removes what it pulled right after the failure)."""
    import contextlib

    from bolo import logs
    from bolo.core import Bolo
    from bolo.errors import ComponentMissing
    from bolo.text import phonemize

    monkeypatch.setattr(phonemize, "get_phonemizer", lambda: None)
    monkeypatch.setattr(phonemize, "get_uccharan", lambda: None)
    real_open = logs.open_shared
    handles: list[object] = []

    def tracking(path: Path, binary: bool = False) -> object:
        fh = real_open(path, binary)
        if path.name == "manifest.json":
            handles.append(fh)
        return fh

    monkeypatch.setattr(logs, "open_shared", tracking)
    core = Bolo(settings)  # type: ignore[arg-type]
    await core.start()
    try:
        core.catalog.add(
            ModelSpec.from_dict(
                {
                    "id": "needs-espeak", "task": "tts", "engine": "fake", "kind": "fake-tts", "family": "Piper",
                    "languages": ["ru"], "license": "MIT", "sample_rate": 24000, "requires": ["phonemizer"],
                    "options": {"language": "ru-RU"}, "source": {"local": "/nonexistent"},
                }
            )
        )  # fmt: skip
        mdir = core.store.model_dir("needs-espeak")
        mdir.mkdir(parents=True)
        (mdir / "manifest.json").write_text('{"id": "needs-espeak", "complete": true, "files": []}', "utf-8")
        assert core.store.is_installed("needs-espeak")  # read once here, maybe again by the refresh
        core.advisor.invalidate()  # as after a pull: the next request starts the background refresh
        with pytest.raises(ComponentMissing):
            await core.synthesize_all("Hello.", model="needs-espeak")
        assert await core.remove("needs-espeak") is True and not mdir.exists()
        for _ in range(100):
            if not core.advisor._refreshing:
                break
            await asyncio.sleep(0.02)
        assert handles and all(fh.closed for fh in handles)  # type: ignore[attr-defined]
        fds = Path("/proc/self/fd")
        if fds.is_dir():  # Linux: no descriptor of this process is the manifest
            held: list[str] = []
            for fd in fds.iterdir():
                with contextlib.suppress(OSError):  # the listing's own descriptor is gone by now
                    held.append(str(fd.readlink()))
            assert not [p for p in held if p.endswith("manifest.json")]
    finally:
        await core.stop()


_PULL = """
import sys
from pathlib import Path
from bolo.store import HubClient, ModelStore, Source
store = ModelStore(Path(sys.argv[1]), HubClient(sys.argv[2]))
manifest = store.pull("same-model", Source(urls=[{"path": "m.onnx", "url": sys.argv[3]}]))
print(manifest["size_bytes"], manifest["installed_at"])
"""


def test_two_processes_pulling_the_same_model_fetch_it_once(hub: str, tmp_path: Path) -> None:
    """The server pulling a model on first use while ``bolo pull`` fetches the same one (WinError 32 on both sides
    on Windows: two writers of one .part file): the second waits for the first and takes the finished model. The
    file has no checksum in the source, so a second fetch of it could not be skipped otherwise."""
    Hub.slow = 2.0
    url = f"{hub}/org/model/resolve/main/model.int8.onnx"
    data = tmp_path / "data"
    args = [sys.executable, "-c", _PULL, str(data), hub, url]
    first = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 60
    while not Hub.requests and time.monotonic() < deadline:  # the first is downloading
        time.sleep(0.02)
    second = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    outs = [p.communicate(timeout=120) for p in (first, second)]
    assert [p.returncode for p in (first, second)] == [0, 0], [err for _, err in outs]
    assert outs[0][0] == outs[1][0]  # the same manifest: the second took the first's
    assert [path for path, _ in Hub.requests] == ["/org/model/resolve/main/model.int8.onnx"]  # fetched once
    store = ModelStore(data)
    assert (store.model_dir("same-model") / "m.onnx").read_bytes() == BIG
    assert not list((data / "blobs" / "partial").glob("*.part"))


def test_threads_pulling_the_same_model_fetch_it_once(hub: str, tmp_path: Path) -> None:
    Hub.slow = 1.0
    store = ModelStore(tmp_path, HubClient(hub))
    src = Source(urls=[{"path": "m.onnx", "url": f"{hub}/org/model/resolve/main/model.int8.onnx"}])
    results: list[dict[str, object]] = []
    progress: list[tuple[int, int]] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(store.pull("m", src, progress=lambda d, t: progress.append((d, t))))
        )
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == 2 and results[0] == results[1] and len(Hub.requests) == 1
    assert (len(BIG), len(BIG)) in progress  # the one that waited reports the finished model


def test_a_repull_leaves_files_in_place_and_renames_wait_out_sharing_violations(
    hub: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows: a model the server has loaded keeps its files open, so ``bolo pull`` of it again must not delete
    them (they are the same files already); and a rename into place that meets a file held for a moment by an
    antivirus scanner (WinError 32) is tried again."""
    from bolo import locks

    store = ModelStore(tmp_path, HubClient(hub))
    store.pull("my-model", SOURCE)
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if store.models in self.parents:
            raise _in_use(self.parent)  # held open by the loaded model
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    real_replace = Path.replace
    refused: list[Path] = []

    def replace(self: Path, target: Path) -> Path:
        if len(refused) < 2:
            refused.append(self)
            raise _in_use(self.parent)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(locks.time, "sleep", lambda s: None)
    manifest = store.pull("my-model", SOURCE)
    assert manifest["complete"] and len(refused) == 2
    assert (store.model_dir("my-model") / "model.int8.onnx").read_bytes() == BIG


def test_the_token_hf_auth_login_saved_is_used(tmp_path: Path) -> None:
    """A gated pull failed after `hf auth login` (it asked for --hf-token or BOLO_HF_TOKEN): the saved token is
    read where the Hugging Face libraries read it."""
    from bolo.store import saved_hf_token

    home = tmp_path / "hf"
    home.mkdir()
    (home / "token").write_text("hf_saved_by_login\n")
    assert saved_hf_token({"HF_HOME": str(home)}) == "hf_saved_by_login"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("hf_from_token_path")
    assert saved_hf_token({"HF_HOME": str(home), "HF_TOKEN_PATH": str(elsewhere)}) == "hf_from_token_path"
    (tmp_path / "huggingface").mkdir()
    (tmp_path / "huggingface" / "token").write_text("hf_in_the_cache")
    assert saved_hf_token({"XDG_CACHE_HOME": str(tmp_path)}) == "hf_in_the_cache"  # ~/.cache/huggingface
    assert saved_hf_token({"HF_HOME": str(tmp_path / "nothing")}) is None
    (home / "token").write_text("  \n")
    assert saved_hf_token({"HF_HOME": str(home)}) is None
