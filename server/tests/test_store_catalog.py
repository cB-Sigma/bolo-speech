from __future__ import annotations

import hashlib
import http.server
import json
import re
import threading
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
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def hub() -> Iterator[str]:
    Hub.requests, Hub.truncate_once, Hub.corrupt, Hub.gated = [], False, False, False
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
    # entries whose files are not published yet: described, but without checksums and never offered
    pending = [s for s in every if s.status == "pending"]
    # (a release asset of this repository, or a repository of the project's Hugging Face organisation)
    assert all(
        s.validated == "untested"
        and (
            (
                s.source.urls
                and s.source.urls[0]["url"].startswith("https://github.com/cb-Sigma/bolo/releases/download/")
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
        if s.source.repo:
            assert len(s.source.revision) == 40, f"{s.id}: revisions are pinned to a commit"
            assert "espeak-ng-data/*" in s.source.exclude or s.engine != "sherpa-onnx" or s.source.include, s.id
        else:  # plain URLs: the revision is part of the URL and the large files carry a checksum
            assert s.source.urls, s.id
            for u in s.source.urls:
                assert re.search(r"/resolve/[0-9a-f]{40}/", u["url"]) or (u.get("extract") and u.get("sha256")), s.id
                assert u.get("sha256") or (u.get("size") or 0) < 1_000_000, f"{s.id}: {u['path']} needs a sha256"
        assert s.size_bytes > 100_000, s.id
        assert s.description and s.source_url.startswith("https://"), s.id
        assert s.engine in ("sherpa-onnx", "faster-whisper", "onnxruntime", "plugin:torch", "tts-server"), s.id
        if s.task == "tts":
            assert s.sample_rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000), s.id
        if s.validated == "cpu":
            assert s.measured.get("rtf") is not None and s.measured.get("hardware"), s.id
        if "non-commercial" in s.license.lower() or "-nc" in s.license.lower():
            assert s.commercial_use is False, s.id
    tasks = {s.task for s in specs}
    assert tasks == {"stt", "tts", "vad", "diarization", "turn", "separation"}
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
