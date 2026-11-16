"""Closed character sets, local and archived voices, upstream Piper conversion, pending entries, end-of-turn detection."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import io
import json
import sys
import tarfile
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import websockets

from bolo.audio.features import whisper_log_mel
from bolo.catalog import Catalog, CatalogError, ModelSpec
from bolo.config import Settings
from bolo.core import Bolo
from bolo.custom import register_local_model
from bolo.errors import DownloadError, InvalidRequest, ModelNotPublished, UnsupportedCharacters
from bolo.piper import convert_piper_dir, metadata_bytes, register_piper_voice
from bolo.store import HubClient, ModelStore, Source, extract_archive
from bolo.text.normalize import fit_to_charset
from conftest import LiveServer, speechlike

DEVANAGARI = frozenset("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंँ्।, ")


# -- closed character sets ----------------------------------------------------
def test_text_that_fits_passes_through_and_punctuation_is_mapped() -> None:
    assert fit_to_charset("नमक, पानी।", DEVANAGARI, "mai") == "नमक, पानी।"
    assert fit_to_charset("नमक; पानी?", DEVANAGARI, "mai") == "नमक, पानी"
    assert fit_to_charset("नमक — पानी…", DEVANAGARI, "mai") == "नमक , पानी"
    latin = frozenset("abcdefghijklmnopqrstuvwxyz .,")
    assert fit_to_charset("Hello, World", latin, "en") == "hello, world"


def test_digits_are_spelled_out_where_a_number_normaliser_exists() -> None:
    latin = frozenset("abcdefghijklmnopqrstuvwxyz .,-")
    assert fit_to_charset("room 42", latin, "en") == "room forty-two"
    hindi = fit_to_charset("कमरा 42", DEVANAGARI | frozenset("ब"), "hi")
    assert not any(ch.isdigit() for ch in hindi) and hindi.startswith("कमरा ")
    # a related language may opt into Hindi number words (catalog option `numbers: hi`)
    assert fit_to_charset("कमरा 42", DEVANAGARI | frozenset("ब"), "bho", numbers="hi") == hindi


def test_unsupported_characters_are_named() -> None:
    with pytest.raises(UnsupportedCharacters) as err:
        fit_to_charset("कमरा 42", DEVANAGARI, "mai")
    assert "'4', '2'" in err.value.message and "no number normaliser for language 'mai'" in err.value.message
    assert err.value.status == 400 and err.value.code == "unsupported_characters"
    with pytest.raises(UnsupportedCharacters, match="'D', 'e', 'l', 'h', 'i'"):
        fit_to_charset("हम Delhi", DEVANAGARI, "bho", numbers="hi")
    # digits the voice does have a symbol for are left alone
    assert fit_to_charset("कमरा ४२", DEVANAGARI | frozenset("४२"), "mai") == "कमरा ४२"


# -- archives -----------------------------------------------------------------
def make_tar(members: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return buf.getvalue()


def test_archives_are_unpacked_and_a_single_top_folder_is_stripped(tmp_path: Path) -> None:
    archive = tmp_path / "voice.tar.gz"
    archive.write_bytes(
        make_tar({"voice/model.onnx": b"M" * 10, "voice/tokens.txt": b"a 1\n", "voice/sub/x.json": b"{}"})
    )
    assert extract_archive(archive, tmp_path / "out", "voice.tar.gz") == ["model.onnx", "sub/x.json", "tokens.txt"]
    assert (tmp_path / "out" / "tokens.txt").read_bytes() == b"a 1\n"
    flat = tmp_path / "flat.zip"
    with zipfile.ZipFile(flat, "w") as z:
        z.writestr("model.onnx", b"M")
        z.writestr("tokens.txt", b"a 1\n")
    assert extract_archive(flat, tmp_path / "out2", "flat.zip") == ["model.onnx", "tokens.txt"]


@pytest.mark.parametrize(
    "members, links",
    [
        ({"../evil.txt": b"x"}, None),
        ({"voice/../../evil.txt": b"x", "voice/ok": b"y"}, None),
        ({"/etc/evil": b"x"}, None),
        ({"voice/model.onnx": b"x"}, {"voice/link": "/etc/passwd"}),
    ],
)
def test_archives_cannot_write_outside_the_model_directory(
    tmp_path: Path, members: dict[str, bytes], links: dict[str, str] | None
) -> None:
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(make_tar(members, links))
    with pytest.raises(DownloadError):
        extract_archive(archive, tmp_path / "out", "bad.tar.gz")
    assert not (tmp_path / "evil.txt").exists() and not list((tmp_path / "out").rglob("*link*"))
    with pytest.raises(DownloadError, match="unsupported archive"):
        extract_archive(archive, tmp_path / "out", "bad.rar")


class Assets(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    seen_authorization: list[str | None] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).seen_authorization.append(self.headers.get("Authorization"))
        data = self.files.get(self.path.lstrip("/"))
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def assets() -> Iterator[str]:
    Assets.files, Assets.seen_authorization = {}, []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Assets)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_pull_of_an_archive_verifies_extracts_and_keeps_the_token_private(assets: str, tmp_path: Path) -> None:
    data = make_tar({"v/model.onnx": b"M" * 5000, "v/tokens.txt": b"a 1\n", "v/LICENSE": b"MIT"})
    Assets.files["v.tar.gz"] = data
    store = ModelStore(tmp_path, HubClient("http://hub.invalid", token="hf_secret"))
    digest = hashlib.sha256(data).hexdigest()
    src = Source(
        urls=[{"path": "v.tar.gz", "url": f"{assets}/v.tar.gz", "size": len(data), "sha256": digest, "extract": True}]
    )
    progress: list[tuple[int, int]] = []
    manifest = store.pull("voice", src, progress=lambda done, total: progress.append((done, total)))
    assert sorted(f["path"] for f in manifest["files"]) == ["LICENSE", "model.onnx", "tokens.txt"]
    assert manifest["meta"]["archive"]["sha256"] == digest and progress[-1] == (len(data), len(data))
    assert (store.model_dir("voice") / "model.onnx").read_bytes() == b"M" * 5000
    assert not list(store.blobs.glob("sha256-*")), "the archive is not kept once it is unpacked"
    assert Assets.seen_authorization == [None], "the Hugging Face token must not be sent to other hosts"
    bad = Source(urls=[{"path": "v.tar.gz", "url": f"{assets}/v.tar.gz", "sha256": "0" * 64, "extract": True}])
    with pytest.raises(DownloadError):
        ModelStore(tmp_path / "b", HubClient("https://huggingface.co")).pull("voice", bad)


# -- pending entries ----------------------------------------------------------
def entry(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "v1", "task": "tts", "engine": "sherpa-onnx", "kind": "vits", "family": "VITS", "languages": ["mai"],
        "license": "MIT", "source": {"urls": [{"path": "v.tar.gz", "url": "https://example.org/v.tar.gz", "extract": True}]},
    }  # fmt: skip
    return {**base, **kw}


async def test_pending_entries_are_marked_and_cannot_be_pulled(core: Bolo) -> None:
    with pytest.raises(CatalogError, match="pinned with sha256"):
        ModelSpec.from_dict(entry())
    with pytest.raises(CatalogError, match="status"):
        ModelSpec.from_dict(entry(status="soon"))
    spec = ModelSpec.from_dict(entry(status="pending"))
    core.catalog.add(spec)
    assert "v1" not in [s.id for s in core.catalog.search("tts")]
    assert "v1" in [s.id for s in core.catalog.search("tts", include_pending=True)]
    # the API's catalog lists them, marked: a console can show "not published yet" instead of nothing
    listed = {m["id"]: m for m in core.catalog_api()}
    assert listed["v1"]["status"] == "pending" and listed["v1"]["installed"] is False
    assert "v1" not in [m["id"] for m in core.catalog_api(installed=True)]
    assert core.model_api(spec)["status"] == "pending"
    with pytest.raises(ModelNotPublished, match="not published yet"):
        core.start_pull("v1")
    with pytest.raises(ModelNotPublished):
        async for _ in core.synthesize("नमक", model="v1"):
            pass
    packaged = Catalog.load()
    pending = [s for s in packaged.search(include_pending=True) if s.status == "pending"]
    # entries still waiting for their release must not have been run here
    assert all(s.validated == "untested" for s in pending)


# -- local directories --------------------------------------------------------
async def test_a_local_directory_is_registered_in_place(core: Bolo, tmp_path: Path) -> None:
    voice = tmp_path / "My Voice"
    voice.mkdir()
    (voice / "model.onnx").write_bytes(b"M" * 100)
    (voice / "tokens.txt").write_bytes(b"a 1\n")
    with pytest.raises(InvalidRequest, match="Pass --kind"):
        register_local_model(core, str(voice))
    spec = register_local_model(core, str(voice), "vits", languages=["mai"], options={"numbers": "hi"})
    assert spec.id == "local-my-voice" and spec.custom and spec.task == "tts" and spec.source.local == str(voice)
    assert spec.options == {"numbers": "hi", "language": "mai"}
    assert register_local_model(core, str(voice), "vits") is spec
    manifest = core.store.pull(spec.id, spec.source)
    assert core.store.is_installed(spec.id) and manifest["size_bytes"] == 104
    assert (core.store.model_dir(spec.id) / "tokens.txt").read_text(encoding="utf-8") == "a 1\n"
    # it survives a restart (custom.yaml) and removing it leaves the user's files alone
    assert Catalog.load(core.settings.data_dir).resolve("local-my-voice").source.local == str(voice)  # type: ignore[union-attr]
    assert await core.remove(spec.id) and (voice / "model.onnx").is_file()
    with pytest.raises(InvalidRequest, match="not a directory"):
        register_local_model(core, str(tmp_path / "nope"), "vits")


@pytest.mark.parametrize("junction_works", [True, False])
async def test_a_local_directory_without_symlink_rights(
    core: Bolo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, junction_works: bool
) -> None:
    """Windows without Developer Mode or administrator rights cannot make directory symlinks (WinError 1314): bolo
    falls back to a directory junction there, and to a copy when that fails too (or on other systems)."""
    from bolo import store

    if junction_works and sys.platform != "win32":
        pytest.skip("directory junctions exist on Windows only")
    voice = tmp_path / "Voice"
    voice.mkdir()
    (voice / "model.onnx").write_bytes(b"M" * 100)
    (voice / "tokens.txt").write_bytes(b"a 1\n")

    def no_symlink(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", no_symlink)
    if not junction_works:

        def no_junction(target: Path, link: Path) -> None:
            raise OSError("no junctions here")

        monkeypatch.setattr(store, "_create_junction", no_junction)
    spec = register_local_model(core, str(voice), "vits")
    manifest = core.store.pull(spec.id, spec.source)
    mdir = core.store.model_dir(spec.id)
    assert manifest["size_bytes"] == 104 and (mdir / "tokens.txt").read_bytes() == b"a 1\n"
    assert store.is_dir_link(mdir) is junction_works
    if junction_works:
        assert mdir.resolve() == voice.resolve()
    assert await core.remove(spec.id) and not mdir.exists() and (voice / "model.onnx").is_file()


# -- upstream Piper voices ----------------------------------------------------
def read_metadata(blob: bytes) -> dict[str, str]:
    """Parse ModelProto.metadata_props entries (field 14) back out of protobuf bytes."""

    def varint(pos: int) -> tuple[int, int]:
        value = shift = 0
        while True:
            byte = blob[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value, pos

    out: dict[str, str] = {}
    pos = 0
    while pos < len(blob):
        assert blob[pos] == 0x72
        length, pos = varint(pos + 1)
        chunk, pos = blob[pos : pos + length], pos + length
        klen = chunk[1]
        key = chunk[2 : 2 + klen].decode()
        out[key] = chunk[4 + klen :].decode()
    return out


def test_piper_voice_is_converted_without_touching_the_download(tmp_path: Path) -> None:
    assert read_metadata(metadata_bytes({"model_type": "vits", "n_speakers": 9})) == {
        "model_type": "vits",
        "n_speakers": "9",
    }
    blob = tmp_path / "blob"
    blob.write_bytes(b"ONNX-GRAPH")
    voice = tmp_path / "voice"
    voice.mkdir()
    (voice / "te_IN-x-medium.onnx").hardlink_to(blob)
    config = {
        "audio": {"sample_rate": 22050, "quality": "medium"}, "espeak": {"voice": "te"}, "num_speakers": 2,
        "speaker_id_map": {"b": 1, "a": 0}, "language": {"name_english": "Telugu"},
        "phoneme_id_map": {"_": [0], "a": [5], " ": [3], "aɪ": [161], "\n": [9]},
    }  # fmt: skip
    (voice / "te_IN-x-medium.onnx.json").write_text(json.dumps(config), encoding="utf-8")
    info = convert_piper_dir(voice)
    assert info == {
        "espeak_voice": "te",
        "sample_rate": 22050,
        "num_speakers": 2,
        "speakers": ["a", "b"],
        "quality": "medium",
    }
    assert (voice / "tokens.txt").read_text(encoding="utf-8") == "_ 0\na 5\n  3\n"  # one code point per line only
    model = (voice / "model.onnx").read_bytes()
    assert model.startswith(b"ONNX-GRAPH") and not (voice / "te_IN-x-medium.onnx").exists()
    meta = read_metadata(model[len(b"ONNX-GRAPH") :])
    assert (
        meta["comment"] == "piper" and meta["voice"] == "te" and meta["n_speakers"] == "2" and meta["has_espeak"] == "1"
    )
    assert blob.read_bytes() == b"ONNX-GRAPH", "the shared download must not be modified"
    assert convert_piper_dir(voice) == info, "running it again is harmless"
    (tmp_path / "empty").mkdir()
    with pytest.raises(DownloadError, match="not an upstream Piper voice"):
        convert_piper_dir(tmp_path / "empty")


async def test_piper_names_become_catalog_entries(core: Bolo) -> None:
    spec = register_piper_voice(core, "xx_XX-nobody-high")
    assert spec.id == "piper-xx-xx-nobody-high" and spec.requires == ["phonemizer"] and spec.source.convert == "piper"
    assert spec.source.urls[0]["url"].endswith("/xx/xx_XX/nobody/high/xx_XX-nobody-high.onnx")
    assert register_piper_voice(core, "xx_XX-nobody-high") is spec
    # voices that are already in the catalog are found by their upstream name
    assert register_piper_voice(core, "te_IN-venkatesh-medium").id == "piper-te-in-venkatesh-medium"
    assert not register_piper_voice(core, "te_IN-venkatesh-medium").custom
    with pytest.raises(InvalidRequest, match="expected piper:"):
        register_piper_voice(core, "not a voice")


# -- end-of-turn detection ----------------------------------------------------
def test_whisper_features_have_the_expected_shape_and_range() -> None:
    audio = np.concatenate([np.zeros(16000 * 6, dtype=np.float32), speechlike(2.0)])
    features = whisper_log_mel(audio, normalize=True)
    assert features.shape == (80, 800) and features.dtype == np.float32
    assert features.max() - features.min() <= 2.0 + 1e-5  # clamped to 8 (log10) below the peak, divided by 4
    bin_220hz = int(np.argmax(features[:, 700]))
    assert bin_220hz < 12, "a 220 Hz tone sits in the lowest mel bands"


async def turns(url: str, mode: dict[str, Any], audio: np.ndarray) -> list[dict[str, Any]]:
    pcm = (audio * 32767).astype("<i2").tobytes()
    events: list[dict[str, Any]] = []
    async with websockets.connect(url.replace("http", "ws") + "/v1/realtime?intent=transcription") as ws:
        json.loads(await ws.recv())
        session = {
            "type": "transcription",
            "audio": {"input": {"transcription": {"model": "fake-stt"}, "turn_detection": mode}},
        }
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        async def read() -> None:
            async for message in ws:
                events.append(json.loads(message))

        reader = asyncio.create_task(read())
        for i in range(0, len(pcm), 4800):
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + 4800]).decode()})
            )
            await asyncio.sleep(0.05)
        # wait until the server has gone quiet (a fixed pause was too short on a busy machine), at most 10 s
        loop = asyncio.get_running_loop()
        deadline, seen, quiet_since = loop.time() + 10.0, -1, loop.time()
        while loop.time() < deadline:
            if len(events) != seen:
                seen, quiet_since = len(events), loop.time()
            elif loop.time() - quiet_since >= 1.0:
                break
            await asyncio.sleep(0.05)
        reader.cancel()
    return events


async def test_semantic_vad_keeps_a_turn_open_across_a_pause(settings: Settings) -> None:
    settings.default_turn = "fake-turn"  # says "finished" once it has heard three seconds of sound
    live = LiveServer(settings)
    live.start()
    try:
        silence = np.zeros(int(24000 * 0.8), dtype=np.float32)
        audio = np.concatenate(
            [speechlike(2.0, 24000), silence, speechlike(2.0, 24000), np.zeros(24000 * 2, dtype=np.float32)]
        )
        plain = await turns(live.url, {"type": "server_vad", "silence_duration_ms": 400}, audio)
        assert [e["type"] for e in plain].count("input_audio_buffer.speech_stopped") == 2
        smart = await turns(live.url, {"type": "semantic_vad"}, audio)
        types = [e["type"] for e in smart]
        assert (
            types.count("input_audio_buffer.speech_started") == 1
            and types.count("input_audio_buffer.speech_stopped") == 1
        )
        stopped = next(e for e in smart if e["type"] == "input_audio_buffer.speech_stopped")
        assert 4500 < stopped["audio_end_ms"] < 5400, "one turn that spans the pause"
        # an unfinished turn still ends when the pause outlasts the patience that `eagerness` sets
        short = np.concatenate([speechlike(1.5, 24000), np.zeros(24000 * 3, dtype=np.float32)])
        eager = await turns(live.url, {"type": "semantic_vad", "eagerness": "high"}, short)
        assert [e["type"] for e in eager].count("input_audio_buffer.speech_stopped") == 1
        import httpx

        info = httpx.get(live.url + "/api/v1/info").json()["capabilities"]["realtime"]
        assert info["semantic_vad"] == {"model": "fake-turn", "available": True, "installed": True}
        assert info["turn_detection"] == ["server_vad", "semantic_vad"] and info["sip"] is False
        assert (
            info["webrtc"]["endpoint"] == "/v1/realtime/calls"
            and info["client_secrets"] == "/v1/realtime/client_secrets"
        )
    finally:
        live.stop()


async def test_semantic_vad_without_a_model_is_plain_silence_detection(live: LiveServer) -> None:
    # the default end-of-turn model is not installed here and auto_pull is off
    audio = np.concatenate([speechlike(2.0, 24000), np.zeros(24000 * 2, dtype=np.float32)])
    events = await turns(live.url, {"type": "semantic_vad"}, audio)
    assert [e["type"] for e in events].count("input_audio_buffer.speech_stopped") == 1
