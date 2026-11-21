"""Zero-shot audio classification (``/v1/audio/classify`` with labels) and classifiers trained from examples
(``/v1/audio/classifiers``), with the fake CLAP model and the fake speaker embedder and tagger as encoders; the
CLAP ONNX engine's padding and text cache with stand-in sessions."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from bolo import classifiers
from bolo.audio.encode import encode_audio
from bolo.cli import main
from bolo.core import Bolo
from bolo.engines.clap_onnx import ClapOnnx, fill
from bolo.errors import InvalidRequest
from bolo.obs.trace import Trace
from conftest import LiveServer

SR = 16000


def tone(hz: float, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = rng.uniform(0, 2 * np.pi)
    return (0.3 * np.sin(2 * np.pi * hz * t + phase) + rng.normal(0, 0.003, len(t))).astype(np.float32)


def noise(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.2, int(seconds * SR)).astype(np.float32)


def wav(x: np.ndarray, name: str = "clip.wav") -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(encode_audio(x, SR, "wav")), "audio/wav")


def test_parse_labels() -> None:
    assert classifiers.parse_labels(["a dog barking, hold music"]) == ["a dog barking", "hold music"]
    assert classifiers.parse_labels(['["a, b", "c"]']) == ["a, b", "c"]
    assert classifiers.parse_labels(["speech\nmusic", "music", "noise"]) == ["speech", "music", "noise"]
    assert classifiers.parse_labels(["", None]) == []
    with pytest.raises(InvalidRequest):
        classifiers.parse_labels(['["a", 1]'])
    with pytest.raises(InvalidRequest):
        classifiers.parse_labels([",".join(str(i) for i in range(101))])


def test_prompt_text() -> None:
    assert classifiers.prompt_text("This is a sound of {}.", "rain") == "This is a sound of rain."
    assert classifiers.prompt_text("{label}!", "rain") == "rain!"
    assert classifiers.prompt_text("", "rain") == "rain"


def test_fill() -> None:
    x = np.array([1, 2, 3], dtype=np.float32)
    assert fill(x, 8, "repeatpad").tolist() == [1, 2, 3, 1, 2, 3, 0, 0]  # LAION: repeated whole, then zeros
    assert fill(x, 8, "repeat").tolist() == [1, 2, 3, 1, 2, 3, 1, 2]  # MS-CLAP: repeated and cut
    assert fill(np.arange(10, dtype=np.float32), 4, "repeatpad").tolist() == [0, 1, 2, 3]
    assert fill(np.zeros(0, dtype=np.float32), 3, "repeat").tolist() == [0, 0, 0]


def test_pieces() -> None:
    assert classifiers._pieces(0, 100, 40) == [(0, 40), (40, 80), (80, 100)]
    assert classifiers._pieces(0, 85, 40) == [(0, 40), (40, 85)]  # a last piece under a quarter joins the one before
    assert classifiers._pieces(10, 30, 40) == [(10, 30)]


class _Session:
    """Stands in for an onnxruntime session: counts runs, returns one-hot rows."""

    def __init__(self, dim: int) -> None:
        self.dim, self.runs, self.shapes = dim, 0, []

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs += 1
        first = next(iter(feeds.values()))
        self.shapes.append(first.shape)
        return [np.eye(self.dim, dtype=np.float32)[np.arange(first.shape[0]) % self.dim]]


def test_clap_onnx_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"<pad>": 0, "[UNK]": 1, "a": 2, "dog": 3, "<|endoftext|>": 4}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    for name in ("audio.onnx", "text.onnx"):
        (tmp_path / name).write_bytes(b"stand-in")
    cfg = {"sample_rate": 100, "clip_samples": 50, "padding": "repeat", "logit_scale": 33.0, "max_tokens": 3,
           "pad_id": 0, "append_eos": True, "embedding_dim": 4, "prompt": "a {}"}  # fmt: skip
    (tmp_path / "clap.json").write_text(json.dumps(cfg))
    sessions: list[_Session] = []

    def create_session(*_a: Any, **_k: Any) -> _Session:
        sessions.append(_Session(4))
        return sessions[-1]

    monkeypatch.setattr("bolo.ort_providers.create_session", create_session)
    from bolo.engines.base import LoadOptions

    model = ClapOnnx(tmp_path, {}, LoadOptions())
    assert model.clip_s == 0.5 and model.logit_scale == 33.0 and model.prompt == "a {}"
    audio, text = sessions
    out = model.embed_audio([np.ones(10, np.float32)] * 6)  # two runs: four clips, then two
    assert out.shape == (6, 4) and audio.runs == 2 and audio.shapes == [(4, 50), (2, 50)]
    ids, mask = model._tokens(["a dog", "dog dog dog dog"])
    assert ids.tolist() == [[2, 3, 4], [3, 3, 3]] and mask.tolist() == [[1, 1, 1], [1, 1, 1]]  # cut at 3 tokens
    ids, mask = model._tokens(["a"])
    assert ids.tolist() == [[2, 4]] and mask.tolist() == [[1, 1]]
    model.embed_text(["a", "dog", "a"])
    assert text.runs == 1 and text.shapes == [(2, 2)]  # each text once
    model.embed_text(["dog", "a"])
    assert text.runs == 1 and model.cached_texts == 2  # from the cache
    model.embed_text(["a dog"])
    assert text.runs == 2


async def test_zero_shot(core: Bolo) -> None:
    labels = ["a low hum", "a high whistle", "static noise"]
    for clip, want in ((tone(100), "a low hum"), (tone(4200), "a high whistle"), (noise(), "static noise")):
        result = await classifiers.zero_shot(core, clip, SR, "fake-clap", labels, Trace("t", "test", "fake-clap"))
        assert result.scores[0][0] == want
        assert abs(sum(s for _, s, _ in result.scores) - 1.0) < 1e-5
        assert all(sim is not None and -1.0 <= sim <= 1.0 for _, _, sim in result.scores)
    with pytest.raises(InvalidRequest, match="at least two labels"):
        await classifiers.zero_shot(core, tone(100), SR, "fake-clap", ["one"], Trace("t", "test", "fake-clap"))


async def test_zero_shot_text_cache_and_prompt(core: Bolo) -> None:
    labels = ["a low hum", "a high whistle"]
    result = await classifiers.zero_shot(core, tone(100), SR, "fake-clap", labels, Trace("t", "t", "m"))
    assert result.prompt == "the sound of {}"
    async with core.manager.use("fake-clap") as entry:
        model = entry.model
    runs = model.text_runs  # type: ignore[attr-defined]
    await classifiers.zero_shot(core, tone(4200), SR, "fake-clap", labels, Trace("t", "t", "m"))
    # the fake model does not cache, the engine's ClapOnnx does; the route asks once per request for all labels
    assert model.text_runs == runs + 2  # type: ignore[attr-defined]
    raw = await classifiers.zero_shot(core, tone(100), SR, "fake-clap", labels, Trace("t", "t", "m"), prompt="")
    assert raw.prompt == "" and raw.scores[0][0] == "a low hum"


async def test_zero_shot_windows(core: Bolo) -> None:
    x = np.concatenate([tone(100, 3.0), tone(4200, 3.0)])
    labels = ["low", "high"]
    result = await classifiers.zero_shot(core, x, SR, "fake-clap", labels, Trace("t", "t", "m"), window=3.0)
    assert [(w.start, w.end) for w in result.windows] == [(0.0, 3.0), (3.0, 6.0)]
    assert [w.scores[0][0] for w in result.windows] == ["low", "high"]
    # a recording longer than the model's clip (2 s for the fake) is heard in pieces whose embeddings are averaged
    assert {label for label, _, _ in result.scores} == {"low", "high"} and len(result.scores) == 2
    with pytest.raises(InvalidRequest):
        await classifiers.zero_shot(core, x, SR, "fake-clap", labels, Trace("t", "t", "m"), window=0.1)
    with pytest.raises(InvalidRequest):
        await classifiers.zero_shot(core, x, SR, "fake-clap", labels, Trace("t", "t", "m"), window=2.0, hop=3.0)


def test_fit_head_separates_and_cross_validates() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (20, 8)) + np.array([3, 0, 0, 0, 0, 0, 0, 0])
    b = rng.normal(0, 1, (20, 8)) - np.array([3, 0, 0, 0, 0, 0, 0, 0])
    x = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)
    for kind in ("logistic", "mlp"):
        head = classifiers.fit_head(x, y, ["a", "b"], kind)
        assert (np.argmax(head.probabilities(x), axis=1) == y).mean() > 0.95
        cv = classifiers.cross_validate(x, y, ["a", "b"], kind)
        assert cv is not None and cv["folds"] == 5 and cv["accuracy"] > 0.9
        assert np.array(cv["confusion"]).sum() == 40
    # the same data gives the same head
    h1, h2 = classifiers.fit_head(x, y, ["a", "b"]), classifiers.fit_head(x, y, ["a", "b"])
    assert np.array_equal(h1.weights[0][0], h2.weights[0][0])
    assert classifiers.cross_validate(x[:21], y[:21], ["a", "b"], "logistic") is None  # one example of b


def test_head_save_load(tmp_path: Path) -> None:
    x = np.random.default_rng(0).normal(0, 1, (12, 5)).astype(np.float32)
    y = np.array([0, 1, 2] * 4)
    head = classifiers.fit_head(x, y, ["a", "b", "c"], "mlp")
    head.save(tmp_path / "head.npz")
    again = classifiers.Head.load(tmp_path / "head.npz", ["a", "b", "c"])
    assert len(again.weights) == 2 and np.allclose(again.probabilities(x), head.probabilities(x))


def _examples(n: int, seed: int = 0) -> list[tuple[str, np.ndarray, int]]:
    out = []
    for i in range(n):
        out.append(("music", tone(150 + 10 * i, 1.0, seed + i), SR))
        out.append(("speech", tone(2500 + 40 * i, 1.0, seed + 100 + i), SR))
    return out


async def test_train_classify_delete(core: Bolo) -> None:
    out = await classifiers.train(core, "Tone-Kind", _examples(12), encoder="fake-clap")
    assert out["id"] == "tone-kind" and out["labels"] == ["music", "speech"] and out["examples"] == {
        "music": 12, "speech": 12,
    }  # fmt: skip
    assert out["cross_validation"]["accuracy"] == 1.0 and out["warnings"] == [] and out["dim"] == 16
    spec = core.catalog.resolve("tone-kind")
    assert spec is not None and spec.kind == "classifier_head" and spec.custom and core.store.is_installed("tone-kind")
    assert any(m["id"] == "tone-kind" for m in core.catalog_api(installed=True))  # what `bolo list` shows
    trace = Trace("t", "t", "m")
    result = await classifiers.classify_request(core, tone(180, 1.5, 99), SR, trace, classifier="tone-kind")
    assert result.classifier == "tone-kind" and result.model == "fake-clap" and result.scores[0][0] == "music"
    assert result.scores[0][2] is None
    # its id also works as the model
    result = await classifiers.classify_request(core, tone(2600, 1.5, 98), SR, trace, model="tone-kind", window=1.0)
    assert result.scores[0][0] == "speech" and len(result.windows) == 2
    with pytest.raises(classifiers.ClassifierExists):
        await classifiers.train(core, "tone-kind", _examples(12), encoder="fake-clap")
    again = await classifiers.train(core, "tone-kind", _examples(3), encoder="fake-clap", kind="mlp", replace=True)
    assert again["head"] == "mlp" and any("too little data" in w for w in again["warnings"])
    assert [c["id"] for c in classifiers.list_classifiers(core)] == ["tone-kind"]
    await classifiers.delete(core, "tone-kind")
    assert core.catalog.resolve("tone-kind") is None and not core.store.model_dir("tone-kind").exists()
    with pytest.raises(classifiers.ClassifierNotFound):
        await classifiers.classify_request(core, tone(180), SR, trace, classifier="tone-kind")


async def test_train_checks(core: Bolo) -> None:
    with pytest.raises(InvalidRequest, match="two classes"):
        await classifiers.train(core, "one", [("a", tone(100), SR)] * 3, encoder="fake-clap")
    with pytest.raises(classifiers.ClassifierExists, match="id of a model"):
        await classifiers.train(core, "fake-clap", _examples(2), encoder="fake-clap")
    with pytest.raises(InvalidRequest, match="cannot be an encoder"):
        await classifiers.train(core, "x", _examples(2), encoder="fake-stt")
    with pytest.raises(InvalidRequest, match="invalid classifier name"):
        await classifiers.train(core, "bad name!", _examples(2), encoder="fake-clap")
    with pytest.raises(InvalidRequest, match=r"shorter than 0\.1 s"):
        await classifiers.train(core, "x", [*_examples(2), ("a", tone(100, 0.01), SR)], encoder="fake-clap")


@pytest.mark.parametrize("encoder", ["fake-speaker-embedding", "fake-tagging"])
async def test_other_encoders(core: Bolo, encoder: str) -> None:
    """A speaker embedder (its "embed") and an audio tagger (its class probabilities) as encoders."""
    out = await classifiers.train(core, f"by-{encoder}", _examples(6), encoder=encoder)
    assert out["encoder"] == encoder and out["dim"] >= 2
    result = await classifiers.classify_request(core, tone(160, 1.0, 7), SR, Trace("t", "t", "m"), classifier=out["id"])
    assert result.model == encoder and {s[0] for s in result.scores} == {"music", "speech"}


def test_classify_routes(live: LiveServer) -> None:
    url = live.url + "/v1/audio/classify"
    r = httpx.post(
        url, files={"file": wav(tone(100))}, data={"model": "fake-clap", "labels": "a low hum, a high whistle"},
        timeout=30,
    )  # fmt: skip
    body = r.json()
    assert r.status_code == 200, body
    assert r.headers["x-bolo-model"] == "fake-clap" and body["object"] == "audio.classification"
    assert body["label"] == "a low hum" and body["classifier"] is None and body["prompt"] == "the sound of {}"
    assert [s["label"] for s in body["scores"]] == ["a low hum", "a high whistle"] and "similarity" in body["scores"][0]
    r = httpx.post(
        url, files={"file": wav(np.concatenate([tone(100, 3.0), noise(3.0)]))},
        data={"model": "fake-clap", "labels": '["low", "noise", "high"]', "window": "3", "top_k": "1"}, timeout=30,
    )  # fmt: skip
    body = r.json()
    assert [s["label"] for s in body["segments"]] == ["low", "noise"] and len(body["segments"][0]["scores"]) == 1
    r = httpx.post(url, files={"file": wav(tone(100))}, data={"model": "fake-clap"}, timeout=30)
    assert r.status_code == 400 and "labels" in r.text
    r = httpx.post(url, files={"file": wav(tone(100))}, data={"classifier": "nope"}, timeout=30)
    assert r.status_code == 404 and r.json()["error"]["code"] == "classifier_not_found"

    # train from multipart examples, then classify with it, list it, delete it
    files: list[tuple[str, Any]] = []
    data: dict[str, Any] = {"name": "tones", "encoder": "fake-clap", "label": []}
    for i, (label, x, _) in enumerate(_examples(10)):
        files.append(("file", wav(x, f"{i}.wav")))
        data["label"].append(label)
    r = httpx.post(live.url + "/v1/audio/classifiers", files=files, data=data, timeout=60)
    assert r.status_code == 201, r.text
    assert r.json()["cross_validation"]["accuracy"] == 1.0
    r = httpx.post(url, files={"file": wav(tone(2700, 1.0, 5))}, data={"classifier": "tones"}, timeout=30)
    body = r.json()
    assert body["label"] == "speech" and body["classifier"] == "tones" and r.headers["x-bolo-classifier"] == "tones"
    assert "similarity" not in body["scores"][0]
    listing = httpx.get(live.url + "/v1/audio/classifiers", timeout=10).json()["data"]
    assert [c["id"] for c in listing] == ["tones"]
    assert httpx.get(live.url + "/v1/audio/classifiers/tones", timeout=10).json()["encoder"] == "fake-clap"
    doc = httpx.get(live.url + "/api/v1/models/tones/show", timeout=10).json()  # a local model like any other
    assert doc["task"] == "audio-classify" and doc["kind"] == "classifier_head" and doc["installed"]
    assert httpx.delete(live.url + "/v1/audio/classifiers/tones", timeout=10).json()["deleted"] is True
    assert httpx.get(live.url + "/v1/audio/classifiers/tones", timeout=10).status_code == 404
    r = httpx.post(live.url + "/v1/audio/classifiers", files=files[:2], data={"name": "x", "label": "a"}, timeout=30)
    assert r.status_code == 400 and "label" in r.text


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BOLO_FAKE_ENGINE", "true")
    monkeypatch.setenv("BOLO_AUTO_PULL", "false")
    clip = tmp_path / "low.wav"
    clip.write_bytes(encode_audio(tone(100), SR, "wav"))
    assert main(["classify", str(clip), "-m", "fake-clap", "--labels", "low hum,high whistle"]) == 0
    assert capsys.readouterr().out.startswith("fake-clap: low hum 1.00")
    for label, hz in (("music", 150), ("speech", 2500)):
        folder = tmp_path / label
        folder.mkdir()
        for i in range(4):
            (folder / f"{i}.wav").write_bytes(encode_audio(tone(hz + 10 * i, 1.0, i), SR, "wav"))
    args = ["classifier", "train", "tones", "--class", f"music={tmp_path / 'music'}", "--encoder", "fake-clap"]
    assert main(args + ["--class", f"speech={tmp_path / 'speech'}"]) == 0
    out = capsys.readouterr().out
    assert "trained classifier tones" in out and "cross-validated accuracy: 100.0%" in out
    assert "warning: too little data" in out
    assert main(args) == 2 and "two classes" in capsys.readouterr().err
    assert main(["classifier", "list"]) == 0 and "tones" in capsys.readouterr().out
    assert main(["list"]) == 0 and "tones" in capsys.readouterr().out
    assert main(["classify", str(clip), "--classifier", "tones", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "music"
    assert main(["classifier", "rm", "tones"]) == 0
    assert main(["classifier", "show", "tones"]) == 1 and "no trained classifier" in capsys.readouterr().err


async def test_pipeline_step(core: Bolo) -> None:
    """``classify`` as a pipeline step: zero-shot with labels, and a trained classifier as the step's model."""
    from bolo.pipelines import Audio

    await classifiers.train(core, "tones", _examples(6), encoder="fake-clap")
    definition = {
        "id": "classify",
        "steps": [
            {"id": "zs", "type": "classify", "model": "fake-clap", "options": {"labels": ["low hum", "high whistle"]}},
            {"id": "trained", "type": "classify", "model": "tones"},
            {"id": "by_option", "type": "classify", "options": {"classifier": "tones"}},
        ],
        "outputs": {"label": "{{steps.zs.label}}", "kind": "{{steps.trained.label}}"},
    }
    run = core.pipelines.runner.prepare(definition, audio=Audio(tone(120, 2.0, 3), SR))
    await core.pipelines.run(run)
    doc = run.to_dict()
    assert doc["status"] == "ok", doc["error"]
    assert doc["outputs"] == {"label": "low hum", "kind": "music"}
    by = {s["id"]: s for s in doc["steps"]}
    assert by["trained"]["model"] == "tones" and by["by_option"]["output"]["label"] == "music"
    assert "similarity" in by["zs"]["output"]["scores"][0]
