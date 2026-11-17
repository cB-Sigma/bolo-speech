"""MaskGCT: language choice, length rule, voices, the import guard, and the load path with fakes of the authors'
entry points (``maskgct_utils``: ``g2p_``, ``build_*``, ``MaskGCT_Inference_Pipeline`` with ``text2semantic``,
``extract_acoustic_code`` and ``semantic2acoustic``; ``utils.util.load_config``; transformers' w2v-bert classes)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bolo_torch import upstream
from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED
from bolo_torch.models import maskgct
from bolo_torch.models.maskgct import Backend, MaskGCT, target_seconds, text_language
from bolo_torch.runtime import InvalidRequest, LoadContext


def test_language_by_script_and_the_length_rule() -> None:
    assert text_language("Hello there.", "en") == "en"
    assert text_language("你好，世界。", "en") == "en"  # Amphion's g2p_ reads Chinese and English (mixed) for both
    assert text_language("こんにちは", "en") == "ja" and text_language("안녕하세요", "en") == "ko"
    assert text_language("Bonjour.", "fr") == "fr" and text_language("Hallo.", "de") == "de"
    assert text_language("Hola.", "es") == "en"  # not a MaskGCT language: the g2p's default
    assert text_language("Guten Morgen, wie geht es dir?", "en") == "de"  # bolo's text-language guess
    assert text_language("Bonjour, comment allez-vous ?", "en") == "fr"
    # text2semantic's rule: the reference's phonemes per second, applied to the new text; speed shortens it
    assert target_seconds(48000, 30, 20, 1.0) == pytest.approx(2.0)
    assert target_seconds(48000, 30, 20, 2.0) == pytest.approx(1.0)


class FakeBackend(Backend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str, str, float]] = []

    def phones(self, text: str, language: str) -> int:
        return len(text)

    def reference_samples(self, ref_audio: str) -> int:
        return 32000

    def generate(self, ref_audio: str, ref_text: str, ref_language: str, text: str, language: str,
                 seconds: float) -> Any:  # fmt: skip
        self.calls.append((ref_audio, ref_text, ref_language, text, language, seconds))
        return np.full(int(seconds * 24000), 0.25, np.float32)


def test_cloned_voices_sentences_and_speed(tmp_path: Path) -> None:
    backend = FakeBackend()
    voices = [{"id": "heart", "ref_audio": "/r/heart.wav", "ref_text": "Ten chars."},
              {"id": "anne", "ref_audio": "/r/anne.wav", "ref_text": "Ref.", "language": "fr"}]  # fmt: skip
    model = MaskGCT(backend, voices)
    assert model.sample_rate == 24000 and [v.id for v in model.voices] == ["heart", "anne"]
    chunks = list(model.synthesize("One two. 你好。", speed=2.0))
    assert len(chunks) == 2
    first = backend.calls[0]
    assert first[:5] == ("/r/heart.wav", "Ten chars.", "en", "One two.", "en")
    assert first[5] == pytest.approx(target_seconds(32000, 10, 8, 2.0))
    list(model.synthesize("Bonjour.", voice="ANNE"))
    assert backend.calls[-1][4] == "fr"  # the voice's language for text in Latin script
    with pytest.raises(InvalidRequest, match="unknown voice"):
        list(model.synthesize("x", voice="nobody"))
    with pytest.raises(InvalidRequest, match="reference voice"):
        MaskGCT(backend, [])
    assert list(model.synthesize("A. B.", cancelled=lambda: True)) == []


def test_another_top_level_utils_package_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = types.ModuleType("utils")
    other.__file__ = str(tmp_path / "elsewhere" / "utils.py")
    monkeypatch.setitem(sys.modules, "utils", other)
    with pytest.raises(InvalidRequest, match="top-level package 'utils'"):
        maskgct.check_top_level(tmp_path / "bundle")
    mine = types.ModuleType("utils")
    mine.__file__ = str(tmp_path / "bundle" / "utils" / "__init__.py")
    monkeypatch.setitem(sys.modules, "utils", mine)
    maskgct.check_top_level(tmp_path / "bundle")  # the bundle's own: fine


def test_phonemizer_is_pointed_at_espeakng_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = types.ModuleType("espeakng_loader")
    loader.get_library_path = lambda: "/x/libespeak-ng.so"  # type: ignore[attr-defined]
    loader.get_data_path = lambda: "/x/espeak-ng-data"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
    monkeypatch.delenv("PHONEMIZER_ESPEAK_LIBRARY", raising=False)
    monkeypatch.delenv("ESPEAK_DATA_PATH", raising=False)
    maskgct.point_phonemizer_at_espeak()
    import os

    assert os.environ["PHONEMIZER_ESPEAK_LIBRARY"] == "/x/libespeak-ng.so"
    assert os.environ["ESPEAK_DATA_PATH"] == "/x/espeak-ng-data"
    monkeypatch.setenv("PHONEMIZER_ESPEAK_LIBRARY", "/mine.so")
    maskgct.point_phonemizer_at_espeak()
    assert os.environ["PHONEMIZER_ESPEAK_LIBRARY"] == "/mine.so"  # the operator's choice stands


# -- the load path ----------------------------------------------------------------------------------------------


class FakeNet(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def cond_emb(self, codes: torch.Tensor) -> torch.Tensor:
        return codes


class FakePipeline:
    """``MaskGCT_Inference_Pipeline``: built here without its constructor; its methods read the attributes the
    constructor would set and return what the real ones return (semantic codes at 50 Hz, 24 kHz audio)."""

    def text2semantic(self, prompt_speech: Any, prompt_text: str, prompt_language: str, target_text: str,
                      target_language: str, target_len: float | None = None, n_timesteps: int = 50, cfg: float = 2.5,
                      rescale_cfg: float = 0.75) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        for name in ("processor", "semantic_model", "semantic_mean", "semantic_std", "semantic_codec", "t2s_model"):
            assert hasattr(self, name), name
        self.t2s = (prompt_text, prompt_language, target_text, target_language, target_len, n_timesteps, cfg)
        prompt = torch.zeros(1, len(prompt_speech) // 320, dtype=torch.long)
        new = torch.ones(1, int((target_len or 1.0) * 50), dtype=torch.long)
        return torch.cat([prompt, new], dim=-1), prompt

    def extract_acoustic_code(self, speech: torch.Tensor) -> torch.Tensor:
        assert speech.dim() == 2 and speech.shape[0] == 1
        return torch.zeros(1, speech.shape[1] // 480, 12, dtype=torch.long)

    def semantic2acoustic(self, combine_semantic_code: torch.Tensor, acoustic_code: torch.Tensor,
                          n_timesteps: list[int] | None = None, cfg: float = 2.5,
                          rescale_cfg: float = 0.75) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
        self.s2a = (list(n_timesteps or []), cfg, rescale_cfg)
        new = combine_semantic_code.shape[1] - acoustic_code.shape[1]  # both 50 frames per second
        return np.zeros(1), np.full(new * 480, 0.1, np.float32)


def fake_utils() -> types.ModuleType:
    built: dict[str, Any] = {}
    utils = types.ModuleType("models.tts.maskgct.maskgct_utils")
    utils.g2p_ = lambda text, language: (text, list(range(len(text))))  # type: ignore[attr-defined]
    utils.build_semantic_codec = lambda cfg, dev: built.setdefault("semantic_codec", FakeNet("semantic"))  # type: ignore[attr-defined]
    utils.build_t2s_model = lambda cfg, dev: built.setdefault("t2s", FakeNet("t2s"))  # type: ignore[attr-defined]
    utils.build_s2a_model = lambda cfg, dev: FakeNet(f"s2a{cfg.num_quantizer}")  # type: ignore[attr-defined]
    utils.build_acoustic_codec = lambda cfg, dev: (FakeNet("encoder"), FakeNet("decoder"))  # type: ignore[attr-defined]
    utils.MaskGCT_Inference_Pipeline = FakePipeline  # type: ignore[attr-defined]
    return utils


def ns(**kw: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kw)


@pytest.fixture
def fake_libraries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {"loaded": []}
    code = tmp_path / "code"
    (code / Path(maskgct.STATS).parent).mkdir(parents=True)
    torch.save({"mean": torch.zeros(4), "var": torch.full((4,), 4.0)}, str(code / maskgct.STATS))
    monkeypatch.setattr(upstream, "ensure", lambda name, **_: seen.setdefault("bundle", name) and code)
    monkeypatch.setattr(maskgct, "import_pipeline", lambda root: seen.setdefault("utils", fake_utils()))
    cfg = ns(model=ns(semantic_codec=ns(), t2s_model=ns(), acoustic_codec=ns(),
                      s2a_model=ns(s2a_1layer=ns(num_quantizer=1), s2a_full=ns(num_quantizer=12))))  # fmt: skip
    util = types.ModuleType("utils.util")
    util.load_config = lambda path: seen.setdefault("config", path) and cfg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.util", util)

    class W2vBert:
        @classmethod
        def from_pretrained(cls, path: str) -> Any:
            seen["w2v"] = path
            return FakeNet("w2v-bert")

    class Extractor:
        @classmethod
        def from_pretrained(cls, path: str) -> str:
            seen["extractor"] = path
            return "feature extractor"

    transformers = types.ModuleType("transformers")
    transformers.Wav2Vec2BertModel = W2vBert  # type: ignore[attr-defined]
    transformers.SeamlessM4TFeatureExtractor = Extractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_model = lambda net, path: seen["loaded"].append(
        (net.name, Path(path).relative_to(tmp_path / "model").as_posix())
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    librosa = types.ModuleType("librosa")
    librosa.load = lambda path, sr: (np.zeros(sr * 2, np.float32), sr)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "librosa", librosa)

    def snapshot(repo: str, revision: str | None = None, allow: list[str] | None = None, **_: Any) -> Path:
        seen["semantic"] = (repo, revision, allow)
        return tmp_path / "w2v-bert-2.0"

    monkeypatch.setattr(maskgct, "hub_snapshot", snapshot)
    return seen


def model_folder(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    for rel in maskgct.WEIGHTS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return root


def test_load_follows_maskgct_inference(tmp_path: Path, fake_libraries: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    options = {"voices": [{"id": "heart", "name": "Heart", "reference": "en-short"}]}
    model = maskgct.load(LoadContext("maskgct", "tts", root, options))
    seen = fake_libraries
    assert seen["bundle"] == "maskgct" and seen["config"].endswith(maskgct.CONFIG)
    # w2v-bert-2.0 and its feature extractor from the pinned download, never the Hub's latest revision
    assert seen["semantic"][:2] == (maskgct.SEMANTIC_REPO, maskgct.SEMANTIC_REVISION)
    assert seen["w2v"] == seen["extractor"] == str(tmp_path / "w2v-bert-2.0")
    # every network gets its weights from the model folder, the 1-layer semantic-to-acoustic model included
    assert sorted(path for _, path in seen["loaded"]) == sorted(maskgct.WEIGHTS)
    by_path = {path: name for name, path in seen["loaded"]}
    assert by_path["s2a_model/s2a_model_1layer/model.safetensors"] == "s2a1"
    pipeline = model.backend.pipeline  # type: ignore[attr-defined]
    assert torch.equal(pipeline.semantic_std, torch.full((4,), 2.0))  # sqrt of the statistics' variance
    chunks = list(model.synthesize("Hello there."))
    assert len(chunks) == 1 and chunks[0].dtype == np.float32
    prompt_text, prompt_language, target, language, seconds, steps, cfg = pipeline.t2s
    assert (prompt_language, target, language, steps, cfg) == ("en", "Hello there.", "en", 25, 2.5)
    assert pipeline.s2a == ([25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2.5, 0.75)  # maskgct_inference's defaults
    assert seconds == pytest.approx(2.0 * len("Hello there.") / len(prompt_text))


def test_missing_weights_and_languages_are_refused(tmp_path: Path, fake_libraries: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    voices = {"voices": [{"id": "heart", "reference": "en-short"}]}
    with pytest.raises(InvalidRequest, match="speaks"):
        maskgct.load(LoadContext("maskgct", "tts", root, {**voices, "language": "hi"}))
    (root / "s2a_model/s2a_model_1layer/model.safetensors").unlink()
    with pytest.raises(InvalidRequest, match="s2a_model_1layer"):
        maskgct.load(LoadContext("maskgct", "tts", root, voices))


def test_the_bundle_and_kind() -> None:
    bundle = upstream.get("maskgct")
    (part,) = bundle.parts
    assert part.repo == "open-mmlab/Amphion" and len(part.commit) == 40 and part.license.startswith("MIT")
    for rel in ("models/tts/maskgct/maskgct_utils.py", maskgct.CONFIG, maskgct.STATS,
                "models/tts/maskgct/g2p/sources/g2p_chinese_model/poly_bert_model.onnx"):  # fmt: skip
        assert rel in part.files, rel
    assert LOADERS["maskgct"] == ("bolo_torch.models.maskgct", "load") and "maskgct" not in NOT_IMPLEMENTED
