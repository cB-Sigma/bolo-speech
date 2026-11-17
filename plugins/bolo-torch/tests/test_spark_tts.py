"""Spark-TTS: prompts as the authors' ``SparkTTS`` builds them, token parsing, voices, and the load path with fakes of
the real entry points (transformers' Auto classes, ``sparktts.models.audio_tokenizer.BiCodecTokenizer``,
``sparktts.utils.token_parser``, ``sparktts.utils.file.load_config``)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bolo_torch import upstream
from bolo_torch.engine import LOADERS, NOT_IMPLEMENTED
from bolo_torch.models import spark_tts
from bolo_torch.models.spark_tts import Backend, SparkTTS, clone_prompt, control_prompt, speed_level
from bolo_torch.runtime import InvalidRequest, LoadContext

# the authors' maps (sparktts/utils/token_parser.py at the pinned commit), the parts the recipe uses
MAPS = types.SimpleNamespace(
    TASK_TOKEN_MAP={"tts": "<|task_tts|>", "controllable_tts": "<|task_controllable_tts|>"},
    LEVELS_MAP={"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4},
    GENDER_MAP={"female": 0, "male": 1},
)
GLOBALS = list(range(100, 132))  # 32 global tokens, as BiCodec's speaker encoder makes


def speech(semantic: list[int], global_ids: list[int] | None = None) -> str:
    """What the language model writes (special tokens are skipped when decoding, the bicodec ones are not)."""
    out = "".join(f"<|bicodec_global_{i}|>" for i in global_ids or [])
    return out + "".join(f"<|bicodec_semantic_{i}|>" for i in semantic)


class FakeBackend(Backend):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.tokenized: list[str] = []
        self.decoded: list[tuple[list[int], list[int]]] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "<|task_controllable_tts|>" in prompt:
            return speech([5, 6, 7], GLOBALS)  # a created voice: the model writes the global tokens itself
        return speech([8, 9])

    def tokenize(self, path: str) -> tuple[list[int], list[int]]:
        self.tokenized.append(path)
        return list(range(200, 232)), [1, 2, 3]

    def detokenize(self, global_ids: list[int], semantic_ids: list[int]) -> Any:
        self.decoded.append((global_ids, semantic_ids))
        return np.full(320 * len(semantic_ids), 0.5, np.float32)  # BiCodec: 320 samples per semantic token


def test_prompts_are_the_authors() -> None:
    assert control_prompt("Hi.", "male", "low", "high", MAPS) == (
        "<|task_controllable_tts|><|start_content|>Hi.<|end_content|><|start_style_label|>"
        "<|gender_1|><|pitch_label_1|><|speed_label_3|><|end_style_label|>"
    )
    assert clone_prompt("Hi.", [1, 2], [3], "Ref text.", MAPS) == (
        "<|task_tts|><|start_content|>Ref text.Hi.<|end_content|><|start_global_token|><|bicodec_global_1|>"
        "<|bicodec_global_2|><|end_global_token|><|start_semantic_token|><|bicodec_semantic_3|>"
    )
    assert clone_prompt("Hi.", [1], [3], None, MAPS) == (
        "<|task_tts|><|start_content|>Hi.<|end_content|><|start_global_token|><|bicodec_global_1|><|end_global_token|>"
    )
    with pytest.raises(InvalidRequest):
        control_prompt("Hi.", "robot", "low", "high", MAPS)
    assert [speed_level(x) for x in (0.5, 0.8, 1.0, 1.3, 2.0)] == ["very_low", "low", "moderate", "high", "very_high"]


def test_created_and_cloned_voices(tmp_path: Path) -> None:
    backend = FakeBackend()
    clip = tmp_path / "ref.wav"
    voices = [
        {"id": "female", "name": "Female", "gender": "f", "spark_gender": "female", "pitch": "high"},
        {"id": "heart", "name": "Heart", "ref_audio": str(clip), "ref_text": "Reference words."},
    ]
    model = SparkTTS(backend, voices, MAPS)
    assert [v.id for v in model.voices] == ["female", "heart"] and model.sample_rate == 16000
    chunks = list(model.synthesize("One. Two.", speed=1.3))  # the first voice is the default
    assert len(chunks) == 2 and all(c.dtype == np.float32 and len(c) == 960 for c in chunks)
    assert "<|gender_0|><|pitch_label_3|><|speed_label_3|>" in backend.prompts[0]
    assert backend.decoded[0] == (GLOBALS, [5, 6, 7])  # the global tokens the model wrote
    chunks = list(model.synthesize("Three. Four.", voice="HEART"))
    assert backend.tokenized == [str(clip)]  # the reference is tokenized once, then reused
    assert backend.prompts[-1].startswith("<|task_tts|><|start_content|>Reference words.Four.")
    assert backend.decoded[-1] == (list(range(200, 232)), [8, 9])  # the reference's global tokens
    with pytest.raises(InvalidRequest, match="unknown voice"):
        list(model.synthesize("x", voice="nobody"))
    stop = list(model.synthesize("A. B. C.", cancelled=lambda: True))
    assert stop == []


def test_a_sentence_without_speech_tokens_is_skipped() -> None:
    class Silent(FakeBackend):
        def generate(self, prompt: str) -> str:
            return "no speech here"

    model = SparkTTS(Silent(), [dict(v) for v in spark_tts.DEFAULT_VOICES], MAPS)
    assert list(model.synthesize("Hello.")) == []


# -- the load path, with fakes of the real entry points ------------------------------------------------------


class FakeInputs(dict):  # type: ignore[type-arg]
    """``BatchEncoding``: a mapping with ``input_ids`` and ``to``."""

    def __init__(self, ids: torch.Tensor) -> None:
        super().__init__(input_ids=ids)
        self.input_ids = ids

    def to(self, device: str) -> FakeInputs:
        return self


class FakeTokenizer:
    """A vocabulary of whole strings: id i is ``vocab[i]``."""

    def __init__(self) -> None:
        self.vocab: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str) -> FakeTokenizer:
        assert path.endswith("LLM")
        return FAKE_TOKENIZER  # the language model writes into the same vocabulary

    def __call__(self, texts: list[str], return_tensors: str = "pt") -> FakeInputs:
        self.vocab.append(texts[0])
        return FakeInputs(torch.tensor([[len(self.vocab) - 1]]))

    def batch_decode(self, sequences: list[torch.Tensor], skip_special_tokens: bool = False) -> list[str]:
        assert skip_special_tokens
        return ["".join(self.vocab[int(i)] for i in seq) for seq in sequences]


class FakeLM:
    loaded: list[FakeLM] = []

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []
        self.device: Any = None

    @classmethod
    def from_pretrained(cls, path: str, torch_dtype: Any = None) -> FakeLM:
        model = cls(FAKE_TOKENIZER)
        model.dtype = torch_dtype
        FakeLM.loaded.append(model)
        return model

    def to(self, device: str) -> FakeLM:
        self.device = device
        return self

    def eval(self) -> FakeLM:
        return self

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        self.tokenizer.vocab.append(speech([4, 5], GLOBALS))
        return torch.cat([input_ids, torch.tensor([[len(self.tokenizer.vocab) - 1]])], dim=1)


FAKE_TOKENIZER = FakeTokenizer()


class FakeBiCodecTokenizer:
    """``BiCodecTokenizer(model_dir, device)``: ``tokenize(path)`` gives global tokens (1, 1, 32) and semantic tokens
    (1, n); ``detokenize(global (1, 32), semantic (1, n))`` gives the waveform as a numpy array."""

    def __init__(self, model_dir: Path, device: torch.device | None = None) -> None:
        self.model_dir = model_dir
        self.device = device
        self.seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def tokenize(self, audio_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.arange(32).reshape(1, 1, 32), torch.tensor([[7, 8, 9]])

    def detokenize(self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor) -> np.ndarray:
        self.seen.append((tuple(global_tokens.shape), tuple(semantic_tokens.shape)))
        return np.zeros(320 * semantic_tokens.shape[-1], np.float32)


def model_folder(tmp_path: Path) -> Path:
    root = tmp_path / "Spark-TTS-0.5B"
    for name in spark_tts.MODEL_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("{}")
    return root


@pytest.fixture
def fake_libraries(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"bundles": []}
    monkeypatch.setattr(upstream, "ensure", lambda name, **_: seen["bundles"].append(name))
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = FakeLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    for name in ("einops", "einx", "omegaconf", "soxr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    audio_tokenizer = types.ModuleType("sparktts.models.audio_tokenizer")
    audio_tokenizer.BiCodecTokenizer = FakeBiCodecTokenizer  # type: ignore[attr-defined]
    parser = types.ModuleType("sparktts.utils.token_parser")
    parser.__dict__.update(vars(MAPS))
    files = types.ModuleType("sparktts.utils.file")
    files.load_config = lambda path: json.loads('{"sample_rate": 16000}') if path.endswith("config.yaml") else {}  # type: ignore[attr-defined]
    for name, module in (
        ("sparktts.models.audio_tokenizer", audio_tokenizer),
        ("sparktts.utils.token_parser", parser),
        ("sparktts.utils.file", files),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    FakeLM.loaded.clear()
    FAKE_TOKENIZER.vocab.clear()
    return seen


def test_load_follows_the_recipe(tmp_path: Path, fake_libraries: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    ctx = LoadContext("spark-tts", "tts", root, {"voices": [{"id": "deep", "spark_gender": "male", "pitch": "low"}]})
    model = spark_tts.load(ctx)
    assert fake_libraries["bundles"] == ["spark-tts"]  # the authors' code, pinned, before anything is imported
    assert [v.id for v in model.voices] == ["deep"] and model.sample_rate == 16000
    chunks = list(model.synthesize("Hello there."))
    assert len(chunks) == 1 and len(chunks[0]) == 640
    lm = FakeLM.loaded[-1]
    # the reference's sampling
    assert lm.calls[-1] == {"do_sample": True, "max_new_tokens": 3000, "temperature": 0.8, "top_k": 50, "top_p": 0.95}
    codec = model.backend.audio_tokenizer  # type: ignore[attr-defined]
    assert codec.model_dir == root and str(codec.device) == "cpu"
    assert codec.seen[-1] == ((1, 32), (1, 2))  # detokenize(global (1, 32), semantic (1, n)), as SparkTTS.inference
    # a cloned voice: the global tokens come from the reference (BiCodecTokenizer.tokenize gives (1, 1, 32))
    assert model.backend.tokenize("x.wav") == (list(range(32)), [7, 8, 9])


def test_default_voices_and_bolos_reference_clips(tmp_path: Path, fake_libraries: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    options = {"voices": [{"id": "heart", "name": "Heart", "reference": "en-short"}]}
    model = spark_tts.load(LoadContext("spark-tts", "tts", root, options))
    ids = [v.id for v in model.voices]
    assert ids[:2] == ["female", "male"]  # no created voice in the catalog: the two defaults
    assert "heart" in ids  # bolo's built-in clip with its transcript
    heart = model.voice("heart")
    assert Path(heart["ref_audio"]).is_file() and heart["ref_text"]


def test_a_folder_without_the_model_files_is_refused(tmp_path: Path, fake_libraries: dict[str, Any]) -> None:
    root = model_folder(tmp_path)
    (root / "BiCodec" / "model.safetensors").unlink()
    with pytest.raises(InvalidRequest, match=r"BiCodec/model\.safetensors"):
        spark_tts.load(LoadContext("spark-tts", "tts", root, {}))


def test_the_bundle_is_pinned() -> None:
    bundle = upstream.get("spark-tts")
    (part,) = bundle.parts
    assert part.repo == "SparkAudio/Spark-TTS" and len(part.commit) == 40 and part.license.startswith("Apache-2.0")
    assert "sparktts/models/audio_tokenizer.py" in part.files and "LICENSE" in part.files
    assert all(len(sha) == 64 for sha in part.files.values()) and bundle.extra == "spark-tts"
    assert LOADERS["spark-tts"] == ("bolo_torch.models.spark_tts", "load") and "spark-tts" not in NOT_IMPLEMENTED
