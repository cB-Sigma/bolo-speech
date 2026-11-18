"""whisper.cpp's DTW word times stay off with a pywhispercpp whose whisper.cpp aborts on a short segment."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from bolo.engines import whispercpp
from bolo.engines.base import LoadOptions
from bolo.engines.whispercpp import WhisperCppSTT, dtw_is_safe


def test_dtw_is_safe_only_after_the_last_binding_that_aborts() -> None:
    assert not dtw_is_safe("1.5.1") and not dtw_is_safe("1.4.0") and not dtw_is_safe(None) and not dtw_is_safe("")
    assert dtw_is_safe("1.5.2") and dtw_is_safe("1.6.0") and dtw_is_safe("2.0")


class FakeCppModel:
    def __init__(self, path: str, context_params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.context_params = context_params or {}


def load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binding: str, spec: dict[str, Any] | None = None) -> Any:
    pw = types.SimpleNamespace(WHISPER_AHEADS_LARGE_V3_TURBO=7, WHISPER_AHEADS_N_TOP_MOST=1)
    monkeypatch.setitem(sys.modules, "_pywhispercpp", pw)
    monkeypatch.setattr(whispercpp, "_binding_version", lambda: binding)
    stt = WhisperCppSTT.__new__(WhisperCppSTT)
    model = stt._load(FakeCppModel, tmp_path / "ggml-large-v3-turbo.bin", LoadOptions(device="coreml"), spec or {})
    return stt, model


def test_a_binding_that_aborts_loads_without_dtw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stt, model = load(monkeypatch, tmp_path, "1.5.1")
    assert stt.dtw is False and "dtw_token_timestamps" not in model.context_params
    assert model.context_params == {"use_gpu": True}  # the GPU it was placed on, still


def test_a_fixed_binding_loads_with_dtw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stt, model = load(monkeypatch, tmp_path, "1.6.0")
    assert stt.dtw is True and model.context_params["dtw_token_timestamps"] is True
    stt, model = load(monkeypatch, tmp_path, "1.6.0", {"dtw": False})  # the catalog can still turn it off
    assert stt.dtw is False and "dtw_token_timestamps" not in model.context_params
