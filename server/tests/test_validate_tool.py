"""tools/validate_models.py: the machine it names, the speech it needs, which check each task gets, and a cleanup
that Windows blocks (WinError 32) not ending the run."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bolo import hardware as hwmod
from bolo.core import Bolo

TOOL = Path(__file__).resolve().parents[1] / "tools" / "validate_models.py"
pytestmark = pytest.mark.skipif(not TOOL.is_file(), reason="needs the repository checkout (tools/)")


@pytest.fixture(scope="module")
def vm() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_models", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_models"] = module
    spec.loader.exec_module(module)
    return module


def tone(freq: float, seconds: float = 1.5, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)


def test_the_cpu_name_comes_from_bolo_hardware(vm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows there is no /proc/cpuinfo: the name comes from bolo.hardware's detection (the registry there)."""
    laptop = hwmod.Hardware(
        system="windows",
        cpu=hwmod.Cpu(model="AMD Ryzen 7 5800H with Radeon Graphics", architecture="x86_64", logical_cores=16),
        gpus=[
            hwmod.Gpu("amd", "AMD Radeon(TM) Graphics", integrated=True, backend="dml"),
            hwmod.Gpu("nvidia", "NVIDIA GeForce RTX 3060 Laptop GPU", backend="cuda"),
        ],
        device="cpu",
    )
    monkeypatch.setattr(hwmod, "detect", lambda *a, **k: laptop)
    text = vm.hardware()
    assert text.startswith("16 vCPU x86_64 (AMD Ryzen 7 5800H with Radeon Graphics)")
    assert "RTX 3060" in text and "not used" in text and "Radeon(TM)" not in text
    plain = hwmod.Hardware(cpu=hwmod.Cpu(model="Intel Xeon", architecture="x86_64", logical_cores=4))
    assert vm.hardware(plain) == "4 vCPU x86_64 (Intel Xeon), no GPU"


def test_speech_without_an_english_voice(vm: ModuleType) -> None:
    with pytest.raises(RuntimeError, match="--en-voice"):
        vm.sample_speech({})
    hindi = (tone(200), 16000, "text", "piper-hi")
    assert vm.sample_speech({"hi": hindi}) is hindi  # another language's speech rather than a KeyError
    english = (tone(300), 16000, "text", "kokoro")
    assert vm.sample_speech({"hi": hindi, "en": english}) is english


async def test_each_task_gets_its_own_check(vm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    for kind in ("stt", "tts", "vad", "diarization", "turn"):

        async def fake(*a: Any, _kind: str = kind, **k: Any) -> dict[str, Any]:
            called.append(_kind)
            return {}

        monkeypatch.setattr(vm, f"validate_{kind}", fake)
    args = argparse.Namespace(stt="fake-stt")
    for task in ("stt", "tts", "vad", "diarization", "turn"):
        await vm.validate(None, SimpleNamespace(task=task), args, Path(), {})
    assert called == ["stt", "tts", "vad", "diarization", "turn"]  # diarization and turn no longer go to VAD
    with pytest.raises(RuntimeError, match="no validation for separation"):
        await vm.validate(None, SimpleNamespace(task="separation"), args, Path(), {})


async def test_diarization_check_runs_the_model(vm: ModuleType, core: Bolo) -> None:
    spec = core.catalog.resolve("fake-diarization")
    synth = {"en": (tone(150), 16000, "", "a"), "hi": (tone(420), 16000, "", "b")}
    out = await vm.validate_diarization(core, spec, synth)
    assert out["turns"] >= 2 and out["speakers"] == 2 and out["rtf"] >= 0


async def test_a_blocked_cleanup_is_recorded_and_the_run_goes_on(
    vm: ModuleType, core: Bolo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a failed load the tool removes the model it pulled; Windows' WinError 32 on manifest.json ended the
    whole run. Now the store waits it out a few times, and the tool records what is left and goes on."""
    from bolo import store as store_mod

    mdir = core.store.model_dir("fake-tts")
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_text('{"id": "fake-tts", "complete": true, "files": []}', encoding="utf-8")

    def in_use(path: Path) -> None:
        exc = PermissionError(13, "The process cannot access the file because it is being used by another process")
        exc.filename = str(path / "manifest.json")
        exc.winerror = 32  # type: ignore[attr-defined]
        raise exc

    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)
    core.store.remover = in_use
    problem = await vm.cleanup(core, "fake-tts")
    assert problem and "manifest.json" in problem and "in use by another program" in problem

    async def os_error(name: str) -> bool:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(core, "remove", os_error)
    assert "PermissionError" in (await vm.cleanup(core, "fake-tts") or "")
