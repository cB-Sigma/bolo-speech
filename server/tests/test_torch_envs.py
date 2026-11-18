"""GPU builds of PyTorch for the bolo-torch plugin's environments (bolo.torch_envs, `bolo torch`, the torch-*
components). Fakes only: no pip, no nvidia-smi, no download."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from bolo import torch_envs
from bolo.config import Settings
from bolo.core import Bolo
from bolo.torch_envs import Gpu, TorchEnv

NVIDIA_SMI = """Wed Nov 18 15:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.83                 Driver Version: 572.83         CUDA Version: 12.8     |
+-----------------------------------------------------------------------------------------+
"""


@pytest.fixture(autouse=True)
def _host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here depends on the machine the tests run on: no uv on PATH and a Linux platform, unless a test
    says otherwise."""
    from bolo import tts_runtimes

    monkeypatch.setattr(torch_envs, "which", lambda name: None)
    monkeypatch.setattr(tts_runtimes, "platform_key", lambda: "linux-x86_64")


def test_pip_or_uv() -> None:
    assert torch_envs.pip_install("py") == ["py", "-m", "pip", "install"]
    torch_envs.which = lambda name: "/usr/bin/uv" if name == "uv" else None  # (restored by the fixture)
    assert torch_envs.pip_install("py") == ["/usr/bin/uv", "pip", "install", "--python", "py"]


def page(index: str, package: str) -> str:
    """PyTorch index pages as they read: links to wheels, ``+`` written as ``%2B``."""
    versions = {
        ("cu130", "torch"): ["2.14.1"],
        ("cu128", "torch"): ["2.9.0", "2.9.1", "2.11.0"],
        ("cu126", "torch"): ["2.6.0", "2.9.0", "2.14.1"],
        ("cu128", "torchaudio"): ["2.9.0", "2.9.1", "2.11.0"],
        ("cu128", "torchvision"): ["0.24.0", "0.24.1", "0.26.0"],
        ("cu126", "torchaudio"): ["2.6.0", "2.9.0", "2.11.0"],
        ("cu126", "torchvision"): ["0.21.0", "0.24.0", "0.29.1"],
    }.get((index, package), [])
    links = []
    for v in versions:
        for plat in ("win_amd64", "manylinux_2_28_x86_64"):
            name = f"{package}-{v}%2B{index}-cp312-cp312-{plat}.whl"
            links.append(f'<a href="/whl/{index}/{name}#sha256=00">{name.replace("%2B", "+")}</a>')
        links.append(f'<a href="x">{package}-{v}%2B{index}-cp311-cp311-win_amd64.whl</a>')
    return "\n".join(links)


def windows_env(torch: str = "2.9.0+cpu", audio: str | None = "2.9.0+cpu", vision: str | None = None) -> TorchEnv:
    return TorchEnv(
        ["plugin:torch"], r"C:\bolo-env\Scripts\python.exe", "cp312", "win-amd64", "0.1.6",
        {"torch": torch, "torchaudio": audio, "torchvision": vision},
    )  # fmt: skip


def test_the_index_follows_the_driver() -> None:
    def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
        if list(cmd) == ["nvidia-smi"]:
            return 0, NVIDIA_SMI
        return 0, "NVIDIA GeForce RTX 4060 Laptop GPU\n"

    gpu = torch_envs.detect_gpu(runner, which=lambda name: name if name == "nvidia-smi" else None)
    assert gpu == Gpu("nvidia", "NVIDIA GeForce RTX 4060 Laptop GPU", (12, 8)) and gpu.target == "cuda"
    assert torch_envs.index_for("cuda", gpu)[0] == "cu128"  # not cu129 or cu130: the driver runs CUDA 12.8 at most
    assert torch_envs.index_for("cuda", Gpu("nvidia", None, (12, 4)))[:2] == ["cu124", "cu121"]
    assert torch_envs.index_for("cuda", Gpu())[0] == "cu130"  # driver unknown: newest first
    assert torch_envs.index_for("rocm", Gpu("amd", None, (6, 4)))[0] == "rocm6.4"
    assert torch_envs.index_for("xpu") == ["xpu"] and torch_envs.index_for("cpu") == ["cpu"]
    assert torch_envs.detect_gpu(runner, which=lambda name: None) == Gpu()
    with pytest.raises(ValueError):
        torch_envs.index_for("tpu")


def test_builds_and_labels() -> None:
    assert torch_envs.build_of("2.15.0+cpu") == "cpu"
    assert torch_envs.build_of("2.15.0", "win32") == "cpu"  # PyPI's Windows wheel
    assert torch_envs.build_of("2.15.0", "linux") == "pypi-cuda"  # PyPI's Linux wheel is a CUDA build
    assert torch_envs.family_of("cu128") == "cuda" and torch_envs.family_of("rocm6.4") == "rocm"
    assert torch_envs.build_label("cpu", "RTX 4060") == "CPU build, GPU not used"
    assert torch_envs.build_label("cu128") == "CUDA 12.8 build" and torch_envs.build_label("cu130") == "CUDA 13.0 build"


def test_the_plan_keeps_the_installed_version_and_matches_torchaudio() -> None:
    gpu = Gpu("nvidia", "RTX 4060", (12, 8))
    p = torch_envs.plan(windows_env(), "cuda", gpu, page)
    assert p.index == "cu128" and p.pins == {"torch": "2.9.0", "torchaudio": "2.9.0"} and p.error is None
    assert p.command() == [
        r"C:\bolo-env\Scripts\python.exe", "-m", "pip", "install", "torch==2.9.0+cu128", "torchaudio==2.9.0+cu128",
        "--index-url", "https://download.pytorch.org/whl/cu128", "--extra-index-url", "https://pypi.org/simple",
    ]  # fmt: skip
    # 2.14.1 is not on cu128 (the best index for this driver) but is on cu126: kept, from there
    p = torch_envs.plan(windows_env("2.14.1+cpu", None, "0.29.1+cpu"), "cuda", gpu, page)
    assert p.index == "cu126" and p.pins == {"torch": "2.14.1", "torchvision": "0.29.1"}
    # a version no index of this driver has: the newest of the best index, said in a note
    p = torch_envs.plan(windows_env("2.15.0+cpu", "2.11.0+cpu"), "cuda", gpu, page)
    assert p.index == "cu128" and p.pins == {"torch": "2.11.0", "torchaudio": "2.11.0"} and p.notes
    p = torch_envs.plan(windows_env(), "cuda", gpu, page, latest=True)
    assert p.pins["torch"] == "2.11.0"
    assert torch_envs.plan(windows_env("2.9.0+cu128"), "cuda", gpu, page).done
    p = torch_envs.plan(windows_env(), "rocm", Gpu("amd", None, (6, 4)), page)
    assert p.error and "no rocm build" in p.error and p.command() == []
    other = TorchEnv(["plugin:torch"], "py", "cp311", "linux-aarch64", "0.1.6", {"torch": "2.9.0+cpu"})
    assert torch_envs.plan(other, "cuda", gpu, page).error  # no wheels for that Python and platform


def test_environments_are_found_from_the_plugin_commands(tmp_path: Path) -> None:
    venv = tmp_path / "glm" / ("Scripts" if sys.platform == "win32" else "bin")
    venv.mkdir(parents=True)
    script = venv / ("bolo-torch.exe" if sys.platform == "win32" else "bolo-torch")
    script.write_text("", encoding="utf-8")
    script.chmod(0o755)
    python = venv / ("python.exe" if sys.platform == "win32" else "python")
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    assert torch_envs.python_of([str(script), "serve"]) == str(python)  # a console script: the Python beside it
    assert torch_envs.python_of(["/opt/env/bin/python3.12", "-m", "bolo_torch"]) == "/opt/env/bin/python3.12"
    assert torch_envs.python_of(["no-such-program"]) is None
    probes: list[str] = []

    def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
        probes.append(cmd[0])
        has = cmd[0] != "/opt/other/bin/python"
        return 0, json.dumps({
            "tag": "cp312", "platform": "linux-x86_64", "bolo_torch": "0.1.6" if has else None,
            "torch": "2.9.0+cpu", "torchaudio": None, "torchvision": None,
        })  # fmt: skip

    commands = {
        "plugin:torch": [str(python), "-m", "bolo_torch"],
        "plugin:glm": [str(script)],  # the same environment: one entry, both plugins
        "plugin:other": ["/opt/other/bin/python", "-m", "other_plugin"],  # no bolo-torch: not listed
    }
    envs = torch_envs.discover(commands, runner)
    assert [(e.plugins, e.python, e.torch) for e in envs] == [
        (["plugin:torch", "plugin:glm"], str(python), "2.9.0+cpu")
    ]
    assert len(probes) == 2
    settings_commands = torch_envs.plugin_commands(["glm=/opt/glm/bin/python -m bolo_torch"], entry_points=False)
    assert settings_commands == {"plugin:glm": ["/opt/glm/bin/python", "-m", "bolo_torch"]}
    lines = torch_envs.doctor_lines(envs, Gpu("nvidia", "RTX 4060", (12, 8)))
    assert len(lines) == 1 and "torch 2.9.0+cpu, a CPU build" in lines[0] and "bolo torch use cuda" in lines[0]
    assert torch_envs.doctor_lines(envs, Gpu()) == []  # no GPU: nothing to say


def test_apply_stops_the_plugin_installs_and_checks() -> None:
    gpu = Gpu("nvidia", "RTX 4060", (12, 8))
    plans = [torch_envs.plan(windows_env(), "cuda", gpu, page)]
    ran: list[list[str]] = []
    stopped: list[list[str]] = []

    def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
        ran.append(list(cmd))
        if "-c" in cmd:
            return 0, json.dumps({"version": "2.9.0+cu128", "cuda": True, "hip": None, "xpu": False,
                                  "device": "NVIDIA GeForce RTX 4060 Laptop GPU"})  # fmt: skip
        return 0, "Successfully installed torch-2.9.0+cu128"

    (result,) = torch_envs.apply(plans, runner, stopped.append)
    assert stopped == [["plugin:torch"]] and ran[0] == plans[0].command()
    assert result["ok"] and result["detail"].startswith("torch 2.9.0+cu128 on NVIDIA GeForce RTX 4060 Laptop GPU")

    def no_gpu(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
        if "-c" in cmd:
            return 0, json.dumps({"version": "2.9.0+cu128", "cuda": False, "hip": None, "xpu": False, "device": None})
        return 0, ""

    (result,) = torch_envs.apply(plans, no_gpu)
    assert not result["ok"] and "does not reach the GPU" in result["detail"]


def test_the_cli_dry_run_prints_the_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from bolo import cli

    monkeypatch.setattr(torch_envs, "plugin_commands", lambda plugins, entry_points=True: {"plugin:torch": ["py"]})
    monkeypatch.setattr(torch_envs, "discover", lambda commands, runner=None: [windows_env()])
    monkeypatch.setattr(torch_envs, "detect_gpu", lambda runner=None: Gpu("nvidia", "RTX 4060", (12, 8)))
    monkeypatch.setattr(torch_envs, "fetch_page", page)
    monkeypatch.setattr(torch_envs, "apply", lambda *a, **k: pytest.fail("a dry run installs nothing"))
    assert cli.main(["torch", "use", "cuda", "--dry-run", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "GPU: nvidia RTX 4060, CUDA 12.8" in out and "torch 2.9.0+cpu (CPU build, GPU not used)" in out
    assert "torch==2.9.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128" in out
    assert cli.main(["components", "install", "torch-cuda", "--dry-run", "--data-dir", str(tmp_path)]) == 0
    assert "torch==2.9.0+cu128" in capsys.readouterr().out
    assert cli.main(["torch", "--data-dir", str(tmp_path)]) == 0
    assert "WARNING" in capsys.readouterr().out and cli.main(["torch", "use", "--data-dir", str(tmp_path)]) == 1


async def test_the_component_installs_the_gpu_build_and_restarts_the_plugin(settings: Settings) -> None:
    core = Bolo(settings)
    await core.start()
    try:
        manager = core.components
        env = windows_env()
        installed = {"torch": "2.9.0+cpu"}

        def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
            if cmd[0] == "nvidia-smi":
                return 0, NVIDIA_SMI if len(cmd) == 1 else "RTX 4060\n"
            if "-c" in cmd and cmd[2] == torch_envs.VERIFY:
                return 0, json.dumps({"version": installed["torch"], "cuda": True, "hip": None, "xpu": False,
                                      "device": "RTX 4060"})  # fmt: skip
            installed["torch"] = "2.9.0+cu128"
            return 0, ""

        manager.torch_runner, manager.torch_page = runner, page
        found = [env]
        manager.torch_state = lambda fresh=False: (found, torch_envs.detect_gpu(runner, lambda n: n))  # type: ignore[method-assign]
        (row,) = [c for c in manager.status() if c["id"] == "torch-cuda"]
        assert row["id"] == "torch-cuda" and not row["installed"] and row["installable"]
        assert row["envs"][0]["label"] == "CPU build, GPU not used" and row["size_bytes"] == torch_envs.DOWNLOAD_BYTES
        stopped: list[list[str]] = []

        async def stop(plugins: list[str]) -> None:
            stopped.append(plugins)

        manager.stop_plugins = stop  # type: ignore[method-assign]
        job = manager.start_install("torch-cuda", accept_license=False)
        for _ in range(200):
            if job.state in ("done", "error"):
                break
            await asyncio.sleep(0.02)
        assert job.state == "done", job.error
        assert stopped == [["plugin:torch"]]
        found[:] = [windows_env("2.9.0+cu128", "2.9.0+cu128")]
        (row,) = [c for c in manager.status() if c["id"] == "torch-cuda"]
        assert row["installed"] and row["envs"][0]["label"] == "CUDA 12.8 build"
    finally:
        await core.stop()


def test_triton_windows_follows_the_torch_version() -> None:
    assert torch_envs.triton_windows_requirement("2.9.1+cu128") == "triton-windows>=3.5,<3.6"
    assert torch_envs.triton_windows_requirement("2.4.1+cu124") == "triton-windows>=3.1,<3.2"
    assert torch_envs.triton_windows_requirement("2.6.0") == "triton-windows>=3.2,<3.3"
    assert torch_envs.triton_windows_requirement("2.15.0+cu130") is None  # no series known: not guessed
    assert torch_envs.triton_windows_requirement(None) is None


def test_torch_use_cuda_on_windows_adds_triton() -> None:
    gpu = Gpu("nvidia", "RTX 3060", (12, 8))
    plans = [torch_envs.plan(windows_env(), "cuda", gpu, page)]
    ran: list[list[str]] = []

    def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
        ran.append(list(cmd))
        if "-c" in cmd:
            return 0, json.dumps({"version": "2.9.0+cu128", "cuda": True, "hip": None, "xpu": False, "device": "RTX"})
        return 0, ""

    (result,) = torch_envs.apply(plans, runner)
    assert result["ok"] and ran[-1][-1] == "triton-windows>=3.5,<3.6" and "triton-windows" in result["detail"]
    linux = TorchEnv(["plugin:torch"], "/env/bin/python", "cp312", "linux-x86_64", "0.1.6", {"torch": "2.9.0+cpu"})
    ran.clear()
    torch_envs.apply([torch_envs.plan(linux, "cuda", gpu, page)], runner)
    assert not any("triton-windows" in " ".join(c) for c in ran)  # Linux's CUDA torch brings Triton


async def test_the_triton_component_on_windows(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from bolo import tts_runtimes

    monkeypatch.setattr(tts_runtimes, "platform_key", lambda: "windows-x86_64")
    core = Bolo(settings)
    await core.start()
    try:
        manager = core.components
        targets = [{"name": "moshi", "python": r"C:\data\moshi\Scripts\python.exe", "torch": "2.9.1+cu128",
                    "triton": False}]  # fmt: skip
        manager.triton_targets = lambda: targets  # type: ignore[method-assign]
        ran: list[list[str]] = []

        def runner(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
            if "install" not in cmd:
                return 1, ""  # nvidia-smi, probes: nothing here
            ran.append(list(cmd))
            targets[0]["triton"] = True
            return 0, ""

        manager.torch_runner = runner
        (row,) = [c for c in manager.status() if c["id"] == "triton-windows"]
        assert not row["installed"] and row["installable"] and "speed-up off" in row["envs"][0]["label"]
        job = manager.start_install("triton-windows", accept_license=False)
        for _ in range(200):
            if job.state in ("done", "error"):
                break
            await asyncio.sleep(0.02)
        assert job.state == "done", job.error
        assert ran[0] == [r"C:\data\moshi\Scripts\python.exe", "-m", "pip", "install", "triton-windows>=3.5,<3.6"]
        (row,) = [c for c in manager.status() if c["id"] == "triton-windows"]
        assert row["installed"]
        monkeypatch.setattr(tts_runtimes, "platform_key", lambda: "linux-x86_64")
        assert not [c for c in manager.status() if c["id"] == "triton-windows"]  # Linux: not listed
    finally:
        await core.stop()
