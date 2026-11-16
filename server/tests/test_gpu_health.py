"""A lost GPU (bolo.gpu_health): the check against fake NVML and a fake nvidia-smi, what the model manager does
about it (degraded models, CPU fallback or failing requests, `bolo ps`, /healthz, doctor), and a fake llama-server
that prints CUDA errors. Plus the Vulkan environment for hybrid AMD + NVIDIA laptops."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo import cli, gpu_health, tts_runtimes, winhw
from bolo.accel import Accelerators
from bolo.catalog import ModelSpec
from bolo.config import Settings, load_settings
from bolo.core import Bolo
from bolo.engines.base import LoadOptions
from bolo.engines.tts_server import ServerProcess, TTSServerEngine, server_env
from bolo.errors import EngineUnavailable
from bolo.gpu_health import GpuHealth, GpuMonitor, check, gpu_error_line
from conftest import LiveServer
from test_placement import FakeNvml, add, write_gguf
from test_tts_server import fake_binary

LOST_SMI = (
    "Unable to determine the device handle for GPU0: 0000:01:00.0: GPU is lost.  Reboot the system to recover "
    "this GPU\n"
)


class NVMLError(Exception):
    """pynvml's error: the code in ``value``, the text in ``str()``."""

    def __init__(self, value: int, text: str) -> None:
        super().__init__(text)
        self.value = value


class LostNvml(FakeNvml):
    def nvmlDeviceGetHandleByIndex(self, i: int) -> int:
        raise NVMLError(gpu_health.NVML_ERROR_GPU_IS_LOST, "GPU is lost")


class NoGpuNvml(FakeNvml):
    def nvmlDeviceGetCount(self) -> int:
        return 0


def smi(code: int | None, text: str) -> Any:
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float) -> tuple[int | None, str]:
        calls.append(argv)
        return code, text

    run.calls = calls  # type: ignore[attr-defined]
    return run


# -- the check ----------------------------------------------------------------------------------------------------
def test_nvml_says_whether_the_gpu_is_there() -> None:
    ok = check(nvml=FakeNvml(), which=lambda _: None)
    assert ok.state == "ok" and ok.gpus == 1 and ok.source == "nvml" and not ok.lost and ok.checked_at
    lost = check(nvml=LostNvml(), which=lambda _: None)
    assert lost.lost and lost.source == "nvml" and "GPU is lost" in (lost.reason or "")
    gone = check(nvml=NoGpuNvml(), which=lambda _: None)
    assert gone.lost and "no NVIDIA GPU" in (gone.reason or "")
    # with no GPU model loaded, finding none is not a loss (doctor on a machine without one)
    assert check(nvml=NoGpuNvml(), which=lambda _: None, expect_gpu=False).state == "none"


def test_nvidia_smi_says_whether_the_gpu_is_there() -> None:
    which = {"nvidia-smi": "/usr/bin/nvidia-smi"}.get
    lost = check(nvml=None, which=which, runner=smi(15, LOST_SMI))
    assert lost.lost and lost.source == "nvidia-smi" and lost.reason == f"nvidia-smi: {LOST_SMI.strip()}"
    none = check(nvml=None, which=which, runner=smi(6, "No devices were found\n"))
    assert none.lost and none.reason == "nvidia-smi: No devices were found"
    assert check(nvml=None, which=which, runner=smi(6, "No devices were found\n"), expect_gpu=False).state == "none"
    fine = check(nvml=None, which=which, runner=smi(0, "0, NVIDIA GeForce RTX 3060 Laptop GPU, 1500 MiB\n"))
    assert fine.state == "ok" and fine.gpus == 1
    hung = check(nvml=None, which=which, runner=smi(None, ""))
    assert hung.state == "unknown" and "did not answer" in (hung.reason or "")
    assert check(nvml=None, which=lambda _: None).state == "unknown"
    # NVML answers for itself when it can; nvidia-smi is not started then
    runner = smi(15, LOST_SMI)
    assert check(nvml=FakeNvml(), which=which, runner=runner).state == "ok" and runner.calls == []  # type: ignore[attr-defined]


def test_gpu_error_lines() -> None:
    for line in (
        "CUDA error: unspecified launch failure",
        "ggml_cuda_compute_forward: GGML_OP_MUL_MAT failed",
        "CUDA error: cudaErrorDevicesUnavailable",
        "cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or unavailable",
        "ggml_vulkan: Device lost: vk::DeviceLostError ErrorDeviceLost",
    ):
        assert gpu_error_line(line), line
    for line in (
        "ggml_cuda_init: found 1 CUDA devices:",
        "  Device 0: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, VMM: yes",
        "CUDA error: out of memory",
        "load_tensors: offloaded 29/29 layers to GPU",
    ):
        assert not gpu_error_line(line), line


def test_notice_and_dict() -> None:
    h = GpuHealth("lost", "nvidia-smi: No devices were found", "nvidia-smi", fallback="cpu")
    assert h.notice() == "GPU LOST (nvidia-smi: No devices were found): reboot to recover; bolo is using the CPU"
    h.fallback = "fail"
    assert "fail" in (h.notice() or "")
    d = h.to_dict()
    assert d["lost"] is True and d["state"] == "lost" and d["notice"] == h.notice()
    assert GpuHealth("ok").notice() is None and GpuHealth("ok").to_dict()["fallback"] is None


def test_settings() -> None:
    s = load_settings(env={"BOLO_GPU_LOST_FALLBACK": "fail", "BOLO_GPU_HEALTH_INTERVAL_S": "5"})
    assert s.gpu_lost_fallback == "fail" and s.gpu_health_interval_s == 5.0
    assert Settings().gpu_lost_fallback == "cpu" and Settings().gpu_health_interval_s == 30.0
    with pytest.raises(ValueError, match="gpu_lost_fallback"):
        load_settings(env={"BOLO_GPU_LOST_FALLBACK": "gpu"})


# -- the monitor --------------------------------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.answer = GpuHealth("ok", None, "nvml", gpus=1)
        self.calls = 0

    def __call__(self) -> GpuHealth:
        self.calls += 1
        return GpuHealth(self.answer.state, self.answer.reason, self.answer.source, self.answer.gpus)


async def until(condition: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


async def test_monitor_checks_periodically_and_acts_once() -> None:
    checker, lost = Checker(), []

    async def on_lost(h: GpuHealth) -> None:
        lost.append(h)

    monitor = GpuMonitor(interval_s=0.05, has_gpu_models=lambda: True, on_lost=on_lost, checker=checker)
    try:
        monitor.ensure_running()
        await until(lambda: checker.calls >= 2)
        assert monitor.health.state == "ok" and not lost
        checker.answer = GpuHealth("lost", "NVML: GPU is lost", "nvml")
        await until(lambda: monitor.lost)
        await asyncio.sleep(0.2)
        assert len(lost) == 1 and lost[0].lost_at and monitor.health.fallback == "cpu"
        calls = checker.calls
        await asyncio.sleep(0.2)
        assert checker.calls == calls  # lost stays lost: no more checks
    finally:
        await monitor.close()


async def test_monitor_stops_without_gpu_models() -> None:
    checker = Checker()
    gpu_models = [True]

    async def on_lost(h: GpuHealth) -> None:
        pass

    monitor = GpuMonitor(interval_s=0.02, has_gpu_models=lambda: gpu_models[0], on_lost=on_lost, checker=checker)
    monitor.ensure_running()
    await until(lambda: checker.calls >= 1)
    gpu_models[0] = False
    assert monitor._task is not None
    await asyncio.wait_for(monitor._task, 2.0)
    await monitor.close()


async def test_a_slow_request_asks_for_a_check_now() -> None:
    checker = Checker()

    async def on_lost(h: GpuHealth) -> None:
        pass

    monitor = GpuMonitor(interval_s=3600, has_gpu_models=lambda: True, on_lost=on_lost, checker=checker)
    try:
        monitor.ensure_running()
        await asyncio.sleep(0.05)
        assert checker.calls == 0
        # from a worker thread, as requests finish: 11.8 and 12.1 are this model's normal, 29.4 is not
        for rtf in (11.8, 12.1, 13.0):
            await asyncio.to_thread(monitor.note_rtf, "svara", rtf)
        await asyncio.sleep(0.1)
        assert checker.calls == 0
        await asyncio.to_thread(monitor.note_rtf, "svara", 29.4)
        await until(lambda: checker.calls == 1)
        await asyncio.to_thread(monitor.note_rtf, "svara", 60.0)  # within the rate limit: no second check
        await asyncio.sleep(0.1)
        assert checker.calls == 1
    finally:
        await monitor.close()


# -- the model manager --------------------------------------------------------------------------------------------
@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch) -> FakeNvml:
    from bolo.engines.fake import FakeEngine

    nv = FakeNvml()
    real = FakeEngine.load

    def load(self: FakeEngine, kind: str, task: str, path: Path, spec: dict[str, Any], opts: LoadOptions) -> Any:
        model = real(self, kind, task, path, spec, opts)
        model.seen_device = opts.device  # type: ignore[attr-defined]
        return model

    monkeypatch.setattr(FakeEngine, "load", load)
    return nv


async def gpu_core(settings: Settings, nv: FakeNvml, checker: Checker) -> Bolo:
    settings.device = "cuda"
    core = Bolo(settings)
    core.manager.accel = Accelerators(nvml=nv, which=lambda _: None, max_age_s=0)
    core.manager.gpu._checker = checker
    await core.start()
    return core


async def test_lost_gpu_moves_models_to_the_cpu(settings: Settings, gpu: FakeNvml) -> None:
    checker = Checker()
    core = await gpu_core(settings, gpu, checker)
    try:
        add(core, "voice", task="tts")
        add(core, "cpu-model")
        core.store.set_model_settings("cpu-model", {"device": "cpu"})
        voice = await core.manager.load("voice")
        voice.keep_alive_s = -1
        await core.manager.load("cpu-model")
        assert voice.model.seen_device == "cuda:0" and core.manager.has_gpu_models()  # type: ignore[attr-defined]
        assert (await core.manager.gpu.check_now()).state == "ok"
        assert core.health()["status"] == "ok"

        checker.answer = GpuHealth("lost", f"nvidia-smi: {LOST_SMI.strip()}", "nvidia-smi")
        health = await core.manager.gpu.check_now()
        assert health.lost and health.moved == ["voice"] and health.degraded == []
        moved = core.manager.loaded["voice"]
        assert moved is not voice and moved.model.seen_device == "cpu"  # type: ignore[attr-defined]
        assert moved.placement.device == "cpu" and moved.processor() == "100% CPU" and moved.keep_alive_s == -1
        assert "the GPU was lost" in (moved.placement.notice or "") and "reboot to recover" in moved.placement.notice
        assert moved.degraded is None and not core.manager.has_gpu_models()
        # a model loaded from now on goes to the CPU
        add(core, "later")
        assert (await core.manager.load("later")).placement.device == "cpu"

        ps = core.ps()
        assert ps["system"]["gpu_health"]["lost"] is True
        text = cli.format_ps(ps)
        assert "GPU LOST" in text.splitlines()[0]
        assert "reboot to recover; bolo is using the CPU" in text and "moved to the CPU: voice" in text
        row = next(m for m in core.stats()["models"] if m["id"] == "voice")
        assert row["device"] == "cpu" and row["degraded"] is None
        info = core.health()
        assert info["status"] == "degraded" and info["gpu"]["moved_to_cpu"] == ["voice"]
        assert core.info()["health"]["gpu"]["lost"] is True
    finally:
        await core.stop()


async def test_lost_gpu_with_fallback_fail(settings: Settings, gpu: FakeNvml) -> None:
    settings.gpu_lost_fallback = "fail"
    checker = Checker()
    core = await gpu_core(settings, gpu, checker)
    try:
        add(core, "voice", task="tts")
        voice = await core.manager.load("voice")
        checker.answer = GpuHealth("lost", "NVML: GPU is lost", "nvml")
        health = await core.manager.gpu.check_now()
        assert health.lost and health.degraded == ["voice"] and health.moved == []
        assert core.manager.loaded["voice"] is voice and voice.degraded and "GPU was lost" in voice.degraded
        with pytest.raises(EngineUnavailable, match="gpu_lost_fallback = fail"):
            async with core.manager.use("voice"):
                pass
        add(core, "other")
        with pytest.raises(EngineUnavailable, match="GPU LOST"):
            await core.manager.load("other")
        text = cli.format_ps(core.ps())
        assert "GPU LOST" in text.splitlines()[0] and "voice" in text
        row = next(line for line in text.splitlines() if line.startswith("voice"))
        assert "GPU LOST" in row and "models placed on it fail" in text
    finally:
        await core.stop()


def test_healthz_and_info(settings: Settings) -> None:
    srv = LiveServer(settings)
    srv.start()
    try:
        assert httpx.get(f"{srv.url}/healthz").json() == {"status": "ok"}
        assert httpx.get(f"{srv.url}/api/v1/info").json()["health"]["status"] == "ok"
        srv.core.manager.gpu.health = GpuHealth("lost", "NVML: GPU is lost", "nvml", fallback="cpu")
        r = httpx.get(f"{srv.url}/healthz")
        assert r.status_code == 200 and r.json()["status"] == "degraded"
        assert r.json()["notice"] == "GPU LOST (NVML: GPU is lost): reboot to recover; bolo is using the CPU"
        info = httpx.get(f"{srv.url}/api/v1/info").json()["health"]
        assert info["status"] == "degraded" and info["gpu"]["reason"] == "NVML: GPU is lost"
        assert httpx.get(f"{srv.url}/api/v1/ps").json()["system"]["gpu_health"]["lost"] is True
    finally:
        srv.stop()


def test_doctor_reports_a_lost_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    def lost(**kw: Any) -> GpuHealth:
        return GpuHealth("lost", f"nvidia-smi: {LOST_SMI.strip()}", "nvidia-smi")

    which = {"nvidia-smi": "nvidia-smi"}.get
    cli._doctor_gpu_health(Settings(device="cuda"), which=which, checker=lost)
    out = capsys.readouterr().out
    assert out.startswith("gpu health: GPU LOST (nvidia-smi: Unable to determine the device handle")
    assert "reboot to recover; bolo is using the CPU" in out
    cli._doctor_gpu_health(Settings(device="cuda"), which=which, checker=lambda **kw: GpuHealth("ok", None, "nvml", 1))
    assert capsys.readouterr().out == "gpu health: ok (NVML: 1 NVIDIA GPU answer)\n"


# -- a managed llama-server that prints CUDA errors --------------------------------------------------------------
async def test_llama_server_cuda_errors_move_it_to_the_cpu(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_ENV_FILE", str(env_file))
    settings.llama_server_bin = str(fake_binary(tmp_path, "llama-server"))
    checker = Checker()
    core = await gpu_core(settings, FakeNvml(), checker)
    try:
        spec = ModelSpec.from_dict({
            "id": "svara", "task": "tts", "engine": "tts-server", "kind": "tts-server", "family": "X",
            "languages": ["en"], "license": "MIT", "sample_rate": 16000,
            "options": {"runtime": "llama-server", "model_file": "m.gguf"}, "source": {"repo": "x/y"},
        })  # fmt: skip
        core.catalog.add(spec)
        model_dir = core.store.model_dir(spec.id)
        model_dir.mkdir(parents=True, exist_ok=True)
        write_gguf(model_dir / "m.gguf", blocks=28)
        (model_dir / "manifest.json").write_text(
            json.dumps({"id": "svara", "files": [], "complete": True}), encoding="utf-8"
        )
        entry = await core.manager.load("svara")
        proc = entry.model.process  # type: ignore[attr-defined]
        assert entry.placement.device == "cuda:0" and proc.gpu and "-ngl" in proc.argv
        assert "--device" not in proc.argv
        assert "CUDA_VISIBLE_DEVICES" not in json.loads(env_file.read_text(encoding="utf-8"))
        # a request makes it print llama.cpp's CUDA error: the GPU counts as lost, and the model moves
        out = await asyncio.to_thread(lambda: list(entry.model.synthesize("cuda error now")))
        assert out
        await until(lambda: core.manager.gpu.health.moved == ["svara"], timeout=40)
        health = core.manager.gpu.health
        assert health.lost and health.source == "llama-server:svara" and "CUDA error" in (health.reason or "")
        assert checker.calls == 0  # a server's own report needs no driver check
        assert not proc.running
        moved = core.manager.loaded["svara"]
        new = moved.model.process  # type: ignore[attr-defined]
        assert moved.placement.device == "cpu" and new.running and not new.gpu
        assert new.argv[new.argv.index("--device") + 1] == "none" and "-ngl" not in new.argv
        assert json.loads(env_file.read_text(encoding="utf-8"))["CUDA_VISIBLE_DEVICES"] == "-1"
        assert len(await asyncio.to_thread(lambda: list(moved.model.synthesize("Hello.")))) > 0
    finally:
        await core.stop()


def test_cpu_server_does_not_report_gpu_errors(tmp_path: Path) -> None:
    """A server started for the CPU prints CUDA's failure to start (GPUs hidden): that is not a lost GPU."""
    heard: list[tuple[str, str]] = []

    class Ear:
        def report(self, source: str, line: str) -> None:
            heard.append((source, line))

    ear = Ear()
    gpu_health._monitors.add(ear)  # type: ignore[arg-type]
    try:
        settings = Settings(data_dir=tmp_path / "data", llama_server_bin=str(fake_binary(tmp_path, "llama-server")))
        engine = TTSServerEngine()
        engine.configure(settings)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "m.gguf").write_bytes(b"GGUF")
        spec = {"runtime": "llama-server", "model_file": "m.gguf", "model_id": "m", "sample_rate": 16000}
        for device, reports in (("cpu", 0), ("cuda:0", 1)):
            model = engine.load("tts-server", "tts", model_dir, spec, LoadOptions(threads=1, device=device))
            try:
                list(model.synthesize("cuda error now"))
                list(model.synthesize("cuda error now"))  # once per start
                deadline = time.monotonic() + 5
                while len(heard) < reports and time.monotonic() < deadline:
                    time.sleep(0.02)
                time.sleep(0.2)
                assert len(heard) == reports, device
            finally:
                model.close()
        assert heard[0][0] == "llama-server:m" and heard[0][1].startswith("CUDA error")
    finally:
        gpu_health._monitors.discard(ear)  # type: ignore[arg-type]


# -- Vulkan on a hybrid AMD + NVIDIA laptop ----------------------------------------------------------------------
def adapter(vendor_id: int, name: str, index: int = 0, software: bool = False) -> winhw.Adapter:
    return winhw.Adapter(index, name, vendor_id, 512 * 1024**2, 8 * 1024**3, software)


AMD_IGPU = adapter(0x1002, "AMD Radeon(TM) Graphics", 0)
RTX = adapter(0x10DE, "NVIDIA GeForce RTX 3060 Laptop GPU", 1)


def test_vulkan_build_on_a_windows_laptop_with_amd_graphics() -> None:
    var = winhw.AMD_SWITCHABLE_LAYER_VAR
    assert var == "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"
    hybrid = [AMD_IGPU, RTX]
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=hybrid, environ={}) == {var: "1"}
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=lambda: hybrid, environ={}) == {var: "1"}
    # the operator's own value (any value, any letter case) is left alone
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=hybrid, environ={var: "0"}) == {}
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=hybrid, environ={var.lower(): "1"}) == {}
    # only the Vulkan build, only on Windows, only with an AMD adapter
    assert winhw.vulkan_child_env("cuda", windows=True, adapters=hybrid, environ={}) == {}
    assert winhw.vulkan_child_env("cpu", windows=True, adapters=hybrid, environ={}) == {}
    assert winhw.vulkan_child_env("vulkan", windows=False, adapters=hybrid, environ={}) == {}
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=[RTX], environ={}) == {}
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=[adapter(0x1002, "x", software=True)]) == {}
    assert winhw.vulkan_child_env("vulkan", windows=True, adapters=lambda: None, environ={}) == {}


def test_server_env_for_each_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    var = winhw.AMD_SWITCHABLE_LAYER_VAR
    monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(winhw, "dxgi_adapters", lambda include_software=False: [AMD_IGPU, RTX])
    monkeypatch.setattr(winhw, "_windows", lambda: True)
    binary = tmp_path / "vulkan" / "llama-server.exe"
    binary.parent.mkdir()
    binary.write_bytes(b"MZ")
    (binary.parent / "ggml-vulkan.dll").write_bytes(b"MZ")
    assert tts_runtimes.flavour_of(None, "llama-server", binary) == "vulkan"
    env = server_env(binary, "vulkan")
    assert env[var] == "1" and str(binary.parent) in env["LD_LIBRARY_PATH"]
    assert var not in server_env(binary, "cuda") and var not in server_env(binary, "cpu")
    lost = server_env(binary, "cuda", gpu_lost=True)
    assert lost["CUDA_VISIBLE_DEVICES"] == "-1"
    monkeypatch.setenv(var, "0")
    assert var not in server_env(binary, "vulkan")


def test_vulkan_env_reaches_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Vulkan build bolo installed, started for a model: the child's environment has the variable."""
    var = winhw.AMD_SWITCHABLE_LAYER_VAR
    monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(winhw, "dxgi_adapters", lambda include_software=False: [AMD_IGPU, RTX])
    monkeypatch.setattr(winhw, "_windows", lambda: True)
    env_file = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_ENV_FILE", str(env_file))
    settings = Settings(data_dir=tmp_path / "data", device="vulkan")
    folder = tts_runtimes.install_dir(settings, "llama-server", "vulkan")
    binary = fake_binary(folder, "llama-server")
    # where the release puts it: the version folder's "vulkan" build, found as that build
    assert tts_runtimes.find_binary(settings, "llama-server")[0] == binary
    assert tts_runtimes.flavour_of(settings, "llama-server", binary) == "vulkan"
    engine = TTSServerEngine()
    engine.configure(settings)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_bytes(b"GGUF")
    spec = {"runtime": "llama-server", "model_file": "m.gguf", "model_id": "m", "sample_rate": 16000}
    model = engine.load("tts-server", "tts", model_dir, spec, LoadOptions(threads=1, device="vulkan"))
    try:
        assert json.loads(env_file.read_text(encoding="utf-8"))[var] == "1"
        assert model.process.flavour == "vulkan"  # type: ignore[attr-defined]
    finally:
        model.close()


def test_server_that_hangs_names_the_vulkan_layer(tmp_path: Path) -> None:
    from fakebin import fake_program

    script = fake_program(tmp_path / "hangs", "import time\nprint('ggml_vulkan: Found 2 Vulkan devices', flush=True)\n"
                          "time.sleep(60)\n")  # fmt: skip
    proc = ServerProcess("hangs", [str(script)], 1, start_timeout_s=1, flavour="vulkan")
    with pytest.raises(EngineUnavailable, match=r"not ready after 1 s.*Vulkan build can hang"):
        proc.ensure_started()


def test_list_devices_probe_timeout_is_a_clear_error(tmp_path: Path) -> None:
    from bolo import hardware as hw
    from test_hardware import X86_AVX2, FakeHost, cpuinfo

    binary = fake_binary(tmp_path / "vk", "llama-server")
    (binary.parent / "ggml-vulkan.dll").write_bytes(b"MZ")
    host = FakeHost(files={"/proc/cpuinfo": cpuinfo(X86_AVX2, 8)}, cores=8)
    host.hangs.add(f"{binary} --list-devices")
    found = hw.detect(Settings(data_dir=tmp_path / "data", llama_server_bin=str(binary)), host, deep=True).runtimes[
        "llama-server"
    ]
    assert found["gpu_devices"] is None
    assert "--list-devices` did not answer within 20 s" in found["probe_error"]
    assert "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1=1" in found["probe_error"]
    lines = cli._hardware_lines(hw.detect(Settings(data_dir=tmp_path / "data", llama_server_bin=str(binary)), host,
                                          deep=True).to_dict() | {"speed": {}})  # fmt: skip
    assert any(line.startswith("warning: llama-server: `") for line in lines)


def test_real_probe_kills_a_command_that_hangs(tmp_path: Path) -> None:
    from bolo.hardware import Host
    from fakebin import fake_program

    script = fake_program(tmp_path / "hang", "import time\ntime.sleep(60)\n")
    t0 = time.monotonic()
    out, why = Host().probe([str(script)], timeout=1.0)
    assert out is None and why == "did not answer within 1 s" and time.monotonic() - t0 < 15
    out, why = Host().probe([str(fake_program(tmp_path / "ok", "print('Available devices:')\n"))], timeout=10.0)
    assert why is None and out is not None and "Available devices" in out
    out, why = Host().probe([str(tmp_path / "missing")], timeout=1.0)
    assert out is None and why


def test_threads_do_not_leak_monitors() -> None:
    before = len(gpu_health._monitors)

    async def on_lost(h: GpuHealth) -> None:
        pass

    def make() -> None:
        GpuMonitor(interval_s=1, has_gpu_models=lambda: False, on_lost=on_lost)

    t = threading.Thread(target=make)
    t.start()
    t.join()
    import gc

    gc.collect()
    assert len(gpu_health._monitors) <= before  # (others may go too)
