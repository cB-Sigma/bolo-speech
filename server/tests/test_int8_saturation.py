"""The float Whisper decoder on x86 CPUs without AVX-512 VNNI (``options.float_decoder_without_vnni``).

On an AMD EPYC 7763 (AVX2, no VNNI) sherpa-onnx's int8 whisper-large-v3-turbo decoder turned Telugu, Bengali and
Tamil into nonsense, while the same files gave the right text on Intel Xeons with AVX-512 VNNI and on ARM; with the
float32 decoder the EPYC gave the right text. The CPU is mocked here, so these hold on any machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from bolo import hardware
from bolo.catalog import Catalog
from bolo.core import source_here
from bolo.engines.sherpa import whisper_decoder
from test_hardware import FakeHost

EPYC = hardware.Cpu(architecture="x86_64", features=["sse4.2", "avx", "avx2", "fma", "f16c"])
XEON = hardware.Cpu(architecture="x86_64", features=["avx2", "avx512", "avx512_vnni"])
M4 = hardware.Cpu(architecture="arm64", features=["neon", "dotprod", "fp16"])


def test_which_cpus_may_saturate_int8_products() -> None:
    assert hardware.int8_may_saturate(EPYC)
    assert not hardware.int8_may_saturate(XEON)
    assert not hardware.int8_may_saturate(M4)
    # AVX-VNNI alone (no AVX-512) is not trusted, and an x86 CPU that cannot be read counts as one that may
    assert hardware.int8_may_saturate(hardware.Cpu(architecture="x86_64", features=["avx2", "avx_vnni"]))
    assert hardware.int8_may_saturate(hardware.Cpu(architecture="x86_64"))


def test_an_intel_mac_reports_avx512_vnni() -> None:
    host = FakeHost(
        system="darwin",
        commands={
            "sysctl -n machdep.cpu.features": "FPU SSE4.2 AVX1.0",
            "sysctl -n machdep.cpu.leaf7_features": "AVX2 AVX512F AVX512VNNI",
        },
    )
    cpu, _ = hardware.detect_cpu(host)
    assert "avx512_vnni" in (cpu.features or []) and not hardware.int8_may_saturate(cpu)


@pytest.fixture
def turbo(tmp_path: Path) -> Path:
    for name in ("turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-decoder.onnx", "turbo-tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


OPTIONS = {"float_decoder_without_vnni": "*-decoder.onnx"}


def test_the_float_decoder_is_loaded_where_int8_may_saturate(turbo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "this_cpu", lambda: EPYC)
    assert Path(whisper_decoder(turbo, OPTIONS)).name == "turbo-decoder.onnx"
    # an entry without the option keeps its int8 decoder everywhere
    assert Path(whisper_decoder(turbo, {})).name == "turbo-decoder.int8.onnx"
    for cpu in (XEON, M4):
        monkeypatch.setattr(hardware, "this_cpu", lambda cpu=cpu: cpu)
        assert Path(whisper_decoder(turbo, OPTIONS)).name == "turbo-decoder.int8.onnx"


def test_a_copy_pulled_without_the_float_decoder_says_how_to_get_it(
    turbo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (turbo / "turbo-decoder.onnx").unlink()
    monkeypatch.setattr(hardware, "this_cpu", lambda: EPYC)
    with caplog.at_level(logging.WARNING, logger="bolo.engines.sherpa"):
        assert Path(whisper_decoder(turbo, OPTIONS)).name == "turbo-decoder.int8.onnx"
    assert "bolo pull" in caplog.text and "AVX-512 VNNI" in caplog.text


def test_the_float_decoder_is_downloaded_only_where_it_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = Catalog.load().resolve("whisper-large-v3-turbo")
    assert spec is not None and spec.source is not None
    assert spec.options.get("float_decoder_without_vnni") == "*-decoder.onnx"
    monkeypatch.setattr(hardware, "this_cpu", lambda: EPYC)
    assert source_here(spec).include == [*spec.source.include, "*-decoder.onnx"]
    for cpu in (XEON, M4):
        monkeypatch.setattr(hardware, "this_cpu", lambda cpu=cpu: cpu)
        assert source_here(spec).include == spec.source.include
    # an entry without the option is pulled as it is
    other = Catalog.load().resolve("whisper-tiny")
    assert other is not None and source_here(other) is other.source
