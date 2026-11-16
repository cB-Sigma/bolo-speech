"""The recommender on a small fixed catalog and described machines, then through the core, the API and the CLI.

The rule under test throughout: bolo recommends and warns, it never restricts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from bolo import recommend as rec
from bolo.catalog import ModelSpec
from bolo.cli import main
from bolo.config import Settings
from bolo.core import Bolo
from bolo.hardware import Container, Cpu, Disk, Gpu, Hardware, Memory
from conftest import LiveServer, wav_file

GB = 1024**3
MB = 1024**2


def spec(model_id: str, task: str = "stt", **kw: Any) -> ModelSpec:
    entry: dict[str, Any] = {
        "id": model_id,
        "task": task,
        "engine": "sherpa-onnx",
        "kind": "whisper" if task == "stt" else "vits",
        "family": model_id.partition("-")[0],
        "languages": ["en"],
        "license": "Apache-2.0",
        "size_bytes": 200 * MB,
        "source": {"repo": f"example/{model_id}"},
    }
    rtf = kw.pop("rtf", None)
    if rtf is not None:
        entry.update(validated="cpu", measured={"rtf": rtf, "threads": 4, **kw.pop("measured", {})})
    entry.update(kw)
    return ModelSpec.from_dict(entry)


def machine(
    cores: float = 4,
    ram_gb: float = 16,
    features: list[str] | None = None,
    arch: str = "x86_64",
    gpus: list[Gpu] | None = None,
    device: str = "cpu",
    limit_gb: float | None = None,
    system: str = "linux",
    runtimes: dict[str, dict[str, Any]] | None = None,
) -> Hardware:
    feats = ["avx2", "avx512", "avx512_vnni", "amx"] if features is None else features
    return Hardware(
        system=system,
        cpu=Cpu("Test CPU", "Test", arch, int(cores) or 1, int(cores) or 1, cores, feats),
        memory=Memory(int(ram_gb * GB), int(ram_gb * GB * 0.8), int(limit_gb * GB) if limit_gb else None),
        disk=Disk("/data", 500 * GB, 100 * GB),
        gpus=gpus or [],
        runtimes=runtimes or {},
        container=Container(detected=False),
        device=device,
        threads=int(min(4, cores)) or 1,
    )


def nvidia(vram_gb: float) -> Gpu:
    return Gpu("nvidia", "RTX Test", int(vram_gb * GB), int(vram_gb * GB), "550", backend="cuda", source="nvidia-smi")


#: a small catalog with one of everything the recommender has to tell apart
CATALOG = [
    spec("tiny-en", rtf=0.05, tier="realtime", family="Tiny", size_bytes=80 * MB),
    spec("base-multi", rtf=0.12, tier="balanced", family="Base", languages=["en", "hi", "de", "fr"]),
    spec("large-multi", rtf=0.9, tier="accuracy", family="Large", size_bytes=1500 * MB,
         languages=[f"l{i}" for i in range(40)] + ["en", "hi"], measured={"wer": 0.03}),
    spec("large-en-nc", rtf=0.3, tier="accuracy", family="LargeNC", size_bytes=1200 * MB, license="CC-BY-NC-4.0",
         commercial_use=False),
    spec("big-en", rtf=0.7, tier="accuracy", family="Big", size_bytes=900 * MB),
    spec("hindi-only", rtf=0.2, tier="balanced", family="Hindi", languages=["hi"]),
    spec("untested-en", tier="balanced", family="Tiny2", size_bytes=160 * MB),  # never measured
    spec("slow-en", rtf=3.0, tier="accuracy", family="Slow"),
    spec("huge-en", rtf=0.5, tier="accuracy", family="Huge", size_bytes=40 * GB),
    spec("gpu-en", tier="accuracy", family="GpuOnly", size_bytes=6 * GB, engine="tts-server", kind="tts-server",
         options={"runtime": "llama-server", "hardware": {"gpu": "required", "vram_gb": 8.3}}),
    spec("cloud-en", tier="accuracy", family="Cloud", engine="remote", kind="remote-stt", cloud=True, size_bytes=0,
         license="Proprietary", license_url="https://example.com/terms", source={}),
    spec("pending-en", tier="balanced", family="Pending", status="pending"),
    spec("voice-en", "tts", rtf=0.1, family="Voice"),
]  # fmt: skip


def run(q: rec.Query, hw: Hardware | None = None, specs: list[ModelSpec] | None = None, **kw: Any) -> dict[str, Any]:
    """Rank the fixed catalog the way the advisor does, without a core."""
    hw = hw or machine()
    specs = CATALOG if specs is None else specs
    speed = kw.pop("speed", None) or rec.heuristic_speed(hw)
    siblings = rec.Siblings(specs)
    kept, hidden = rec.select(specs, q)
    candidates = [rec.Candidate(s, rec.assess(s, hw, speed, siblings, bool(q.realtime))) for s in kept]
    ordered, recommended, default = rec.rank(candidates, q)
    return {
        "ids": [c.spec.id for c in ordered],
        "levels": {c.spec.id: c.fit.level for c in ordered},
        "fits": {c.spec.id: c.fit for c in ordered},
        "recommended": recommended,
        "default": default,
        "hidden": hidden,
    }


# -- speed ------------------------------------------------------------------------------------------------------
def test_heuristic_speed_by_cpu() -> None:
    reference = rec.heuristic_speed(machine())
    assert reference.factor == pytest.approx(1.0) and reference.source == "heuristic"
    assert reference.spread == rec.HEURISTIC_SPREAD and "not measured" in reference.note
    avx2 = rec.heuristic_speed(machine(features=["avx", "avx2"]))
    old = rec.heuristic_speed(machine(features=["sse4.2"]))
    two_cores = rec.heuristic_speed(machine(cores=2))
    pi = rec.heuristic_speed(machine(arch="arm64", features=["neon"]))
    mac = rec.heuristic_speed(machine(arch="arm64", features=["neon", "dotprod"], system="macos"))
    quota = rec.heuristic_speed(machine(cores=1.5))
    assert mac.factor > reference.factor > avx2.factor > two_cores.factor > old.factor
    assert pi.factor == pytest.approx(0.3) and quota.factor < two_cores.factor
    unknown = rec.heuristic_speed(Hardware())
    assert unknown.factor > 0 and "could not be read" in unknown.note  # still an answer, and it says how rough


def test_benchmark_replaces_the_heuristic() -> None:
    hw = machine()
    bench = {"speed": 2.0, "threads": 4, "workload": "onnxruntime", "at": "2026-11-14T10:00:00Z"}
    speed = rec.speed_for(hw, bench)
    assert speed.source == "benchmark" and speed.factor == 2.0 and speed.spread == rec.BENCHMARK_SPREAD
    assert rec.speed_for(hw, None).source == "heuristic"
    # taken with another thread count than bolo uses now: not comparable, so the heuristic again
    assert rec.speed_for(hw, {**bench, "threads": 2}).source == "heuristic"
    fit = rec.assess(CATALOG[1], hw, speed)
    assert fit.expected_rtf == pytest.approx(0.06) and fit.scaled_by == "benchmark"
    assert fit.rtf_range == pytest.approx((0.12 / (2 * 1.33), 0.12 / (2 * 0.75)))
    rough = rec.assess(CATALOG[1], hw, rec.heuristic_speed(hw))
    assert rough.rtf_range is not None and fit.rtf_range is not None
    assert rough.rtf_range[1] / rough.rtf_range[0] > fit.rtf_range[1] / fit.rtf_range[0]  # the estimate is wider
    assert "micro-benchmark" in fit.reason and "rough estimate" in rough.reason


# -- fit --------------------------------------------------------------------------------------------------------
def test_fit_levels_on_the_reference_machine() -> None:
    out = run(rec.Query("stt"))
    assert out["levels"] == {
        "tiny-en": "comfortable",
        "base-multi": "comfortable",
        "large-en-nc": "comfortable",
        "hindi-only": "comfortable",
        "untested-en": "comfortable",
        "large-multi": "usable",
        "big-en": "usable",
        "cloud-en": "cloud",
        "slow-en": "slow",
        "huge-en": "unlikely",
        "gpu-en": "unlikely",
        "pending-en": "unavailable",
    }
    fits = out["fits"]
    measured = fits["base-multi"]
    assert measured.rtf_basis == "measured" and measured.expected_rtf == pytest.approx(0.12)
    assert measured.device == "cpu" and measured.memory_kind == "ram" and measured.memory_basis == "estimated"
    assert measured.memory_needed_bytes == int(200 * MB * 1.2 + 150 * MB)
    assert "8x faster than real time" in measured.reason and measured.reason.endswith(".")
    assert "This machine" not in measured.reason and measured.reason.count(".") <= 6  # one plain sentence
    assert "40.0 GB" not in fits["huge-en"].reason and "this machine has 16.0 GB" in fits["huge-en"].reason
    assert "needs a GPU" in fits["gpu-en"].reason and "never measured" in fits["gpu-en"].reason
    assert "hardware does not matter" in fits["cloud-en"].reason and fits["cloud-en"].device == "cloud"
    assert "not published" in fits["pending-en"].reason


def test_untested_models_say_they_are_estimates() -> None:
    fits = run(rec.Query("stt"))["fits"]
    est = fits["untested-en"]
    # base-multi (200 MB, RTF 0.12) is the measured model of the same architecture nearest in size (160 MB)
    assert est.rtf_basis == "estimated" and est.expected_rtf == pytest.approx(0.12 * 160 / 200)
    assert "Not measured" in est.reason and "base-multi" in est.reason
    assert est.rtf_range is not None and est.rtf_range[1] / est.rtf_range[0] == pytest.approx(16.0)
    # no sibling and no hint: a guess, said to be one, and no invented number
    lonely = rec.assess(spec("lonely", kind="unique", tier="accuracy"), machine(), rec.heuristic_speed(machine()))
    assert lonely.level == "slow" and lonely.expected_rtf is None and lonely.rtf_basis == "unknown"
    assert "guess" in lonely.reason and "Never measured" in lonely.reason
    hinted = spec("hinted", options={"hardware": {"gpu": "recommended", "vram_gb": 2.0}}, kind="unique2")
    slow = rec.assess(hinted, machine(), rec.heuristic_speed(machine()))
    assert slow.level == "slow" and "slow on a CPU" in slow.reason and slow.expected_rtf is None


def test_fit_follows_the_machine() -> None:
    base = CATALOG[1]
    weak = machine(cores=2, features=["avx", "avx2"], ram_gb=4)
    out = run(rec.Query("stt"), weak)
    assert out["fits"]["base-multi"].expected_rtf == pytest.approx(0.12 / (0.7 * 0.5**0.8))
    assert out["levels"]["large-multi"] == "slow" and out["levels"]["slow-en"] == "unlikely"
    assert out["levels"]["large-en-nc"] == "usable"
    # memory: 1.5 GB of files on a 2 GB container does not fit; 1.2 GB of files on 2.5 GB is tight
    boxed = machine(ram_gb=64, limit_gb=2)
    tight = rec.assess(CATALOG[2], boxed, rec.heuristic_speed(boxed))
    assert tight.level == "unlikely" and "container's limit" in tight.reason and "2.0 GB" in tight.reason
    four = machine(ram_gb=2.5)
    snug = rec.assess(CATALOG[3], four, rec.heuristic_speed(four))
    assert snug.level == "usable" and "tight" in snug.reason
    blind = machine()
    blind.memory = Memory()
    unknown = rec.assess(base, blind, rec.heuristic_speed(blind))
    assert unknown.level == "comfortable" and "could not be read" in unknown.reason
    # live use is stricter than batch
    assert rec.assess(spec("x", rtf=0.3), machine(), rec.heuristic_speed(machine())).level == "comfortable"
    assert rec.assess(spec("x", rtf=0.3), machine(), rec.heuristic_speed(machine()), live=True).level == "usable"
    assert rec.assess(spec("x", rtf=0.8), machine(), rec.heuristic_speed(machine()), live=True).level == "slow"


def test_windows_gpu_speech_to_text_is_faster_whisper() -> None:
    """On Windows with device = "cuda", sherpa-onnx models are judged on the CPU (its CUDA wheel cannot run inside
    bolo there) and faster-whisper on the GPU, where it was measured 6-19x faster than on the CPU."""
    sherpa = spec("sherpa-en", rtf=0.3, tier="accuracy", family="Sherpa", options={"hardware": {"gpu": "recommended"}})
    fw = spec("fw-en", tier="accuracy", family="Fw", engine="faster-whisper", kind="ct2-whisper", size_bytes=1500 * MB)
    windows = machine(gpus=[nvidia(6)], device="cuda", system="windows")
    assert rec.gpu_for(sherpa, windows) == (None, rec.SHERPA_WINDOWS_CUDA)
    fit = rec.assess(sherpa, windows, rec.heuristic_speed(windows))
    assert fit.device == "cpu" and rec.SHERPA_WINDOWS_CUDA in fit.notes
    fit = rec.assess(fw, windows, rec.heuristic_speed(windows))
    assert fit.device == "gpu" and fit.level == "comfortable" and "measured on an RTX 3060 Laptop" in fit.reason
    out = run(rec.Query("stt", "en"), windows, [sherpa, fw])
    assert "fw-en" in out["recommended"] and out["fits"]["fw-en"].device == "gpu"
    linux = machine(gpus=[nvidia(6)], device="cuda")
    assert rec.gpu_for(sherpa, linux)[0] is not None  # elsewhere sherpa-onnx's CUDA build works
    no_ct2_cuda = machine(gpus=[nvidia(6)], device="cuda", system="windows", runtimes={"ctranslate2": {"gpu": False}})
    assert rec.assess(fw, no_ct2_cuda, rec.heuristic_speed(no_ct2_cuda)).device == "cpu"


def test_gpu_models() -> None:
    gpu_model = next(s for s in CATALOG if s.id == "gpu-en")
    # a GPU that bolo is not told to use: still the CPU verdict, with the remark
    idle = machine(gpus=[nvidia(24)])
    fit = rec.assess(gpu_model, idle, rec.heuristic_speed(idle))
    assert fit.level == "unlikely" and any('"cpu"' in n for n in fit.notes)
    used = machine(gpus=[nvidia(24)], device="cuda")
    fit = rec.assess(gpu_model, used, rec.heuristic_speed(used))
    assert fit.level == "usable" and fit.device == "gpu" and fit.memory_kind == "vram"
    assert fit.memory_needed_bytes == int(8.3 * GB) and fit.memory_available_bytes == 24 * GB
    assert fit.expected_rtf is None and "never measured" in fit.reason and "guess" in fit.reason
    small = machine(gpus=[nvidia(6)], device="cuda")
    fit = rec.assess(gpu_model, small, rec.heuristic_speed(small))
    assert fit.level == "unlikely" and "8.3 GB" in fit.reason and "6.0 GB" in fit.reason
    cpu_build = machine(gpus=[nvidia(24)], device="cuda", runtimes={"llama-server": {"gpu": False}})
    fit = rec.assess(gpu_model, cpu_build, rec.heuristic_speed(cpu_build))
    assert fit.level == "unlikely" and any("CPU build" in n for n in fit.notes)
    # a model that is fine on the CPU keeps its measured verdict next to a GPU
    fit = rec.assess(CATALOG[0], used, rec.heuristic_speed(used))
    assert fit.level == "comfortable" and fit.device == "cpu" and any("RTX Test" in n for n in fit.notes)
    # the torch plugin picks its own device
    torch_model = spec("torch-en", engine="plugin:torch", kind="t", size_bytes=3 * GB,
                       options={"hardware": {"gpu": "recommended", "vram_gb": 4.0}})  # fmt: skip
    assert rec.assess(torch_model, idle, rec.heuristic_speed(idle)).device == "gpu"
    no_cuda = machine(gpus=[nvidia(24)], runtimes={"plugin:torch": {"cuda": False}})
    assert rec.assess(torch_model, no_cuda, rec.heuristic_speed(no_cuda)).device == "cpu"
    mac = machine(arch="arm64", system="macos", ram_gb=16, features=["neon"],
                  gpus=[Gpu("apple", "Apple M2 GPU", 16 * GB, 8 * GB, unified_memory=True, backend="metal")])  # fmt: skip
    fit = rec.assess(torch_model, mac, rec.heuristic_speed(mac))
    assert fit.device == "gpu" and "shared with the system" in fit.reason


# -- ranking ----------------------------------------------------------------------------------------------------
def test_ranking_never_drops_a_model() -> None:
    every = {s.id for s in CATALOG if s.task == "stt"}
    for hw in (machine(), machine(cores=1, ram_gb=1, features=["sse4.2"]), Hardware()):
        for realtime in (None, True, False):
            out = run(rec.Query("stt", realtime=realtime), hw)
            assert set(out["ids"]) == every and len(out["ids"]) == len(every)
            assert all(v == 0 for v in out["hidden"].values())
    # on a machine nothing fits, everything is still listed, the unsuitable last, each with its reason
    out = run(rec.Query("stt"), machine(cores=1, ram_gb=1, features=["sse4.2"]))
    order = [out["levels"][i] for i in out["ids"]]
    assert order == sorted(order, key=["comfortable", "usable", "cloud", "slow", "unlikely", "unavailable"].index)
    assert all(out["fits"][i].reason for i in out["ids"])
    assert out["ids"][-1] == "pending-en" and out["levels"]["huge-en"] == "unlikely"


def test_ranking_order_and_marks() -> None:
    out = run(rec.Query("stt", "en"))
    assert out["ids"][-4:] == ["slow-en", "gpu-en", "huge-en", "pending-en"] or out["ids"][-4:] == [
        "slow-en", "huge-en", "gpu-en", "pending-en",
    ]  # fmt: skip
    assert out["ids"].index("cloud-en") < out["ids"].index("slow-en")
    # the default is measured, comfortable and made for the language; an estimate never beats a measurement
    assert out["default"] == "tiny-en" and out["recommended"][0] == "tiny-en"
    assert len(out["recommended"]) <= 3 and "untested-en" not in out["recommended"][:1]
    assert all(out["levels"][i] in ("comfortable", "usable") for i in out["recommended"])
    families = [next(s.family for s in CATALOG if s.id == i) for i in out["recommended"]]
    assert len(set(families)) == len(families)
    assert not {"cloud-en", "pending-en", "huge-en", "gpu-en", "slow-en"} & set(out["recommended"])
    # batch work: the most capable model that keeps up, even if it is only "usable"
    batch = run(rec.Query("stt", "en", realtime=False))
    assert batch["default"] == "big-en" and batch["ids"].index("big-en") < batch["ids"].index("tiny-en")
    # a licence that forbids commercial use never makes the default while a free one fits as well
    assert out["ids"].index("large-en-nc") > out["ids"].index("tiny-en")
    # live: the fast, small model first; the slow accuracy model is not recommended
    live = run(rec.Query("stt", "en", realtime=True))
    assert live["default"] == "tiny-en" and "big-en" not in live["recommended"]
    assert live["levels"]["big-en"] == "slow" and live["levels"]["large-multi"] == "slow"
    # no language asked: the broad model is the general suggestion
    assert run(rec.Query("stt"))["default"] == "base-multi"


def test_no_language_prefers_english_and_multilingual_voices() -> None:
    """Without a language, a voice of one other language is not a general suggestion, however fast it measured."""
    voices = [
        spec("nepali-voice", "tts", languages=["ne"], rtf=0.02),
        spec("gujarati-voice", "tts", languages=["gu"], rtf=0.03),
        spec("english-voice", "tts", languages=["en"], rtf=0.2),
        spec("british-voice", "tts", languages=["en-GB"], rtf=0.25),
        spec("many-voice", "tts", languages=["hi", "bn", "ta", "te", "mr"], rtf=0.3),
    ]
    out = run(rec.Query("tts"), specs=voices)
    assert set(out["recommended"]) == {"english-voice", "british-voice", "many-voice"}
    assert set(out["ids"][-2:]) == {"nepali-voice", "gujarati-voice"}  # still listed, after the general ones
    for realtime in (True, False):
        live = run(rec.Query("tts", realtime=realtime), specs=voices)
        assert not {"nepali-voice", "gujarati-voice"} & set(live["recommended"])
    # with a language asked for, that language's voice is the pick
    assert run(rec.Query("tts", "ne"), specs=voices)["default"] == "nepali-voice"
    # recognisers are not affected
    assert not rec._other_language(spec("nepali-stt", languages=["ne"]), rec.Query("stt"))
    assert not rec._other_language(voices[0], rec.Query("tts", "ne")) and rec._other_language(
        voices[0], rec.Query("tts")
    )


def test_nothing_fits_still_names_the_best_of_the_rest() -> None:
    weak = machine(cores=2, ram_gb=8, features=["avx", "avx2"])
    out = run(rec.Query("stt", "en"), weak, [s for s in CATALOG if s.id in ("slow-en", "large-multi", "huge-en")])
    assert out["recommended"] == [] and out["default"] == "large-multi" and out["levels"]["large-multi"] == "slow"
    hopeless = run(rec.Query("stt", "en"), weak, [s for s in CATALOG if s.id in ("huge-en", "gpu-en")])
    assert hopeless["recommended"] == [] and hopeless["default"] is None and len(hopeless["ids"]) == 2


def test_filters() -> None:
    licence = run(rec.Query("stt", commercial=True))
    assert "large-en-nc" not in licence["ids"] and licence["hidden"]["commercial"] == 1
    assert "cloud-en" in licence["ids"]
    hindi = run(rec.Query("stt", "hi"))
    assert set(hindi["ids"]) == {"base-multi", "large-multi", "hindi-only"} and hindi["hidden"]["language"] == 9
    assert hindi["default"] == "hindi-only"  # made for the language, ahead of the ones that also cover it
    assert set(run(rec.Query("stt", "hi-IN"))["ids"]) == set(hindi["ids"])
    local = run(rec.Query("stt", local_only=True))
    assert "cloud-en" not in local["ids"] and local["hidden"]["local_only"] == 1
    small = run(rec.Query("stt", max_size_bytes=300 * MB))
    assert "huge-en" not in small["ids"] and small["hidden"]["max_size"] == 5
    assert run(rec.Query("tts"))["ids"] == ["voice-en"]
    # language-independent models pass a language filter
    assert rec.speaks(spec("vad", "vad", languages=[]), "ta") and rec.speaks(spec("m", languages=["multi"]), "ta")
    assert rec.normalize_task("S2S") == "speech-to-speech" and rec.normalize_task("stt") == "stt"
    s2s = spec("talker", "tts", features=["speech-to-speech"])
    assert rec.in_task(s2s, "speech-to-speech") and not rec.in_task(CATALOG[0], "speech-to-speech")


def test_quality_and_warnings() -> None:
    assert rec.quality(CATALOG[2]) == {
        "metric": "wer",
        "label": "word error rate",
        "value": 0.03,
        "note": "measured by the maintainers on that model's own test audio; not comparable across models",
    }
    assert rec.quality(CATALOG[0]) is None
    hw = machine()
    speed = rec.heuristic_speed(hw)
    fine = rec.fit_warnings(CATALOG[0], rec.assess(CATALOG[0], hw, speed))
    assert fine == []
    slow_spec = next(s for s in CATALOG if s.id == "slow-en")
    slow = rec.fit_warnings(slow_spec, rec.assess(slow_spec, hw, speed))
    assert [w["code"] for w in slow] == ["hardware_fit"] and slow[0]["level"] == "slow"
    assert "slower than real time" in slow[0]["message"] and "not a restriction" in slow[0]["message"]
    huge_spec = next(s for s in CATALOG if s.id == "huge-en")
    huge = rec.fit_warnings(huge_spec, rec.assess(huge_spec, hw, speed), disk_free=10 * GB, pulling=True)
    assert [w["code"] for w in huge] == ["hardware_fit", "disk_space"] and "10.0 GB free" in huge[1]["message"]
    assert rec.fit_warnings(huge_spec, rec.assess(huge_spec, hw, speed), disk_free=10 * GB)[0]["level"] == "unlikely"
    doc = rec.assess(CATALOG[1], hw, speed).to_dict()
    assert {"level", "label", "expected_rtf", "rtf_range", "rtf_basis", "scaled_by", "device", "memory", "reason",
            "recommended", "recommended_for", "default_for", "notes", "engine_available"} == set(doc)  # fmt: skip
    assert json.loads(json.dumps(doc)) == doc


# -- through the core -------------------------------------------------------------------------------------------
SMALL = machine(cores=2, ram_gb=4, features=["avx", "avx2"])


def small_core(settings: Settings, **kw: Any) -> Bolo:
    for key, value in kw.items():
        setattr(settings, key, value)
    core = Bolo(settings)
    core.advisor.use_hardware(SMALL)
    return core


def test_catalog_entries_carry_a_fit(settings: Settings) -> None:
    core = small_core(settings)
    models = {m["id"]: m for m in core.catalog_api()}
    assert len(models) == len(core.catalog) and all(m["fit"] is not None for m in models.values())
    assert {m["fit"]["level"] for m in models.values()} <= set(rec.ALL_LEVELS)
    assert all(m["fit"]["reason"] for m in models.values())
    whisper = models["whisper-base"]["fit"]
    assert whisper["rtf_basis"] == "measured" and whisper["scaled_by"] == "heuristic" and whisper["expected_rtf"] > 0.11
    assert models["openai-whisper-1"]["fit"]["level"] == "cloud"
    assert models["fish-drama-3-audiocpp"]["fit"]["level"] == "unlikely"  # 24 GB of weights, wants a 32 GB GPU
    # every task has a default pick, and no task has more than three general recommendations
    for task in ("stt", "tts", "vad", "diarization"):
        mine = [m for m in models.values() if m["task"] == task]
        assert sum(m["fit"]["recommended"] for m in mine) <= 3
        assert sum("*" in m["fit"]["default_for"] for m in mine) == 1
    hindi = [m for m in models.values() if m["task"] == "tts" and "hi" in m["fit"]["recommended_for"]]
    assert 1 <= len(hindi) <= 3 and all("hi" in m["languages"] or not m["languages"] for m in hindi)
    assert core.model_api(core.manager.resolve("whisper-base"))["fit"] == models["whisper-base"]["fit"]
    assert core.info()["capabilities"]["recommendations"]["restricts"] is False


def test_advisor_recommend_and_setup(settings: Settings) -> None:
    core = small_core(settings)
    answer = core.advisor.recommend(rec.Query("stt", "hi", commercial=True, local_only=True))
    total = len([s for s in core.catalog.all() if s.task == "stt"])
    assert answer["total"] + sum(answer["filtered_out"].values()) == total
    assert answer["query"]["language"] == "hi" and answer["speed"]["source"] == "heuristic"
    assert [r["rank"] for r in answer["results"]] == list(range(1, answer["total"] + 1))
    assert all(r["commercial_use"] and not r["cloud"] for r in answer["results"])
    assert answer["default"] and answer["results"][0]["default"] and answer["results"][0]["recommended"]
    limited = core.advisor.recommend(rec.Query("stt"), 5)
    assert len(limited["results"]) == 5 and limited["total"] > 5
    for use, tasks in (
        ("live", ["stt", "tts", "vad", "turn"]),
        ("files", ["stt", "diarization"]),
        ("voiceover", ["tts"]),
    ):
        setup = core.advisor.setup(use, "en")
        assert [p["task"] for p in setup["picks"]] == tasks and setup["use"] == use
        chosen = [p["model"] for p in setup["picks"] if p["model"]]
        assert chosen and all(not m["cloud"] for m in chosen)
        assert set(setup["pull"]) == {m["id"] for m in chosen if not m["installed"]}
        assert setup["download_bytes"] == sum(m["size_bytes"] for m in chosen if not m["installed"])
    live = core.advisor.setup("live", "en")["picks"][0]["model"]
    assert live["fit"]["level"] in ("comfortable", "usable") and live["fit"]["expected_rtf"] <= rec.RTF_USABLE_LIVE


def test_benchmark_is_stored_and_used(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    core = small_core(settings)
    assert core.advisor.speed().source == "heuristic"
    before = core.advisor.fit(core.manager.resolve("whisper-base")).expected_rtf
    fake = {"version": 1, "speed": 4.0, "threads": SMALL.threads, "workload": "numpy", "ms": 1.0, "reference_ms": 4.0,
            "at": "2026-11-14T10:00:00Z", "cpu_model": SMALL.cpu.model, "logical_cores": SMALL.cpu.logical_cores,
            "usable_cores": SMALL.cpu.usable_cores}  # fmt: skip
    monkeypatch.setattr("bolo.hardware.run_benchmark", lambda hw: dict(fake))
    assert core.advisor.benchmark()["speed"] == 4.0
    assert json.loads((settings.data_dir / "hardware-benchmark.json").read_text(encoding="utf-8"))["speed"] == 4.0
    speed = core.advisor.speed()
    assert speed.source == "benchmark" and speed.factor == 4.0
    after = core.advisor.fit(core.manager.resolve("whisper-base"))
    assert before is not None and after.expected_rtf is not None and after.expected_rtf < before
    assert after.scaled_by == "benchmark"
    again = small_core(settings)  # a new process finds the stored result
    assert again.advisor.speed().source == "benchmark"


def test_request_path_never_waits_for_a_recomputation(settings: Settings) -> None:
    core = small_core(settings)
    advisor = core.advisor
    spec = core.manager.resolve("parler-tts-tiny-v1")
    computed: list[int] = []
    kicked: list[int] = []
    compute = advisor._compute
    advisor._compute = lambda *a: (computed.append(1), compute(*a))[1]  # type: ignore[method-assign]
    advisor.refresh_soon = lambda: kicked.append(1)  # type: ignore[method-assign]
    # nothing is known yet: a request gets no notice, computes nothing, and only asks for a background refresh
    assert advisor.cached_fit(spec) is None and advisor.warnings(spec, wait=False) == []
    assert computed == [] and kicked == [1, 1]
    first = advisor.fits()
    assert computed == [1] and advisor.cached_fit(spec) is first[spec.id]
    assert advisor.warnings(spec, wait=False)[0]["level"] == "unlikely" and kicked == [1, 1]
    advisor.invalidate()  # a model was installed: the fits are due for a refresh
    assert advisor.cached_fit(spec) is first[spec.id] and computed == [1] and kicked == [1, 1, 1]
    assert advisor.fit(spec, stale_ok=True) is first[spec.id] and computed == [1]  # the catalog listing too
    assert advisor.fit(spec) is not first[spec.id] and computed == [1, 1]  # a ranking wants fresh ones
    advisor.use_hardware(machine(cores=64, ram_gb=256))  # another machine: the old fits are not reused
    assert advisor.cached_fit(spec) is None and computed == [1, 1]


def test_background_refresh_fills_the_cache(settings: Settings) -> None:
    core = small_core(settings)
    spec = core.manager.resolve("whisper-base")
    assert core.advisor.cached_fit(spec) is None  # starts the computation in a thread
    for _ in range(200):
        if core.advisor.cached_fit(spec) is not None:
            break
        time.sleep(0.02)
    fit = core.advisor.cached_fit(spec)
    assert fit is not None and fit.level in rec.LEVELS


# -- warnings: attached, never blocking ---------------------------------------------------------------------------
def poor_fit(core: Bolo, *ids: str) -> None:
    """Make the fake models look like ones this machine cannot run well."""
    real, cached = core.advisor.fit, core.advisor.cached_fit
    poor = rec.Fit("unlikely", "Needs about 40.0 GB of memory; this machine has 4.0 GB.", device="cpu")

    def fit(model: ModelSpec, stale_ok: bool = False) -> rec.Fit:
        return poor if model.id in ids else real(model, stale_ok)

    def cached_fit(model: ModelSpec) -> rec.Fit | None:
        return poor if model.id in ids else cached(model)

    core.advisor.fit = fit  # type: ignore[method-assign]
    core.advisor.cached_fit = cached_fit  # type: ignore[method-assign]


def test_load_and_requests_warn_but_work(live: LiveServer) -> None:
    poor_fit(live.core, "fake-stt", "fake-tts")
    loaded = httpx.post(live.url + "/api/v1/models/fake-stt/load").json()
    assert loaded["loaded"] is True and loaded["id"] == "fake-stt"
    assert [w["code"] for w in loaded["warnings"]] == ["hardware_fit"]
    assert "'fake-stt' is unlikely to fit" in loaded["warnings"][0]["message"]
    fine = httpx.post(live.url + "/api/v1/models/fake-vad/load").json()
    assert fine["loaded"] is True and "warnings" not in fine
    # the request is served, and its trace says what to expect
    r = httpx.post(live.url + "/v1/audio/transcriptions", files={"file": wav_file(1.0)}, data={"model": "fake-stt"})
    assert r.status_code == 200 and r.json()["text"]
    r = httpx.post(live.url + "/v1/audio/speech", json={"model": "fake-tts", "input": "Hello there.", "voice": "alpha"})
    assert r.status_code == 200 and len(r.content) > 1000
    traces = httpx.get(live.url + "/api/v1/requests").json()["requests"]
    by_kind = {t["kind"]: t for t in traces}
    assert by_kind["stt"]["status"] == "ok" and by_kind["stt"]["warnings"][0]["code"] == "hardware_fit"
    assert by_kind["tts"]["status"] == "ok" and by_kind["tts"]["warnings"][0]["model"] == "fake-tts"


def test_pull_warns_but_starts(live: LiveServer, monkeypatch: pytest.MonkeyPatch) -> None:
    core = live.core
    core.advisor.use_hardware(SMALL)
    pulled: list[str] = []
    monkeypatch.setattr(core, "_pull_blocking", lambda spec, job: pulled.append(spec.id))
    monkeypatch.setattr(core.store, "is_installed", lambda model_id: False)
    r = httpx.post(live.url + "/api/v1/models/pull", json={"id": "parler-tts-tiny-v1"})
    assert r.status_code == 202
    body = r.json()
    assert body["warnings"][0]["code"] == "hardware_fit" and body["warnings"][0]["level"] == "unlikely"
    job = httpx.get(live.url + f"/api/v1/jobs/{body['job_id']}").json()
    assert job["warnings"] == body["warnings"]
    for _ in range(100):
        if pulled:
            break
        time.sleep(0.02)
    assert pulled == ["parler-tts-tiny-v1"]  # the download went ahead
    ok = httpx.post(live.url + "/api/v1/models/pull", json={"id": "silero-vad"})
    assert ok.status_code == 202 and "warnings" not in ok.json()


def test_warnings_can_be_turned_off(settings: Settings) -> None:
    core = small_core(settings, hardware_warnings=False)
    poor_fit(core, "fake-stt")
    assert core.advisor.warnings(core.manager.resolve("fake-stt")) == []
    assert core.advisor.warnings(core.manager.resolve("parler-tts-tiny-v1"), pulling=True) == []
    assert core.info()["capabilities"]["recommendations"]["warnings"] is False
    loud = small_core(settings, hardware_warnings=True)
    assert loud.advisor.warnings(loud.manager.resolve("parler-tts-tiny-v1"), pulling=True)
    # the fit itself is still reported: only the notices go away
    assert core.model_api(core.manager.resolve("parler-tts-tiny-v1"))["fit"]["level"] == "unlikely"


# -- API shapes -------------------------------------------------------------------------------------------------
def test_hardware_api(live: LiveServer) -> None:
    doc = httpx.get(live.url + "/api/v1/hardware").json()
    assert {"system", "cpu", "memory", "disk", "gpus", "gpus_known", "runtimes", "container", "device", "threads",
            "notes", "detected_at", "deep", "summary", "speed"} <= set(doc)  # fmt: skip
    assert set(doc["cpu"]) == {"model", "vendor", "architecture", "physical_cores", "logical_cores", "usable_cores", "features"}  # fmt: skip
    assert set(doc["memory"]) == {"total_bytes", "available_bytes", "limit_bytes", "usable_bytes"}
    assert set(doc["container"]) == {"detected", "kind", "virtual_machine", "hypervisor", "cpu_quota", "memory_limit_bytes"}  # fmt: skip
    assert doc["speed"]["source"] in ("heuristic", "benchmark") and doc["speed"]["factor"] > 0
    assert doc["deep"] is False and isinstance(doc["summary"], str)
    assert httpx.get(live.url + "/api/v1/hardware", params={"refresh": "true"}).json()["deep"] is True
    # unknown values are null, not invented
    live.core.advisor.use_hardware(Hardware())
    blank = httpx.get(live.url + "/api/v1/hardware").json()
    assert blank["cpu"]["model"] is None and blank["memory"]["total_bytes"] is None and blank["speed"]["factor"] > 0


def test_recommendations_api(live: LiveServer) -> None:
    live.core.advisor.use_hardware(SMALL)
    r = httpx.get(live.url + "/api/v1/recommendations", params={"task": "tts", "language": "hi", "limit": 4})
    assert r.status_code == 200
    doc = r.json()
    assert {"query", "hardware", "speed", "default", "recommended", "total", "filtered_out", "note", "results"} == set(
        doc
    )
    assert len(doc["results"]) == 4 and doc["total"] > 4 and len(doc["recommended"]) <= 3
    row = doc["results"][0]
    assert {"rank", "id", "task", "family", "engine", "tier", "languages", "size_bytes", "installed", "license",
            "commercial_use", "cloud", "validated", "quality", "recommended", "default", "fit"} <= set(row)  # fmt: skip
    assert row["id"] == doc["default"] and row["fit"]["level"] in ("comfortable", "usable")
    full = httpx.get(live.url + "/api/v1/recommendations", params={"task": "tts"}).json()
    assert full["total"] == len(full["results"]) == len([s for s in live.core.catalog.all() if s.task == "tts"])
    assert full["results"][-1]["fit"]["level"] in ("unlikely", "unavailable")  # listed, last, never dropped
    flags = httpx.get(
        live.url + "/api/v1/recommendations",
        params={"task": "stt", "commercial": "true", "local_only": "1", "realtime": "true", "max_size_mb": 100},
    ).json()
    assert flags["query"] == {"task": "stt", "language": None, "commercial": True, "local_only": True,
                              "realtime": True, "max_size_bytes": 100 * MB}  # fmt: skip
    assert all(r["size_bytes"] <= 100 * MB and r["commercial_use"] for r in flags["results"])
    s2s = httpx.get(live.url + "/api/v1/recommendations", params={"task": "s2s"}).json()
    assert s2s["query"]["task"] == "speech-to-speech" and s2s["total"] >= 1
    bad = httpx.get(live.url + "/api/v1/recommendations", params={"task": "cooking"})
    assert bad.status_code == 400 and "unknown task" in bad.text
    # without a task: an overview of every task, each in the shape of a single task's answer
    overview = httpx.get(live.url + "/api/v1/recommendations", params={"language": "hi", "limit": 2})
    assert overview.status_code == 200
    doc = overview.json()
    assert set(doc) == {"query", "hardware", "speed", "tasks"} and doc["query"]["task"] is None
    assert doc["query"]["language"] == "hi" and list(doc["tasks"]) == list(rec.TASKS)
    tts = doc["tasks"]["tts"]
    assert {"default", "recommended", "total", "filtered_out", "note", "results"} == set(tts)
    assert len(tts["results"]) == 2 and tts["results"][0]["id"] == tts["default"] and tts["total"] > 2
    assert len(httpx.get(live.url + "/api/v1/recommendations").json()["tasks"]["stt"]["results"]) == 3
    assert httpx.get(live.url + "/api/v1/recommendations", params={"task": ""}).json()["tasks"]
    setup = httpx.get(live.url + "/api/v1/recommendations/setup", params={"use": "files", "language": "en"}).json()
    assert [p["task"] for p in setup["picks"]] == ["stt", "diarization"] and setup["local_only"] is True
    assert setup["picks"][1]["optional"] is True and isinstance(setup["download_bytes"], int)
    assert httpx.get(live.url + "/api/v1/recommendations/setup", params={"use": "karaoke"}).status_code == 400
    catalog = httpx.get(live.url + "/api/v1/catalog", params={"task": "vad"}).json()["models"]
    assert all({"level", "expected_rtf", "reason"} <= set(m["fit"]) for m in catalog)
    one = httpx.get(live.url + "/api/v1/models/whisper-base").json()
    assert one["fit"]["level"] in rec.LEVELS and one["fit"]["rtf_basis"] == "measured"


def test_benchmark_api(live: LiveServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bolo.hardware.run_benchmark",
        lambda hw: {
            "version": 1,
            "speed": 0.5,
            "threads": hw.threads,
            "workload": "numpy",
            "ms": 12.0,
            "reference_ms": 6.0,
            "at": "2026-11-14T10:00:00Z",
        },
    )
    doc = httpx.post(live.url + "/api/v1/hardware/benchmark").json()
    assert doc["benchmark"]["speed"] == 0.5 and doc["speed"]["source"] == "benchmark" and doc["speed"]["factor"] == 0.5
    assert httpx.get(live.url + "/api/v1/hardware").json()["speed"]["benchmark"]["ms"] == 12.0


# -- command line -----------------------------------------------------------------------------------------------
@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BOLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("BOLO_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bolo.hardware.detect", lambda settings=None, host=None, deep=False, cache=None: SMALL)
    return tmp_path


def test_cli_recommend(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["recommend"]) == 0
    out = capsys.readouterr().out
    assert "this machine:" in out and "cpu: Test CPU (x86_64)" in out and "gpu: none found" in out
    assert "speed against the reference machine: 0.40x" in out and "no benchmark stored" in out
    assert "DEFAULT PICK" in out and "Speech to text, live" in out and "Text to speech" in out
    assert "recommendations, not restrictions" in out

    assert main(["recommend", "--task", "stt", "--language", "en", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "Speech to text (en):" in out and "RTF HERE (RANGE)" in out and " *" in out and "LICENCE" in out
    assert "more, the least suitable last" in out and "nothing is left out because of hardware" in out
    assert main(["recommend", "--task", "tts", "--all", "--commercial", "--local-only", "--batch", "--why"]) == 0
    out = capsys.readouterr().out
    assert "commercial use only, local only, batch" in out and "unlikely to fit" in out and "non-commercial" not in out
    assert "moss-tts-local-v1.5-audiocpp" in out  # the ones that cannot fit are printed too, last

    assert main(["recommend", "--task", "stt", "--language", "hi", "--realtime", "--json", "--limit", "3"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["hardware"]["cpu"]["model"] == "Test CPU" and doc["recommendations"]["query"]["realtime"] is True
    assert len(doc["recommendations"]["results"]) == 3
    assert main(["recommend", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["recommendations"]) == set(rec.TASKS)

    assert main(["recommend", "--use", "live", "--language", "en"]) == 0
    out = capsys.readouterr().out
    assert "Live conversation in en:" in out and "bolo pull " in out and "(optional)" in out
    assert main(["recommend", "--task", "cooking"]) == 1
    assert "unknown task" in capsys.readouterr().err
    assert main(["recommend", "--use", "karaoke"]) == 1


def test_cli_recommend_benchmark(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "bolo.hardware.run_benchmark",
        lambda hw: {
            "version": 1,
            "speed": 1.25,
            "threads": hw.threads,
            "workload": "numpy",
            "ms": 4.88,
            "reference_ms": 6.1,
            "at": "2026-11-14T10:00:00Z",
            "cpu_model": hw.cpu.model,
            "logical_cores": hw.cpu.logical_cores,
            "usable_cores": hw.cpu.usable_cores,
        },
    )
    assert main(["recommend", "--benchmark", "--task", "vad"]) == 0
    captured = capsys.readouterr()
    assert "1.25x the reference machine" in captured.err and "stored in" in captured.err
    assert "speed against the reference machine: 1.25x, measured by the micro-benchmark" in captured.out
    assert "no benchmark stored" not in captured.out
    assert (cli_env / "data" / "hardware-benchmark.json").is_file()
    assert main(["recommend", "--task", "vad"]) == 0  # the stored result is used from then on
    assert "measured by the micro-benchmark" in capsys.readouterr().out


def test_cli_doctor_and_pull_warning(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "this machine:" in out and "recommended setup for this machine" in out
    assert "live speech to text:" in out and "text to speech:" in out

    pulled: list[str] = []

    def fake_pull(self: Any, model_id: str, source: Any, progress: Any = None, cancelled: Any = None) -> dict[str, Any]:
        pulled.append(model_id)
        return {"size_bytes": 1, "files": []}

    monkeypatch.setattr("bolo.store.ModelStore.pull", fake_pull)
    assert main(["pull", "parler-tts-tiny-v1"]) == 0
    captured = capsys.readouterr()
    assert (
        pulled == ["parler-tts-tiny-v1"] and "installed parler-tts-tiny-v1" in captured.out
    )  # warned, and pulled all the same
    assert "warning: 'parler-tts-tiny-v1' is unlikely to fit or to run well on this machine" in captured.err
    assert "not a restriction" in captured.err
    monkeypatch.setenv("BOLO_HARDWARE_WARNINGS", "false")
    assert main(["pull", "parler-tts-tiny-v1"]) == 0
    assert "warning:" not in capsys.readouterr().err
    monkeypatch.delenv("BOLO_HARDWARE_WARNINGS")
    assert main(["pull", "silero-vad"]) == 0
    assert "warning:" not in capsys.readouterr().err
