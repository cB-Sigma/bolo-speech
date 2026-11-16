"""Which catalog models suit this machine: a fit level per model, a ranking per task, and warnings.

The rule of this module: **recommend, never restrict**. Nothing here hides a model, refuses a pull or blocks a
load. A model that is unlikely to fit sorts last and says why; the user can still pick it.

How a fit is computed, what is measured and what is a guess: docs/recommendations.md.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bolo import hardware as hwmod
from bolo.catalog import ModelSpec
from bolo.hardware import Gpu, Hardware, gigabytes

if TYPE_CHECKING:
    from bolo.core import Bolo

#: hardware fit levels, best first; "cloud" and "unavailable" are not about this machine's hardware
LEVELS = ("comfortable", "usable", "slow", "unlikely")
ALL_LEVELS = (*LEVELS, "cloud", "unavailable")
LEVEL_LABEL = {
    "comfortable": "comfortable",
    "usable": "usable",
    "slow": "slow",
    "unlikely": "unlikely to fit",
    "cloud": "cloud",
    "unavailable": "not available yet",
}
#: tasks the recommender takes; "speech-to-speech" is a feature of some catalog entries, not a catalog task
TASKS = ("stt", "tts", "diarization", "vad", "turn", "separation", "speech-to-speech")
TASK_ALIASES = {"s2s": "speech-to-speech", "asr": "stt", "diarize": "diarization", "separate": "separation"}
S2S = "speech-to-speech"

#: real-time factor (processing time / audio duration) up to which a level applies
RTF_COMFORTABLE = 0.4
RTF_USABLE = 1.0
RTF_SLOW = 5.0
#: live use leaves less room: the model shares the turn with endpointing, a language model and a voice
RTF_COMFORTABLE_LIVE = 0.25
RTF_USABLE_LIVE = 0.6

#: how far the real figure may be from the expected one, as multipliers of the speed factor
BENCHMARK_SPREAD = (0.75, 1.33)
HEURISTIC_SPREAD = (0.5, 2.0)
#: an estimate taken from another model of the same architecture is this much wider again
SIBLING_SPREAD = (0.5, 2.0)
#: a sibling this many times larger or smaller is too different to estimate from
SIBLING_MAX_RATIO = 4.0

#: speed of one core against the reference machine's (an Intel Xeon with AVX-512, VNNI and AMX), by instruction
#: set. Documented guesses, not measurements: `bolo recommend --benchmark` replaces them.
CORE_SPEED = {
    "x86_avx512_vnni": 1.0,
    "x86_avx512": 0.9,
    "x86_avx2": 0.7,
    "x86_avx": 0.45,
    "x86_sse": 0.3,
    "x86_unknown": 0.7,
    "apple_silicon": 1.4,
    "arm_dotprod": 0.7,
    "arm_neon": 0.3,
    "unknown": 0.5,
}

#: uses the "pick for me" flow knows: the tasks it proposes, and whether each must keep up with live speech
USES: dict[str, dict[str, Any]] = {
    "live": {
        "label": "Live conversation",
        "tasks": [("stt", True, False), ("tts", True, False), ("vad", True, True), ("turn", True, True)],
    },
    "files": {
        "label": "Transcription of files",
        "tasks": [("stt", False, False), ("diarization", False, True)],
    },
    "voiceover": {"label": "Voice-over", "tasks": [("tts", False, False)]},
}


# -- speed ------------------------------------------------------------------------------------------------------
@dataclass
class Speed:
    """How fast this machine runs inference against the reference machine the catalog was measured on."""

    #: > 1: faster than the reference. The catalog's real-time factors are divided by it.
    factor: float
    source: str  # "benchmark" | "heuristic"
    spread: tuple[float, float]
    note: str
    benchmark: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": round(self.factor, 3),
            "source": self.source,
            "spread": list(self.spread),
            "note": self.note,
            "benchmark": self.benchmark,
            "reference": "4 vCPUs of an Intel Xeon at 2.1 GHz (AVX-512, AMX), no GPU, 4 inference threads",
        }


def core_class(hw: Hardware) -> str:
    feats = hw.cpu.features
    arch = hw.cpu.architecture
    if arch == "x86_64":
        if feats is None:
            return "x86_unknown"
        if "avx512" in feats:
            return "x86_avx512_vnni" if ("avx512_vnni" in feats or "amx" in feats) else "x86_avx512"
        if "avx2" in feats:
            return "x86_avx2"
        return "x86_avx" if "avx" in feats else "x86_sse"
    if arch == "arm64":
        if hw.system == "macos" or (hw.cpu.vendor or "").lower() == "apple":
            return "apple_silicon"
        return "arm_dotprod" if feats and ("dotprod" in feats or "sve" in feats) else "arm_neon"
    return "unknown"


def effective_threads(hw: Hardware) -> float:
    """Inference threads that really get a core: bolo's thread count, capped by the cores this process may use."""
    usable = hw.cpu.usable_cores
    threads = float(hw.threads) if hw.threads else (min(4.0, usable) if usable else 4.0)
    return max(0.25, min(threads, usable) if usable else threads)


def heuristic_speed(hw: Hardware) -> Speed:
    cls = core_class(hw)
    threads = effective_threads(hw)
    factor = CORE_SPEED[cls] * hwmod.thread_scale(threads)
    unknown = [
        what
        for what, missing in (
            ("the CPU's instruction sets", hw.cpu.features is None),
            ("the number of usable cores", hw.cpu.usable_cores is None),
        )
        if missing
    ]
    note = (
        f"estimated from the CPU ({cls.replace('_', ' ')}, {hwmod._num(threads)} inference threads), not measured; "
        "`bolo recommend --benchmark` measures it"
    )
    if unknown:
        note += f" ({' and '.join(unknown)} could not be read, so this is rougher still)"
    return Speed(factor, "heuristic", HEURISTIC_SPREAD, note)


def speed_for(hw: Hardware, benchmark: dict[str, Any] | None) -> Speed:
    """The benchmark's speed when one is stored for this machine and thread count, else the heuristic."""
    if benchmark and (hw.threads is None or benchmark.get("threads") in (None, hw.threads)):
        if benchmark.get("workload") == "numpy":
            what = f"numpy: one core timed, {benchmark.get('threads')} threads applied by rule"
        else:
            what = f"{benchmark.get('workload')}, {benchmark.get('threads')} threads"
        note = f"measured by the micro-benchmark ({what}) on {str(benchmark.get('at', ''))[:10]}"
        return Speed(float(benchmark["speed"]), "benchmark", BENCHMARK_SPREAD, note, benchmark)
    return heuristic_speed(hw)


# -- fit --------------------------------------------------------------------------------------------------------
@dataclass
class Fit:
    level: str
    reason: str
    device: str | None = None  # cpu | gpu | cloud
    expected_rtf: float | None = None
    rtf_range: tuple[float, float] | None = None
    #: "measured" (the maintainers ran this model; scaled to this machine), "estimated" (from a measured model of
    #: the same architecture) or "unknown"
    rtf_basis: str = "unknown"
    scaled_by: str | None = None  # benchmark | heuristic
    memory_needed_bytes: int | None = None
    memory_available_bytes: int | None = None
    memory_kind: str | None = None  # ram | vram
    memory_basis: str | None = None  # catalog | estimated
    engine_available: bool = True
    notes: list[str] = field(default_factory=list)
    #: marks from the per-task rankings: "*" stands for "any language"
    recommended_for: list[str] = field(default_factory=list)
    default_for: list[str] = field(default_factory=list)

    @property
    def recommended(self) -> bool:
        return "*" in self.recommended_for

    def to_dict(self) -> dict[str, Any]:
        memory = None
        if self.memory_needed_bytes is not None:
            memory = {
                "needed_bytes": self.memory_needed_bytes,
                "available_bytes": self.memory_available_bytes,
                "kind": self.memory_kind,
                "basis": self.memory_basis,
            }
        return {
            "level": self.level,
            "label": LEVEL_LABEL[self.level],
            "expected_rtf": _round_rtf(self.expected_rtf),
            "rtf_range": [_round_rtf(self.rtf_range[0]), _round_rtf(self.rtf_range[1])] if self.rtf_range else None,
            "rtf_basis": self.rtf_basis,
            "scaled_by": self.scaled_by,
            "device": self.device,
            "memory": memory,
            "engine_available": self.engine_available,
            "reason": self.reason,
            "notes": list(self.notes),
            "recommended": self.recommended,
            "recommended_for": list(self.recommended_for),
            "default_for": list(self.default_for),
        }


def _round_rtf(x: float | None) -> float | None:
    if x is None:
        return None
    if x >= 10:
        return round(x)
    return round(x, 2) if x >= 0.1 else round(x, 3)


def _rtf_text(x: float) -> str:
    r = _round_rtf(x)
    return f"{r:g}"


def hint(spec: ModelSpec) -> dict[str, Any]:
    """The catalog's hardware hint (``options.hardware``), or ``{}``."""
    h = spec.options.get("hardware")
    return dict(h) if isinstance(h, dict) else {}


def measured_rtf(spec: ModelSpec) -> float | None:
    rtf = spec.measured.get("rtf") if spec.measured else None
    return float(rtf) if isinstance(rtf, (int, float)) and rtf > 0 else None


def ram_need(spec: ModelSpec) -> tuple[int | None, str | None]:
    """Memory the model takes on a CPU: the catalog's figure, else an estimate from its files."""
    h = hint(spec)
    if isinstance(h.get("ram_gb"), (int, float)):
        return int(float(h["ram_gb"]) * 1024**3), "catalog"
    if not spec.size_bytes:
        return None, None
    # what bolo's own memory budget assumes (files x 1.2), and never less than the growth the maintainers saw
    need = spec.size_bytes * 1.2 + 150 * 1024**2
    seen = spec.measured.get("memory_mb") if spec.measured else None
    if isinstance(seen, (int, float)):
        need = max(need, float(seen) * 1024**2)
    if isinstance(h.get("vram_gb"), (int, float)):
        need = max(need, float(h["vram_gb"]) * 1024**3)  # the same weights sit in RAM without a GPU
    return int(need), "estimated"


def vram_need(spec: ModelSpec) -> tuple[int | None, str | None]:
    h = hint(spec)
    if isinstance(h.get("vram_gb"), (int, float)):
        return int(float(h["vram_gb"]) * 1024**3), "catalog"
    if not spec.size_bytes:
        return None, None
    return int(spec.size_bytes * 1.3 + 0.5 * 1024**3), "estimated"


def gpu_for(spec: ModelSpec, hw: Hardware) -> tuple[Gpu | None, str | None]:
    """The GPU this model would run on with the current settings, and a remark when one is present but unused."""
    cuda, metal = hw.best_gpu("cuda"), hw.best_gpu("metal")
    any_gpu = cuda or metal
    if any_gpu is None:
        return None, None
    device = (hw.device or "cpu").lower()
    if spec.engine == "plugin:torch":
        rt = hw.runtimes.get("plugin:torch", {})
        chosen = str(rt.get("device") or "auto")
        if chosen == "cpu":
            return None, "the torch plugin is set to the CPU"
        if cuda is not None and rt.get("cuda") is False:
            return None, "the torch plugin's PyTorch build has no CUDA support"
        return any_gpu, None
    if device in ("cpu", "", "auto"):
        return None, 'a GPU was found, but the `device` setting is "cpu"'
    if spec.engine == "faster-whisper":
        if cuda is None or hw.runtimes.get("ctranslate2", {}).get("gpu") is False:
            return None, "CTranslate2 here has no CUDA support"
        return cuda, None
    if spec.engine == "tts-server":
        runtime = str(spec.options.get("runtime") or "llama-server")
        if hw.runtimes.get(runtime, {}).get("gpu") is False:
            return None, f"the installed {runtime} is a CPU build"
        return any_gpu, None
    if spec.engine in ("sherpa-onnx", "onnxruntime"):
        return (cuda, None) if device.startswith("cuda") and cuda is not None else (None, None)
    return None, None


def rtf_level(rtf: float, live: bool = False) -> str:
    comfortable, usable = (RTF_COMFORTABLE_LIVE, RTF_USABLE_LIVE) if live else (RTF_COMFORTABLE, RTF_USABLE)
    if rtf <= comfortable:
        return "comfortable"
    if rtf <= usable:
        return "usable"
    return "slow" if rtf <= RTF_SLOW else "unlikely"


def _worse(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


class Siblings:
    """Measured models by architecture, to estimate the speed of one that was never run."""

    def __init__(self, specs: list[ModelSpec]) -> None:
        self._by_key: dict[tuple[str, ...], list[ModelSpec]] = {}
        for spec in specs:
            if measured_rtf(spec) is not None and spec.size_bytes:
                self._by_key.setdefault(self._key(spec), []).append(spec)

    @staticmethod
    def _key(spec: ModelSpec) -> tuple[str, ...]:
        key: tuple[str, ...] = (spec.task, spec.engine, spec.kind)
        # one loader kind covers unrelated architectures behind a server or the torch plugin: same family only
        if spec.engine in ("tts-server", "plugin:torch"):
            key = (*key, spec.family)
        return key

    def estimate(self, spec: ModelSpec) -> tuple[float, ModelSpec] | None:
        """``(rtf on the reference machine, the model it was taken from)``, scaled by the difference in size."""
        if not spec.size_bytes:
            return None
        best: tuple[float, ModelSpec] | None = None
        for other in self._by_key.get(self._key(spec), []):
            if other.id == spec.id:
                continue
            ratio = spec.size_bytes / other.size_bytes
            distance = abs(math.log(ratio))
            if distance <= math.log(SIBLING_MAX_RATIO) and (best is None or distance < best[0]):
                best = (distance, other)
        if best is None:
            return None
        other = best[1]
        rtf = measured_rtf(other)
        assert rtf is not None
        return rtf * (spec.size_bytes / other.size_bytes), other


def assess(
    spec: ModelSpec,
    hw: Hardware,
    speed: Speed,
    siblings: Siblings | None = None,
    live: bool = False,
    engine_available: bool = True,
) -> Fit:
    """How well ``spec`` suits this machine. Pure: the same inputs give the same answer."""
    fit = _assess(spec, hw, speed, siblings, live)
    fit.engine_available = engine_available
    if not engine_available and fit.level not in ("cloud", "unavailable"):
        fit.notes.append(f"its engine ({spec.engine}) is not installed here")
    return fit


def _assess(spec: ModelSpec, hw: Hardware, speed: Speed, siblings: Siblings | None, live: bool) -> Fit:
    if spec.status == "pending":
        return Fit("unavailable", "Its files are not published yet, so it cannot be pulled; hardware is not the issue.")
    if not spec.runnable:
        return Fit("unavailable", f"No runtime bolo has can run it yet: {spec.not_runnable_reason}")
    if spec.cloud:
        return Fit(
            "cloud",
            "Runs at the provider, so this machine's hardware does not matter; audio or text leaves this server.",
            device="cloud",
        )
    if spec.builtin:
        return Fit("comfortable", "A built-in test model: no files, no real inference.", device="cpu")

    h = hint(spec)
    wants = str(h.get("gpu") or "")  # "", optional, recommended, required
    gpu, gpu_remark = gpu_for(spec, hw)
    cpu = _assess_cpu(spec, hw, speed, siblings, live, wants)
    if gpu_remark and wants in ("recommended", "required"):
        cpu.notes.append(gpu_remark)
    if gpu is None:
        return cpu

    need, basis = vram_need(spec)
    total = gpu.vram_total_bytes
    budget = int(total * 0.7) if total is not None and gpu.unified_memory else total
    name = gpu.name or "the GPU"
    if need is not None and budget is not None and need > budget:
        cpu.notes.append(f"needs about {gigabytes(need)} of GPU memory and {name} has {gigabytes(total)}")
        if wants == "required":
            cpu.level = "unlikely"
            cpu.reason = (
                f"Needs a GPU with about {gigabytes(need)} of memory ({_basis_words(basis)}); "
                f"{name} has {gigabytes(total)}."
            )
        return cpu
    if wants not in ("recommended", "required") and cpu.level in ("comfortable", "usable"):
        cpu.notes.append(f"{name} can run it too; GPU speed was never measured by the maintainers")
        return cpu
    fit = Fit("usable", "", device="gpu", memory_kind="vram", memory_basis=basis)
    fit.memory_needed_bytes = need
    fit.memory_available_bytes = total
    size = f"about {gigabytes(need)} of {gigabytes(total)}" if need is not None and total is not None else "unknown"
    shared = " (memory shared with the system)" if gpu.unified_memory else ""
    fit.reason = (
        f"Should fit on {name}{shared}: needs {size} ({_basis_words(basis)}); "
        "its speed on a GPU was never measured by the maintainers, so the level is a guess."
    )
    if need is not None and gpu.vram_free_bytes is not None and need > gpu.vram_free_bytes:
        fit.notes.append(f"only {gigabytes(gpu.vram_free_bytes)} of GPU memory is free right now")
    return fit


def _basis_words(basis: str | None) -> str:
    return "the catalog's estimate" if basis == "catalog" else "estimated from its files"


def _assess_cpu(spec: ModelSpec, hw: Hardware, speed: Speed, siblings: Siblings | None, live: bool, wants: str) -> Fit:
    fit = Fit("usable", "", device="cpu", memory_kind="ram")
    need, basis = ram_need(spec)
    usable = hw.memory.usable_bytes
    fit.memory_needed_bytes, fit.memory_basis = need, basis
    fit.memory_available_bytes = usable

    # -- speed
    lo, hi = speed.spread
    ref = measured_rtf(spec)
    source: ModelSpec | None = None
    if ref is not None:
        fit.rtf_basis = "measured"
    elif wants != "required" and siblings is not None:
        est = siblings.estimate(spec)
        if est is not None:
            ref, source = est
            fit.rtf_basis = "estimated"
            lo, hi = lo * SIBLING_SPREAD[0], hi * SIBLING_SPREAD[1]
    by = "by the micro-benchmark" if speed.source == "benchmark" else "by a rough estimate from the CPU"
    if ref is not None:
        expected = ref / speed.factor
        fit.expected_rtf = expected
        fit.rtf_range = (ref / (speed.factor * hi), ref / (speed.factor * lo))
        fit.scaled_by = speed.source
        fit.level = rtf_level(expected, live)
        pace = _pace(expected)
        if source is None:
            fit.reason = (
                f"{pace} here (real-time factor ~{_rtf_text(expected)}, scaled {by} from the "
                f"{_rtf_text(ref)} measured on the reference machine)"
            )
        else:
            fit.reason = (
                f"Not measured: {pace[0].lower()}{pace[1:]} here (real-time factor ~{_rtf_text(expected)}), estimated from "
                f"{source.id}, a measured model of the same architecture, and scaled {by}"
            )
    elif wants == "required":
        fit.level = "unlikely"
        fit.reason = "The catalog says it needs a GPU and no usable one was found here; it was never measured"
    elif wants == "recommended":
        fit.level = "slow"
        fit.reason = (
            "Never measured; the catalog says it is slow on a CPU and wants a GPU, and no usable one was found "
            "here, so expect slower than real time"
        )
    elif spec.task in ("vad", "turn") or (spec.size_bytes and spec.size_bytes < 40 * 1024**2):
        fit.level = "comfortable"
        fit.reason = "Never measured, but it is a small model: the level is a guess from its size"
    else:
        fit.level = "slow" if spec.tier == "accuracy" else "usable"
        fit.reason = (
            "Never measured, and no measured model of the same architecture to estimate from: the level is a guess "
            f"from its catalog tier ({spec.tier}) and size"
        )

    # -- memory
    if need is not None and usable is not None:
        if need > usable * 0.9:
            fit.level = "unlikely"
            limit = " (the container's limit)" if hw.memory.limit_bytes else ""
            fit.reason = (
                f"Needs about {gigabytes(need)} of memory ({_basis_words(basis)}); this machine has "
                f"{gigabytes(usable)}{limit}"
            )
        else:
            tight = need > usable * 0.6
            if tight and fit.level == "comfortable":
                fit.level = "usable"
            fit.reason += f"; needs about {gigabytes(need)} of the {gigabytes(usable)} of memory" + (
                ", which is tight" if tight else ""
            )
            free = hw.memory.available_bytes
            if free is not None and need > free:
                fit.notes.append(f"only {gigabytes(free)} of memory is free right now")
    elif need is not None:
        fit.reason += f"; needs about {gigabytes(need)} of memory, and this machine's memory could not be read"
    fit.reason += "."
    return fit


def _pace(rtf: float) -> str:
    """ "About 12x faster than real time", for a sentence that goes on with "here"."""
    if rtf <= 0:
        return "Of unknown speed"
    if rtf < 1:
        return f"About {1 / rtf:.0f}x faster than real time" if 1 / rtf >= 1.5 else "Barely faster than real time"
    return f"About {rtf:.0f}x slower than real time" if rtf >= 1.5 else "A little slower than real time"


# -- ranking ----------------------------------------------------------------------------------------------------
@dataclass
class Query:
    task: str
    language: str | None = None
    commercial: bool = False  # only models whose licence allows commercial use
    local_only: bool = False  # no cloud models
    #: True: must keep up with live speech; False: batch work (files, voice-over); None: not said
    realtime: bool | None = None
    max_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "language": self.language,
            "commercial": self.commercial,
            "local_only": self.local_only,
            "realtime": self.realtime,
            "max_size_bytes": self.max_size_bytes,
        }


def normalize_task(task: str) -> str:
    t = (task or "").strip().lower().replace("_", "-")
    return TASK_ALIASES.get(t, t)


def base_language(code: str) -> str:
    return code.strip().lower().replace("_", "-").split("-")[0]


def speaks(spec: ModelSpec, language: str | None) -> bool:
    """True when the model covers ``language`` (or is language independent, or no language was asked for)."""
    if not language or not spec.languages:
        return True
    want = base_language(language)
    return any(base_language(code) == want or code.lower() in ("multi", "multilingual") for code in spec.languages)


def in_task(spec: ModelSpec, task: str, s2s_model: str | None = None) -> bool:
    if task == S2S:
        return S2S in spec.features or (s2s_model is not None and spec.id == s2s_model)
    return spec.task == task


def quality(spec: ModelSpec) -> dict[str, Any] | None:
    """What the catalog knows about how good the model is. Error rates come from each model's own test audio,
    so they say whether a model works, not which of two models is better."""
    m = spec.measured or {}
    for key, label in (
        ("wer", "word error rate"),
        ("cer", "character error rate"),
        ("round_trip_wer", "round-trip word error rate"),
        ("round_trip_cer", "round-trip character error rate"),
        ("der", "diarization error rate"),
    ):
        value = m.get(key)
        if isinstance(value, (int, float)):
            return {
                "metric": key,
                "label": label,
                "value": float(value),
                "note": "measured by the maintainers on that model's own test audio; not comparable across models",
            }
    return None


def _poor_voice(spec: ModelSpec) -> bool:
    """A voice whose speech a recogniser could not read back (round trip), in the maintainers' run."""
    if spec.task != "tts":
        return False
    m = spec.measured or {}
    value = m.get("round_trip_wer", m.get("round_trip_cer"))
    return isinstance(value, (int, float)) and value > 0.25


@dataclass
class Candidate:
    spec: ModelSpec
    fit: Fit
    installed: bool = False
    is_default: bool = False  # the model the server's settings name for this task


def _other_language(spec: ModelSpec, q: Query) -> bool:
    """No language was asked for and this is a voice of one or a few languages other than English: no general
    suggestion. English and multilingual voices come first (with a language, every candidate already speaks it)."""
    if spec.task != "tts" or q.language:
        return False
    english = any(str(lang).lower().replace("_", "-").split("-")[0] == "en" for lang in spec.languages)
    return not (english or len(spec.languages) > 3)


def _rank_key(c: Candidate, q: Query) -> tuple[Any, ...]:
    spec, fit = c.spec, c.fit
    if q.realtime is False:  # batch: anything that keeps up is fine
        bucket = {"comfortable": 0, "usable": 0, "cloud": 1, "slow": 2, "unlikely": 3, "unavailable": 4}[fit.level]
    else:
        bucket = {"comfortable": 0, "usable": 1, "cloud": 2, "slow": 3, "unlikely": 4, "unavailable": 5}[fit.level]
    basis = {"measured": 0, "estimated": 1}.get(fit.rtf_basis, 2)
    if q.realtime:
        tier = {"realtime": 0, "balanced": 1, "accuracy": 2}[spec.tier] - (1 if spec.streaming else 0)
    else:  # the most capable model that fits
        tier = {"accuracy": 0, "balanced": 1, "realtime": 2}[spec.tier]
    # a recogniser made for the language asked for comes before one that covers a hundred (a voice is not
    # worse at a language for having others); no language asked: the broader model is the safer suggestion
    count = len(spec.languages)
    breadth = 0 if count <= 3 else (1 if count <= 25 else 2)
    specialist = (breadth if spec.task == "stt" else 0) if q.language else -breadth
    rtf = fit.expected_rtf if fit.expected_rtf is not None else math.inf
    late = rtf if q.realtime else -float(spec.params or spec.size_bytes or 0)
    return (
        bucket,
        spec.builtin,
        not fit.engine_available,
        _other_language(spec, q),
        basis,
        _poor_voice(spec),
        specialist,
        not spec.commercial_use,
        tier,
        not c.is_default,
        not c.installed,
        ALL_LEVELS.index(fit.level),
        late,
        spec.id,
    )


def rank(
    candidates: list[Candidate], q: Query, max_recommended: int = 3
) -> tuple[list[Candidate], list[str], str | None]:
    """``(ordered, recommended ids, default id)``. Every candidate stays in the list; the unlikely ones sort last."""
    ordered = sorted(candidates, key=lambda c: _rank_key(c, q))
    recommended: list[str] = []
    families: set[str] = set()
    # general suggestions first; voices of one other language only fill the places that are left
    for c in [
        *(c for c in ordered if not _other_language(c.spec, q)),
        *(c for c in ordered if _other_language(c.spec, q)),
    ]:
        if len(recommended) >= max_recommended:
            break
        if c.fit.level not in ("comfortable", "usable") or c.spec.builtin or c.spec.family in families:
            continue
        families.add(c.spec.family)
        recommended.append(c.spec.id)
    default = recommended[0] if recommended else None
    if default is None:
        # nothing is expected to keep up: the best of what merely runs is still worth naming
        default = next((c.spec.id for c in ordered if c.fit.level == "slow" and not c.spec.builtin), None)
    return ordered, recommended, default


def select(specs: list[ModelSpec], q: Query, s2s_model: str | None = None) -> tuple[list[ModelSpec], dict[str, int]]:
    """Apply the filters the user asked for (never a hardware one). Returns what is kept and what each filter hid."""
    kept: list[ModelSpec] = []
    hidden = {"language": 0, "commercial": 0, "local_only": 0, "max_size": 0}
    for spec in specs:
        if not in_task(spec, q.task, s2s_model):
            continue
        if not speaks(spec, q.language):
            hidden["language"] += 1
        elif q.commercial and not spec.commercial_use:
            hidden["commercial"] += 1
        elif q.local_only and spec.cloud:
            hidden["local_only"] += 1
        elif q.max_size_bytes is not None and spec.size_bytes > q.max_size_bytes:
            hidden["max_size"] += 1
        else:
            kept.append(spec)
    return kept, hidden


def result_dict(c: Candidate, position: int, recommended: bool, default: bool) -> dict[str, Any]:
    spec = c.spec
    return {
        "rank": position,
        "id": spec.id,
        "task": spec.task,
        "family": spec.family,
        "engine": spec.engine,
        "tier": spec.tier,
        "streaming": spec.streaming,
        "languages": spec.languages,
        "size_bytes": spec.size_bytes,
        "installed": c.installed,
        "license": spec.license,
        "commercial_use": spec.commercial_use,
        "cloud": spec.cloud,
        "validated": spec.validated,
        "quality": quality(spec),
        "recommended": recommended,
        "default": default,
        "fit": c.fit.to_dict(),
    }


def fit_warnings(
    spec: ModelSpec, fit: Fit, disk_free: int | None = None, pulling: bool = False
) -> list[dict[str, Any]]:
    """Notices for someone about to pull or load ``spec``. They inform; nothing is ever refused because of them."""
    out: list[dict[str, Any]] = []
    if fit.level in ("slow", "unlikely"):
        what = (
            "is unlikely to fit or to run well on this machine"
            if fit.level == "unlikely"
            else "is expected to be slower than real time on this machine"
        )
        out.append(
            {
                "code": "hardware_fit",
                "model": spec.id,
                "level": fit.level,
                "message": f"'{spec.id}' {what}. {fit.reason} This is a recommendation, not a restriction.",
            }
        )
    if pulling and disk_free is not None and spec.size_bytes and spec.size_bytes > disk_free:
        out.append(
            {
                "code": "disk_space",
                "model": spec.id,
                "level": "unlikely",
                "message": f"'{spec.id}' is a {gigabytes(spec.size_bytes)} download and the data directory has "
                f"{gigabytes(disk_free)} free.",
            }
        )
    return out


# -- the server's view ------------------------------------------------------------------------------------------
class Advisor:
    """Hardware, speed and fits for one running bolo, cached. Owned by the core (``core.advisor``)."""

    #: fits are recomputed after this long (engines get installed, memory limits change)
    TTL_S = 60.0

    def __init__(self, core: Bolo) -> None:
        self.core = core
        self._lock = threading.RLock()
        self._hardware: Hardware | None = None
        self._speed: Speed | None = None
        self._fits: dict[str, Fit] = {}
        self._fits_at = 0.0
        self._fits_size = -1
        #: bumped when the cached fits must not be used any more (other hardware, another speed factor)
        self._generation = 0
        self._refreshing = False
        self._warned: set[str] = set()

    # -- hardware and speed
    def hardware(self, refresh: bool = False, deep: bool = False) -> Hardware:
        with self._lock:
            if self._hardware is not None and not refresh:
                return self._hardware
        # detection can take seconds (nvidia-smi, the torch plugin's doctor): never while holding the lock,
        # which catalog requests wait on
        found = hwmod.detect(self.core.settings, deep=deep)
        with self._lock:
            if self._hardware is None or refresh:
                self._hardware = found
                self._speed = None
                self._drop_fits()
            return self._hardware

    def use_hardware(self, hw: Hardware, speed: Speed | None = None) -> None:
        """Replace what was detected (tests, and anyone who wants to see the catalog as another machine would)."""
        with self._lock:
            self._hardware, self._speed = hw, speed
            self._drop_fits()

    def speed(self) -> Speed:
        hw = self.hardware()
        with self._lock:
            if self._speed is None:
                stored, why_not = hwmod.benchmark_status(self.core.settings.data_dir, hw)
                self._speed = speed_for(hw, stored)
                if why_not and self._speed.source != "benchmark":  # say so, rather than silently estimate
                    self._speed.note += f"; {why_not}"
            return self._speed

    def benchmark(self) -> dict[str, Any]:
        """Run the micro-benchmark (a few seconds of CPU), store it in the data directory and use it from now on."""
        hw = self.hardware()
        result = hwmod.run_benchmark(hw)
        try:
            hwmod.save_benchmark(self.core.settings.data_dir, result)
        except OSError as exc:
            result["not_saved"] = str(exc)
        with self._lock:
            self._speed = speed_for(hw, result)
            self._drop_fits()
        return result

    def hardware_api(self, refresh: bool = False) -> dict[str, Any]:
        hw = self.hardware(refresh=refresh, deep=refresh)
        return {**hw.to_dict(), "summary": hw.summary(), "speed": self.speed().to_dict()}

    # -- fits
    def _engine_ok(self, cache: dict[str, bool], engine: str) -> bool:
        if engine not in cache:
            try:
                cache[engine] = bool(self.core.engines.status(engine).available)
            except Exception:
                cache[engine] = False
        return cache[engine]

    def _defaults(self) -> dict[str, str]:
        s = self.core.settings
        names = {
            "stt": s.default_stt, "tts": s.default_tts, "vad": s.default_vad, "diarization": s.default_diarization,
            "separation": s.default_separation, "turn": s.default_turn, S2S: s.s2s_model,
        }  # fmt: skip
        out: dict[str, str] = {}
        for task, name in names.items():
            spec = self.core.catalog.resolve(name) if name else None
            if spec is not None:
                out[task] = spec.id
        return out

    def _live_defaults(self) -> dict[str, str]:
        s = self.core.settings
        out = self._defaults()
        for task, name in (("stt", s.default_realtime_stt), ("tts", s.default_realtime_tts)):
            spec = self.core.catalog.resolve(name) if name else None
            if spec is not None:
                out[task] = spec.id
        return out

    def _installed(self, spec: ModelSpec) -> bool:
        try:
            return bool(self.core.manager.is_installed(spec))
        except Exception:
            return False

    def _drop_fits(self) -> None:
        """The cached fits are wrong now (other hardware, another speed factor). Call with the lock held."""
        self._fits = {}
        self._generation += 1

    def invalidate(self) -> None:
        """The cached fits are due for a refresh (a model was installed or removed). Until the next catalog or
        recommendation request recomputes them, requests in flight keep using the ones there are."""
        with self._lock:
            self._fits_at = 0.0

    def fits(self, stale_ok: bool = False) -> dict[str, Fit]:
        """The fit of every catalog entry, with the "recommended for this machine" marks, cached.

        With ``stale_ok`` (the catalog listing) fits that are due for a refresh are returned as they are and
        refreshed in the background.
        """
        specs = self.core.catalog.all()
        with self._lock:
            usable = bool(self._fits) and len(specs) == self._fits_size
            fresh = time.monotonic() - self._fits_at < self.TTL_S
            found = self._fits if usable and (fresh or stale_ok) else None
            generation = self._generation
        if found is not None:
            if not fresh:
                self.refresh_soon()
            return found
        hw, speed = self.hardware(), self.speed()
        fits = self._compute(specs, hw, speed)  # not under the lock: nothing waits for this
        with self._lock:
            if generation == self._generation:
                self._fits, self._fits_at, self._fits_size = fits, time.monotonic(), len(specs)
        return fits

    def cached_fit(self, spec: ModelSpec) -> Fit | None:
        """The fit of ``spec`` if it is already known. For the request path: this never detects hardware and
        never computes anything; when there is nothing yet it starts the computation in the background."""
        with self._lock:
            fit = self._fits.get(spec.id)
            due = not self._fits or time.monotonic() - self._fits_at >= self.TTL_S
        if due:
            self.refresh_soon()
        return fit

    def refresh_soon(self) -> None:
        """Compute the fits in a background thread (one at a time), off the event loop and off any request."""
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True

        def work() -> None:
            try:
                with self._lock:
                    self._fits_at = 0.0
                self.fits()
            except Exception:  # advice must never take anything down
                pass
            finally:
                self._refreshing = False

        threading.Thread(target=work, name="bolo-hardware", daemon=True).start()

    def _compute(self, specs: list[ModelSpec], hw: Hardware, speed: Speed) -> dict[str, Fit]:
        siblings = Siblings(specs)
        engines: dict[str, bool] = {}
        fits = {s.id: assess(s, hw, speed, siblings, False, self._engine_ok(engines, s.engine)) for s in specs}
        installed = {s.id: self._installed(s) for s in specs}
        defaults = self._defaults()
        s2s = self.core.settings.s2s_model
        for task in TASKS:
            if task == S2S:
                continue  # its models are text-to-speech entries too: they keep the marks of that ranking
            in_this = [s for s in specs if in_task(s, task, s2s)]
            # the languages of each model once: `speaks` for every model and language is most of the work otherwise
            covers = {s.id: {base_language(code) for code in s.languages} for s in in_this}
            languages = sorted(set().union(*covers.values()) - {"multi", "multilingual"}) if covers else []
            for language in [None, *languages]:
                q = Query(task, language)
                candidates = [
                    Candidate(s, fits[s.id], installed[s.id], defaults.get(task) == s.id)
                    for s in in_this
                    if language is None
                    or not covers[s.id]
                    or language in covers[s.id]
                    or not covers[s.id].isdisjoint(("multi", "multilingual"))
                ]
                _ordered, recommended, default = rank(candidates, q)
                mark = language or "*"
                for model_id in recommended:
                    if mark not in fits[model_id].recommended_for:
                        fits[model_id].recommended_for.append(mark)
                if default is not None and mark not in fits[default].default_for:
                    fits[default].default_for.append(mark)
        return fits

    def fit(self, spec: ModelSpec, stale_ok: bool = False) -> Fit:
        """The fit of one entry; see :meth:`fits`. May detect the hardware and compute: not for the request path."""
        fits = self.fits(stale_ok)
        if spec.id in fits:
            return fits[spec.id]
        # an entry added a moment ago (a provider:model id used by name)
        return assess(spec, self.hardware(), self.speed(), Siblings(self.core.catalog.all()))

    # -- recommendations
    def recommend(self, q: Query, limit: int | None = None) -> dict[str, Any]:
        s2s = self.core.settings.s2s_model
        kept, hidden = select(self.core.catalog.all(), q, s2s)
        hw, speed = self.hardware(), self.speed()
        defaults = self._live_defaults() if q.realtime else self._defaults()
        if q.realtime:  # live use has stricter levels than the cached, general fits
            siblings = Siblings(self.core.catalog.all())
            engines: dict[str, bool] = {}
            fits = {s.id: assess(s, hw, speed, siblings, True, self._engine_ok(engines, s.engine)) for s in kept}
        else:
            fits = {s.id: self.fit(s) for s in kept}
        candidates = [Candidate(s, fits[s.id], self._installed(s), defaults.get(q.task) == s.id) for s in kept]
        ordered, recommended, default = rank(candidates, q)
        results = [result_dict(c, i + 1, c.spec.id in recommended, c.spec.id == default) for i, c in enumerate(ordered)]
        note = None
        if default is None:
            note = "No model for this task is expected to run on this machine as it is; the list says why for each."
        elif default not in recommended:
            note = "Nothing is expected to keep up with real time here; the default is the best of what runs slowly."
        return {
            "query": q.to_dict(),
            "hardware": hw.summary(),
            "speed": speed.to_dict(),
            "default": default,
            "recommended": recommended,
            "total": len(results),
            # hidden by the filters that were asked for; hardware never hides a model
            "filtered_out": hidden,
            "note": note,
            "results": results[:limit] if limit else results,
        }

    def setup(
        self, use: str, language: str | None = None, commercial: bool = False, local_only: bool = True
    ) -> dict[str, Any]:
        """Models for one use ("live", "files", "voiceover") and language: a pick per task, with alternatives."""
        plan = USES[use]
        picks: list[dict[str, Any]] = []
        for task, live, optional in plan["tasks"]:
            q = Query(task, language, commercial=commercial, local_only=local_only, realtime=live)
            answer = self.recommend(q)
            by_id = {r["id"]: r for r in answer["results"]}
            chosen = by_id.get(answer["default"]) if answer["default"] else None
            picks.append(
                {
                    "task": task,
                    "optional": optional,
                    "realtime": live,
                    "model": chosen,
                    "alternatives": [by_id[i] for i in answer["recommended"] if i != answer["default"]],
                    "note": answer["note"],
                }
            )
        chosen_models = [p["model"] for p in picks if p["model"] is not None]
        to_pull = [m for m in chosen_models if not m["installed"] and not m["cloud"]]
        return {
            "use": use,
            "label": plan["label"],
            "language": language,
            "commercial": commercial,
            "local_only": local_only,
            "hardware": self.hardware().summary(),
            "speed": self.speed().to_dict(),
            "picks": picks,
            "pull": [m["id"] for m in to_pull],
            "download_bytes": sum(int(m["size_bytes"]) for m in to_pull),
        }

    # -- warnings
    def warnings(self, spec: ModelSpec, pulling: bool = False, wait: bool = True) -> list[dict[str, Any]]:
        """Notices for a pull or load of ``spec``; empty when ``hardware_warnings`` is off. Never raises.

        ``wait=False`` is for the request path and the event loop: it answers from what is already known and
        never detects hardware or computes fits (then a notice can be missing right after start-up)."""
        if not self.core.settings.hardware_warnings:
            return []
        try:
            fit = self.fit(spec, stale_ok=True) if wait else self.cached_fit(spec)
            if fit is None:
                return []
            with self._lock:
                known = self._hardware
            free = known.disk.free_bytes if pulling and known is not None else None
            return fit_warnings(spec, fit, free, pulling)
        except Exception:
            return []

    def warn_once(self, spec: ModelSpec) -> list[dict[str, Any]]:
        """The warnings of ``spec``, logged the first time it is loaded in this process."""
        found = self.warnings(spec, wait=False)
        if found and spec.id not in self._warned:
            self._warned.add(spec.id)
            for w in found:
                self.core.log("warning", str(w["message"]))
        return found
