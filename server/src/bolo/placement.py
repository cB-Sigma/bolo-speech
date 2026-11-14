"""Where a model runs: on the accelerator, on the CPU, or split between them, decided per model at load time.

The rule is Ollama's: with an accelerator (``device`` other than ``cpu``) a model goes on the GPU when its estimated
memory fits in the free accelerator memory minus ``gpu_reserve_mb`` (and in ``gpu_memory_budget_mb`` when set), and
otherwise on the CPU, with a notice. Nothing is refused. How a model can be split depends on its runtime:

========================= ==========================================================================================
runtime                   granularity
========================= ==========================================================================================
llama.cpp (llama-server)  layers: ``-ngl N`` computed from free memory, the GGUF's layer count and its size
ONNX Runtime, several     graphs: each graph (encoder, decoder, ...) gets its own execution provider; the largest
graphs (DiCoW, LFM3)      go on the GPU while they fit, the rest stay on the CPU; with ``onnx_split = "auto"`` the
                          largest graph left over is cut in two (:mod:`bolo.onnx_split`)
ONNX Runtime, one graph   with ``onnx_split = "auto"``: cut in two at a layer boundary, the front on the GPU and the
(Nemotron, Smart Turn)    back on the CPU (:mod:`bolo.onnx_split`); else whole
PyTorch plugin            ``device_map="auto"`` with ``max_memory`` from the free memory (accelerate), reported back
everything else           whole model on one device (sherpa-onnx owns its sessions: one provider per model;
                          CTranslate2/faster-whisper: one device per model)
========================= ==========================================================================================

Several NVIDIA GPUs (``gpus``, ``gpu_spread``): a model that names no GPU (``device = "cuda"``) goes on the allowed
GPU with the most memory available. A llama.cpp model that does not fit whole on that GPU is spread over several by
layers (llama-server's ``--tensor-split``), as Ollama does; with ``gpu_spread`` every llama.cpp model is spread over
all allowed GPUs. Other runtimes cannot split a model across GPUs: they take one GPU each.

The pure functions here take the free memory as a number, so tests drive them with made-up figures.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

MB = 1024 * 1024

#: ``device`` families each engine can use; anything else runs on the CPU with a notice
_ORT = frozenset({"cuda", "metal", "rocm", "dml", "directml", "openvino", "migraphx"})
ENGINE_FAMILIES: dict[str, frozenset[str]] = {
    "sherpa-onnx": frozenset({"cuda", "metal"}),
    "faster-whisper": frozenset({"cuda"}),
    "onnxruntime": _ORT,
    "tts-server": frozenset({"cuda", "metal", "rocm", "vulkan"}),
    "fake": frozenset({"cuda", "metal", "rocm", "vulkan", *_ORT}),
}
#: model kinds of the onnxruntime engine made of several graphs, placed graph by graph
GRAPH_KINDS = ("dicow_onnx", "lfm3_audio_onnx")
#: share of a llama.cpp model's size added for its KV cache and compute buffers when offloading layers
LLAMA_OVERHEAD = 0.10
#: ONNX Runtime and the other in-process runtimes need about this much more than the files on the accelerator
GRAPH_OVERHEAD = 1.2
#: single-graph ONNX Runtime models that may be cut in two (bolo.onnx_split)
SPLITTABLE_KINDS = ("smart_turn", "nemotron_diarization")


def family(device: str | None) -> str:
    name = (device or "cpu").strip().lower().partition(":")[0]
    if name in ("cuda", "gpu"):
        return "cuda"
    if name in ("coreml", "mps", "metal"):
        return "metal"
    if name == "directml":
        return "dml"
    return name or "cpu"


def engine_families(engine_id: str) -> frozenset[str]:
    if engine_id.startswith("plugin:"):
        return frozenset({"cuda", "metal"})
    return ENGINE_FAMILIES.get(engine_id, frozenset())


def split_mode(engine_id: str, kind: str, options: dict[str, Any]) -> str:
    """``layers``, ``graphs``, ``device_map`` or ``whole``."""
    if engine_id == "fake" and options.get("split"):
        return str(options["split"])
    if engine_id == "tts-server" and str(options.get("runtime") or "llama-server") == "llama-server":
        return "layers"
    if engine_id == "onnxruntime" and kind in GRAPH_KINDS:
        return "graphs"
    if engine_id.startswith("plugin:"):
        return "device_map"
    return "whole"


def processor(gpu_bytes: int, total_bytes: int) -> str:
    """Ollama's PROCESSOR column: "100% GPU", "100% CPU", or "35%/65% CPU/GPU" (by bytes)."""
    if total_bytes <= 0 or gpu_bytes <= 0:
        return "100% CPU"
    if gpu_bytes >= total_bytes:
        return "100% GPU"
    cpu = min(99, max(1, round(100 * (total_bytes - gpu_bytes) / total_bytes)))
    return f"{cpu}%/{100 - cpu}% CPU/GPU"


@dataclass
class Placement:
    #: the device the engine is given: "cpu", "cuda:0", "coreml", ...
    device: str = "cpu"
    #: what the settings (or a pin) asked for
    requested: str = "cpu"
    #: how the model is split: whole | layers | graphs | device_map
    split: str = "whole"
    gpu_bytes: int = 0
    cpu_bytes: int = 0
    #: llama.cpp: layers offloaded (``-ngl``) and the model's layer count (repeating layers plus the output layer)
    gpu_layers: int | None = None
    total_layers: int | None = None
    #: the operator set ``-ngl`` in the server arguments: bolo left it alone
    operator_layers: bool = False
    #: ONNX Runtime models made of several graphs: role -> device
    graphs: dict[str, str] = field(default_factory=dict)
    #: PyTorch plugin: accelerate's ``max_memory`` (device -> bytes)
    max_memory: dict[str, int] | None = None
    #: ONNX Runtime graphs cut in two (bolo.onnx_split): role -> bytes of weights its front half may put on the GPU
    graph_split: dict[str, int] = field(default_factory=dict)
    #: why the model is not (all) on the accelerator, when it was asked for
    notice: str | None = None
    #: the free accelerator memory was known when the decision was made
    measured: bool = True
    #: CUDA GPUs the model's weights are on (several: llama.cpp layers spread with ``--tensor-split``)
    gpus: list[int] = field(default_factory=list)
    #: planned accelerator bytes per GPU
    gpu_bytes_by_gpu: dict[int, int] = field(default_factory=dict)
    #: llama.cpp across several GPUs: each GPU's share of the layers (in ``gpus`` order) and the main GPU
    tensor_split: list[float] | None = None
    main_gpu: int | None = None

    @property
    def on_gpu(self) -> bool:
        return self.gpu_bytes > 0

    def processor(self) -> str:
        return processor(self.gpu_bytes, self.gpu_bytes + self.cpu_bytes)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "device": self.device,
            "requested": self.requested,
            "split": self.split,
            "gpu_bytes": self.gpu_bytes,
            "cpu_bytes": self.cpu_bytes,
            "processor": self.processor(),
            "notice": self.notice,
            "free_memory_known": self.measured,
        }
        if self.total_layers is not None:
            out.update(gpu_layers=self.gpu_layers, total_layers=self.total_layers, operator_layers=self.operator_layers)
        if self.graphs:
            out["graphs"] = dict(self.graphs)
        if self.max_memory is not None:
            out["max_memory"] = dict(self.max_memory)
        if self.graph_split:
            out["graph_split"] = dict(self.graph_split)
        out["gpus"] = list(self.gpus)
        if self.gpu_bytes_by_gpu:
            out["gpu_bytes_by_gpu"] = {str(k): v for k, v in sorted(self.gpu_bytes_by_gpu.items())}
        if self.tensor_split is not None:
            out.update(tensor_split=list(self.tensor_split), main_gpu=self.main_gpu)
        return out


# -- llama.cpp: GGUF layers and -ngl ---------------------------------------------------------------------------
@dataclass
class LayerInfo:
    block_count: int
    #: bytes of the GGUF file holding the layers
    file_bytes: int
    #: bytes that go to the accelerator first, whatever the layer count (a multimodal projector, ``-mm``)
    extra_bytes: int = 0

    @property
    def total_layers(self) -> int:
        return self.block_count + 1  # the repeating blocks plus the output layer, as llama.cpp counts for -ngl

    @property
    def per_layer(self) -> float:
        return self.file_bytes / self.total_layers


_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read(fh: IO[bytes], fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _gguf_string(fh: IO[bytes]) -> str:
    n = _read(fh, "<Q")
    if n > 1 << 24:
        raise ValueError("implausible GGUF string length")
    return fh.read(n).decode("utf-8", errors="replace")


def _gguf_value(fh: IO[bytes], vtype: int) -> Any:
    if vtype in _GGUF_SCALARS:
        return _read(fh, _GGUF_SCALARS[vtype])
    if vtype == 8:
        return _gguf_string(fh)
    if vtype == 9:
        item_type = _read(fh, "<I")
        count = _read(fh, "<Q")
        if item_type in _GGUF_SCALARS:  # skip without decoding (token lists hold hundreds of thousands)
            fh.seek(struct.calcsize(_GGUF_SCALARS[item_type]) * count, 1)
            return None
        for _ in range(count):
            _gguf_value(fh, item_type)
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_block_count(path: Path) -> int | None:
    """``<architecture>.block_count`` from a GGUF file's header, or None when it cannot be read."""
    try:
        with Path(path).open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            version = _read(fh, "<I")
            if version < 2:
                return None
            _read(fh, "<Q")  # tensors
            kv_count = _read(fh, "<Q")
            arch: str | None = None
            counts: dict[str, int] = {}
            for _ in range(kv_count):
                key = _gguf_string(fh)
                vtype = _read(fh, "<I")
                value = _gguf_value(fh, vtype)
                if key == "general.architecture":
                    arch = str(value)
                elif key.endswith(".block_count") and isinstance(value, int):
                    counts[key] = int(value)
                if arch is not None and f"{arch}.block_count" in counts:
                    return counts[f"{arch}.block_count"]
            if arch is not None and f"{arch}.block_count" in counts:
                return counts[f"{arch}.block_count"]
            return next(iter(counts.values()), None)
    except (OSError, ValueError, struct.error):
        return None


def operator_layers(args: list[str]) -> int | None:
    """The ``-ngl`` an operator put in the server arguments (``-ngl 20``, ``--n-gpu-layers=20``), or None."""
    names = ("-ngl", "--n-gpu-layers", "--gpu-layers")
    for i, a in enumerate(args):
        for name in names:
            if a == name and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    return 999  # "auto", "all": llama.cpp decides; counted as all
            if a.startswith(name + "="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    return 999
    return None


def offload_layers(layers: LayerInfo, available_bytes: int, overhead: float = LLAMA_OVERHEAD) -> int:
    """How many layers fit in ``available_bytes`` (Ollama's arithmetic, simplified): the projector first, then a
    share of the model for the KV cache and compute buffers, then whole layers."""
    usable = available_bytes - layers.extra_bytes - int(layers.file_bytes * overhead)
    if usable <= 0:
        return 0
    if usable >= layers.file_bytes:
        return layers.total_layers
    return max(0, min(layers.total_layers, math.floor(usable / layers.per_layer)))


def layer_split(layers: LayerInfo, ngl: int) -> tuple[int, int]:
    """``(gpu_bytes, cpu_bytes)`` with ``ngl`` layers offloaded."""
    n = max(0, min(ngl, layers.total_layers))
    total = layers.file_bytes + layers.extra_bytes
    if n == 0:
        return 0, total
    gpu = round(n * layers.per_layer) + layers.extra_bytes
    if n >= layers.total_layers:
        gpu = total
    return gpu, total - gpu


# -- ONNX Runtime: graph by graph ------------------------------------------------------------------------------
def assign_graphs(graphs: dict[str, int], available_bytes: int, overhead: float = GRAPH_OVERHEAD) -> dict[str, bool]:
    """Role -> on the accelerator. Largest graph first (the encoder of an encoder-decoder model, the decoder of a
    language model), each while it fits in what is left; the rest stay on the CPU."""
    left = available_bytes
    out: dict[str, bool] = {}
    for role, size in sorted(graphs.items(), key=lambda kv: (-kv[1], kv[0])):
        need = int(size * overhead)
        if need <= left:
            out[role] = True
            left -= need
        else:
            out[role] = False
    return out


# -- several GPUs ----------------------------------------------------------------------------------------------
def best_gpu(available: dict[int, int | None]) -> int:
    """The GPU with the most memory available (unknown counts as none); the lowest number on a tie."""
    return min(available, key=lambda i: (-(available[i] or 0), i))


def spread_layers(layers: LayerInfo, available: dict[int, int], overhead: float = LLAMA_OVERHEAD) -> dict[int, int]:
    """llama.cpp layers per GPU when a model is spread over several (``--tensor-split``): as many layers as fit in
    all of them together, shared in proportion to what each has available. The projector and the share for the KV
    cache and buffers are counted on the GPU with the most memory (llama.cpp's main GPU)."""
    usable = {i: max(0, int(a)) for i, a in available.items()}
    if not usable:
        return {}
    main = best_gpu(dict(usable))
    usable[main] = max(0, usable[main] - layers.extra_bytes - int(layers.file_bytes * overhead))
    room = sum(usable.values())
    if room <= 0:
        return {}
    n = layers.total_layers if room >= layers.file_bytes else max(0, math.floor(room / layers.per_layer))
    n = min(n, sum(math.floor(u / layers.per_layer) for u in usable.values()) if n < layers.total_layers else n)
    if n <= 0:
        return {}
    # largest remainder: each GPU's share of n, never more layers than fit on it
    exact = {i: n * u / room for i, u in usable.items()}
    out = {i: math.floor(x) for i, x in exact.items()}
    for i in sorted(exact, key=lambda i: (-(exact[i] - out[i]), i)):
        if sum(out.values()) >= n:
            break
        out[i] += 1
    return {i: k for i, k in out.items() if k > 0}


def plan_spread(
    *,
    model_id: str,
    requested: str,
    layers: LayerInfo,
    available: dict[int, int],
) -> Placement:
    """A llama.cpp model spread over several GPUs by layers."""
    per_gpu = spread_layers(layers, available)
    n = sum(per_gpu.values())
    p = Placement(device="cpu", requested=requested, split="layers", total_layers=layers.total_layers, gpu_layers=n)
    p.gpu_bytes, p.cpu_bytes = layer_split(layers, n)
    if n <= 0:
        free = sum(max(0, a) for a in available.values()) // MB
        p.notice = f"'{model_id}' does not fit in {free} MB on GPUs {', '.join(map(str, sorted(available)))}: it runs on the CPU"
        return p
    gpus = sorted(per_gpu)
    main = best_gpu({i: available[i] for i in gpus})
    p.device = f"cuda:{main}"
    p.gpus = gpus
    p.main_gpu = main
    p.tensor_split = [round(per_gpu[i] / n, 4) for i in gpus]
    weights = p.gpu_bytes - layers.extra_bytes if n else 0
    p.gpu_bytes_by_gpu = {i: int(weights * per_gpu[i] / n) for i in gpus}
    p.gpu_bytes_by_gpu[main] += p.gpu_bytes - sum(p.gpu_bytes_by_gpu.values())  # the projector and rounding
    if n < layers.total_layers:
        p.notice = f"'{model_id}': {n} of {layers.total_layers} layers on GPUs {', '.join(map(str, gpus))}; the rest on the CPU"
    return p


# -- the decision ----------------------------------------------------------------------------------------------
def plan(
    *,
    model_id: str,
    engine_id: str,
    requested: str,
    device: str,
    split: str,
    need_bytes: int,
    available_bytes: int | None,
    layers: LayerInfo | None = None,
    graphs: dict[str, int] | None = None,
    operator_ngl: int | None = None,
    device_index: int = 0,
    onnx_split: bool = False,
) -> Placement:
    """Place one model.

    ``onnx_split``: ONNX Runtime graphs that do not fit whole may be cut in two (the ``onnx_split`` setting is
    ``auto`` and the ``onnx`` package is installed).

    ``requested`` is the ``device`` asked for (``auto`` included), ``device`` the concrete accelerator it resolved to
    (``cpu`` when there is none), ``need_bytes`` the model's estimated memory, ``available_bytes`` what may be used
    on the accelerator (free memory minus the reserve, capped by the GPU budget), or None when unknown.
    """
    p = Placement(device="cpu", requested=requested, split="whole", cpu_bytes=need_bytes)
    if family(device) == "cpu":
        if family(requested) not in ("cpu", "auto"):
            p.notice = f"{requested} is not available here: '{model_id}' runs on the CPU"
        return p
    fam = family(device)
    if fam not in engine_families(engine_id):
        p.notice = f"the {engine_id} engine cannot use {device}: '{model_id}' runs on the CPU"
        return p
    p.measured = available_bytes is not None
    avail = available_bytes if available_bytes is not None else 1 << 62
    mb = MB

    if split == "layers" and layers is not None:
        p.split = "layers"
        p.total_layers = layers.total_layers
        if operator_ngl is not None:
            p.operator_layers = True
            n = min(operator_ngl, layers.total_layers)
        else:
            n = offload_layers(layers, avail) if available_bytes is not None else layers.total_layers
        p.gpu_layers = n
        p.gpu_bytes, p.cpu_bytes = layer_split(layers, n)
        if n > 0:
            p.device = device
        if not p.operator_layers and n < layers.total_layers:
            free = f"{max(0, avail) // mb} MB" if available_bytes is not None else "unknown memory"
            p.notice = (
                f"'{model_id}': {n} of {layers.total_layers} layers on {device} ({free} available); the rest on the CPU"
                if n
                else f"'{model_id}' does not fit in {free} on {device}: it runs on the CPU"
            )
        return p

    if split == "graphs" and graphs:
        p.split = "graphs"
        placed = assign_graphs(graphs, avail)
        total = sum(graphs.values())
        p.graphs = {role: (device if on else "cpu") for role, on in placed.items()}
        gpu = sum(graphs[r] for r, on in placed.items() if on)
        left = avail - sum(int(graphs[r] * GRAPH_OVERHEAD) for r, on in placed.items() if on)
        rest = [r for r, on in placed.items() if not on]
        if onnx_split and rest and left > 0 and available_bytes is not None:
            # the largest graph that did not fit is cut in two: its front takes what is left
            role = max(rest, key=lambda r: (graphs[r], r))
            budget = int(left / GRAPH_OVERHEAD)
            p.graphs[role] = device
            p.graph_split[role] = budget
            gpu += min(budget, graphs[role])
        p.gpu_bytes = int(need_bytes * gpu / total) if total else 0
        p.cpu_bytes = need_bytes - p.gpu_bytes
        if gpu:
            p.device = device
        if p.graph_split:
            role, budget = next(iter(p.graph_split.items()))
            p.notice = f"'{model_id}': graph {role} is cut in two, about {budget // mb} MB of it on {device}" + (
                f"; graphs {', '.join(sorted(r for r in rest if r != role))} run on the CPU" if len(rest) > 1 else ""
            )
        elif gpu < total:
            on_cpu = ", ".join(sorted(r for r, on in placed.items() if not on))
            p.notice = (
                f"'{model_id}': graphs {on_cpu} run on the CPU ({max(0, avail) // mb} MB available on {device})"
                if gpu
                else f"'{model_id}' does not fit in {max(0, avail) // mb} MB on {device}: it runs on the CPU"
            )
        return p

    if split == "device_map" and need_bytes > avail > 0:
        p.split = "device_map"
        p.device = device
        p.max_memory = {str(device_index): int(avail), "cpu": int(max(need_bytes - avail, 0) * 2 + 1024 * mb)}
        p.gpu_bytes, p.cpu_bytes = int(avail), int(need_bytes - avail)
        p.notice = f"'{model_id}' is split across {device} and the CPU ({avail // mb} MB available on {device})"
        return p

    if need_bytes <= avail:
        p.device = device
        p.gpu_bytes, p.cpu_bytes = need_bytes, 0
        return p
    if onnx_split and split == "whole" and engine_id == "onnxruntime" and available_bytes is not None and avail > 0:
        # one graph that does not fit: cut it at a layer boundary, the front's weights within the free memory
        budget = int(avail / GRAPH_OVERHEAD)
        p.split = "onnx_split"
        p.device = device
        p.graph_split = {"model": budget}
        p.gpu_bytes = min(int(budget * GRAPH_OVERHEAD), need_bytes)
        p.cpu_bytes = need_bytes - p.gpu_bytes
        p.notice = f"'{model_id}' is cut in two: about {budget // mb} MB of weights on {device}, the rest on the CPU"
        return p
    p.notice = (
        f"'{model_id}' needs about {need_bytes // mb} MB and {max(0, avail) // mb} MB is available on {device}: "
        "it runs on the CPU"
    )
    return p
