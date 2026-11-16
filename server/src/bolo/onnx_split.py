"""Split one ONNX graph in two at a layer boundary: the front half on the accelerator, the back half on the CPU.

Used by the scheduler (:mod:`bolo.placement`) when a single-graph ONNX Runtime model (or the largest graph of a
multi-graph one) does not fit in free accelerator memory whole and the ``onnx_split`` setting is ``auto``.

How a cut point is chosen
    ONNX graphs list their nodes in topological order, so every prefix of the node list is a valid "front" and
    the rest a valid "back". For each prefix bolo counts the initializer bytes the front needs and the tensors that
    cross the boundary (activations the back reads from the front). A cut point is refused when a control-flow
    node (If, Loop, Scan) would read a value from the other half through its subgraph, when a crossing tensor's
    element type is unknown, or when either half would be empty. Among the rest, the cut keeps the fewest crossing
    tensors (a layer boundary: in a transformer, the hidden state and the few tensors every layer shares) and, among
    those, puts the most weights in the front that still fit the GPU share. Initializers both halves use are copied
    into both.

The halves are written next to the model (``.bolo-split/<sha>-<cut>/front.onnx``, ``back.onnx``; external data
for halves over 1.5 GB) and reused while the model file stays the same. :class:`SplitSession` runs them as two ONNX
Runtime sessions behind the interface engines use (``run``, ``get_inputs``, ``get_outputs``). The crossing tensors
come back to the host between the halves; IO binding would only help when both halves sit on one device, which is
never the case here, so plain ``run`` is used.

Needs the ``onnx`` package (``bolo-speech[onnx-split]``). Tested on CPU for both halves (outputs compared with the
whole graph); not on a GPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("bolo.onnx_split")

CONTROL_FLOW = {"If", "Loop", "Scan", "SequenceMap"}
#: halves larger than this are saved with their weights in a separate file (protobuf's limit is 2 GB)
EXTERNAL_DATA_ABOVE = 1_500_000_000
SPLIT_DIR = ".bolo-split"


class SplitError(RuntimeError):
    """No usable cut point, or the halves could not be built."""


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


@dataclass
class CutPoint:
    #: nodes[:index] form the front
    index: int
    front_bytes: int
    back_bytes: int
    #: tensors the back reads from the front, in a stable order
    crossing: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "front_bytes": self.front_bytes,
            "back_bytes": self.back_bytes,
            "crossing": list(self.crossing),
        }


# -- graph analysis ---------------------------------------------------------------------------------------------
def _tensor_bytes(t: Any) -> int:
    from onnx import helper

    n = 1
    for d in t.dims:
        n *= int(d)
    try:
        size = helper.tensor_dtype_to_np_dtype(t.data_type).itemsize
    except Exception:
        size = 4
    return int(n * size)


def _subgraph_reads(node: Any) -> set[str]:
    """Names a control-flow node's subgraphs read from the enclosing scope."""
    out: set[str] = set()
    for attr in node.attribute:
        graphs = ([attr.g] if attr.HasField("g") else []) + list(attr.graphs)
        for g in graphs:
            defined = {i.name for i in g.input} | {i.name for i in g.initializer}
            for n in g.node:
                defined.update(n.output)
            for n in g.node:
                out.update(x for x in n.input if x and x not in defined)
                out.update(x for x in _subgraph_reads(n) if x not in defined)
    return out


@dataclass
class _Graph:
    model: Any
    #: compute nodes, in order: the cut runs between them
    nodes: list[Any]
    init_bytes: dict[str, int]
    inputs: list[str]  # real graph inputs (not initializers)
    outputs: list[str]
    #: per compute node: what it reads (graph inputs, compute outputs, the initializers behind its weight nodes,
    #: subgraph reads); weight-node outputs are replaced by the initializers they come from
    reads: list[set[str]]
    implicit: list[set[str]]  # per compute node: subgraph reads only
    elem_types: dict[str, int]
    #: nodes computed from initializers and constants only (DequantizeLinear of a quantized weight, a Constant, a
    #: Transpose of a weight): copied into whichever half reads them, never cut across
    weight_nodes: list[Any]
    #: weight-node output -> the initializers it is computed from
    weight_roots: dict[str, set[str]]
    #: per compute node: the weight-node outputs it reads (directly or in subgraphs)
    weight_reads: list[set[str]]


def _analyse(model: Any) -> _Graph:
    import onnx

    with contextlib.suppress(Exception):
        model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    init_bytes = {t.name: _tensor_bytes(t) for t in g.initializer}
    inputs = [i.name for i in g.input if i.name not in init_bytes]
    elem: dict[str, int] = {}
    for vi in [*g.input, *g.output, *g.value_info]:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type:
            elem[vi.name] = vi.type.tensor_type.elem_type
    outputs = {o.name for o in g.output}
    weight_roots: dict[str, set[str]] = {}
    weight_nodes, nodes, reads, implicit, weight_reads = [], [], [], [], []
    for n in g.node:
        has_graph = any(a.HasField("g") or len(a.graphs) for a in n.attribute)
        ins = [x for x in n.input if x]
        if not has_graph and all(x in init_bytes or x in weight_roots for x in ins) and not set(n.output) & outputs:
            roots: set[str] = set()
            for x in ins:
                roots |= {x} if x in init_bytes else weight_roots[x]
            for o in n.output:
                if o:
                    weight_roots[o] = roots
            weight_nodes.append(n)
            continue
        sub = _subgraph_reads(n) if n.op_type in CONTROL_FLOW or has_graph else set()
        direct = set(ins) | sub
        w = {x for x in direct if x in weight_roots}
        expanded = {x for x in direct if x not in weight_roots}
        for x in w:
            expanded |= weight_roots[x]
        nodes.append(n)
        implicit.append({x for x in sub if x not in weight_roots})
        reads.append(expanded)
        weight_reads.append(w)
    return _Graph(
        model, nodes, init_bytes, inputs, [o.name for o in g.output], reads, implicit, elem, weight_nodes,
        weight_roots, weight_reads,
    )  # fmt: skip


def cut_points(model: Any) -> list[CutPoint]:
    """Every valid cut point of a model, in node order."""
    return _cut_points(_analyse(model))


def _cut_points(g: _Graph) -> list[CutPoint]:
    """Linear in the graph's size: the crossing tensors, weights and refused ranges are kept up to date as the cut
    moves one node at a time."""
    n = len(g.nodes)
    if n < 2:
        return []
    produced_at: dict[str, int] = {}
    for i, node in enumerate(g.nodes):
        for o in node.output:
            if o:
                produced_at[o] = i
    last_read: dict[str, int] = {}
    for i, r in enumerate(g.reads):
        for x in r:
            last_read[x] = i
    # cut k is refused when a subgraph reads a value made on the other side: producer p, reader i, p < k <= i
    refused = [0] * (n + 2)
    for i in range(n):
        for x in g.implicit[i]:
            p = produced_at.get(x)
            if p is not None and p != i:
                lo, hi = min(p, i), max(p, i)
                refused[lo + 1] += 1
                refused[hi + 1] -= 1
    init_last: dict[int, int] = {}
    for x, size in g.init_bytes.items():
        if x in last_read:
            init_last[last_read[x]] = init_last.get(last_read[x], 0) + size
    used_total = sum(size for x, size in g.init_bytes.items() if x in last_read)
    dying: dict[int, list[str]] = {}
    for x, made in produced_at.items():
        last = last_read.get(x, -1)
        if last > made:
            dying.setdefault(last, []).append(x)
    out: list[CutPoint] = []
    seen_inits: set[str] = set()
    front = 0
    back = used_total
    live: set[str] = set()
    blocked = 0
    for k in range(1, n):
        blocked += refused[k]
        prev = k - 1
        for x in g.reads[prev]:
            if x in g.init_bytes and x not in seen_inits:
                seen_inits.add(x)
                front += g.init_bytes[x]
        back -= init_last.get(prev, 0)  # weights read last by the node that just moved to the front
        live.difference_update(dying.get(prev, ()))
        live.update(o for o in g.nodes[prev].output if o and last_read.get(o, -1) > prev)
        if blocked > 0 or front == 0 or back == 0:
            continue
        if any(x not in g.elem_types for x in live):
            continue
        crossing = sorted(live, key=lambda x: (produced_at[x], x))
        out.append(CutPoint(k, front, back, crossing))
    return out


def choose_cut(points: Sequence[CutPoint], front_budget: int, window: int = 12) -> CutPoint | None:
    """The cut that puts the most weights in the front within ``front_budget``, among layer boundaries: points
    where no cut within ``window`` nodes has fewer tensors crossing (inside a layer, queries, keys, values and
    residuals are alive at once; between layers only the hidden state and what every layer shares)."""
    ordered = sorted(points, key=lambda p: p.index)
    counts = [len(p.crossing) for p in ordered]
    idx = [p.index for p in ordered]
    boundaries: list[CutPoint] = []
    lo = 0
    for i, p in enumerate(ordered):
        while idx[lo] < p.index - window:
            lo += 1
        hi = i
        while hi + 1 < len(ordered) and idx[hi + 1] <= p.index + window:
            hi += 1
        if counts[i] <= min(counts[lo : hi + 1]):
            boundaries.append(p)
    fitting = [p for p in boundaries if 0 < p.front_bytes <= front_budget and p.back_bytes > 0]
    if not fitting:
        return None
    return max(fitting, key=lambda p: (p.front_bytes, -len(p.crossing), -p.index))


# -- building the halves ---------------------------------------------------------------------------------------
def _value_info(g: _Graph, name: str) -> Any:
    from onnx import helper

    for vi in [*g.model.graph.input, *g.model.graph.output, *g.model.graph.value_info]:
        if vi.name == name:
            return vi
    return helper.make_tensor_value_info(name, g.elem_types[name], None)


def build_halves(model: Any, cut: CutPoint) -> tuple[Any, Any]:
    """The front and back models of ``model`` cut at ``cut``."""
    from onnx import helper

    g = _analyse(model)
    k = cut.index
    producer = {o: n for n in g.weight_nodes for o in n.output if o}

    def with_weights(compute: list[Any], wanted: set[str]) -> tuple[list[Any], set[str]]:
        """The weight nodes the compute nodes need (transitively), in the model's order, ahead of them."""
        need: set[int] = set()
        stack = list(wanted)
        reads: set[str] = set()
        while stack:
            x = stack.pop()
            w = producer.get(x)
            if w is None or id(w) in need:
                continue
            need.add(id(w))
            for i in w.input:
                if i:
                    reads.add(i)
                    stack.append(i)
        return [w for w in g.weight_nodes if id(w) in need] + compute, reads

    front_nodes, front_w = with_weights(g.nodes[:k], set().union(*g.weight_reads[:k]))
    back_nodes, back_w = with_weights(g.nodes[k:], set().union(*g.weight_reads[k:]))
    front_reads = set().union(*g.reads[:k]) | front_w | {x for n in g.nodes[:k] for x in n.input if x}
    back_reads = set().union(*g.reads[k:]) | back_w | {x for n in g.nodes[k:] for x in n.input if x}
    front_made = {o for n in front_nodes for o in n.output if o}
    back_made = {o for n in back_nodes for o in n.output if o}
    inits = {t.name: t for t in g.model.graph.initializer}
    # graph outputs no node makes (an initializer or an input handed straight through): the front returns them
    passthrough = [o for o in g.outputs if o not in front_made and o not in back_made]
    front_reads |= set(passthrough)
    front_made |= set(passthrough)

    front_outputs = list(cut.crossing) + [o for o in g.outputs if o in front_made and o not in cut.crossing]
    front = helper.make_graph(
        front_nodes,
        g.model.graph.name + "_front",
        [_value_info(g, x) for x in g.inputs if x in front_reads],
        [_value_info(g, x) for x in front_outputs],
        initializer=[inits[x] for x in inits if x in front_reads],
        value_info=[v for v in g.model.graph.value_info if v.name in front_made and v.name not in passthrough],
    )
    back_inputs = list(cut.crossing) + [x for x in g.inputs if x in back_reads and x not in cut.crossing]
    back = helper.make_graph(
        back_nodes,
        g.model.graph.name + "_back",
        [_value_info(g, x) for x in back_inputs],
        [_value_info(g, x) for x in g.outputs if x in back_made],
        initializer=[inits[x] for x in inits if x in back_reads],
        value_info=[v for v in g.model.graph.value_info if v.name in back_made and v.name not in back_inputs],
    )

    def wrap(graph: Any) -> Any:
        m = helper.make_model(graph, opset_imports=list(g.model.opset_import))
        m.ir_version = g.model.ir_version
        m.functions.extend(g.model.functions)
        for p in g.model.metadata_props:
            m.metadata_props.append(p)
        return m

    return wrap(front), wrap(back)


def load_with_data(path: Path) -> Any:
    """The model with its external weights read in. onnx's own loader refuses weight files with several hard links,
    which is how bolo's blob store keeps one copy of each file; this reads them directly, from files inside the
    model's directory only."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path), load_external_data=False)
    base = path.parent.resolve()
    handles: dict[Path, Any] = {}
    try:
        tensors = list(model.graph.initializer)
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.HasField("t"):
                    tensors.append(attr.t)
        for t in tensors:
            if t.data_location != TensorProto.EXTERNAL:
                continue
            info = {e.key: e.value for e in t.external_data}
            where = (base / info["location"]).resolve()
            if base not in where.parents:
                raise SplitError(f"{path.name}: external data outside the model directory ({info['location']})")
            fh = handles.get(where) or handles.setdefault(where, where.open("rb"))
            fh.seek(int(info.get("offset", 0)))
            data = fh.read(int(info["length"])) if "length" in info else fh.read()
            t.raw_data = data
            t.data_location = TensorProto.DEFAULT
            del t.external_data[:]
    finally:
        for fh in handles.values():
            fh.close()
    return model


def _fingerprint(path: Path) -> str:
    """The model file's SHA-256, plus the size and time of external weight files next to it."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    for extra in sorted(path.parent.glob(path.name + "*")):
        if extra != path and extra.is_file():
            st = extra.stat()
            h.update(f"{extra.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _save(model: Any, path: Path) -> None:
    import onnx

    big = sum(_tensor_bytes(t) for t in model.graph.initializer) > EXTERNAL_DATA_ABOVE
    if big:
        onnx.save_model(model, str(path), save_as_external_data=True, location=path.name + ".data")
    else:
        onnx.save_model(model, str(path))


def split_model(
    path: str | os.PathLike[str], front_budget: int, cache_root: Path | None = None
) -> tuple[Path, Path, CutPoint]:
    """Cut the model at ``path`` so the front's weights fit ``front_budget`` bytes; returns the two halves' paths
    (built once, then reused from the cache) and the cut point. Raises :class:`SplitError` when there is no usable
    cut point."""

    path = Path(path)
    root = cache_root or path.parent / SPLIT_DIR
    sha = _fingerprint(path)
    points: list[CutPoint] | None = None
    # the cut points of this model file were found before: choose among them without loading the model
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        for meta_file in sorted(root.glob(f"{sha}-*/split.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("source") == path.name:
                points = [CutPoint(**c) for c in meta["candidates"]]
                break
    if points is not None:
        cut = choose_cut(points, front_budget)
        if cut is None:
            raise SplitError(f"{path.name}: no cut point puts a non-empty front within {front_budget >> 20} MB")
        d = root / f"{sha}-{cut.index}"
        if (d / "front.onnx").is_file() and (d / "back.onnx").is_file():
            return d / "front.onnx", d / "back.onnx", cut
    model = load_with_data(path)  # with its external data: the halves carry their own weights
    g = _analyse(model)
    points = _cut_points(g)
    cut = choose_cut(points, front_budget)
    if cut is None:
        raise SplitError(
            f"{path.name}: no cut point puts a non-empty front within {front_budget >> 20} MB "
            f"({len(points)} valid cut points)"
        )
    d = root / f"{sha}-{cut.index}"
    d.mkdir(parents=True, exist_ok=True)
    front, back = build_halves(model, cut)
    _save(front, d / "front.onnx")
    _save(back, d / "back.onnx")
    (d / "split.json").write_text(
        json.dumps({"source": path.name, "cut": cut.to_dict(), "candidates": [p.to_dict() for p in points]}, indent=1),
        encoding="utf-8",
    )
    log.info(
        "split %s at node %d: %d MB in front, %d MB behind, %d tensors cross",
        path.name, cut.index, cut.front_bytes >> 20, cut.back_bytes >> 20, len(cut.crossing),
    )  # fmt: skip
    return d / "front.onnx", d / "back.onnx", cut


# -- running the halves ----------------------------------------------------------------------------------------
class SplitSession:
    """Two ONNX Runtime sessions that answer like one (``run``, ``get_inputs``, ``get_outputs``)."""

    def __init__(self, front: Any, back: Any, inputs: list[str], outputs: list[str], cut: CutPoint) -> None:
        self.front, self.back, self.cut = front, back, cut
        self._front_in = {i.name for i in front.get_inputs()}
        self._back_in = {i.name for i in back.get_inputs()}
        self._front_out = {o.name for o in front.get_outputs()}
        args = {a.name: a for a in [*front.get_inputs(), *back.get_inputs()]}
        self._inputs = [args[n] for n in inputs if n in args]
        outs = {a.name: a for a in [*front.get_outputs(), *back.get_outputs()]}
        self._outputs = [outs[n] for n in outputs if n in outs]

    def get_inputs(self) -> list[Any]:
        return list(self._inputs)

    def get_outputs(self) -> list[Any]:
        return list(self._outputs)

    def get_providers(self) -> list[str]:
        return [*self.front.get_providers(), *self.back.get_providers()]

    def run(self, output_names: Sequence[str] | None, input_feed: dict[str, Any], run_options: Any = None) -> list[Any]:
        wanted = list(output_names) if output_names else [o.name for o in self._outputs]
        front_feed = {k: v for k, v in input_feed.items() if k in self._front_in}
        front_names = [o.name for o in self.front.get_outputs()]
        values = dict(zip(front_names, self.front.run(front_names, front_feed, run_options)))
        back_wanted = [n for n in wanted if n not in self._front_out]
        if back_wanted:
            back_feed = {k: v for k, v in input_feed.items() if k in self._back_in}
            back_feed.update({k: values[k] for k in self.cut.crossing})
            values.update(zip(back_wanted, self.back.run(back_wanted, back_feed, run_options)))
        return [values[n] for n in wanted]


def open_split(
    path: str | os.PathLike[str],
    front_budget: int,
    front_providers: list[Any],
    back_providers: list[Any] | None = None,
    session_options: Callable[[], Any] | None = None,
    cache_root: Path | None = None,
) -> SplitSession:
    """Split ``path`` (or reuse its cached halves) and open both halves."""
    import onnx

    front_path, back_path, cut = split_model(path, front_budget, cache_root)
    whole = onnx.load(str(path), load_external_data=False)
    inits = {t.name for t in whole.graph.initializer}
    inputs = [i.name for i in whole.graph.input if i.name not in inits]
    outputs = [o.name for o in whole.graph.output]

    def opts() -> Any:
        return session_options() if session_options else default_options()

    from bolo.ort_providers import create_session
    from bolo.ort_providers import session_options as default_options

    front = create_session(str(front_path), opts(), front_providers)
    back = create_session(str(back_path), opts(), back_providers or ["CPUExecutionProvider"])
    return SplitSession(front, back, inputs, outputs, cut)


def make_session(
    path: str | os.PathLike[str],
    providers: list[Any],
    session_options: Callable[[], Any],
    split_budget: int | None = None,
) -> Any:
    """An ``InferenceSession`` for ``path``, or with ``split_budget`` (bytes of weights the accelerator takes) a
    :class:`SplitSession`: front on ``providers``, back on the CPU. A split that cannot be made or opened falls back
    to one whole session on the CPU, with a warning, so a placement decision never stops a model from loading."""
    from bolo.ort_providers import create_session

    if split_budget is not None and split_budget > 0:
        try:
            return open_split(path, split_budget, providers, ["CPUExecutionProvider"], session_options)
        except Exception as exc:
            log.warning("%s: not split (%s); it runs whole on the CPU", Path(path).name, exc)
            providers = ["CPUExecutionProvider"]
    return create_session(str(path), session_options(), providers)
