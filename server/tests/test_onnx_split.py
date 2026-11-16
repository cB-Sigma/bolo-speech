"""Splitting one ONNX graph in two (bolo.onnx_split): the halves together answer exactly as the whole graph does.

Both halves run on the CPU here (CI has no GPU); the "accelerator" side is the CPU provider or a fake list.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from bolo import onnx_split  # noqa: E402
from bolo.onnx_split import SplitError, SplitSession, choose_cut, cut_points, make_session, split_model  # noqa: E402

CPU = ["CPUExecutionProvider"]
D = 16


def tiny_transformer(layers: int = 6, d: int = D, seed: int = 0, with_if: bool = False, quantized: bool = False) -> Any:
    """A transformer-shaped graph: per layer a matmul, a bias, a GELU-ish nonlinearity, a residual and a layer
    norm; a scale input every layer reads, an initializer every layer shares, an early output and an output that is an initializer.
    ``quantized``: int8 weights dequantized by nodes at the start of the graph, as quantized exports have them."""
    rng = np.random.default_rng(seed)
    inits = [numpy_helper.from_array(rng.standard_normal((d,)).astype(np.float32) * 0.1, "shared_bias")]
    nodes = []
    dequant = []
    h = "x"
    for i in range(layers):
        w = numpy_helper.from_array((rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32), f"w{i}")
        if quantized:
            q = np.clip(np.round(numpy_helper.to_array(w) * 64), -127, 127).astype(np.int8)
            w = numpy_helper.from_array(q, f"w{i}.q")
            inits.append(numpy_helper.from_array(np.array(1 / 64, np.float32), f"w{i}.scale"))
            dequant.append(helper.make_node("DequantizeLinear", [f"w{i}.q", f"w{i}.scale"], [f"w{i}"]))
        g = numpy_helper.from_array(np.ones((d,), np.float32), f"ln{i}.g")
        b = numpy_helper.from_array(np.zeros((d,), np.float32), f"ln{i}.b")
        inits += [w, g, b]
        nodes += [
            helper.make_node("MatMul", [h, f"w{i}"], [f"l{i}.mm"]),
            helper.make_node("Add", [f"l{i}.mm", "shared_bias"], [f"l{i}.add"]),
            helper.make_node("Tanh", [f"l{i}.add"], [f"l{i}.act"]),
            helper.make_node("Mul", [f"l{i}.act", "scale"], [f"l{i}.scaled"]),
            helper.make_node("Add", [h, f"l{i}.scaled"], [f"l{i}.res"]),
            helper.make_node("LayerNormalization", [f"l{i}.res", f"ln{i}.g", f"ln{i}.b"], [f"l{i}.out"], axis=-1),
        ]
        h = f"l{i}.out"
    nodes = dequant + nodes
    inits.append(numpy_helper.from_array(np.arange(3, dtype=np.float32), "constant_out"))
    outputs = [
        helper.make_tensor_value_info(h, TensorProto.FLOAT, [1, None, d]),
        helper.make_tensor_value_info("l0.out", TensorProto.FLOAT, [1, None, d]),
        helper.make_tensor_value_info("constant_out", TensorProto.FLOAT, [3]),
    ]
    if with_if:
        # a late If whose branches read an early activation from the enclosing graph
        then_g = helper.make_graph(
            [helper.make_node("Identity", ["l0.out"], ["t"])], "then", [],
            [helper.make_tensor_value_info("t", TensorProto.FLOAT, None)],
        )  # fmt: skip
        else_g = helper.make_graph(
            [helper.make_node("Neg", ["l0.out"], ["e"])], "else", [],
            [helper.make_tensor_value_info("e", TensorProto.FLOAT, None)],
        )  # fmt: skip
        nodes += [
            helper.make_node("ReduceSum", [h], ["sum"], keepdims=0),
            helper.make_node("Greater", ["sum", "zero"], ["cond"]),
            helper.make_node("If", ["cond"], ["branch"], then_branch=then_g, else_branch=else_g),
        ]
        inits.append(numpy_helper.from_array(np.array(0, np.float32), "zero"))
        outputs.append(helper.make_tensor_value_info("branch", TensorProto.FLOAT, [1, None, d]))
    graph = helper.make_graph(
        nodes,
        "tiny",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, None, d]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
        ],
        outputs,
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def feeds(seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"x": rng.standard_normal((1, 7, D)).astype(np.float32), "scale": np.array([0.5], np.float32)}


def whole(path: Path) -> Any:
    return ort.InferenceSession(str(path), providers=CPU)


def layer_bytes(quantized: bool = False) -> int:
    return (D * D + 4 if quantized else D * D * 4) + 2 * D * 4  # one layer's weights


def test_cut_points_are_layer_boundaries_and_crossing_is_small() -> None:
    points = cut_points(tiny_transformer())
    assert points
    # at a layer boundary only the hidden state crosses (scale and shared_bias are an input and an initializer)
    boundaries = [p for p in points if p.crossing and p.crossing[-1].endswith(".out")]
    assert boundaries and min(len(p.crossing) for p in points) == 1
    cut = choose_cut(points, 3 * layer_bytes() + D * 4)
    assert cut is not None and cut.crossing == ["l2.out"]  # three layers in front: the most that fit
    assert cut.front_bytes == 3 * layer_bytes() + D * 4  # the shared bias counts in both halves
    assert cut.back_bytes == 3 * layer_bytes() + D * 4
    assert choose_cut(points, 10) is None  # not even one layer fits


def no_fusion() -> Any:
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return o


@pytest.mark.parametrize("budget_layers", [1, 3, 5])
@pytest.mark.parametrize("optimised", [False, True])
def test_split_matches_the_whole_graph(tmp_path: Path, budget_layers: int, optimised: bool) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(tiny_transformer(), str(path))
    options = ort.SessionOptions if optimised else no_fusion
    session = make_session(path, CPU, options, split_budget=budget_layers * layer_bytes() + D * 4)
    assert isinstance(session, SplitSession)
    assert session.cut.crossing == [f"l{budget_layers - 1}.out"]
    ref = ort.InferenceSession(str(path), sess_options=options(), providers=CPU)
    assert [i.name for i in session.get_inputs()] == [i.name for i in ref.get_inputs()]
    assert [o.name for o in session.get_outputs()] == [o.name for o in ref.get_outputs()]
    f = feeds()
    got, want = session.run(None, f), ref.run(None, f)
    for a, b in zip(got, want, strict=True):
        if optimised:
            # ONNX Runtime fuses nodes differently on each side of the cut: float rounding, not a different result
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(a, b)  # the same kernels, node by node: bit for bit
    # one output, the early one (made by the front half), and outputs in the order asked for
    np.testing.assert_allclose(session.run(["l0.out"], f)[0], ref.run(["l0.out"], f)[0], rtol=1e-5, atol=1e-6)
    last = ref.get_outputs()[0].name
    a, b = session.run([last, "l0.out"], f)
    np.testing.assert_allclose(a, ref.run([last], f)[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(b, ref.run(["l0.out"], f)[0], rtol=1e-5, atol=1e-6)


def test_quantized_weights_go_with_the_layer_that_uses_them(tmp_path: Path) -> None:
    """Dequantize nodes at the start of the graph are not cut points: each goes to the half that reads it."""
    model = tiny_transformer(quantized=True)
    points = cut_points(model)
    assert all(not any(x.startswith("w") for x in p.crossing) for p in points)
    cut = choose_cut(points, 2 * layer_bytes(True) + D * 4)
    assert cut is not None and cut.crossing == ["l1.out"]
    path = tmp_path / "q.onnx"
    onnx.save(model, str(path))
    session = make_session(path, CPU, no_fusion, split_budget=2 * layer_bytes(True) + D * 4)
    assert isinstance(session, SplitSession)
    ref = ort.InferenceSession(str(path), sess_options=no_fusion(), providers=CPU)
    for a, b in zip(session.run(None, feeds()), ref.run(None, feeds()), strict=True):
        np.testing.assert_array_equal(a, b)
    front = onnx.load(str(onnx_split.split_model(path, 2 * layer_bytes(True) + D * 4)[0]))
    assert {n.output[0] for n in front.graph.node if n.op_type == "DequantizeLinear"} == {"w0", "w1"}


def test_control_flow_reading_across_is_refused(tmp_path: Path) -> None:
    model = tiny_transformer(with_if=True)
    points = cut_points(model)
    # the If reads l0.out through its branches: no cut between layer 0's output and the If is allowed
    produced = {o: i for i, n in enumerate(model.graph.node) for o in n.output}
    if_at = next(i for i, n in enumerate(model.graph.node) if n.op_type == "If")
    assert all(not (produced["l0.out"] < p.index <= if_at) for p in points)
    assert all(p.index <= produced["l0.out"] for p in points)
    path = tmp_path / "if.onnx"
    onnx.save(model, str(path))
    # the only cuts left are inside layer 0, before l0.out: the split still answers as the whole graph does
    session = make_session(path, CPU, no_fusion, split_budget=3 * layer_bytes())
    assert isinstance(session, SplitSession) and session.cut.index <= produced["l0.out"]
    ref = ort.InferenceSession(str(path), sess_options=no_fusion(), providers=CPU)
    for a, b in zip(session.run(None, feeds()), ref.run(None, feeds()), strict=True):
        np.testing.assert_array_equal(a, b)
    # nothing fits: no split; make_session falls back to one whole session on the CPU
    with pytest.raises(SplitError, match="no cut point"):
        split_model(path, 100)
    assert not isinstance(make_session(path, CPU, ort.SessionOptions, split_budget=100), SplitSession)


def test_external_data_and_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ext.onnx"
    onnx.save_model(
        tiny_transformer(layers=4), str(path), save_as_external_data=True, location="ext.onnx.data", size_threshold=0
    )
    assert (tmp_path / "ext.onnx.data").is_file()
    budget = 2 * layer_bytes() + D * 4
    front, back, cut = split_model(path, budget)
    assert front.parent.parent == tmp_path / onnx_split.SPLIT_DIR and cut.crossing == ["l1.out"]
    s = onnx_split.open_split(path, budget, CPU)
    np.testing.assert_allclose(s.run(None, feeds())[0], whole(path).run(None, feeds())[0], rtol=1e-5, atol=1e-6)

    # the same model and budget again: the cached halves, without loading the model
    def no_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model was loaded again")

    monkeypatch.setattr(onnx, "load", no_load)
    assert split_model(path, budget)[:2] == (front, back)
    monkeypatch.undo()
    # another budget: another cut, cached next to the first
    front3, _, cut3 = split_model(path, 3 * layer_bytes() + D * 4)
    assert front3 != front and cut3.crossing == ["l2.out"]
    # a changed model file is split again
    onnx.save_model(tiny_transformer(layers=4, seed=9), str(path), save_as_external_data=True,
                    location="ext.onnx.data", size_threshold=0)  # fmt: skip
    assert split_model(path, budget)[0] != front


def test_providers_reach_each_half(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(tiny_transformer(), str(path))
    seen: list[Any] = []
    real = ort.InferenceSession

    def recording(p: str, sess_options: Any = None, providers: Any = None, **kw: Any) -> Any:
        seen.append((Path(p).name, providers, sess_options.get_session_config_entry("session.x64quantprecision")))
        return real(p, sess_options=sess_options, providers=CPU)  # this machine only has the CPU

    monkeypatch.setattr(ort, "InferenceSession", recording)
    gpu = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    # options made without bolo.ort_providers: both halves still ask for exact int8 products
    make_session(path, gpu, ort.SessionOptions, split_budget=2 * layer_bytes() + D * 4)
    assert seen == [("front.onnx", gpu, "1"), ("back.onnx", ["CPUExecutionProvider"], "1")]


REAL = {"smart-turn-v3": "*.onnx", "nemotron-3-diarization": "onnx/*.onnx"}


@pytest.mark.parametrize("model_id", sorted(REAL))
def test_real_model_if_cached(model_id: str, tmp_path: Path) -> None:
    """Real bolo models, when a data directory here has them (BOLO_DATA_DIR or ~/.cache/bolo): the cut graph
    answers as the whole one does."""
    roots = [os.environ.get("BOLO_DATA_DIR"), str(Path.home() / ".cache" / "bolo")]
    found = next((f for r in roots if r for f in sorted((Path(r) / "models" / model_id).glob(REAL[model_id]))), None)
    if found is None:
        pytest.skip(f"{model_id} is not downloaded here")
    ref = whole(found)
    size = sum(onnx_split._tensor_bytes(t) for t in onnx_split.load_with_data(found).graph.initializer)
    session = onnx_split.open_split(found, size // 2, CPU, cache_root=tmp_path)
    assert 0 < session.cut.front_bytes <= size // 2
    rng = np.random.default_rng(0)
    feed: dict[str, np.ndarray] = {}
    for inp in ref.get_inputs():
        sizes = {"num_mel_frames": 200, "num_frames": 25, "num_cached_frames": 0}
        shape = [d if isinstance(d, int) else sizes.get(str(d), 1) for d in inp.shape]
        feed[inp.name] = (
            np.ones(shape, np.int64) if "int64" in inp.type else rng.standard_normal(shape).astype(np.float32)
        )
    for a, b in zip(session.run(None, feed), ref.run(None, feed), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


# -- placement and the engines -------------------------------------------------------------------------------------
def test_plan_cuts_a_graph_that_does_not_fit() -> None:
    from bolo.placement import plan

    mb = 1024 * 1024
    args: dict[str, Any] = {
        "model_id": "nemotron", "engine_id": "onnxruntime", "requested": "cuda", "device": "cuda", "split": "whole",
        "need_bytes": 600 * mb, "available_bytes": 240 * mb,
    }  # fmt: skip
    p = plan(**args, onnx_split=True)
    assert (p.split, p.device, p.graph_split) == ("onnx_split", "cuda", {"model": 200 * mb})
    assert p.gpu_bytes == 240 * mb and p.cpu_bytes == 360 * mb and p.processor() == "60%/40% CPU/GPU"
    assert "cut in two" in (p.notice or "")
    assert plan(**args).device == "cpu"  # onnx_split off: the CPU, as before
    assert plan(**{**args, "available_bytes": 0}, onnx_split=True).device == "cpu"  # nothing free: nothing to cut
    assert plan(**{**args, "engine_id": "sherpa-onnx"}, onnx_split=True).device == "cpu"  # sherpa owns its sessions

    # several graphs: the encoder does not fit, the decoder does; what is left goes to the encoder's front half
    g = plan(**{**args, "split": "graphs"}, graphs={"encoder": 400 * mb, "decoder": 100 * mb}, onnx_split=True)
    assert g.graphs == {"encoder": "cuda", "decoder": "cuda"} and g.graph_split == {"encoder": 100 * mb}
    assert "encoder is cut in two" in (g.notice or "")


def tiny_turn_model(path: Path) -> Path:
    """Shaped like Smart Turn's interface: 80x800 log-mel in, one probability out."""
    rng = np.random.default_rng(3)
    inits = [
        numpy_helper.from_array((rng.standard_normal((80, 32)) / 9).astype(np.float32), "w1"),
        numpy_helper.from_array((rng.standard_normal((32, 32)) / 6).astype(np.float32), "w2"),
        numpy_helper.from_array((rng.standard_normal((32, 32)) / 6).astype(np.float32), "w3"),
        numpy_helper.from_array((rng.standard_normal((32, 1)) / 6).astype(np.float32), "w4"),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["input_features"], ["pooled"], axes=[2], keepdims=0),
        helper.make_node("MatMul", ["pooled", "w1"], ["h1"]),
        helper.make_node("Tanh", ["h1"], ["a1"]),
        helper.make_node("MatMul", ["a1", "w2"], ["h2"]),
        helper.make_node("Tanh", ["h2"], ["a2"]),
        helper.make_node("MatMul", ["a2", "w3"], ["h3"]),
        helper.make_node("Tanh", ["h3"], ["a3"]),
        helper.make_node("MatMul", ["a3", "w4"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["probability"]),
    ]
    graph = helper.make_graph(
        nodes, "turn", [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, 80, 800])],
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 1])], initializer=inits,
    )  # fmt: skip
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path / "turn.onnx"))
    return path


def test_engine_loads_a_cut_graph_and_answers_the_same(tmp_path: Path) -> None:
    from bolo.engines.base import LoadOptions
    from bolo.engines.onnx_turn import SmartTurn

    d = tiny_turn_model(tmp_path / "turn")
    audio = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.1
    whole_model = SmartTurn(d, {"model": "turn.onnx"}, LoadOptions(threads=1))
    cut = SmartTurn(d, {"model": "turn.onnx"}, LoadOptions(threads=1, graph_split={"model": 80 * 32 * 4 + 32 * 32 * 4}))
    assert isinstance(cut._session, SplitSession) and cut._session.cut.crossing  # front: two layers' weights
    assert cut.probability(audio) == pytest.approx(whole_model.probability(audio), abs=1e-6)


async def test_the_manager_cuts_a_model_that_does_not_fit_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end with a fake NVML: the placement asks for a split, the engine builds it, ps reports it."""
    from bolo import cuda_libs
    from bolo.accel import Accelerators

    # this ONNX Runtime has no CUDA provider, so the front half really runs on the CPU and the load would (rightly)
    # move the placement to the CPU (bolo.cuda_libs, tests/test_cuda_libs.py): here only the split's plumbing counts
    monkeypatch.setattr(cuda_libs, "record", lambda *args, **kwargs: None)
    from bolo.catalog import ModelSpec
    from bolo.config import Settings
    from bolo.core import Bolo
    from test_placement import FakeNvml

    mb = 1024 * 1024
    settings = Settings(data_dir=tmp_path / "data", device="cuda", gpu_reserve_mb=0, auto_pull=False)
    core = Bolo(settings)
    # 30 KB free on a fake card for a model said to need about 40 KB: cut in two
    core.manager.accel = Accelerators(nvml=FakeNvml(total=1 * mb, used=1 * mb - 30_000), which=lambda _: None)
    spec = ModelSpec.from_dict({
        "id": "tiny-turn", "task": "turn", "engine": "onnxruntime", "kind": "smart_turn", "family": "X",
        "languages": [], "license": "MIT", "builtin": True, "size_bytes": 34_000, "options": {"model": "turn.onnx"},
    })  # fmt: skip
    core.catalog.add(spec)
    tiny_turn_model(core.store.model_dir(spec.id))
    await core.start()
    try:
        entry = await core.manager.load("tiny-turn")
        assert entry.placement.split == "onnx_split" and entry.placement.graph_split["model"] > 0
        assert isinstance(entry.model._session, SplitSession)  # type: ignore[attr-defined]
        assert "CPU/GPU" in entry.placement.processor()
        row = core.ps()["models"][0]
        assert row["split"] == "onnx_split" and row["placement"]["graph_split"]["model"] > 0
        settings.onnx_split = "off"
        await core.manager.unload("tiny-turn")
        again = await core.manager.load("tiny-turn")
        assert again.placement.device == "cpu" and not isinstance(again.model._session, SplitSession)  # type: ignore[attr-defined]
    finally:
        await core.stop()
