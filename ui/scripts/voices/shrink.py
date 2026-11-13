"""Shrink a VITS voice for the browser without changing what it computes in a way anyone can hear.

Weights are stored smaller and expanded back to 32-bit floats when the browser loads the model,
so ONNX Runtime Web still computes in fp32:

* convolution weights outside the text encoder: 8-bit integers with one scale per output
  channel (a ``DequantizeLinear`` node in front of each);
* everything else (the text encoder, embeddings, norms): 16-bit floats (a ``Cast`` node).

The text encoder is kept at 16 bits because 8-bit weights there changed the voice audibly
(measured as log-mel distance on the loudest frames: 0.64 with an 8-bit text encoder, 0.014 for
an 8-bit decoder, 0.021 for an 8-bit flow, 0.005 for fp16; see docs/ui-server-notes.md).

Usage (a library function, called by build.py):  shrink(src, dst, keep_prefixes)
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 1024  # small tensors are not worth a node


def shrink(src: str, dst: str, keep_prefixes: tuple[str, ...]) -> dict[str, int]:
    model = onnx.load(src)
    graph = model.graph
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for k, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node, k))
    added: list[onnx.NodeProto] = []
    counts = {"int8": 0, "fp16": 0}
    for tensor in list(graph.initializer):
        if tensor.data_type != TensorProto.FLOAT:
            continue
        weights = numpy_helper.to_array(tensor)
        if weights.size < MIN_ELEMENTS:
            continue
        name = tensor.name
        users = consumers.get(name, [])
        conv_weight = bool(users) and all(n.op_type in ("Conv", "ConvTranspose") and k == 1 for n, k in users)
        graph.initializer.remove(tensor)
        if conv_weight and not name.startswith(keep_prefixes):
            # ConvTranspose weights are [in, out/groups, k]: the output channel is axis 1.
            axis = 1 if users[0][0].op_type == "ConvTranspose" else 0
            reduce = tuple(i for i in range(weights.ndim) if i != axis)
            scale = np.abs(weights).max(axis=reduce) / 127.0
            scale[scale == 0] = 1e-12
            shape = [1] * weights.ndim
            shape[axis] = -1
            quantised = np.clip(np.round(weights / scale.reshape(shape)), -127, 127).astype(np.int8)
            graph.initializer.append(numpy_helper.from_array(quantised, name + "__q8"))
            graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), name + "__scale"))
            added.append(
                helper.make_node(
                    "DequantizeLinear", [name + "__q8", name + "__scale"], [name], axis=axis, name=name + "__dq"
                )
            )
            counts["int8"] += weights.size
        else:
            graph.initializer.append(numpy_helper.from_array(weights.astype(np.float16), name + "__f16"))
            added.append(helper.make_node("Cast", [name + "__f16"], [name], to=TensorProto.FLOAT, name=name + "__cast"))
            counts["fp16"] += weights.size
    for node in reversed(added):
        graph.node.insert(0, node)
    if max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")) < 13:
        raise SystemExit(f"{src}: opset 13 or later is needed for per-channel DequantizeLinear")
    onnx.save(model, dst)
    return counts
