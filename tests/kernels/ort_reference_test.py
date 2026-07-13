#!/usr/bin/env python3
"""Differential tests: MetalEP kernels against ORT's CPU EP."""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def make_model(
    op_type: str,
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 24,
) -> bytes:
    node = helper.make_node(
        op_type,
        [value.name for value in inputs],
        [value.name for value in outputs],
        domain=domain,
        **(attributes or {}),
    )
    imports = [helper.make_opsetid("", opset)]
    if domain:
        imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(
        helper.make_graph([node], f"mps_{op_type}", inputs, outputs),
        opset_imports=imports,
    )
    model.ir_version = 11
    return model.SerializeToString()


def compare(
    name: str,
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    cpu = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    metal = ort.InferenceSession(
        model, options, providers=["MetalEP", "CPUExecutionProvider"]
    )
    expected = cpu.run(None, feeds)
    actual = metal.run(None, feeds)
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: output count differs")
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol, err_msg=f"{name} output {index}"
        )
    print(f"PASS {name}")


def tensor(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def rotary_caches(max_seq: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    angles = pos * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def gqa_case(
    name: str,
    *,
    batch: int,
    seq: int,
    past: int,
    num_heads: int,
    kv_heads: int,
    head: int,
    do_rotary: int,
    interleaved: int = 0,
    local_window: int = -1,
) -> None:
    present = past + seq
    max_seq = present + 4
    scale = 1.0 / np.sqrt(head)
    rng = np.random.default_rng(hash((name, seq, past)) & 0xFFFFFFFF)
    q = rng.standard_normal((batch, seq, num_heads * head)).astype(np.float32)
    k = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    v = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    past_k = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    past_v = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    seqlens_k = np.full((batch,), present - 1, dtype=np.int32)
    total = np.array([present], dtype=np.int32)
    cos, sin = rotary_caches(max_seq, head)

    attrs = {
        "num_heads": num_heads,
        "kv_num_heads": kv_heads,
        "scale": float(scale),
        "do_rotary": do_rotary,
        "rotary_interleaved": interleaved,
    }
    if local_window >= 0:
        attrs["local_window_size"] = local_window

    inputs = [
        tensor("query", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("key", TensorProto.FLOAT, [batch, seq, kv_heads * head]),
        tensor("value", TensorProto.FLOAT, [batch, seq, kv_heads * head]),
        tensor("past_key", TensorProto.FLOAT, [batch, kv_heads, past, head]),
        tensor("past_value", TensorProto.FLOAT, [batch, kv_heads, past, head]),
        tensor("seqlens_k", TensorProto.INT32, [batch]),
        tensor("total_sequence_length", TensorProto.INT32, [1]),
        tensor("cos_cache", TensorProto.FLOAT, [max_seq, head // 2]),
        tensor("sin_cache", TensorProto.FLOAT, [max_seq, head // 2]),
    ]
    outputs = [
        tensor("attn_output", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", TensorProto.FLOAT, [batch, kv_heads, present, head]),
        tensor("present_value", TensorProto.FLOAT, [batch, kv_heads, present, head]),
    ]
    feeds = {
        "query": q,
        "key": k,
        "value": v,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total,
        "cos_cache": cos,
        "sin_cache": sin,
    }
    compare(
        name,
        make_model(
            "GroupQueryAttention",
            inputs,
            outputs,
            domain="com.microsoft",
            attributes=attrs,
        ),
        feeds,
        rtol=2e-3,
        atol=2e-3,
    )


def gqa_fused_case(
    name: str,
    *,
    batch: int,
    seq: int,
    past: int,
    num_heads: int,
    kv_heads: int,
    head: int,
    do_rotary: int,
) -> None:
    """Force GQA to consume a device-resident intermediate: Add(query, bias) -> GQA.

    In a multi-node fused subgraph the Metal EP keeps the Add output on-device and feeds it
    straight into GQA, exercising the batched-encoder path with device-resident operands."""
    present = past + seq
    max_seq = present + 4
    scale = 1.0 / np.sqrt(head)
    rng = np.random.default_rng(hash((name, seq, past)) & 0xFFFFFFFF)
    q = rng.standard_normal((batch, seq, num_heads * head)).astype(np.float32)
    bias = np.zeros((batch, seq, num_heads * head), dtype=np.float32)
    k = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    v = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    past_k = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    past_v = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    seqlens_k = np.full((batch,), present - 1, dtype=np.int32)
    total = np.array([present], dtype=np.int32)
    cos, sin = rotary_caches(max_seq, head)

    inputs = [
        tensor("query", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("bias", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("key", TensorProto.FLOAT, [batch, seq, kv_heads * head]),
        tensor("value", TensorProto.FLOAT, [batch, seq, kv_heads * head]),
        tensor("past_key", TensorProto.FLOAT, [batch, kv_heads, past, head]),
        tensor("past_value", TensorProto.FLOAT, [batch, kv_heads, past, head]),
        tensor("seqlens_k", TensorProto.INT32, [batch]),
        tensor("total_sequence_length", TensorProto.INT32, [1]),
        tensor("cos_cache", TensorProto.FLOAT, [max_seq, head // 2]),
        tensor("sin_cache", TensorProto.FLOAT, [max_seq, head // 2]),
    ]
    outputs = [
        tensor("attn_output", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", TensorProto.FLOAT, [batch, kv_heads, present, head]),
        tensor("present_value", TensorProto.FLOAT, [batch, kv_heads, present, head]),
    ]
    add_node = helper.make_node("Add", ["query", "bias"], ["query_fused"])
    gqa_node = helper.make_node(
        "GroupQueryAttention",
        ["query_fused", "key", "value", "past_key", "past_value",
         "seqlens_k", "total_sequence_length", "cos_cache", "sin_cache"],
        ["attn_raw", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=kv_heads,
        scale=float(scale),
        do_rotary=do_rotary,
        rotary_interleaved=0,
    )
    # Consume attn_raw on-device (mirrors the model's o_proj consuming GQA output).
    post_node = helper.make_node("Add", ["attn_raw", "bias"], ["attn_output"])
    graph = helper.make_graph([add_node, gqa_node, post_node], f"mps_{name}", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 11
    feeds = {
        "query": q, "bias": bias, "key": k, "value": v,
        "past_key": past_k, "past_value": past_v,
        "seqlens_k": seqlens_k, "total_sequence_length": total,
        "cos_cache": cos, "sin_cache": sin,
    }
    compare(name, model.SerializeToString(), feeds, rtol=2e-3, atol=2e-3)


def gqa_differential_checks() -> None:
    common = dict(batch=1, num_heads=4, kv_heads=2, head=16)
    gqa_case("GQA-decode", seq=1, past=5, do_rotary=1, **common)
    gqa_case("GQA-decode-norope", seq=1, past=5, do_rotary=0, **common)
    gqa_case("GQA-decode-batch", seq=1, past=3, do_rotary=1,
             num_heads=4, kv_heads=2, head=16, batch=2)
    gqa_case("GQA-prefill", seq=6, past=0, do_rotary=1, **common)
    gqa_case("GQA-prefill-interleaved", seq=6, past=0, do_rotary=1, interleaved=1, **common)
    gqa_case("GQA-chunked", seq=3, past=4, do_rotary=1, **common)
    gqa_case("GQA-decode-swa", seq=1, past=6, do_rotary=1, local_window=3, **common)
    gqa_case("GQA-prefill-swa", seq=6, past=0, do_rotary=1, local_window=3, **common)
    # Multi-node fused: GQA consumes a device-resident intermediate (Add output).
    gqa_fused_case("GQA-fused-decode", seq=1, past=5, do_rotary=1, **common)
    gqa_fused_case("GQA-fused-prefill", seq=6, past=0, do_rotary=1, **common)
    # Real-model head geometry (Qwen2.5-0.5B): num_heads=14, kv=2, head=64.
    model_geom = dict(batch=1, num_heads=14, kv_heads=2, head=64)
    gqa_fused_case("GQA-fused-decode-h64", seq=1, past=5, do_rotary=1, **model_geom)
    gqa_fused_case("GQA-fused-prefill-h64", seq=26, past=0, do_rotary=1, **model_geom)
    gqa_case("GQA-decode-h64", seq=1, past=40, do_rotary=1, **model_geom)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ort_reference_test.py <libonnxruntime_mps_ep.dylib>", file=sys.stderr)
        return 2
    ort.register_execution_provider_library("MetalEP", os.path.abspath(sys.argv[1]))

    a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    b = np.array([2.0, -4.0, 0.5], dtype=np.float32)
    binary_inputs = [
        tensor("a", TensorProto.FLOAT, [2, 3]),
        tensor("b", TensorProto.FLOAT, [3]),
    ]
    binary_output = [tensor("out", TensorProto.FLOAT, [2, 3])]
    for op in ("Mul", "Sub", "Div"):
        compare(op, make_model(op, binary_inputs, binary_output), {"a": a, "b": b})

    unary_input = [tensor("x", TensorProto.FLOAT, [2, 3])]
    unary_output = [tensor("out", TensorProto.FLOAT, [2, 3])]
    compare("Sigmoid", make_model("Sigmoid", unary_input, unary_output), {"x": a})
    compare(
        "Gelu exact",
        make_model("Gelu", unary_input, unary_output, attributes={"approximate": "none"}),
        {"x": a},
        rtol=3e-5,
        atol=3e-6,
    )
    compare(
        "Gelu tanh",
        make_model("Gelu", unary_input, unary_output, attributes={"approximate": "tanh"}),
        {"x": a},
        rtol=3e-5,
        atol=3e-6,
    )

    add16_inputs = [
        tensor("a", TensorProto.FLOAT16, [2, 3]),
        tensor("b", TensorProto.FLOAT16, [3]),
    ]
    add16_output = [tensor("out", TensorProto.FLOAT16, [2, 3])]
    compare(
        "Add fp16",
        make_model("Add", add16_inputs, add16_output),
        {"a": a.astype(np.float16), "b": b.astype(np.float16)},
        rtol=2e-3,
        atol=2e-3,
    )

    compare(
        "Cast fp32->fp16",
        make_model(
            "Cast",
            [tensor("x", TensorProto.FLOAT, [2, 3])],
            [tensor("out", TensorProto.FLOAT16, [2, 3])],
            attributes={"to": TensorProto.FLOAT16},
        ),
        {"x": a},
        rtol=0,
        atol=0,
    )
    compare(
        "Cast fp16->fp32",
        make_model(
            "Cast",
            [tensor("x", TensorProto.FLOAT16, [2, 3])],
            [tensor("out", TensorProto.FLOAT, [2, 3])],
            attributes={"to": TensorProto.FLOAT},
        ),
        {"x": a.astype(np.float16)},
        rtol=0,
        atol=0,
    )

    compare(
        "Sub int64 scalar",
        make_model(
            "Sub",
            [
                tensor("a", TensorProto.INT64, [3]),
                tensor("b", TensorProto.INT64, []),
            ],
            [tensor("out", TensorProto.INT64, [3])],
        ),
        {
            "a": np.array([5, -2, 9], dtype=np.int64),
            "b": np.array(3, dtype=np.int64),
        },
        rtol=0,
        atol=0,
    )

    qdata = np.empty((2, 16), dtype=np.uint8)
    for row in range(2):
        values = (np.arange(32, dtype=np.uint8) + row) & 0x0F
        qdata[row] = values[0::2] | (values[1::2] << 4)
    compare(
        "GatherBlockQuantized",
        make_model(
            "GatherBlockQuantized",
            [
                tensor("data", TensorProto.UINT8, [2, 16]),
                tensor("indices", TensorProto.INT64, [2]),
                tensor("scales", TensorProto.FLOAT, [2, 2]),
            ],
            [tensor("out", TensorProto.FLOAT, [2, 32])],
            domain="com.microsoft",
            attributes={
                "bits": 4,
                "block_size": 16,
                "gather_axis": 0,
                "quantize_axis": 1,
            },
        ),
        {
            "data": qdata,
            "indices": np.array([1, -2], dtype=np.int64),
            "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
        },
    )
    compare(
        "GatherBlockQuantized fp16 zero-point",
        make_model(
            "GatherBlockQuantized",
            [
                tensor("data", TensorProto.UINT8, [2, 16]),
                tensor("indices", TensorProto.INT64, [2]),
                tensor("scales", TensorProto.FLOAT16, [2, 2]),
                tensor("zero_points", TensorProto.UINT8, [2, 1]),
            ],
            [tensor("out", TensorProto.FLOAT16, [2, 32])],
            domain="com.microsoft",
            attributes={
                "bits": 4,
                "block_size": 16,
                "gather_axis": 0,
                "quantize_axis": 1,
            },
        ),
        {
            "data": qdata,
            "indices": np.array([1, -2], dtype=np.int64),
            "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float16),
            "zero_points": np.array([[0x87], [0x99]], dtype=np.uint8),
        },
        rtol=2e-3,
        atol=2e-3,
    )

    x = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float16)
    compare(
        "RotaryEmbedding half-split fp16",
        make_model(
            "RotaryEmbedding",
            [
                tensor("x", TensorProto.FLOAT16, [1, 1, 1, 4]),
                tensor("cos", TensorProto.FLOAT16, [2, 2]),
                tensor("sin", TensorProto.FLOAT16, [2, 2]),
                tensor("position", TensorProto.INT64, [1, 1]),
            ],
            [tensor("out", TensorProto.FLOAT16, [1, 1, 1, 4])],
            attributes={"interleaved": 0},
            opset=23,
        ),
        {
            "x": x,
            "cos": np.array([[1.0, 1.0], [0.5, 0.25]], dtype=np.float16),
            "sin": np.array([[0.0, 0.0], [0.5, 0.75]], dtype=np.float16),
            "position": np.array([[1]], dtype=np.int64),
        },
        rtol=2e-3,
        atol=2e-3,
    )
    compare(
        "RotaryEmbedding interleaved fp16",
        make_model(
            "RotaryEmbedding",
            [
                tensor("x", TensorProto.FLOAT16, [1, 1, 1, 4]),
                tensor("cos", TensorProto.FLOAT16, [2, 2]),
                tensor("sin", TensorProto.FLOAT16, [2, 2]),
                tensor("position", TensorProto.INT64, [1, 1]),
            ],
            [tensor("out", TensorProto.FLOAT16, [1, 1, 1, 4])],
            attributes={"interleaved": 1},
            opset=23,
        ),
        {
            "x": x,
            "cos": np.array([[1.0, 1.0], [0.5, 0.25]], dtype=np.float16),
            "sin": np.array([[0.0, 0.0], [0.5, 0.75]], dtype=np.float16),
            "position": np.array([[1]], dtype=np.int64),
        },
        rtol=2e-3,
        atol=2e-3,
    )

    compare(
        "Reshape",
        make_model(
            "Reshape",
            [
                tensor("x", TensorProto.FLOAT, [2, 3]),
                tensor("shape", TensorProto.INT64, [2]),
            ],
            [tensor("out", TensorProto.FLOAT, [3, 2])],
        ),
        {"x": a, "shape": np.array([3, 2], dtype=np.int64)},
        rtol=0,
        atol=0,
    )
    compare(
        "Transpose",
        make_model(
            "Transpose",
            [tensor("x", TensorProto.FLOAT, [2, 3])],
            [tensor("out", TensorProto.FLOAT, [3, 2])],
            attributes={"perm": [1, 0]},
        ),
        {"x": a},
        rtol=0,
        atol=0,
    )
    compare(
        "Concat",
        make_model(
            "Concat",
            [
                tensor("a", TensorProto.FLOAT, [2, 2]),
                tensor("b", TensorProto.FLOAT, [2, 1]),
            ],
            [tensor("out", TensorProto.FLOAT, [2, 3])],
            attributes={"axis": 1},
        ),
        {
            "a": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "b": np.array([[5], [6]], dtype=np.float32),
        },
        rtol=0,
        atol=0,
    )

    gqa_differential_checks()

    print("All ORT CPU-EP differential checks passed")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
