#!/usr/bin/env python3
"""MLX op-correctness tests for the MLX-native ONNX Runtime execution provider.

Each ONNX decoder op we translate to MLX is run through the plugin ("MLXExecutionProvider") and compared,
tolerance-gated, against ORT's CPU EP reference. MLX is the SOLE compute path — there are no
hand-written Metal kernels — so this suite validates the ONNX->MLX translation in mlx_backend.cc.

Models are constructed with the ONNX IR (``onnx_ir``: ``ir.Value`` / ``ir.Node`` / ``ir.Graph`` /
``ir.Model``), not ``onnx.helper``.

Only ops the EP CLAIMS and the MLX translator supports (registered in the modular op registry,
src/ep/ops/*.cc) are exercised here: MatMulNBits, GroupQueryAttention (rope in-op), RMSNormalization,
SkipSimplifiedLayerNormalization, GatherBlockQuantized, Softmax, Add, Mul, Sub, Sigmoid, Cast. The
dtype-generic paths (elementwise/activation/softmax/normalization/cast) are additionally exercised in
fp16 (vs ORT CPU) and bf16 (bf16-interior subgraph vs a numpy fp32 reference — ORT CPU has no bf16
kernels). Ops the EP no longer claims (Div, Gelu, RotaryEmbedding, Reshape, Transpose, Concat) are
intentionally NOT tested: they fall back to ORT CPU and would only compare CPU-vs-CPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

DataType = ir.DataType


def tensor(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    """A named, typed, shaped IR value — used for graph inputs and outputs."""
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _attr(name: str, value: object) -> ir.Attr:
    # bool is a subclass of int, but no boolean attributes are used here.
    if isinstance(value, float):
        return ir.AttrFloat32(name, value)
    if isinstance(value, int):
        return ir.AttrInt64(name, int(value))
    raise TypeError(f"unsupported attribute type for {name!r}: {type(value)!r}")


def make_model(
    op_type: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 24,
) -> bytes:
    node = ir.Node(
        domain,
        op_type,
        inputs,
        attributes=[_attr(k, v) for k, v in (attributes or {}).items()],
        outputs=outputs,
    )
    opset_imports = {"": opset}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph(
        inputs, outputs, nodes=[node], name=f"mlx_{op_type}", opset_imports=opset_imports
    )
    model = ir.Model(graph, ir_version=11)
    return ir.to_proto(model).SerializeToString()


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
        model, options, providers=["MLXExecutionProvider", "CPUExecutionProvider"]
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


def run_mlx(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through the MLX EP (CPU fallback available) and return its outputs."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        model, options, providers=["MLXExecutionProvider", "CPUExecutionProvider"]
    )
    return session.run(None, feeds)


def compare_ref(
    name: str,
    model: bytes,
    feeds: dict[str, np.ndarray],
    expected: list[np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare the MLX EP output against a precomputed numpy reference.

    Used for bfloat16 coverage: ORT's CPU EP ships no bf16 kernels for these ops (and the ORT Python
    binding cannot feed bf16 arrays), so we cannot build a CPU reference session. Instead the bf16
    compute is kept INSIDE an MLX-claimed subgraph (fp32 in -> Cast bf16 -> op(bf16) -> Cast fp32
    out) and the fp32 boundary output is compared, tolerance-gated (~1e-2), against a numpy fp32
    reference. This validates the MLX bf16 path end-to-end.
    """
    actual = run_mlx(model, feeds)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol, err_msg=f"{name} output {index}"
        )
    print(f"PASS {name}")


def bf16_interior_model(
    op_type: str,
    float_inputs: list[tuple[str, list[int]]],
    out_shape: list[int],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
) -> bytes:
    """Build fp32-in -> Cast(bf16) -> op(bf16) -> Cast(fp32)-out. Every node (both Casts + the op)
    is MLX-claimable, so the whole subgraph runs in bf16 on MLX while the boundaries stay fp32
    (feedable/readable through the ORT Python binding)."""
    fp_inputs = [tensor(name, DataType.FLOAT, shape) for name, shape in float_inputs]
    bf_inputs = [tensor(f"{name}_bf", DataType.BFLOAT16, shape) for name, shape in float_inputs]
    nodes = [
        ir.Node(
            "", "Cast", [fp], attributes=[ir.AttrInt64("to", int(DataType.BFLOAT16))], outputs=[bf]
        )
        for fp, bf in zip(fp_inputs, bf_inputs, strict=True)
    ]
    bf_out = tensor("y_bf", DataType.BFLOAT16, out_shape)
    nodes.append(
        ir.Node(
            domain,
            op_type,
            bf_inputs,
            attributes=[_attr(k, v) for k, v in (attributes or {}).items()],
            outputs=[bf_out],
        )
    )
    fp_out = tensor("out", DataType.FLOAT, out_shape)
    nodes.append(
        ir.Node(
            "", "Cast", [bf_out], attributes=[ir.AttrInt64("to", int(DataType.FLOAT))], outputs=[fp_out]
        )
    )
    opset_imports = {"": 24}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph(
        fp_inputs, [fp_out], nodes=nodes, name=f"mlx_bf16_{op_type}", opset_imports=opset_imports
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _np_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _np_rms_norm(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * scale


def rotary_caches(max_seq: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    angles = pos * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


# --- GroupQueryAttention (the core decoder op; rope is applied inside the MLX SDPA path) ---------
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
    inputs = [
        tensor("query", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("key", DataType.FLOAT, [batch, seq, kv_heads * head]),
        tensor("value", DataType.FLOAT, [batch, seq, kv_heads * head]),
        tensor("past_key", DataType.FLOAT, [batch, kv_heads, past, head]),
        tensor("past_value", DataType.FLOAT, [batch, kv_heads, past, head]),
        tensor("seqlens_k", DataType.INT32, [batch]),
        tensor("total_sequence_length", DataType.INT32, [1]),
        tensor("cos_cache", DataType.FLOAT, [max_seq, head // 2]),
        tensor("sin_cache", DataType.FLOAT, [max_seq, head // 2]),
    ]
    outputs = [
        tensor("attn_output", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", DataType.FLOAT, [batch, kv_heads, present, head]),
        tensor("present_value", DataType.FLOAT, [batch, kv_heads, present, head]),
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


def gqa_differential_checks() -> None:
    # Real-model head geometry (Qwen2.5-0.5B): num_heads=14, kv=2, head=64.
    model_geom = dict(batch=1, num_heads=14, kv_heads=2, head=64)
    gqa_case("GQA-decode-h64", seq=1, past=40, do_rotary=1, **model_geom)
    gqa_case("GQA-prefill-h64", seq=26, past=0, do_rotary=1, **model_geom)
    gqa_case("GQA-chunked-h64", seq=3, past=8, do_rotary=1, **model_geom)
    common = dict(batch=1, num_heads=4, kv_heads=2, head=16)
    gqa_case("GQA-decode", seq=1, past=5, do_rotary=1, **common)
    gqa_case("GQA-prefill", seq=6, past=0, do_rotary=1, **common)
    gqa_case("GQA-decode-norope", seq=1, past=5, do_rotary=0, **common)


# --- MatMulNBits (int4 block-quantized weight matmul) --------------------------------------------
def matmulnbits_case(name: str, *, M: int, K: int, N: int, block: int = 32) -> None:
    rng = np.random.default_rng(hash((name, M, K, N)) & 0xFFFFFFFF)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    n_blocks = (K + block - 1) // block
    # Packed int4: [N, n_blocks, block/2] uint8, two nibbles per byte.
    b = rng.integers(0, 256, size=(N, n_blocks, block // 2), dtype=np.uint8)
    scales = rng.standard_normal((N * n_blocks,)).astype(np.float32) * 0.05
    inputs = [
        tensor("a", DataType.FLOAT, [1, M, K]),
        tensor("b", DataType.UINT8, [N, n_blocks, block // 2]),
        tensor("scales", DataType.FLOAT, [N * n_blocks]),
    ]
    outputs = [tensor("out", DataType.FLOAT, [1, M, N])]
    compare(
        name,
        make_model(
            "MatMulNBits",
            inputs,
            outputs,
            domain="com.microsoft",
            attributes={"K": K, "N": N, "bits": 4, "block_size": block},
        ),
        {"a": a, "b": b, "scales": scales},
        rtol=2e-3,
        atol=2e-3,
    )


# --- RMSNormalization / SkipSimplifiedLayerNormalization ----------------------------------------
def rmsnorm_case(name: str, *, rows: int, hidden: int) -> None:
    rng = np.random.default_rng(hash((name, rows, hidden)) & 0xFFFFFFFF)
    x = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    scale = rng.standard_normal((hidden,)).astype(np.float32)
    inputs = [
        tensor("x", DataType.FLOAT, [1, rows, hidden]),
        tensor("scale", DataType.FLOAT, [hidden]),
    ]
    outputs = [tensor("out", DataType.FLOAT, [1, rows, hidden])]
    compare(
        name,
        make_model(
            "RMSNormalization",
            inputs,
            outputs,
            attributes={"axis": -1, "epsilon": 1e-6},
        ),
        {"x": x, "scale": scale},
        rtol=2e-3,
        atol=2e-3,
    )


def skip_simplified_layernorm_case(name: str, *, rows: int, hidden: int) -> None:
    rng = np.random.default_rng(hash((name, rows, hidden)) & 0xFFFFFFFF)
    x = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    skip = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    gamma = rng.standard_normal((hidden,)).astype(np.float32)
    # SkipSimplifiedLayerNormalization emits up to 4 outputs; only output 0 (the normed result)
    # is compared here for robustness — declaring a single output makes the trailing ones optional.
    inputs = [
        tensor("x", DataType.FLOAT, [1, rows, hidden]),
        tensor("skip", DataType.FLOAT, [1, rows, hidden]),
        tensor("gamma", DataType.FLOAT, [hidden]),
    ]
    outputs = [tensor("out", DataType.FLOAT, [1, rows, hidden])]
    compare(
        name,
        make_model(
            "SkipSimplifiedLayerNormalization",
            inputs,
            outputs,
            domain="com.microsoft",
            attributes={"epsilon": 1e-6},
        ),
        {"x": x, "skip": skip, "gamma": gamma},
        rtol=2e-3,
        atol=2e-3,
    )


def dtype_generic_checks() -> None:
    """Run the dtype-generic ops (elementwise/activation/softmax/normalization) in fp16 AND bf16.

    fp16 is compared against ORT's CPU EP (which has fp16 kernels for these ops). bf16 keeps the
    compute inside an MLX-claimed subgraph (fp32 boundaries) and is compared against a numpy fp32
    reference at ~1e-2 tolerance (bf16 carries ~3 significant digits).
    """
    rng = np.random.default_rng(7)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    x = rng.standard_normal((2, 5)).astype(np.float32)
    rms_x = rng.standard_normal((1, 4, 8)).astype(np.float32)
    rms_g = rng.standard_normal((8,)).astype(np.float32)

    # ---- fp16: MLX vs ORT CPU (fp16 kernels exist on CPU) ----
    f16 = np.float16
    compare(
        "Mul fp16",
        make_model(
            "Mul",
            [tensor("a", DataType.FLOAT16, [2, 3]), tensor("b", DataType.FLOAT16, [3])],
            [tensor("out", DataType.FLOAT16, [2, 3])],
        ),
        {"a": a.astype(f16), "b": b.astype(f16)},
        rtol=2e-3,
        atol=2e-3,
    )
    compare(
        "Sub fp16",
        make_model(
            "Sub",
            [tensor("a", DataType.FLOAT16, [2, 3]), tensor("b", DataType.FLOAT16, [3])],
            [tensor("out", DataType.FLOAT16, [2, 3])],
        ),
        {"a": a.astype(f16), "b": b.astype(f16)},
        rtol=2e-3,
        atol=2e-3,
    )
    compare(
        "Sigmoid fp16",
        make_model(
            "Sigmoid",
            [tensor("x", DataType.FLOAT16, [2, 5])],
            [tensor("out", DataType.FLOAT16, [2, 5])],
        ),
        {"x": x.astype(f16)},
        rtol=2e-3,
        atol=2e-3,
    )
    compare(
        "Softmax fp16",
        make_model(
            "Softmax",
            [tensor("x", DataType.FLOAT16, [2, 5])],
            [tensor("out", DataType.FLOAT16, [2, 5])],
            attributes={"axis": -1},
        ),
        {"x": x.astype(f16)},
        rtol=2e-3,
        atol=2e-3,
    )
    compare(
        "RMSNormalization fp16",
        make_model(
            "RMSNormalization",
            [tensor("x", DataType.FLOAT16, [1, 4, 8]), tensor("scale", DataType.FLOAT16, [8])],
            [tensor("out", DataType.FLOAT16, [1, 4, 8])],
            attributes={"axis": -1, "epsilon": 1e-6},
        ),
        {"x": rms_x.astype(f16), "scale": rms_g.astype(f16)},
        rtol=3e-3,
        atol=3e-3,
    )

    # ---- bf16: MLX (bf16 interior) vs numpy fp32 reference ----
    bf_rtol, bf_atol = 2e-2, 2e-2
    compare_ref(
        "Add bf16",
        bf16_interior_model("Add", [("a", [2, 3]), ("b", [2, 3])], [2, 3]),
        {"a": a, "b": a * 0.5},
        [a + a * 0.5],
        rtol=bf_rtol,
        atol=bf_atol,
    )
    compare_ref(
        "Mul bf16",
        bf16_interior_model("Mul", [("a", [2, 3]), ("b", [3])], [2, 3]),
        {"a": a, "b": b},
        [a * b],
        rtol=bf_rtol,
        atol=bf_atol,
    )
    compare_ref(
        "Sub bf16",
        bf16_interior_model("Sub", [("a", [2, 3]), ("b", [3])], [2, 3]),
        {"a": a, "b": b},
        [a - b],
        rtol=bf_rtol,
        atol=bf_atol,
    )
    compare_ref(
        "Sigmoid bf16",
        bf16_interior_model("Sigmoid", [("x", [2, 5])], [2, 5]),
        {"x": x},
        [1.0 / (1.0 + np.exp(-x))],
        rtol=bf_rtol,
        atol=bf_atol,
    )
    compare_ref(
        "Softmax bf16",
        bf16_interior_model("Softmax", [("x", [2, 5])], [2, 5], attributes={"axis": -1}),
        {"x": x},
        [_np_softmax(x)],
        rtol=bf_rtol,
        atol=bf_atol,
    )
    compare_ref(
        "RMSNormalization bf16",
        bf16_interior_model(
            "RMSNormalization",
            [("x", [1, 4, 8]), ("scale", [8])],
            [1, 4, 8],
            attributes={"axis": -1, "epsilon": 1e-6},
        ),
        {"x": rms_x, "scale": rms_g},
        [_np_rms_norm(rms_x, rms_g)],
        rtol=bf_rtol,
        atol=bf_atol,
    )
    compare_ref(
        "Cast fp32->bf16->fp32",
        bf16_interior_model("Add", [("a", [2, 3]), ("b", [2, 3])], [2, 3]),
        {"a": a, "b": np.zeros_like(a)},
        [a],
        rtol=bf_rtol,
        atol=bf_atol,
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mlx_op_test.py <libonnxruntime_mlx_ep.dylib>", file=sys.stderr)
        return 2
    ort.register_execution_provider_library("MLXExecutionProvider", os.path.abspath(sys.argv[1]))

    # --- Elementwise (fp32/fp16/int64) ----------------------------------------------------------
    a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    b = np.array([2.0, -4.0, 0.5], dtype=np.float32)
    # IR values are graph-bound, so build fresh inputs/outputs for each model.
    for op in ("Mul", "Sub"):  # fp32 Add is covered by the residual-add pattern below
        compare(
            op,
            make_model(
                op,
                [tensor("a", DataType.FLOAT, [2, 3]), tensor("b", DataType.FLOAT, [3])],
                [tensor("out", DataType.FLOAT, [2, 3])],
            ),
            {"a": a, "b": b},
        )

    compare(
        "Add fp32",
        make_model(
            "Add",
            [tensor("a", DataType.FLOAT, [2, 3]), tensor("b", DataType.FLOAT, [2, 3])],
            [tensor("out", DataType.FLOAT, [2, 3])],
        ),
        {"a": a, "b": a * 0.5},
    )

    compare(
        "Sigmoid",
        make_model(
            "Sigmoid",
            [tensor("x", DataType.FLOAT, [2, 3])],
            [tensor("out", DataType.FLOAT, [2, 3])],
        ),
        {"x": a},
    )

    compare(
        "Add fp16",
        make_model(
            "Add",
            [tensor("a", DataType.FLOAT16, [2, 3]), tensor("b", DataType.FLOAT16, [3])],
            [tensor("out", DataType.FLOAT16, [2, 3])],
        ),
        {"a": a.astype(np.float16), "b": b.astype(np.float16)},
        rtol=2e-3,
        atol=2e-3,
    )

    compare(
        "Cast fp32->fp16",
        make_model(
            "Cast",
            [tensor("x", DataType.FLOAT, [2, 3])],
            [tensor("out", DataType.FLOAT16, [2, 3])],
            attributes={"to": int(DataType.FLOAT16)},
        ),
        {"x": a},
        rtol=0,
        atol=0,
    )
    compare(
        "Cast fp16->fp32",
        make_model(
            "Cast",
            [tensor("x", DataType.FLOAT16, [2, 3])],
            [tensor("out", DataType.FLOAT, [2, 3])],
            attributes={"to": int(DataType.FLOAT)},
        ),
        {"x": a.astype(np.float16)},
        rtol=0,
        atol=0,
    )

    compare(
        "Sub int64 scalar",
        make_model(
            "Sub",
            [tensor("a", DataType.INT64, [3]), tensor("b", DataType.INT64, [])],
            [tensor("out", DataType.INT64, [3])],
        ),
        {"a": np.array([5, -2, 9], dtype=np.int64), "b": np.array(3, dtype=np.int64)},
        rtol=0,
        atol=0,
    )

    # --- Softmax (last-axis) --------------------------------------------------------------------
    compare(
        "Softmax",
        make_model(
            "Softmax",
            [tensor("x", DataType.FLOAT, [2, 5])],
            [tensor("out", DataType.FLOAT, [2, 5])],
            attributes={"axis": -1},
        ),
        {"x": np.random.default_rng(1).standard_normal((2, 5)).astype(np.float32)},
        rtol=2e-3,
        atol=2e-3,
    )

    # --- GatherBlockQuantized (int4 embedding table) --------------------------------------------
    qdata = np.empty((2, 16), dtype=np.uint8)
    for row in range(2):
        values = (np.arange(32, dtype=np.uint8) + row) & 0x0F
        qdata[row] = values[0::2] | (values[1::2] << 4)
    compare(
        "GatherBlockQuantized",
        make_model(
            "GatherBlockQuantized",
            [
                tensor("data", DataType.UINT8, [2, 16]),
                tensor("indices", DataType.INT64, [2]),
                tensor("scales", DataType.FLOAT, [2, 2]),
            ],
            [tensor("out", DataType.FLOAT, [2, 32])],
            domain="com.microsoft",
            attributes={"bits": 4, "block_size": 16, "gather_axis": 0, "quantize_axis": 1},
        ),
        {
            "data": qdata,
            "indices": np.array([1, -2], dtype=np.int64),
            "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
        },
    )
    # NOTE: the asymmetric 4-input (zero_points) GatherBlockQuantized form is intentionally NOT
    # tested here — the EP does not claim it (MLX translates only the symmetric zp=8 form used by the
    # cpu-recipe embedding), so it falls back to ORT CPU. Adding a MLX zero_points path is a follow-up.

    # --- Normalizations -------------------------------------------------------------------------
    rmsnorm_case("RMSNormalization", rows=4, hidden=64)
    skip_simplified_layernorm_case("SkipSimplifiedLayerNormalization", rows=4, hidden=64)

    # --- Quantized matmul -----------------------------------------------------------------------
    matmulnbits_case("MatMulNBits-decode", M=1, K=64, N=32)
    matmulnbits_case("MatMulNBits-prefill", M=8, K=64, N=32)

    # --- Attention ------------------------------------------------------------------------------
    gqa_differential_checks()

    # --- Per-dtype coverage: fp16 (vs ORT CPU) and bf16 (vs numpy ref) --------------------------
    dtype_generic_checks()

    print("All MLX op-correctness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
