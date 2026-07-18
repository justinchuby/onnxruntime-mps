"""ONNX-IR model builders and assertion helpers for the MLX EP op tests.

Models are constructed with the ONNX IR (``onnx_ir``: ``ir.Value`` / ``ir.Node`` /
``ir.Graph`` / ``ir.Model``), never ``onnx.helper``. Comparisons run the model through the MLX EP
(with ORT CPU fallback available) and check the result against either ORT's CPU EP or a numpy
reference, tolerance-gated.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

DataType = ir.DataType
EP_PROVIDERS = ["MLXExecutionProvider", "CPUExecutionProvider"]


# --- IR construction ----------------------------------------------------------------------------
def tensor(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    """A named, typed, shaped IR value — used for graph inputs and outputs."""
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _attr(name: str, value: object) -> ir.Attr:
    # bool is a subclass of int, but no boolean attributes are used here.
    if isinstance(value, float):
        return ir.AttrFloat32(name, value)
    if isinstance(value, int):
        return ir.AttrInt64(name, int(value))
    if isinstance(value, str):
        return ir.AttrString(name, value)
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
    """Build a single-node model. IR values are graph-bound, so pass fresh values per model."""
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
    # Empty-name values are ABSENT optional node inputs (e.g. omitted cos/sin), represented as ""
    # references on the node — they must not appear as graph inputs (two of them would collide).
    graph_inputs = [v for v in inputs if v.name]
    graph = ir.Graph(
        graph_inputs, outputs, nodes=[node], name=f"mlx_{op_type}", opset_imports=opset_imports
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def bf16_interior_model(
    op_type: str,
    float_inputs: list[tuple[str, list[int]]],
    out_shape: list[int],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
) -> bytes:
    """Build fp32-in -> Cast(bf16) -> op(bf16) -> Cast(fp32)-out.

    Every node (both Casts + the op) is MLX-claimable, so the whole subgraph runs in bf16 on MLX
    while the boundaries stay fp32 (feedable/readable through the ORT Python binding, which cannot
    pass bf16 arrays and has no bf16 CPU kernels for these ops).
    """
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
            "",
            "Cast",
            [bf_out],
            attributes=[ir.AttrInt64("to", int(DataType.FLOAT))],
            outputs=[fp_out],
        )
    )
    opset_imports = {"": 24}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph(
        fp_inputs, [fp_out], nodes=nodes, name=f"mlx_bf16_{op_type}", opset_imports=opset_imports
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- Session / assertion helpers ----------------------------------------------------------------
def _session(model: bytes, providers: list[str]) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(model, options, providers=providers)


def run_mlx(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through the MLX EP (CPU fallback available) and return its outputs."""
    return _session(model, EP_PROVIDERS).run(None, feeds)


def run_cpu(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model through ORT's CPU EP."""
    return _session(model, ["CPUExecutionProvider"]).run(None, feeds)


def assert_matches_cpu(
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert the MLX EP output equals ORT's CPU EP output, tolerance-gated."""
    expected = _session(model, ["CPUExecutionProvider"]).run(None, feeds)
    actual = run_mlx(model, feeds)
    assert len(actual) == len(expected), "output count differs"
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {index}")


def assert_matches_ref(
    model: bytes,
    feeds: dict[str, np.ndarray],
    expected: list[np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert the MLX EP output equals a precomputed numpy reference (used for bf16 coverage)."""
    actual = run_mlx(model, feeds)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {index}")


# --- numpy references ---------------------------------------------------------------------------
def np_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def np_rms_norm(x: np.ndarray, scale: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * scale


def rotary_caches(max_seq: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    angles = pos * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


# --- Composite op model builders ----------------------------------------------------------------
def gqa_shared_buffer_model(
    name: str,
    *,
    batch: int,
    seq: int,
    past: int,
    cap: int,
    num_heads: int,
    kv_heads: int,
    head: int,
    do_rotary: int,
    interleaved: int = 0,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """GroupQueryAttention driven with a fixed-capacity SHARED KV buffer.

    Unlike :func:`gqa_model` (growing contract, ``past_key`` sized exactly to the valid past),
    here ``past_key``/``past_value`` are max-length buffers of capacity ``cap`` whose leading
    ``past`` rows are valid and whose tail ``[past, cap)`` is unused. ``total_sequence_length`` =
    ``past + seq`` (valid keys) so ``valid_past = past`` while the buffer capacity ``cap`` exceeds
    it — the shared-buffer path. ``present`` is emitted at the full ``cap`` capacity with the new
    K/V written in place at rows ``[past, past+seq)``. ORT's CPU GQA computes the reference.
    """
    assert cap >= past + seq, "capacity must hold the valid keys"
    valid = past + seq
    max_seq = cap + 4
    scale = 1.0 / np.sqrt(head)
    rng = np.random.default_rng(hash((name, seq, past, cap)) & 0xFFFFFFFF)
    q = rng.standard_normal((batch, seq, num_heads * head)).astype(np.float32)
    k = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    v = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    # Full-capacity buffers: valid rows [0, past) carry history, the tail [past, cap) is buffer
    # slack (filled with a recognizable sentinel so a mis-offset write is visible).
    past_k = np.zeros((batch, kv_heads, cap, head), dtype=np.float32)
    past_v = np.zeros((batch, kv_heads, cap, head), dtype=np.float32)
    past_k[:, :, :past, :] = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    past_v[:, :, :past, :] = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    seqlens_k = np.full((batch,), valid - 1, dtype=np.int32)
    total = np.array([valid], dtype=np.int32)
    cos, sin = rotary_caches(max_seq, head)

    inputs = [
        tensor("query", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("key", DataType.FLOAT, [batch, seq, kv_heads * head]),
        tensor("value", DataType.FLOAT, [batch, seq, kv_heads * head]),
        tensor("past_key", DataType.FLOAT, [batch, kv_heads, cap, head]),
        tensor("past_value", DataType.FLOAT, [batch, kv_heads, cap, head]),
        tensor("seqlens_k", DataType.INT32, [batch]),
        tensor("total_sequence_length", DataType.INT32, [1]),
        tensor("cos_cache", DataType.FLOAT, [max_seq, head // 2]),
        tensor("sin_cache", DataType.FLOAT, [max_seq, head // 2]),
    ]
    outputs = [
        tensor("attn_output", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", DataType.FLOAT, [batch, kv_heads, cap, head]),
        tensor("present_value", DataType.FLOAT, [batch, kv_heads, cap, head]),
    ]
    model = make_model(
        "GroupQueryAttention",
        inputs,
        outputs,
        domain="com.microsoft",
        attributes={
            "num_heads": num_heads,
            "kv_num_heads": kv_heads,
            "scale": float(scale),
            "do_rotary": do_rotary,
            "rotary_interleaved": interleaved,
        },
    )
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
    return model, feeds


def gqa_model(
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
    rope_cache: bool = True,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """GroupQueryAttention. With `rope_cache=False` the cos/sin cache inputs (7,8) are ABSENT
    (empty slots) — the external-rotary form (do_rotary=0) that genai exports emit."""
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

    cos_v = (
        tensor("cos_cache", DataType.FLOAT, [max_seq, head // 2])
        if rope_cache
        else ir.Value(name="")
    )
    sin_v = (
        tensor("sin_cache", DataType.FLOAT, [max_seq, head // 2])
        if rope_cache
        else ir.Value(name="")
    )
    inputs = [
        tensor("query", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("key", DataType.FLOAT, [batch, seq, kv_heads * head]),
        tensor("value", DataType.FLOAT, [batch, seq, kv_heads * head]),
        tensor("past_key", DataType.FLOAT, [batch, kv_heads, past, head]),
        tensor("past_value", DataType.FLOAT, [batch, kv_heads, past, head]),
        tensor("seqlens_k", DataType.INT32, [batch]),
        tensor("total_sequence_length", DataType.INT32, [1]),
        cos_v,
        sin_v,
    ]
    outputs = [
        tensor("attn_output", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", DataType.FLOAT, [batch, kv_heads, present, head]),
        tensor("present_value", DataType.FLOAT, [batch, kv_heads, present, head]),
    ]
    model = make_model(
        "GroupQueryAttention",
        inputs,
        outputs,
        domain="com.microsoft",
        attributes={
            "num_heads": num_heads,
            "kv_num_heads": kv_heads,
            "scale": float(scale),
            "do_rotary": do_rotary,
            "rotary_interleaved": interleaved,
        },
    )
    feeds = {
        "query": q,
        "key": k,
        "value": v,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total,
    }
    if rope_cache:
        feeds["cos_cache"] = cos
        feeds["sin_cache"] = sin
    return model, feeds


def bf16_gqa_model(
    name: str,
    **geometry: int,
) -> tuple[bytes, bytes, dict[str, np.ndarray]]:
    """Build equivalent bf16-interior and fp32-reference GQA models."""
    reference, feeds = gqa_model(name, **geometry)
    batch = geometry["batch"]
    seq = geometry["seq"]
    past = geometry["past"]
    num_heads = geometry["num_heads"]
    kv_heads = geometry["kv_heads"]
    head = geometry["head"]
    present = past + seq
    max_seq = present + 4

    float_specs = [
        ("query", [batch, seq, num_heads * head]),
        ("key", [batch, seq, kv_heads * head]),
        ("value", [batch, seq, kv_heads * head]),
        ("past_key", [batch, kv_heads, past, head]),
        ("past_value", [batch, kv_heads, past, head]),
        ("cos_cache", [max_seq, head // 2]),
        ("sin_cache", [max_seq, head // 2]),
    ]
    fp_inputs = {
        input_name: tensor(input_name, DataType.FLOAT, shape)
        for input_name, shape in float_specs
    }
    bf_inputs = {
        input_name: tensor(f"{input_name}_bf", DataType.BFLOAT16, shape)
        for input_name, shape in float_specs
    }
    nodes = [
        ir.Node(
            "",
            "Cast",
            [fp_inputs[input_name]],
            attributes=[ir.AttrInt64("to", int(DataType.BFLOAT16))],
            outputs=[bf_inputs[input_name]],
        )
        for input_name, _ in float_specs
    ]
    seqlens = tensor("seqlens_k", DataType.INT32, [batch])
    total = tensor("total_sequence_length", DataType.INT32, [1])
    bf_outputs = [
        tensor("attn_output_bf", DataType.BFLOAT16, [batch, seq, num_heads * head]),
        tensor("present_key_bf", DataType.BFLOAT16, [batch, kv_heads, present, head]),
        tensor("present_value_bf", DataType.BFLOAT16, [batch, kv_heads, present, head]),
    ]
    nodes.append(
        ir.Node(
            "com.microsoft",
            "GroupQueryAttention",
            [
                bf_inputs["query"],
                bf_inputs["key"],
                bf_inputs["value"],
                bf_inputs["past_key"],
                bf_inputs["past_value"],
                seqlens,
                total,
                bf_inputs["cos_cache"],
                bf_inputs["sin_cache"],
            ],
            attributes=[
                ir.AttrInt64("num_heads", num_heads),
                ir.AttrInt64("kv_num_heads", kv_heads),
                ir.AttrFloat32("scale", float(1.0 / np.sqrt(head))),
                ir.AttrInt64("do_rotary", geometry["do_rotary"]),
                ir.AttrInt64("rotary_interleaved", geometry.get("interleaved", 0)),
            ],
            outputs=bf_outputs,
        )
    )
    fp_outputs = [
        tensor("attn_output", DataType.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", DataType.FLOAT, [batch, kv_heads, present, head]),
        tensor("present_value", DataType.FLOAT, [batch, kv_heads, present, head]),
    ]
    nodes.extend(
        ir.Node(
            "",
            "Cast",
            [bf],
            attributes=[ir.AttrInt64("to", int(DataType.FLOAT))],
            outputs=[fp],
        )
        for bf, fp in zip(bf_outputs, fp_outputs, strict=True)
    )
    inputs = [fp_inputs[input_name] for input_name, _ in float_specs[:5]]
    inputs.extend([seqlens, total, fp_inputs["cos_cache"], fp_inputs["sin_cache"]])
    graph = ir.Graph(
        inputs,
        fp_outputs,
        nodes=nodes,
        name="mlx_bf16_GroupQueryAttention",
        opset_imports={"": 24, "com.microsoft": 1},
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    return model, reference, feeds


def matmulnbits_model(
    *, M: int, K: int, N: int, block: int = 32
) -> tuple[bytes, dict[str, np.ndarray]]:
    """MatMulNBits (int4 block-quantized weight matmul)."""
    rng = np.random.default_rng(hash((M, K, N)) & 0xFFFFFFFF)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    n_blocks = (K + block - 1) // block
    # Packed int4: [N, n_blocks, block/2] uint8, two nibbles per byte.
    b = rng.integers(0, 256, size=(N, n_blocks, block // 2), dtype=np.uint8)
    scales = rng.standard_normal((N * n_blocks,)).astype(np.float32) * 0.05
    model = make_model(
        "MatMulNBits",
        [
            tensor("a", DataType.FLOAT, [1, M, K]),
            tensor("b", DataType.UINT8, [N, n_blocks, block // 2]),
            tensor("scales", DataType.FLOAT, [N * n_blocks]),
        ],
        [tensor("out", DataType.FLOAT, [1, M, N])],
        domain="com.microsoft",
        attributes={"K": K, "N": N, "bits": 4, "block_size": block},
    )
    return model, {"a": a, "b": b, "scales": scales}


def rmsnorm_model(*, rows: int, hidden: int) -> tuple[bytes, dict[str, np.ndarray]]:
    rng = np.random.default_rng(hash((rows, hidden)) & 0xFFFFFFFF)
    x = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    scale = rng.standard_normal((hidden,)).astype(np.float32)
    model = make_model(
        "RMSNormalization",
        [tensor("x", DataType.FLOAT, [1, rows, hidden]), tensor("scale", DataType.FLOAT, [hidden])],
        [tensor("out", DataType.FLOAT, [1, rows, hidden])],
        attributes={"axis": -1, "epsilon": 1e-6},
    )
    return model, {"x": x, "scale": scale}


def skip_rmsnorm_model(*, rows: int, hidden: int) -> tuple[bytes, dict[str, np.ndarray]]:
    rng = np.random.default_rng(hash((rows, hidden)) & 0xFFFFFFFF)
    x = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    skip = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    gamma = rng.standard_normal((hidden,)).astype(np.float32)
    # SkipSimplifiedLayerNormalization emits up to 4 outputs; declaring a single output makes the
    # trailing ones optional — only output 0 (the normed result) is compared.
    model = make_model(
        "SkipSimplifiedLayerNormalization",
        [
            tensor("x", DataType.FLOAT, [1, rows, hidden]),
            tensor("skip", DataType.FLOAT, [1, rows, hidden]),
            tensor("gamma", DataType.FLOAT, [hidden]),
        ],
        [tensor("out", DataType.FLOAT, [1, rows, hidden])],
        domain="com.microsoft",
        attributes={"epsilon": 1e-6},
    )
    return model, {"x": x, "skip": skip, "gamma": gamma}


def gather_block_quantized_model() -> tuple[bytes, dict[str, np.ndarray]]:
    """GatherBlockQuantized (symmetric int4 embedding table)."""
    qdata = np.empty((2, 16), dtype=np.uint8)
    for row in range(2):
        values = (np.arange(32, dtype=np.uint8) + row) & 0x0F
        qdata[row] = values[0::2] | (values[1::2] << 4)
    model = make_model(
        "GatherBlockQuantized",
        [
            tensor("data", DataType.UINT8, [2, 16]),
            tensor("indices", DataType.INT64, [2]),
            tensor("scales", DataType.FLOAT, [2, 2]),
        ],
        [tensor("out", DataType.FLOAT, [2, 32])],
        domain="com.microsoft",
        attributes={"bits": 4, "block_size": 16, "gather_axis": 0, "quantize_axis": 1},
    )
    feeds = {
        "data": qdata,
        "indices": np.array([1, -2], dtype=np.int64),
        "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
    }
    return model, feeds
