"""MLX op-correctness tests for the MLX-native ONNX Runtime execution provider.

Each ONNX decoder op the modular registry (``src/ep/ops/*.cc``) translates to MLX is run through the
``MLXExecutionProvider`` plugin and compared, tolerance-gated, against a reference:

* fp32 / fp16 cases compare against ORT's **CPU EP** (which has kernels for these ops);
* bf16 cases keep the compute inside an MLX-claimed subgraph (fp32 boundaries) and compare against a
  **numpy fp32 reference** — ORT's CPU EP ships no bf16 kernels and its Python binding can't feed
  bf16 arrays.

Ops the EP does not claim (Div, Gelu, RotaryEmbedding, Reshape, Transpose, Concat) are intentionally
absent: they fall back to ORT CPU and would only compare CPU-vs-CPU. The MLX EP is registered once
per session by ``conftest.py`` from ``ONNXRUNTIME_MLX_EP_LIB``.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m

F16 = np.float16

# Fixed small inputs reused across several fp32/fp16 elementwise cases.
A23 = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
B3 = np.array([2.0, -4.0, 0.5], dtype=np.float32)

# Shared random inputs for the dtype-generic (fp16 + bf16) coverage — drawn once, in order.
_rng = np.random.default_rng(7)
RA = _rng.standard_normal((2, 3)).astype(np.float32)
RB = _rng.standard_normal((3,)).astype(np.float32)
RX = _rng.standard_normal((2, 5)).astype(np.float32)
RMS_X = _rng.standard_normal((1, 4, 8)).astype(np.float32)
RMS_G = _rng.standard_normal((8,)).astype(np.float32)

BF_RTOL = BF_ATOL = 2e-2  # bf16 carries ~3 significant digits


# --- Elementwise / activation (fp32) ------------------------------------------------------------
@pytest.mark.parametrize(
    "op,b_shape,feeds",
    [
        ("Mul", [3], {"a": A23, "b": B3}),
        ("Sub", [3], {"a": A23, "b": B3}),
        ("Add", [2, 3], {"a": A23, "b": A23 * 0.5}),
    ],
)
def test_binary_fp32(op: str, b_shape: list[int], feeds: dict[str, np.ndarray]) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", DT.FLOAT, [2, 3]), m.tensor("b", DT.FLOAT, b_shape)],
        [m.tensor("out", DT.FLOAT, [2, 3])],
    )
    m.assert_matches_cpu(model, feeds)


def test_sigmoid_fp32() -> None:
    model = m.make_model(
        "Sigmoid", [m.tensor("x", DT.FLOAT, [2, 3])], [m.tensor("out", DT.FLOAT, [2, 3])]
    )
    m.assert_matches_cpu(model, {"x": A23})


def test_softmax_fp32() -> None:
    x = np.random.default_rng(1).standard_normal((2, 5)).astype(np.float32)
    model = m.make_model(
        "Softmax",
        [m.tensor("x", DT.FLOAT, [2, 5])],
        [m.tensor("out", DT.FLOAT, [2, 5])],
        attributes={"axis": -1},
    )
    m.assert_matches_cpu(model, {"x": x}, rtol=2e-3, atol=2e-3)


# --- Cast ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "src,dst,x",
    [
        (DT.FLOAT, DT.FLOAT16, A23),
        (DT.FLOAT16, DT.FLOAT, A23.astype(F16)),
    ],
    ids=["fp32->fp16", "fp16->fp32"],
)
def test_cast(src: DT, dst: DT, x: np.ndarray) -> None:
    model = m.make_model(
        "Cast",
        [m.tensor("x", src, [2, 3])],
        [m.tensor("out", dst, [2, 3])],
        attributes={"to": int(dst)},
    )
    m.assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


def test_sub_int64_scalar() -> None:
    model = m.make_model(
        "Sub",
        [m.tensor("a", DT.INT64, [3]), m.tensor("b", DT.INT64, [])],
        [m.tensor("out", DT.INT64, [3])],
    )
    feeds = {"a": np.array([5, -2, 9], dtype=np.int64), "b": np.array(3, dtype=np.int64)}
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


# --- Quantization / gather ----------------------------------------------------------------------
def test_gather_block_quantized() -> None:
    model, feeds = m.gather_block_quantized_model()
    m.assert_matches_cpu(model, feeds)


@pytest.mark.parametrize("M", [1, 8], ids=["decode", "prefill"])
def test_matmulnbits(M: int) -> None:
    model, feeds = m.matmulnbits_model(M=M, K=64, N=32)
    m.assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


# --- Normalization ------------------------------------------------------------------------------
def test_rmsnorm() -> None:
    model, feeds = m.rmsnorm_model(rows=4, hidden=64)
    m.assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


def test_skip_simplified_layernorm() -> None:
    model, feeds = m.skip_rmsnorm_model(rows=4, hidden=64)
    m.assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


# --- GroupQueryAttention (real + toy head geometries, decode / prefill / chunked / no-rope) ------
GQA_CASES = [
    ("decode-h64", dict(batch=1, num_heads=14, kv_heads=2, head=64, seq=1, past=40, do_rotary=1)),
    ("prefill-h64", dict(batch=1, num_heads=14, kv_heads=2, head=64, seq=26, past=0, do_rotary=1)),
    ("chunked-h64", dict(batch=1, num_heads=14, kv_heads=2, head=64, seq=3, past=8, do_rotary=1)),
    ("decode", dict(batch=1, num_heads=4, kv_heads=2, head=16, seq=1, past=5, do_rotary=1)),
    ("prefill", dict(batch=1, num_heads=4, kv_heads=2, head=16, seq=6, past=0, do_rotary=1)),
    ("decode-norope", dict(batch=1, num_heads=4, kv_heads=2, head=16, seq=1, past=5, do_rotary=0)),
]


@pytest.mark.parametrize("name,geom", GQA_CASES, ids=[c[0] for c in GQA_CASES])
def test_gqa(name: str, geom: dict[str, int]) -> None:
    model, feeds = m.gqa_model(name, **geom)
    m.assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


def test_gqa_bf16() -> None:
    model, reference, feeds = m.bf16_gqa_model(
        "bf16-decode-h64",
        batch=1,
        num_heads=14,
        kv_heads=2,
        head=64,
        seq=1,
        past=8,
        do_rotary=1,
    )
    expected = m.run_cpu(reference, feeds)
    m.assert_matches_ref(model, feeds, expected, rtol=2e-2, atol=2e-2)


# --- dtype-generic fp16 (vs ORT CPU) ------------------------------------------------------------
def _fp16_mul() -> tuple[bytes, dict[str, np.ndarray]]:
    return (
        m.make_model(
            "Mul",
            [m.tensor("a", DT.FLOAT16, [2, 3]), m.tensor("b", DT.FLOAT16, [3])],
            [m.tensor("out", DT.FLOAT16, [2, 3])],
        ),
        {"a": RA.astype(F16), "b": RB.astype(F16)},
    )


def _fp16_sub() -> tuple[bytes, dict[str, np.ndarray]]:
    return (
        m.make_model(
            "Sub",
            [m.tensor("a", DT.FLOAT16, [2, 3]), m.tensor("b", DT.FLOAT16, [3])],
            [m.tensor("out", DT.FLOAT16, [2, 3])],
        ),
        {"a": RA.astype(F16), "b": RB.astype(F16)},
    )


def _fp16_sigmoid() -> tuple[bytes, dict[str, np.ndarray]]:
    return (
        m.make_model(
            "Sigmoid", [m.tensor("x", DT.FLOAT16, [2, 5])], [m.tensor("out", DT.FLOAT16, [2, 5])]
        ),
        {"x": RX.astype(F16)},
    )


def _fp16_softmax() -> tuple[bytes, dict[str, np.ndarray]]:
    return (
        m.make_model(
            "Softmax",
            [m.tensor("x", DT.FLOAT16, [2, 5])],
            [m.tensor("out", DT.FLOAT16, [2, 5])],
            attributes={"axis": -1},
        ),
        {"x": RX.astype(F16)},
    )


def _fp16_rmsnorm() -> tuple[bytes, dict[str, np.ndarray]]:
    return (
        m.make_model(
            "RMSNormalization",
            [m.tensor("x", DT.FLOAT16, [1, 4, 8]), m.tensor("scale", DT.FLOAT16, [8])],
            [m.tensor("out", DT.FLOAT16, [1, 4, 8])],
            attributes={"axis": -1, "epsilon": 1e-6},
        ),
        {"x": RMS_X.astype(F16), "scale": RMS_G.astype(F16)},
    )


@pytest.mark.parametrize(
    "builder,rtol,atol",
    [
        (_fp16_mul, 2e-3, 2e-3),
        (_fp16_sub, 2e-3, 2e-3),
        (_fp16_sigmoid, 2e-3, 2e-3),
        (_fp16_softmax, 2e-3, 2e-3),
        (_fp16_rmsnorm, 3e-3, 3e-3),
    ],
    ids=["Mul", "Sub", "Sigmoid", "Softmax", "RMSNormalization"],
)
def test_dtype_fp16(builder, rtol: float, atol: float) -> None:
    model, feeds = builder()
    m.assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


# --- dtype-generic bf16 (MLX bf16 interior vs numpy fp32 reference) ------------------------------
def _bf16_cases() -> list[tuple[str, bytes, dict[str, np.ndarray], list[np.ndarray]]]:
    return [
        (
            "Add",
            m.bf16_interior_model("Add", [("a", [2, 3]), ("b", [2, 3])], [2, 3]),
            {"a": RA, "b": RA * 0.5},
            [RA + RA * 0.5],
        ),
        (
            "Mul",
            m.bf16_interior_model("Mul", [("a", [2, 3]), ("b", [3])], [2, 3]),
            {"a": RA, "b": RB},
            [RA * RB],
        ),
        (
            "Sub",
            m.bf16_interior_model("Sub", [("a", [2, 3]), ("b", [3])], [2, 3]),
            {"a": RA, "b": RB},
            [RA - RB],
        ),
        (
            "Sigmoid",
            m.bf16_interior_model("Sigmoid", [("x", [2, 5])], [2, 5]),
            {"x": RX},
            [1.0 / (1.0 + np.exp(-RX))],
        ),
        (
            "Softmax",
            m.bf16_interior_model("Softmax", [("x", [2, 5])], [2, 5], attributes={"axis": -1}),
            {"x": RX},
            [m.np_softmax(RX)],
        ),
        (
            "RMSNormalization",
            m.bf16_interior_model(
                "RMSNormalization",
                [("x", [1, 4, 8]), ("scale", [8])],
                [1, 4, 8],
                attributes={"axis": -1, "epsilon": 1e-6},
            ),
            {"x": RMS_X, "scale": RMS_G},
            [m.np_rms_norm(RMS_X, RMS_G)],
        ),
        (
            "Cast-roundtrip",
            m.bf16_interior_model("Add", [("a", [2, 3]), ("b", [2, 3])], [2, 3]),
            {"a": RA, "b": np.zeros_like(RA)},
            [RA],
        ),
    ]


BF16_CASES = _bf16_cases()


@pytest.mark.parametrize(
    "model,feeds,expected",
    [(c[1], c[2], c[3]) for c in BF16_CASES],
    ids=[c[0] for c in BF16_CASES],
)
def test_dtype_bf16(
    model: bytes, feeds: dict[str, np.ndarray], expected: list[np.ndarray]
) -> None:
    m.assert_matches_ref(model, feeds, expected, rtol=BF_RTOL, atol=BF_ATOL)
