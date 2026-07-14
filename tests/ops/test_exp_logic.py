"""MLX bitwise and extra-reduction op coverage (opset-17+ expansion).

Covers the two families added in the exp/logic worktree:
  * Bitwise:  BitwiseAnd / BitwiseOr / BitwiseXor / BitwiseNot / BitShift / Xor
  * Reduce2:  ReduceL1 / ReduceLogSum / ReduceLogSumExp / ReduceProd / Hardmax / CumProd /
              Mean / Sum

Each op is parametrized over representative dtypes and shapes and checked against ORT's CPU EP
(exact for integer/bool, tolerance-gated for float).
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m


# --- local builders (string / int64s attributes _models.make_model cannot express) --------------
def _model_with_string_attr(
    op: str,
    dtype: DT,
    shape: list[int],
    attr_name: str,
    attr_value: str,
    n_inputs: int,
    opset: int,
) -> bytes:
    inputs = [m.tensor(f"in{i}", dtype, shape) for i in range(n_inputs)]
    out = m.tensor("out", dtype, shape)
    node = ir.Node(
        "",
        op,
        inputs,
        attributes=[ir.AttrString(attr_name, attr_value)],
        outputs=[out],
    )
    graph = ir.Graph(
        inputs, [out], nodes=[node], name=f"mlx_{op}", opset_imports={"": opset}
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _reduce_attr_model(
    op: str,
    dtype: DT,
    in_shape: list[int],
    out_shape: list[int],
    axes: list[int],
    *,
    keepdims: int,
    opset: int = 13,
) -> bytes:
    x = m.tensor("x", dtype, in_shape)
    out = m.tensor("out", dtype, out_shape)
    node = ir.Node(
        "",
        op,
        [x],
        attributes=[ir.AttrInt64s("axes", axes), ir.AttrInt64("keepdims", keepdims)],
        outputs=[out],
    )
    graph = ir.Graph(
        [x], [out], nodes=[node], name=f"mlx_{op}_axes_attr", opset_imports={"": opset}
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# =============================== Bitwise family =================================================
INT_CASES = [
    (DT.INT8, np.int8),
    (DT.INT32, np.int32),
    (DT.INT64, np.int64),
    (DT.UINT8, np.uint8),
    (DT.UINT32, np.uint32),
]
UINT_CASES = [
    (DT.UINT8, np.uint8),
    (DT.UINT32, np.uint32),
    (DT.UINT64, np.uint64),
]


@pytest.mark.parametrize("op", ["BitwiseAnd", "BitwiseOr", "BitwiseXor"])
@pytest.mark.parametrize("dtype,np_dtype", INT_CASES)
def test_bitwise_binary(op: str, dtype: DT, np_dtype) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", dtype, [2, 3])],
        opset=18,
    )
    rng = np.random.default_rng(1)
    info = np.iinfo(np_dtype)
    feeds = {
        "a": rng.integers(info.min, info.max, size=(2, 3), endpoint=True, dtype=np_dtype),
        "b": rng.integers(info.min, info.max, size=(3,), endpoint=True, dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,np_dtype", INT_CASES)
def test_bitwise_not(dtype: DT, np_dtype) -> None:
    model = m.make_model(
        "BitwiseNot",
        [m.tensor("x", dtype, [2, 3])],
        [m.tensor("out", dtype, [2, 3])],
        opset=18,
    )
    info = np.iinfo(np_dtype)
    x = np.random.default_rng(2).integers(
        info.min, info.max, size=(2, 3), endpoint=True, dtype=np_dtype
    )
    m.assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
@pytest.mark.parametrize("dtype,np_dtype", UINT_CASES)
def test_bitshift(direction: str, dtype: DT, np_dtype) -> None:
    model = _model_with_string_attr(
        "BitShift", dtype, [2, 3], "direction", direction, n_inputs=2, opset=11
    )
    bits = np.iinfo(np_dtype).bits
    rng = np.random.default_rng(3)
    a = rng.integers(0, np.iinfo(np_dtype).max, size=(2, 3), endpoint=True, dtype=np_dtype)
    shift = rng.integers(0, bits, size=(2, 3), dtype=np_dtype)
    m.assert_matches_cpu(model, {"in0": a, "in1": shift}, rtol=0, atol=0)


def test_xor_bool() -> None:
    model = m.make_model(
        "Xor",
        [m.tensor("a", DT.BOOL, [2, 3]), m.tensor("b", DT.BOOL, [3])],
        [m.tensor("out", DT.BOOL, [2, 3])],
    )
    feeds = {
        "a": np.array([[True, False, True], [False, True, False]]),
        "b": np.array([True, True, False]),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


# =============================== Reduction2 family ==============================================
FLOAT_CASES = [(DT.FLOAT, np.float32, 1e-5), (DT.FLOAT16, np.float16, 3e-3)]
NUMERIC_REDUCE_CASES = [
    (DT.FLOAT, np.float32, 1e-5),
    (DT.FLOAT16, np.float16, 5e-3),
    (DT.INT32, np.int32, 0),
    (DT.INT64, np.int64, 0),
]


@pytest.mark.parametrize("dtype,np_dtype,tol", NUMERIC_REDUCE_CASES)
def test_reduce_l1(dtype: DT, np_dtype, tol: float) -> None:
    model = _reduce_attr_model("ReduceL1", dtype, [2, 3, 4], [2, 1, 4], [1], keepdims=1)
    x = (np.random.default_rng(10).standard_normal((2, 3, 4)) * 5).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", NUMERIC_REDUCE_CASES)
def test_reduce_prod(dtype: DT, np_dtype, tol: float) -> None:
    model = _reduce_attr_model("ReduceProd", dtype, [2, 3, 4], [2, 3, 1], [2], keepdims=1)
    if np.issubdtype(np_dtype, np.integer):
        x = np.random.default_rng(11).integers(-3, 4, size=(2, 3, 4)).astype(np_dtype)
    else:
        x = np.random.default_rng(11).uniform(-1.5, 1.5, size=(2, 3, 4)).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_reduce_logsum(dtype: DT, np_dtype, tol: float) -> None:
    model = _reduce_attr_model("ReduceLogSum", dtype, [2, 3, 4], [2, 1, 4], [1], keepdims=1)
    x = np.random.default_rng(12).uniform(0.1, 3.0, size=(2, 3, 4)).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_reduce_logsumexp(dtype: DT, np_dtype, tol: float) -> None:
    model = _reduce_attr_model(
        "ReduceLogSumExp", dtype, [2, 3, 4], [2, 1, 4], [1], keepdims=1
    )
    x = np.random.default_rng(13).standard_normal((2, 3, 4)).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


def test_reduce_prod_axes_input_opset18() -> None:
    model = m.make_model(
        "ReduceProd",
        [m.tensor("x", DT.FLOAT, [2, 3, 4]), m.tensor("axes", DT.INT64, [2])],
        [m.tensor("out", DT.FLOAT, [1, 3, 1])],
        attributes={"keepdims": 1},
        opset=18,
    )
    feeds = {
        "x": np.random.default_rng(14).uniform(0.5, 1.5, size=(2, 3, 4)).astype(np.float32),
        "axes": np.array([0, -1], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds)


def test_reduce_l1_noop_with_empty_axes() -> None:
    model = m.make_model(
        "ReduceL1",
        [m.tensor("x", DT.FLOAT, [2, 3]), m.tensor("axes", DT.INT64, [0])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
        attributes={"keepdims": 1, "noop_with_empty_axes": 1},
        opset=18,
    )
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
    m.assert_matches_cpu(model, {"x": x, "axes": np.empty((0,), dtype=np.int64)})


@pytest.mark.parametrize("axis", [-1, 1])
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_hardmax(axis: int, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Hardmax",
        [m.tensor("x", dtype, [2, 3, 4])],
        [m.tensor("out", dtype, [2, 3, 4])],
        attributes={"axis": axis},
        opset=13,
    )
    # Distinct values (via a permutation) so the argmax has no ties to disambiguate.
    x = np.random.default_rng(15).permutation(24).reshape(2, 3, 4).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "exclusive,reverse", [(0, 0), (1, 0), (0, 1)], ids=["inclusive", "exclusive", "reverse"]
)
@pytest.mark.parametrize(
    "dtype,np_dtype,tol",
    [(DT.FLOAT, np.float32, 1e-5), (DT.INT32, np.int32, 0)],
    ids=["fp32", "int32"],
)
def test_cumprod(exclusive: int, reverse: int, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "CumProd",
        [m.tensor("x", dtype, [2, 4]), m.tensor("axis", DT.INT64, [])],
        [m.tensor("out", dtype, [2, 4])],
        attributes={"exclusive": exclusive, "reverse": reverse},
        opset=26,
    )
    if np.issubdtype(np_dtype, np.integer):
        x = np.array([[1, 2, 3, 4], [2, 1, 3, 1]], dtype=np_dtype)
    else:
        x = np.array([[1.0, 2.0, 0.5, 4.0], [2.0, 1.5, 0.5, 1.0]], dtype=np_dtype)
    feeds = {"x": x, "axis": np.array(-1, dtype=np.int64)}
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


@pytest.mark.parametrize("op", ["Sum", "Mean"])
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_variadic_sum_mean(op: str, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        op,
        [
            m.tensor("a", dtype, [2, 3]),
            m.tensor("b", dtype, [3]),
            m.tensor("c", dtype, [2, 3]),
        ],
        [m.tensor("out", dtype, [2, 3])],
    )
    rng = np.random.default_rng(16)
    feeds = {
        "a": rng.standard_normal((2, 3)).astype(np_dtype),
        "b": rng.standard_normal((3,)).astype(np_dtype),
        "c": rng.standard_normal((2, 3)).astype(np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)
