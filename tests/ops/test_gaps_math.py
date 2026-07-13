"""Remaining Mobius math, activation, logical, and reduction coverage."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m


FLOAT_CASES = [(DT.FLOAT, np.float32, 1e-5), (DT.FLOAT16, np.float16, 3e-3)]


def _initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


def _model(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    initializers: tuple[ir.Value, ...] = (),
    attributes: list[ir.Attr] = (),
    domain: str = "",
    opset: int = 24,
) -> bytes:
    node = ir.Node(domain, op, inputs, attributes=attributes, outputs=outputs)
    graph = ir.Graph(
        [value for value in inputs if value.const_value is None],
        outputs,
        nodes=[node],
        initializers=list(initializers),
        opset_imports={"": opset, **({domain: 1} if domain else {})},
        name=f"mlx_gap_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_mod_fmod(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Mod",
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", dtype, [2, 3])],
        attributes={"fmod": 1},
        opset=13,
    )
    feeds = {
        "a": np.array([[-7.5, 7.5, -4.0], [7.0, -7.0, 4.0]], dtype=np_dtype),
        "b": np.array([3.0, -3.0, 2.5], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


def test_mod_floored_integer() -> None:
    model = m.make_model(
        "Mod",
        [m.tensor("a", DT.INT64, [2, 3]), m.tensor("b", DT.INT64, [3])],
        [m.tensor("out", DT.INT64, [2, 3])],
        attributes={"fmod": 0},
        opset=13,
    )
    feeds = {
        "a": np.array([[-8, 8, -4], [7, -7, 4]], dtype=np.int64),
        "b": np.array([3, -3, 5], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["ArgMin", "ArgMax"])
@pytest.mark.parametrize("select_last", [0, 1], ids=["first", "last"])
def test_argminmax(op: str, select_last: int) -> None:
    model = m.make_model(
        op,
        [m.tensor("x", DT.FLOAT, [2, 2, 4])],
        [m.tensor("out", DT.INT64, [2, 2, 1])],
        attributes={"axis": -1, "keepdims": 1, "select_last_index": select_last},
        opset=13,
    )
    x = np.array(
        [[[1, 4, 4, 2], [3, -1, -1, 5]], [[0, 2, 1, 2], [7, 6, 7, 5]]],
        dtype=np.float32,
    )
    m.assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,np_dtype",
    [(DT.FLOAT, np.float32), (DT.INT64, np.int64)],
    ids=["fp32", "int64"],
)
def test_less_or_equal(dtype: DT, np_dtype) -> None:
    model = m.make_model(
        "LessOrEqual",
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", DT.BOOL, [2, 3])],
        opset=16,
    )
    feeds = {
        "a": np.array([[1, 2, 4], [5, 3, 0]], dtype=np_dtype),
        "b": np.array([1, 3, 2], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_log_softmax(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "LogSoftmax",
        [m.tensor("x", dtype, [2, 3, 4])],
        [m.tensor("out", dtype, [2, 3, 4])],
        attributes={"axis": -1},
        opset=13,
    )
    x = np.linspace(-4, 4, 24, dtype=np.float32).reshape(2, 3, 4).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_elu(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Elu",
        [m.tensor("x", dtype, [2, 3])],
        [m.tensor("out", dtype, [2, 3])],
        attributes={"alpha": 1.25},
    )
    x = np.array([[-3, -1, -0.0], [0.5, 2, 5]], dtype=np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


def test_sigmoid_mul_swish_fold() -> None:
    x = m.tensor("x", DT.FLOAT, [2, 3])
    sigmoid = m.tensor("sigmoid", DT.FLOAT, [2, 3])
    out = m.tensor("out", DT.FLOAT, [2, 3])
    graph = ir.Graph(
        [x],
        [out],
        nodes=[
            ir.Node("", "Sigmoid", [x], outputs=[sigmoid]),
            ir.Node("", "Mul", [x, sigmoid], outputs=[out]),
        ],
        opset_imports={"": 24},
        name="mlx_swish_fold",
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    values = np.array([[-4, -1, 0], [0.5, 2, 5]], dtype=np.float32)
    m.assert_matches_cpu(model, {"x": values})


@pytest.mark.parametrize(
    "values,np_values",
    [
        (DT.FLOAT, np.array([-0.5, 2.0], dtype=np.float32)),
        (DT.INT64, np.array([3, 9], dtype=np.int64)),
    ],
    ids=["fp32-values", "int64-values"],
)
def test_one_hot(values: DT, np_values: np.ndarray) -> None:
    depth = _initializer("depth", np.array(4, dtype=np.int64))
    model = _model(
        "OneHot",
        [m.tensor("indices", DT.INT64, [2, 3]), depth, m.tensor("values", values, [2])],
        [m.tensor("out", values, [2, 4, 3])],
        initializers=(depth,),
        attributes=[ir.AttrInt64("axis", 1)],
        opset=11,
    )
    indices = np.array([[0, 2, -1], [3, 1, 5]], dtype=np.int64)
    m.assert_matches_cpu(model, {"indices": indices, "values": np_values}, rtol=0, atol=0)


@pytest.mark.parametrize("upper", [0, 1], ids=["lower", "upper"])
def test_trilu(upper: int) -> None:
    k = _initializer("k", np.array(1, dtype=np.int64))
    model = _model(
        "Trilu",
        [m.tensor("x", DT.FLOAT16, [2, 3, 4]), k],
        [m.tensor("out", DT.FLOAT16, [2, 3, 4])],
        initializers=(k,),
        attributes=[ir.AttrInt64("upper", upper)],
        opset=14,
    )
    x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    m.assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_round(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Round", [m.tensor("x", dtype, [2, 4])], [m.tensor("out", dtype, [2, 4])], opset=22
    )
    x = np.array([[-2.5, -1.5, -0.5, 0.5], [1.5, 2.5, 3.1, -3.1]], dtype=np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_reduce_l2(dtype: DT, np_dtype, tol: float) -> None:
    axes = _initializer("axes", np.array([0, -1], dtype=np.int64))
    model = _model(
        "ReduceL2",
        [m.tensor("x", dtype, [2, 3, 4]), axes],
        [m.tensor("out", dtype, [1, 3, 1])],
        initializers=(axes,),
        attributes=[ir.AttrInt64("keepdims", 1)],
        opset=18,
    )
    x = np.linspace(-2, 3, 24, dtype=np.float32).reshape(2, 3, 4).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)
