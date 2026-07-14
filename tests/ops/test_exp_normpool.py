"""Correctness tests for the MLX EP normalization + pooling family (normpool.cc).

Each registered op runs through the MLX EP (CPU fallback available) and is compared against ORT's
CPU EP, tolerance-gated (fp32 tight, fp16 loose). Ops left to ORT CPU (MaxUnpool, RoiAlign,
MaxRoiPool, GridSample) are intentionally not exercised here.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType
RNG = np.random.default_rng(71)

_IR_OF = {np.dtype("float32"): DT.FLOAT, np.dtype("float16"): DT.FLOAT16}
DTYPES = [np.float32, np.float16]
DT_IDS = ["fp32", "fp16"]


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


def tol(dt) -> dict:
    return dict(rtol=2e-2, atol=2e-2) if np.dtype(dt) == np.float16 else dict(rtol=1e-4, atol=1e-4)


def sample(shape, dt) -> np.ndarray:
    return (RNG.standard_normal(shape) * 0.5).astype(dt)


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    attrs: list[ir.Attr] | None = None,
    initializers: list[ir.Value] | None = None,
    opset: int = 18,
) -> bytes:
    node = ir.Node("", op, inputs, attributes=list(attrs or []), outputs=outputs)
    graph = ir.Graph(
        [v for v in inputs if v.const_value is None],
        outputs,
        nodes=[node],
        initializers=list(initializers or []),
        opset_imports={"": opset},
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


# --- InstanceNormalization --------------------------------------------------------------------
INSTANCE_SHAPES = [(2, 3, 5, 6), (1, 4, 8)]  # 2D-spatial and 1D-spatial


@pytest.mark.parametrize("dt", DTYPES, ids=DT_IDS)
@pytest.mark.parametrize("shape", INSTANCE_SHAPES, ids=["nchw", "ncl"])
def test_instance_norm(dt, shape) -> None:
    c = shape[1]
    scale = initializer("scale", sample((c,), dt))
    bias = initializer("bias", sample((c,), dt))
    model = build(
        "InstanceNormalization",
        [m.tensor("x", ir_of(dt), list(shape)), scale, bias],
        [m.tensor("y", ir_of(dt), list(shape))],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        initializers=[scale, bias],
    )
    m.assert_matches_cpu(model, {"x": sample(shape, dt)}, **tol(dt))


# --- MeanVarianceNormalization ----------------------------------------------------------------
MVN_CASES = [
    ((2, 3, 4, 4), None),  # default axes [0,2,3]
    ((2, 3, 5), [0, 2]),
    ((4, 6), [-1]),
]


def _mvn_ref(x: np.ndarray, axes) -> np.ndarray:
    xf = x.astype(np.float32)
    raw = [0, 2, 3] if axes is None else list(axes)
    ax = tuple(sorted(a % xf.ndim for a in raw if -xf.ndim <= a < xf.ndim))
    mean = xf.mean(axis=ax, keepdims=True)
    var = (xf**2).mean(axis=ax, keepdims=True) - mean**2
    return ((xf - mean) / (np.sqrt(var) + 1e-9)).astype(x.dtype)


@pytest.mark.parametrize("dt", DTYPES, ids=DT_IDS)
@pytest.mark.parametrize("shape,axes", MVN_CASES, ids=["default", "axes-0-2", "axes-neg1"])
def test_mean_variance_norm(dt, shape, axes) -> None:
    # ORT CPU decomposes MVN into a fp32 function (its 1e-9 epsilon is float), so it cannot run the
    # fp16 graph; fp32 compares against ORT CPU, fp16 against a numpy fp32 reference.
    attrs = [ir.AttrInt64s("axes", axes)] if axes is not None else []
    x = sample(shape, dt)
    model = build(
        "MeanVarianceNormalization",
        [m.tensor("x", ir_of(dt), list(shape))],
        [m.tensor("y", ir_of(dt), list(shape))],
        attrs=attrs,
    )
    if np.dtype(dt) == np.float32:
        m.assert_matches_cpu(model, {"x": x}, **tol(dt))
    else:
        m.assert_matches_ref(model, {"x": x}, [_mvn_ref(x, axes)], **tol(dt))


# --- LRN --------------------------------------------------------------------------------------
LRN_CASES = [
    dict(size=3, alpha=1e-4, beta=0.75, bias=1.0),
    dict(size=5, alpha=2e-4, beta=0.6, bias=1.5),
    dict(size=7, alpha=1e-3, beta=0.5, bias=2.0),
]


@pytest.mark.parametrize("dt", DTYPES, ids=DT_IDS)
@pytest.mark.parametrize("params", LRN_CASES, ids=["s3", "s5", "s7"])
def test_lrn(dt, params) -> None:
    shape = (2, 7, 3, 3)
    model = build(
        "LRN",
        [m.tensor("x", ir_of(dt), list(shape))],
        [m.tensor("y", ir_of(dt), list(shape))],
        attrs=[
            ir.AttrInt64("size", params["size"]),
            ir.AttrFloat32("alpha", params["alpha"]),
            ir.AttrFloat32("beta", params["beta"]),
            ir.AttrFloat32("bias", params["bias"]),
        ],
    )
    m.assert_matches_cpu(model, {"x": sample(shape, dt)}, **tol(dt))


# --- LpPool -----------------------------------------------------------------------------------
def _lp_out_shape(shape, kernel, strides, pads):
    return (
        shape[0],
        shape[1],
        (shape[2] + pads[0] + pads[2] - kernel[0]) // strides[0] + 1,
        (shape[3] + pads[1] + pads[3] - kernel[1]) // strides[1] + 1,
    )


LP_POOL_CASES = [
    dict(kernel=(2, 2), strides=(2, 1), pads=(0, 0, 0, 0), p=2),
    dict(kernel=(2, 3), strides=(2, 1), pads=(0, 1, 0, 1), p=1),
    dict(kernel=(3, 2), strides=(1, 2), pads=(1, 0, 1, 0), p=3),
]


@pytest.mark.parametrize("dt", DTYPES, ids=DT_IDS)
@pytest.mark.parametrize("params", LP_POOL_CASES, ids=["p2", "p1-pad", "p3-pad"])
def test_lp_pool(dt, params) -> None:
    shape = (1, 3, 5, 6)
    out_shape = _lp_out_shape(shape, params["kernel"], params["strides"], params["pads"])
    model = build(
        "LpPool",
        [m.tensor("x", ir_of(dt), list(shape))],
        [m.tensor("y", ir_of(dt), list(out_shape))],
        attrs=[
            ir.AttrInt64s("kernel_shape", params["kernel"]),
            ir.AttrInt64s("strides", params["strides"]),
            ir.AttrInt64s("pads", params["pads"]),
            ir.AttrInt64("p", params["p"]),
        ],
    )
    m.assert_matches_cpu(model, {"x": sample(shape, dt)}, **tol(dt))


# --- GlobalLpPool -----------------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES, ids=DT_IDS)
@pytest.mark.parametrize("p", [1, 2, 3], ids=["p1", "p2", "p3"])
def test_global_lp_pool(dt, p) -> None:
    shape = (2, 3, 4, 5)
    model = build(
        "GlobalLpPool",
        [m.tensor("x", ir_of(dt), list(shape))],
        [m.tensor("y", ir_of(dt), [shape[0], shape[1], 1, 1])],
        attrs=[ir.AttrInt64("p", p)],
    )
    m.assert_matches_cpu(model, {"x": sample(shape, dt)}, **tol(dt))
