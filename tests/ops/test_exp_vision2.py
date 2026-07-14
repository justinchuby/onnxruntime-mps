"""Correctness tests for the MLX EP opset-17+ vision / spatial-transform expansion (vision2.cc).

Each registered op is exercised through the MLX EP (with ORT CPU fallback available) and compared
against ORT's CPU EP, tolerance-gated:

  * GridSample — X[N,C,H,W] sampled at grid[N,Hout,Wout,2]; parametrized over mode
    (linear/nearest) x padding_mode (zeros/border) x align_corners (0/1). The forms the EP leaves to
    CPU (padding_mode="reflection", mode="cubic", the 5-D volumetric form) are not exercised here.
  * AffineGrid — the 2-D form (theta[N,2,3] + constant size[4]); parametrized over align_corners. The
    3-D form is left to CPU.
  * Col2Im — the static (constant image_shape / block_shape) 2-D form; parametrized over
    stride / dilation / pad. Non-float payloads and dynamic shapes are left to CPU.

Shape parameters (AffineGrid `size`, Col2Im `image_shape` / `block_shape`) are supplied as constant
initializers — the only forms the EP claims — mirroring a constant-folded real model.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("float16"): DT.FLOAT16,
}

_TOL = {
    np.dtype("float32"): dict(rtol=1e-3, atol=1e-3),
    np.dtype("float16"): dict(rtol=2e-2, atol=2e-2),
}


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


def tol(dt) -> dict:
    return _TOL[np.dtype(dt)]


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    inits: tuple[ir.Value, ...] = (),
    attrs: list[ir.Attr] | None = None,
    opset: int = 22,
) -> bytes:
    """Single-node model; initializer inputs are pulled out into the graph initializer list."""
    node = ir.Node("", op, inputs, attributes=list(attrs or []), outputs=outputs)
    graph_inputs = [i for i in inputs if i.const_value is None]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=list(inits),
        opset_imports={"": opset},
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- GridSample -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32])
@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample(dt, mode, padding_mode, align_corners):
    N, C, H, W = 2, 3, 4, 5
    Hout, Wout = 3, 4
    rng = np.random.default_rng(hash((mode, padding_mode, align_corners)) & 0xFFFFFFFF)
    x = rng.standard_normal((N, C, H, W)).astype(dt)
    # grid coords deliberately reach outside [-1, 1] to exercise the padding path.
    grid = (rng.random((N, Hout, Wout, 2)).astype(dt) * 2.6 - 1.3).astype(dt)
    model = build(
        "GridSample",
        [m.tensor("x", ir_of(dt), [N, C, H, W]), m.tensor("grid", ir_of(dt), [N, Hout, Wout, 2])],
        [m.tensor("y", ir_of(dt), [N, C, Hout, Wout])],
        attrs=[
            ir.AttrString("mode", mode),
            ir.AttrString("padding_mode", padding_mode),
            ir.AttrInt64("align_corners", align_corners),
        ],
    )
    m.assert_matches_cpu(model, {"x": x, "grid": grid}, **tol(dt))


# --- AffineGrid -----------------------------------------------------------------------------------
@pytest.mark.parametrize("align_corners", [0, 1])
def test_affine_grid(align_corners):
    dt = np.float32
    N, C, H, W = 2, 1, 4, 5
    size = np.array([N, C, H, W], np.int64)
    rng = np.random.default_rng(hash(("affine", align_corners)) & 0xFFFFFFFF)
    # Small rotation + scale + translation per batch.
    theta = rng.standard_normal((N, 2, 3)).astype(dt) * 0.5
    model = build(
        "AffineGrid",
        [m.tensor("theta", ir_of(dt), [N, 2, 3]), initz("size", size)],
        [m.tensor("grid", ir_of(dt), [N, H, W, 2])],
        inits=(initz("size", size),),
        attrs=[ir.AttrInt64("align_corners", align_corners)],
    )
    m.assert_matches_cpu(model, {"theta": theta}, **tol(dt))


# --- Col2Im ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "H, W, kh, kw, strides, dilations, pads",
    [
        (5, 5, 2, 2, [1, 1], [1, 1], [0, 0, 0, 0]),  # simple overlap-add
        (6, 6, 3, 3, [2, 2], [1, 1], [0, 0, 0, 0]),  # strided (non-overlapping blocks)
        (5, 5, 2, 2, [1, 1], [2, 2], [0, 0, 0, 0]),  # dilated kernel
        (4, 4, 3, 3, [1, 1], [1, 1], [1, 1, 1, 1]),  # padded
    ],
)
def test_col2im(H, W, kh, kw, strides, dilations, pads):
    dt = np.float32
    C = 2

    def n_pos(dim, k, s, d, pb, pe):
        return (dim + pb + pe - d * (k - 1) - 1) // s + 1

    nh = n_pos(H, kh, strides[0], dilations[0], pads[0], pads[2])
    nw = n_pos(W, kw, strides[1], dilations[1], pads[1], pads[3])
    L = nh * nw
    rng = np.random.default_rng(hash((H, W, kh, kw, tuple(strides), tuple(dilations), tuple(pads)))
                                & 0xFFFFFFFF)
    data = rng.standard_normal((1, C * kh * kw, L)).astype(dt)
    image_shape = np.array([H, W], np.int64)
    block_shape = np.array([kh, kw], np.int64)
    model = build(
        "Col2Im",
        [
            m.tensor("input", ir_of(dt), [1, C * kh * kw, L]),
            initz("image_shape", image_shape),
            initz("block_shape", block_shape),
        ],
        [m.tensor("out", ir_of(dt), [1, C, H, W])],
        inits=(initz("image_shape", image_shape), initz("block_shape", block_shape)),
        attrs=[
            ir.AttrInt64s("strides", strides),
            ir.AttrInt64s("dilations", dilations),
            ir.AttrInt64s("pads", pads),
        ],
        opset=18,
    )
    m.assert_matches_cpu(model, {"input": data}, **tol(dt))
