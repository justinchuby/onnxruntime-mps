"""MLX op-correctness + claim tests for the elementwise / activation / comparison / logical swath.

Each single-node model is run through the ``MLXExecutionProvider`` plugin. ``assert_mlx_claims`` proves
via ORT per-node profiling that the MLX EP actually translated the op (so the CPU-match check is not a
vacuous CPU-fallback pass), then the output is compared tolerance-gated against ORT's CPU EP. fp16
comparisons are skipped when ORT's CPU EP ships no fp16 kernel for the op (claim is still asserted).

Models are built with the ONNX IR through ``_models`` (unmodified).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from onnx_ir import DataType as DT

import onnxruntime as ort

import _models as m

FLOAT_CASES = [(DT.FLOAT, np.float32, 1e-5), (DT.FLOAT16, np.float16, 3e-3)]


# --- claim helpers ------------------------------------------------------------------------------
def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the MLX EP actually claims (executes) at least one node of ``model``."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_elt_probe"
    sess = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    profile_path = sess.end_profiling()
    try:
        events = json.load(open(profile_path))
    finally:
        os.remove(profile_path)
    providers = {
        e.get("args", {}).get("provider")
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" in providers, (
        f"MLX EP did not claim the node (ran on {providers or 'no EP'}); the CPU-match check would "
        "be vacuous"
    )


def _cpu_can_run(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        ort.InferenceSession(model, options, providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception:
        return False
    return True


def check(model: bytes, feeds: dict[str, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-6):
    """Verify MLX claims the node, then (if CPU has a kernel) that its output matches ORT CPU."""
    assert_mlx_claims(model, feeds)
    if _cpu_can_run(model, feeds):
        m.assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


# --- unary math / rounding / trig ---------------------------------------------------------------
_UNARY_INPUT = {
    "Ceil": [-2.7, -0.1, 0.9, 3.2],
    "Round": [-2.5, -0.5, 0.5, 2.5],  # ties-to-even
    "Reciprocal": [-4.0, -0.5, 0.25, 2.0],
    "Sign": [-3.0, -0.0, 0.5, 4.0],
    "Erf": [-2.0, -0.5, 0.5, 2.0],
    "Sin": [-2.0, -0.5, 0.5, 2.0],
    "Cos": [-2.0, -0.5, 0.5, 2.0],
    "Tan": [-1.0, -0.3, 0.3, 1.0],
    "Sinh": [-2.0, -0.5, 0.5, 2.0],
    "Cosh": [-2.0, -0.5, 0.5, 2.0],
    "Asin": [-0.9, -0.3, 0.3, 0.9],
    "Acos": [-0.9, -0.3, 0.3, 0.9],
    "Atan": [-2.0, -0.5, 0.5, 2.0],
    "Softplus": [-3.0, -0.5, 0.5, 3.0],
    "Softsign": [-3.0, -0.5, 0.5, 3.0],
}


@pytest.mark.parametrize("op", list(_UNARY_INPUT))
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_unary(op: str, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        op, [m.tensor("x", dtype, [2, 2])], [m.tensor("out", dtype, [2, 2])]
    )
    feeds = {"x": np.asarray(_UNARY_INPUT[op]).reshape(2, 2).astype(np_dtype)}
    check(model, feeds, rtol=tol, atol=tol)


# --- activations with attributes ----------------------------------------------------------------
ACT_X = np.array([[-3.0, -0.5, 0.0], [0.5, 2.0, 4.0]], dtype=np.float32)

ACT_CASES = [
    ("LeakyRelu", {}),
    ("LeakyRelu", {"alpha": 0.1}),
    ("Elu", {}),
    ("Elu", {"alpha": 1.5}),
    ("Selu", {}),
    ("Selu", {"alpha": 1.5, "gamma": 1.1}),
    ("Celu", {}),
    ("Celu", {"alpha": 0.7}),
    ("HardSigmoid", {}),
    ("HardSigmoid", {"alpha": 0.15, "beta": 0.4}),
    ("ThresholdedRelu", {}),
    ("ThresholdedRelu", {"alpha": 0.6}),
    ("Gelu", {}),
    ("Gelu", {"approximate": "tanh"}),
]


@pytest.mark.parametrize("op,attrs", ACT_CASES, ids=[f"{o}-{a}" for o, a in ACT_CASES])
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_activation(op: str, attrs: dict, dtype: DT, np_dtype, tol: float) -> None:
    # ONNX constrains Celu to float32 only; its fp16 graph is invalid regardless of EP.
    if op == "Celu" and dtype == DT.FLOAT16:
        pytest.skip("ONNX Celu is float32-only")
    # 'approximate' is a string attribute (not handled by _models._attr), so build the node directly.
    if "approximate" in attrs:
        import onnx_ir as ir

        node = ir.Node(
            "",
            op,
            [m.tensor("x", dtype, [2, 3])],
            attributes=[ir.AttrString("approximate", attrs["approximate"])],
            outputs=[m.tensor("out", dtype, [2, 3])],
        )
        graph = ir.Graph(
            list(node.inputs), list(node.outputs), nodes=[node], name="mlx_gelu",
            opset_imports={"": 24},
        )
        model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    else:
        model = m.make_model(
            op, [m.tensor("x", dtype, [2, 3])], [m.tensor("out", dtype, [2, 3])],
            attributes=attrs,
        )
    check(model, {"x": ACT_X.astype(np_dtype)}, rtol=tol, atol=max(tol, 1e-4))


def test_gelu_com_microsoft() -> None:
    model = m.make_model(
        "Gelu",
        [m.tensor("x", DT.FLOAT, [2, 3])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
        domain="com.microsoft",
    )
    check(model, {"x": ACT_X}, rtol=1e-4, atol=1e-4)


# --- Clip ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_clip_inputs(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Clip",
        [
            m.tensor("x", dtype, [2, 3]),
            m.tensor("min", dtype, []),
            m.tensor("max", dtype, []),
        ],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "x": ACT_X.astype(np_dtype),
        "min": np.array(-1.0, dtype=np_dtype),
        "max": np.array(1.5, dtype=np_dtype),
    }
    check(model, feeds, rtol=tol, atol=tol)


def test_clip_min_only() -> None:
    # Only the min input is provided; max is absent.
    model = m.make_model(
        "Clip",
        [m.tensor("x", DT.FLOAT, [2, 3]), m.tensor("min", DT.FLOAT, [])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
    )
    check(model, {"x": ACT_X, "min": np.array(0.0, dtype=np.float32)})


def test_clip_attr_opset10() -> None:
    # Opset<11: min/max are float attributes rather than inputs.
    model = m.make_model(
        "Clip",
        [m.tensor("x", DT.FLOAT, [2, 3])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
        attributes={"min": -1.0, "max": 1.5},
        opset=10,
    )
    check(model, {"x": ACT_X})


# --- variadic -----------------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["Max", "Min", "Sum", "Mean"])
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_variadic(op: str, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        op,
        [
            m.tensor("a", dtype, [2, 3]),
            m.tensor("b", dtype, [3]),
            m.tensor("c", dtype, [2, 3]),
        ],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, -2, 3], [4, 5, -6]], dtype=np_dtype),
        "b": np.array([0, -4, 2], dtype=np_dtype),
        "c": np.array([[2, -3, 1], [3, 7, -5]], dtype=np_dtype),
    }
    check(model, feeds, rtol=tol, atol=max(tol, 1e-4))


@pytest.mark.parametrize("op", ["Max", "Min", "Sum"])
def test_variadic_single_input(op: str) -> None:
    model = m.make_model(
        op, [m.tensor("a", DT.FLOAT, [2, 3])], [m.tensor("out", DT.FLOAT, [2, 3])]
    )
    check(model, {"a": ACT_X})


@pytest.mark.parametrize("op", ["Max", "Min"])
def test_variadic_int64(op: str) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", DT.INT64, [2, 3]), m.tensor("b", DT.INT64, [3])],
        [m.tensor("out", DT.INT64, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int64),
        "b": np.array([0, -4, 2], dtype=np.int64),
    }
    check(model, feeds, rtol=0, atol=0)


# --- comparisons (bool output) ------------------------------------------------------------------
@pytest.mark.parametrize(
    "op", ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"]
)
@pytest.mark.parametrize(
    "dtype,np_dtype",
    [(DT.FLOAT, np.float32), (DT.INT64, np.int64)],
    ids=["fp32", "int64"],
)
def test_comparison(op: str, dtype: DT, np_dtype) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", DT.BOOL, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
        "b": np.array([2, 2, 5], dtype=np_dtype),
    }
    check(model, feeds, rtol=0, atol=0)


def test_equal_bool() -> None:
    model = m.make_model(
        "Equal",
        [m.tensor("a", DT.BOOL, [4]), m.tensor("b", DT.BOOL, [4])],
        [m.tensor("out", DT.BOOL, [4])],
    )
    feeds = {
        "a": np.array([True, False, True, False]),
        "b": np.array([True, True, False, False]),
    }
    check(model, feeds, rtol=0, atol=0)


# --- logical ------------------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["And", "Or", "Xor"])
def test_logical_binary(op: str) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", DT.BOOL, [2, 3]), m.tensor("b", DT.BOOL, [3])],
        [m.tensor("out", DT.BOOL, [2, 3])],
    )
    feeds = {
        "a": np.array([[True, False, True], [False, True, False]]),
        "b": np.array([True, True, False]),
    }
    check(model, feeds, rtol=0, atol=0)


def test_logical_not() -> None:
    model = m.make_model(
        "Not", [m.tensor("x", DT.BOOL, [2, 3])], [m.tensor("out", DT.BOOL, [2, 3])]
    )
    check(model, {"x": np.array([[True, False, True], [False, True, False]])}, rtol=0, atol=0)


# --- Mod ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_mod_fmod_float(dtype: DT, np_dtype, tol: float) -> None:
    # fmod=1 (C fmod, sign of dividend) — the only valid Mod for floats.
    model = m.make_model(
        "Mod",
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", dtype, [2, 3])],
        attributes={"fmod": 1},
    )
    feeds = {
        "a": np.array([[5.3, -5.3, 5.3], [-5.3, 7.0, -7.0]], dtype=np_dtype),
        "b": np.array([2.0, 2.0, -2.0], dtype=np_dtype),
    }
    check(model, feeds, rtol=tol, atol=max(tol, 1e-3))


@pytest.mark.parametrize("fmod", [0, 1])
def test_mod_int(fmod: int) -> None:
    model = m.make_model(
        "Mod",
        [m.tensor("a", DT.INT64, [2, 3]), m.tensor("b", DT.INT64, [3])],
        [m.tensor("out", DT.INT64, [2, 3])],
        attributes={"fmod": fmod},
    )
    feeds = {
        "a": np.array([[7, -7, 7], [-7, 8, -8]], dtype=np.int64),
        "b": np.array([3, 3, -3], dtype=np.int64),
    }
    # fmod=1 on integers is left to CPU (float-only claim); fmod=0 is claimed.
    if fmod == 0:
        check(model, feeds, rtol=0, atol=0)
    else:
        m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


# --- BitShift -----------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
@pytest.mark.parametrize(
    "dtype,np_dtype", [(DT.UINT32, np.uint32), (DT.UINT8, np.uint8)], ids=["u32", "u8"]
)
def test_bitshift(direction: str, dtype: DT, np_dtype) -> None:
    # 'direction' is a required string attribute — build the node directly.
    import onnx_ir as ir

    node = ir.Node(
        "",
        "BitShift",
        [m.tensor("a", dtype, [4]), m.tensor("b", dtype, [4])],
        attributes=[ir.AttrString("direction", direction)],
        outputs=[m.tensor("out", dtype, [4])],
    )
    graph = ir.Graph(
        list(node.inputs), list(node.outputs), nodes=[node], name="mlx_bitshift",
        opset_imports={"": 24},
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    feeds = {
        "a": np.array([1, 2, 200, 7], dtype=np_dtype),
        "b": np.array([1, 3, 1, 2], dtype=np_dtype),
    }
    check(model, feeds, rtol=0, atol=0)


# --- Cast: integer<->float and integer<->integer conversions -------------------------------------
# ORT Cast to an integer type truncates toward zero; MLX `mlx_astype` does the same. int<->int and
# int->float conversions are exact for in-range values. NaN / out-of-range floats are undefined in
# ONNX and deliberately excluded from these inputs.
_INT_FLOAT_CAST_CASES = [
    # int -> float
    (DT.INT32, DT.FLOAT, np.array([[-5, -1, 0], [1, 7, 123456]], dtype=np.int32)),
    (DT.INT32, DT.FLOAT16, np.array([[-5, -1, 0], [1, 7, 2048]], dtype=np.int32)),
    (DT.INT64, DT.FLOAT, np.array([[-5, -1, 0], [1, 7, 123456]], dtype=np.int64)),
    (DT.INT64, DT.FLOAT16, np.array([[-5, -1, 0], [1, 7, 2048]], dtype=np.int64)),
    # float -> int (truncation toward zero, incl. negatives)
    (DT.FLOAT, DT.INT32, np.array([[-2.9, -0.5, 0.0], [0.5, 2.9, 100.7]], dtype=np.float32)),
    (DT.FLOAT, DT.INT64, np.array([[-2.9, -0.5, 0.0], [0.5, 2.9, 100.7]], dtype=np.float32)),
    (DT.FLOAT16, DT.INT32, np.array([[-2.9, -0.5, 0.0], [0.5, 2.9, 100.5]], dtype=np.float16)),
    (DT.FLOAT16, DT.INT64, np.array([[-2.9, -0.5, 0.0], [0.5, 2.9, 100.5]], dtype=np.float16)),
    # int <-> int
    (DT.INT32, DT.INT64, np.array([[-5, -1, 0], [1, 7, 2147483647]], dtype=np.int32)),
    (DT.INT64, DT.INT32, np.array([[-5, -1, 0], [1, 7, 2147483647]], dtype=np.int64)),
]


@pytest.mark.parametrize(
    "src,dst,x",
    _INT_FLOAT_CAST_CASES,
    ids=[f"{s.name}->{d.name}" for s, d, _ in _INT_FLOAT_CAST_CASES],
)
def test_cast_int_float(src: DT, dst: DT, x: np.ndarray) -> None:
    model = m.make_model(
        "Cast",
        [m.tensor("x", src, list(x.shape))],
        [m.tensor("out", dst, list(x.shape))],
        attributes={"to": int(dst)},
    )
    check(model, {"x": x}, rtol=0, atol=0)


# --- Add / Mul on int32 / int64 ------------------------------------------------------------------
# Element-wise integer Add/Mul, including broadcasting, negatives, and two's-complement overflow
# wraparound (MLX matches ORT CPU / numpy int semantics).
_INT_BINARY_CASES = [
    (DT.INT32, np.int32),
    (DT.INT64, np.int64),
]


@pytest.mark.parametrize("op", ["Add", "Mul"])
@pytest.mark.parametrize("dtype,np_dtype", _INT_BINARY_CASES, ids=["i32", "i64"])
def test_binary_int(op: str, dtype: DT, np_dtype) -> None:
    # (2,3) op (3,) — exercises trailing-suffix broadcasting with negative values.
    model = m.make_model(
        op,
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, -2, 3], [-4, 5, -6]], dtype=np_dtype),
        "b": np.array([2, -4, 7], dtype=np_dtype),
    }
    check(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["Add", "Mul"])
@pytest.mark.parametrize("dtype,np_dtype", _INT_BINARY_CASES, ids=["i32", "i64"])
def test_binary_int_overflow(op: str, dtype: DT, np_dtype) -> None:
    # Values chosen to overflow the dtype: MLX must wrap two's-complement exactly like ORT CPU.
    info = np.iinfo(np_dtype)
    if op == "Add":
        a = np.array([info.max, info.max, info.min], dtype=np_dtype)
        b = np.array([1, info.max, info.min], dtype=np_dtype)
    else:  # Mul
        a = np.array([info.max, info.min, info.max], dtype=np_dtype)
        b = np.array([2, 2, info.max], dtype=np_dtype)
    model = m.make_model(
        op,
        [m.tensor("a", dtype, [3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", dtype, [3])],
    )
    with np.errstate(over="ignore"):
        check(model, {"a": a, "b": b}, rtol=0, atol=0)
