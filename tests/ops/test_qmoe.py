"""QMoE (com.microsoft.QMoE) coverage for the MLX EP.

The EP implements the ``quant_type='int'`` subset: symmetric int4/int8 column- or block-wise expert
weights, top-k softmax routing, and SwiGLU (interleaved, ``swiglu_fusion=1``) / silu / gelu / relu /
identity activation. Each case is proven to actually run on the MLX EP (ORT per-node profiling — a
CPU fallback would make the correctness check vacuous) and then compared, tolerance-gated, against
ORT's CPU EP. If this ORT build lacks a QMoE CPU kernel, the case is skipped rather than failing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def _pack_int4_last(vals: np.ndarray) -> np.ndarray:
    """Pack the last axis of int4 values into uint8 (two nibbles per byte, low then high)."""
    lo = vals[..., 0::2]
    hi = vals[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def _absent() -> ir.Value:
    """An omitted optional node input (empty name — dropped from graph inputs by make_model)."""
    return ir.Value(name="")


def _qmoe_model(
    *,
    E: int,
    H: int,
    I: int,
    K: int,
    bits: int,
    act: str,
    fusion: int,
    block: int = 0,
    fc1b: bool = False,
    fc2b: bool = False,
    T: int = 3,
):
    rng = np.random.default_rng(hash((E, H, I, K, bits, act, fusion, block, fc1b, fc2b, T)) & 0xFFFFFFFF)
    pack = 8 // bits
    F = 2 if fusion > 0 else 1
    FI = F * I

    def qweight(rows_last_packed_shape, k_dim):
        vals = rng.integers(0, 2**bits, size=(*rows_last_packed_shape, k_dim), dtype=np.uint8)
        return _pack_int4_last(vals) if bits == 4 else vals.astype(np.uint8)

    fc1_w = qweight((E, FI), H)  # [E, FI, H/pack]
    fc2_w = qweight((E, H), I)  # [E, H, I/pack]
    if block > 0:
        fc1_s = (rng.random((E, FI, H // block)).astype(np.float32) * 0.2 + 0.02)
        fc2_s = (rng.random((E, H, I // block)).astype(np.float32) * 0.2 + 0.02)
    else:
        fc1_s = (rng.random((E, FI)).astype(np.float32) * 0.2 + 0.02)
        fc2_s = (rng.random((E, H)).astype(np.float32) * 0.2 + 0.02)

    pk_h = (H + pack - 1) // pack
    pk_i = (I + pack - 1) // pack
    inputs = [
        m.tensor("input", DT.FLOAT, [T, H]),
        m.tensor("router", DT.FLOAT, [T, E]),
        m.tensor("fc1w", DT.UINT8, [E, FI, pk_h]),
        m.tensor("fc1s", DT.FLOAT, list(fc1_s.shape)),
    ]
    feeds = {
        "fc1w": fc1_w,
        "fc1s": fc1_s,
        "fc2w": fc2_w,
        "fc2s": fc2_s,
    }
    inputs.append(m.tensor("fc1b", DT.FLOAT, [E, FI]) if fc1b else _absent())
    if fc1b:
        feeds["fc1b"] = (rng.random((E, FI)).astype(np.float32) - 0.5)
    inputs.append(m.tensor("fc2w", DT.UINT8, [E, H, pk_i]))
    inputs.append(m.tensor("fc2s", DT.FLOAT, list(fc2_s.shape)))
    if fc2b:
        inputs.append(m.tensor("fc2b", DT.FLOAT, [E, H]))
        feeds["fc2b"] = (rng.random((E, H)).astype(np.float32) - 0.5)

    attrs: dict[str, object] = {
        "k": K,
        "activation_type": act,
        "expert_weight_bits": bits,
        "normalize_routing_weights": 0,
    }
    if fusion > 0:
        attrs["swiglu_fusion"] = fusion
    if act == "swiglu":
        attrs["swiglu_limit"] = 7.0
        attrs["activation_alpha"] = 1.702
        attrs["activation_beta"] = 1.0
    if block > 0:
        attrs["block_size"] = block

    outputs = [m.tensor("output", DT.FLOAT, [T, H])]
    model = m.make_model("QMoE", inputs, outputs, domain="com.microsoft", attributes=attrs, opset=17)

    feeds["input"] = (rng.random((T, H)).astype(np.float32) * 2 - 1)
    feeds["router"] = (rng.random((T, E)).astype(np.float32) * 4 - 2)
    return model, feeds


def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


def _assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_qmoe_probe"
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
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
        f"MLX EP did not claim QMoE (ran on {providers or 'no EP'})"
    )


def _check(model: bytes, feeds: dict[str, np.ndarray], *, rtol=2e-3, atol=2e-3) -> None:
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks a QMoE kernel in this build")
    _assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    expected = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = m.run_mlx(model, feeds)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


# --- SwiGLU (gpt-oss form): interleaved fusion, int4, top-k -------------------------------------
@pytest.mark.parametrize("K", [1, 2])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_qmoe_swiglu_int4(K: int, bias: bool) -> None:
    model, feeds = _qmoe_model(
        E=4, H=8, I=16, K=K, bits=4, act="swiglu", fusion=1, fc1b=bias, fc2b=bias
    )
    _check(model, feeds)


# --- plain activations (silu/gelu/relu/identity), int4 ------------------------------------------
@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "identity"])
def test_qmoe_activation_int4(act: str) -> None:
    model, feeds = _qmoe_model(E=4, H=8, I=8, K=2, bits=4, act=act, fusion=0)
    _check(model, feeds)


# --- 8-bit weights (fp accumulation → looser tolerance) ----------------------------------------
@pytest.mark.parametrize("act,fusion", [("swiglu", 1), ("silu", 0)])
def test_qmoe_int8(act: str, fusion: int) -> None:
    model, feeds = _qmoe_model(E=4, H=8, I=8, K=2, bits=8, act=act, fusion=fusion, fc2b=True)
    _check(model, feeds, rtol=5e-3, atol=5e-3)


# --- block-wise scales (int4, block_size along K) ----------------------------------------------
@pytest.mark.parametrize("block", [16, 32])
def test_qmoe_blockwise_int4(block: int) -> None:
    model, feeds = _qmoe_model(
        E=2, H=block * 2, I=block * 2, K=1, bits=4, act="swiglu", fusion=1, block=block
    )
    _check(model, feeds)


# --- top-k = num_experts (all experts active) --------------------------------------------------
def test_qmoe_topk_all_experts() -> None:
    model, feeds = _qmoe_model(E=4, H=8, I=8, K=4, bits=4, act="silu", fusion=0)
    _check(model, feeds)
