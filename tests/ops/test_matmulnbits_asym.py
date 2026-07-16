"""MatMulNBits coverage for the MLX EP: the 3-input SYMMETRIC form (implicit zero point 8) across
the supported block sizes AND the 4-input ASYMMETRIC form with an explicit packed-int4 ``zero_points``
input (uint8).

Each case is proven to actually run on the MLX EP (ORT per-node profiling — a CPU fallback would make
the correctness check vacuous) and then compared, tolerance-gated, against ORT's CPU EP. If this ORT
build lacks the (4-input) MatMulNBits kernel, the case is skipped rather than failing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def _pack_int4_last(vals: np.ndarray) -> np.ndarray:
    """Pack the last axis of int4 values into uint8 (two nibbles per byte, low then high)."""
    lo = vals[..., 0::2]
    hi = vals[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def _matmulnbits_model(*, M: int, K: int, N: int, block: int, asymmetric: bool, dtype=DT.FLOAT):
    np_dt = np.float16 if dtype == DT.FLOAT16 else np.float32
    rng = np.random.default_rng(hash((M, K, N, block, asymmetric, int(dtype))) & 0xFFFFFFFF)
    n_blocks = (K + block - 1) // block
    a = rng.standard_normal((1, M, K)).astype(np_dt)
    # Quantized int4 weight [N, n_blocks, block] then packed to [N, n_blocks, block/2] uint8.
    qvals = rng.integers(0, 16, size=(N, n_blocks, block), dtype=np.uint8)
    b = _pack_int4_last(qvals)
    scales = (rng.standard_normal((N * n_blocks,)) * 0.05).astype(np_dt)

    inputs = [
        m.tensor("a", dtype, [1, M, K]),
        m.tensor("b", DT.UINT8, [N, n_blocks, block // 2]),
        m.tensor("scales", dtype, [N * n_blocks]),
    ]
    feeds = {"a": a, "b": b, "scales": scales}

    if asymmetric:
        cols = (n_blocks + 1) // 2
        zp_vals = rng.integers(0, 16, size=(N, n_blocks), dtype=np.uint8)
        # Pad each row to an even count before packing (a trailing padding nibble for odd n_blocks).
        if n_blocks % 2 == 1:
            zp_vals = np.concatenate([zp_vals, np.zeros((N, 1), dtype=np.uint8)], axis=1)
        zp_packed = _pack_int4_last(zp_vals).reshape(N * cols)
        inputs.append(m.tensor("zero_points", DT.UINT8, [N * cols]))
        feeds["zero_points"] = zp_packed

    outputs = [m.tensor("out", dtype, [1, M, N])]
    model = m.make_model(
        "MatMulNBits",
        inputs,
        outputs,
        domain="com.microsoft",
        attributes={"K": K, "N": N, "bits": 4, "block_size": block},
    )
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
    opts.profile_file_prefix = "mlx_nbits_probe"
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
        f"MLX EP did not claim MatMulNBits (ran on {providers or 'no EP'})"
    )


def _check(model: bytes, feeds: dict[str, np.ndarray], *, rtol=2e-3, atol=2e-3) -> None:
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks a MatMulNBits kernel for this form in this build")
    _assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    expected = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = m.run_mlx(model, feeds)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


# --- SYMMETRIC (3-input) across the supported block sizes ---------------------------------------
@pytest.mark.parametrize("block", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 8], ids=["decode", "prefill"])
def test_matmulnbits_symmetric(block: int, M: int) -> None:
    model, feeds = _matmulnbits_model(M=M, K=block * 2, N=32, block=block, asymmetric=False)
    _check(model, feeds)


# --- ASYMMETRIC (4-input packed int4 zero_points) -----------------------------------------------
@pytest.mark.parametrize("block", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 8], ids=["decode", "prefill"])
def test_matmulnbits_asymmetric(block: int, M: int) -> None:
    model, feeds = _matmulnbits_model(M=M, K=block * 3, N=16, block=block, asymmetric=True)
    _check(model, feeds)


# --- fp16 activation/scales/output (q4f16 export form, e.g. gemma-4-E2B decoder) -----------------
@pytest.mark.parametrize("block", [16, 32, 64, 128])
@pytest.mark.parametrize("M", [1, 8], ids=["decode", "prefill"])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_matmulnbits_fp16(block: int, M: int, asym: bool) -> None:
    model, feeds = _matmulnbits_model(
        M=M, K=block * 3, N=16, block=block, asymmetric=asym, dtype=DT.FLOAT16
    )
    # fp16 dequant + matmul accumulates in half — looser tolerance than the fp32 form.
    _check(model, feeds, rtol=6e-2, atol=6e-2)
