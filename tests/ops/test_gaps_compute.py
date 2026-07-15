"""MLX op-correctness tests for the dense-compute coverage gap: ``matmul.cc`` (MatMul, Gemm) and the
standalone ``RotaryEmbedding`` added to ``attention_ext.cc``.

Each case runs the single-node model through the ``MLXExecutionProvider`` and, unless a numpy
reference is used, compares tolerance-gated against ORT's CPU EP. ``assert_mlx_claims`` proves via
ORT per-node profiling that the MLX EP actually translated the op (so the CPU-match check is not a
vacuous CPU-fallback pass), mirroring ``test_attention_ext.py``.

Models are built with the ONNX IR through ``_models`` (which is not modified). ``RotaryEmbedding``
input ordering differs by domain: ai.onnx is ``[X, cos, sin, position_ids?]`` (opset 23), while
com.microsoft is ``[input, position_ids, cos, sin]``.
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

FLOAT = np.float32


# --- helpers ------------------------------------------------------------------------------------
def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the MLX EP actually claims (executes) at least one node of ``model``.

    ``m.assert_matches_cpu`` runs the MLX EP with CPU fallback, so a node the EP declines silently
    runs on CPU and the comparison passes vacuously. ORT per-node profiling confirms an
    ``MLXExecutionProvider`` node ran.
    """
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_gaps_probe"
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


def check(model: bytes, feeds: dict[str, np.ndarray], *, rtol: float = 2e-3, atol: float = 2e-3):
    """Verify MLX claims the node, then that its output matches ORT CPU (tolerance-gated)."""
    assert_mlx_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


def _cpu_can_run(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True when ORT's CPU EP can build+run this model (used to skip dtypes CPU lacks kernels for)."""
    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        ort.InferenceSession(model, options, providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception:
        return False
    return True


# --- MatMul -------------------------------------------------------------------------------------
# name, A shape, B shape (ONNX/numpy-matching batch broadcasting)
MATMUL_CASES = [
    ("2d", [8, 16], [16, 12]),
    ("2d-wide", [1, 64], [64, 32]),
    ("batched-3d", [4, 8, 16], [4, 16, 12]),
    ("broadcast-b-2d", [4, 8, 16], [16, 12]),
    ("broadcast-a-2d", [8, 16], [4, 16, 12]),
    ("batched-4d", [2, 3, 8, 16], [2, 3, 16, 12]),
    ("broadcast-4d", [2, 3, 8, 16], [16, 12]),
]


@pytest.mark.parametrize("case", MATMUL_CASES, ids=[c[0] for c in MATMUL_CASES])
def test_matmul_fp32(case: tuple) -> None:
    name, ashape, bshape = case
    rng = np.random.default_rng(abs(hash(("mm", name))) & 0xFFFFFFFF)
    a = rng.standard_normal(ashape).astype(FLOAT)
    b = rng.standard_normal(bshape).astype(FLOAT)
    out_shape = list(np.matmul(a, b).shape)
    model = m.make_model(
        "MatMul",
        [m.tensor("A", DT.FLOAT, ashape), m.tensor("B", DT.FLOAT, bshape)],
        [m.tensor("Y", DT.FLOAT, out_shape)],
    )
    check(model, {"A": a, "B": b})


def test_matmul_fp16() -> None:
    ashape, bshape = [6, 16], [16, 10]
    rng = np.random.default_rng(11)
    a = rng.standard_normal(ashape).astype(np.float16)
    b = rng.standard_normal(bshape).astype(np.float16)
    model = m.make_model(
        "MatMul",
        [m.tensor("A", DT.FLOAT16, ashape), m.tensor("B", DT.FLOAT16, bshape)],
        [m.tensor("Y", DT.FLOAT16, [6, 10])],
    )
    feeds = {"A": a, "B": b}
    if not _cpu_can_run(model, feeds):
        pytest.skip("ORT CPU has no fp16 MatMul kernel")
    check(model, feeds, rtol=5e-3, atol=5e-3)


def test_matmul_bf16_interior() -> None:
    """bf16 MatMul inside an MLX-claimed subgraph (fp32 boundaries), vs a numpy fp32 reference.

    bf16 keeps ~8 mantissa bits, and a K-wide dot product accumulates per-term rounding, so the
    tolerance is looser than the elementwise bf16 cases elsewhere (the point is to prove the op runs
    in bf16 on MLX and is approximately correct, not to bound accumulation error tightly).
    """
    ashape, bshape = [8, 16], [16, 12]
    rng = np.random.default_rng(12)
    a = rng.standard_normal(ashape).astype(FLOAT)
    b = rng.standard_normal(bshape).astype(FLOAT)
    model = m.bf16_interior_model("MatMul", [("A", ashape), ("B", bshape)], [8, 12])
    ref = [np.matmul(a, b)]
    m.assert_matches_ref(model, {"A": a, "B": b}, ref, rtol=5e-2, atol=1e-1)


# --- Gemm ---------------------------------------------------------------------------------------
# name, M, K, N, transA, transB, alpha, beta, c_shape(None -> no bias)
GEMM_CASES = [
    ("plain", 8, 16, 12, 0, 0, 1.0, 1.0, None),
    ("transA", 8, 16, 12, 1, 0, 1.0, 1.0, None),
    ("transB", 8, 16, 12, 0, 1, 1.0, 1.0, None),
    ("transAB", 8, 16, 12, 1, 1, 1.0, 1.0, None),
    ("alpha", 8, 16, 12, 0, 0, 0.5, 1.0, None),
    ("bias-mn", 8, 16, 12, 0, 0, 1.0, 1.0, [8, 12]),
    ("bias-vec", 8, 16, 12, 0, 0, 1.0, 1.0, [12]),
    ("alpha-beta-bias", 8, 16, 12, 0, 0, 0.75, 0.25, [8, 12]),
    ("transB-alpha-bias", 8, 16, 12, 0, 1, 2.0, 0.5, [12]),
]


@pytest.mark.parametrize("case", GEMM_CASES, ids=[c[0] for c in GEMM_CASES])
def test_gemm_fp32(case: tuple) -> None:
    name, M, K, N, transA, transB, alpha, beta, c_shape = case
    rng = np.random.default_rng(abs(hash(("gemm", name))) & 0xFFFFFFFF)
    a = rng.standard_normal([K, M] if transA else [M, K]).astype(FLOAT)
    b = rng.standard_normal([N, K] if transB else [K, N]).astype(FLOAT)
    inputs = [m.tensor("A", DT.FLOAT, list(a.shape)), m.tensor("B", DT.FLOAT, list(b.shape))]
    feeds = {"A": a, "B": b}
    if c_shape is not None:
        c = rng.standard_normal(c_shape).astype(FLOAT)
        inputs.append(m.tensor("C", DT.FLOAT, c_shape))
        feeds["C"] = c
    model = m.make_model(
        "Gemm",
        inputs,
        [m.tensor("Y", DT.FLOAT, [M, N])],
        attributes={"transA": transA, "transB": transB, "alpha": alpha, "beta": beta},
    )
    check(model, feeds)


def test_gemm_bf16_interior() -> None:
    M, K, N = 8, 16, 12
    rng = np.random.default_rng(21)
    a = rng.standard_normal([M, K]).astype(FLOAT)
    b = rng.standard_normal([K, N]).astype(FLOAT)
    c = rng.standard_normal([M, N]).astype(FLOAT)
    model = m.bf16_interior_model(
        "Gemm",
        [("A", [M, K]), ("B", [K, N]), ("C", [M, N])],
        [M, N],
        attributes={"alpha": 0.5, "beta": 0.5},
    )
    ref = [0.5 * (a @ b) + 0.5 * c]
    m.assert_matches_ref(model, {"A": a, "B": b, "C": c}, ref, rtol=5e-2, atol=1e-1)


# --- RotaryEmbedding ----------------------------------------------------------------------------
def _rope_ai_model(
    *, B, S, N, hd, max_seq, interleaved, rot_dim, layout, with_pos
) -> tuple[bytes, dict[str, np.ndarray]]:
    """ai.onnx opset-23 RotaryEmbedding: [X, cos, sin, position_ids?]."""
    rng = np.random.default_rng(abs(hash(("ai", B, S, N, hd, interleaved, rot_dim, layout))) & 0xFFFFFFFF)
    rot = rot_dim or hd
    half = rot // 2
    if layout == "4d":
        x = rng.standard_normal((B, N, S, hd)).astype(FLOAT)
        x_val = m.tensor("X", DT.FLOAT, [B, N, S, hd])
        out = m.tensor("Y", DT.FLOAT, [B, N, S, hd])
    else:
        x = rng.standard_normal((B, S, N * hd)).astype(FLOAT)
        x_val = m.tensor("X", DT.FLOAT, [B, S, N * hd])
        out = m.tensor("Y", DT.FLOAT, [B, S, N * hd])
    attrs = {"interleaved": interleaved}
    if layout != "4d":
        attrs["num_heads"] = N
    if rot_dim:
        attrs["rotary_embedding_dim"] = rot_dim
    feeds = {"X": x}
    if with_pos:
        cos, sin = m.rotary_caches(max_seq, rot)
        pos = np.tile(np.arange(S, dtype=np.int64), (B, 1))
        inputs = [
            x_val,
            m.tensor("cos", DT.FLOAT, [max_seq, half]),
            m.tensor("sin", DT.FLOAT, [max_seq, half]),
            m.tensor("pos", DT.INT64, [B, S]),
        ]
        feeds.update({"cos": cos, "sin": sin, "pos": pos})
    else:
        # Absent position_ids: caches are per-position [B, S, half].
        cos = rng.standard_normal((B, S, half)).astype(FLOAT)
        sin = rng.standard_normal((B, S, half)).astype(FLOAT)
        inputs = [
            x_val,
            m.tensor("cos", DT.FLOAT, [B, S, half]),
            m.tensor("sin", DT.FLOAT, [B, S, half]),
        ]
        feeds.update({"cos": cos, "sin": sin})
    model = m.make_model("RotaryEmbedding", inputs, [out], attributes=attrs, opset=23)
    return model, feeds


def _rope_ms_model(
    *, B, S, N, hd, max_seq, interleaved, rot_dim, layout, pos_offset
) -> tuple[bytes, dict[str, np.ndarray]]:
    """com.microsoft RotaryEmbedding: [input, position_ids, cos, sin]."""
    rng = np.random.default_rng(abs(hash(("ms", B, S, N, hd, interleaved, rot_dim, layout, pos_offset))) & 0xFFFFFFFF)
    rot = rot_dim or hd
    half = rot // 2
    if layout == "4d":
        x = rng.standard_normal((B, N, S, hd)).astype(FLOAT)
        x_val = m.tensor("input", DT.FLOAT, [B, N, S, hd])
        out = m.tensor("Y", DT.FLOAT, [B, N, S, hd])
    else:
        x = rng.standard_normal((B, S, N * hd)).astype(FLOAT)
        x_val = m.tensor("input", DT.FLOAT, [B, S, N * hd])
        out = m.tensor("Y", DT.FLOAT, [B, S, N * hd])
    cos, sin = m.rotary_caches(max_seq, rot)
    if pos_offset is None:
        pos = np.tile(np.arange(S, dtype=np.int64), (B, 1))
        pos_val = m.tensor("pos", DT.INT64, [B, S])
    else:
        pos = np.array([pos_offset], dtype=np.int64)  # [1] offset form
        pos_val = m.tensor("pos", DT.INT64, [1])
    attrs = {"interleaved": interleaved}
    if layout != "4d":
        attrs["num_heads"] = N
    if rot_dim:
        attrs["rotary_embedding_dim"] = rot_dim
    inputs = [
        x_val,
        pos_val,
        m.tensor("cos", DT.FLOAT, [max_seq, half]),
        m.tensor("sin", DT.FLOAT, [max_seq, half]),
    ]
    model = m.make_model("RotaryEmbedding", inputs, [out], domain="com.microsoft", attributes=attrs)
    return model, {"input": x, "pos": pos, "cos": cos, "sin": sin}


# name, B, S, N, hd, interleaved, rot_dim(0=full), layout, with_pos
ROPE_AI_CASES = [
    ("3d-gather", 1, 6, 4, 16, 0, 0, "3d", True),
    ("3d-interleaved", 1, 6, 4, 16, 1, 0, "3d", True),
    ("4d-gather", 1, 5, 3, 16, 0, 0, "4d", True),
    ("4d-interleaved", 1, 5, 3, 16, 1, 0, "4d", True),
    ("3d-partial", 1, 4, 2, 16, 0, 8, "3d", True),
    ("3d-batch2", 2, 4, 4, 16, 0, 0, "3d", True),
    ("3d-absent-pos", 1, 5, 4, 16, 0, 0, "3d", False),
    ("4d-absent-pos", 1, 5, 3, 16, 0, 0, "4d", False),
]


@pytest.mark.parametrize("case", ROPE_AI_CASES, ids=[c[0] for c in ROPE_AI_CASES])
def test_rotary_embedding_ai(case: tuple) -> None:
    name, B, S, N, hd, interleaved, rot_dim, layout, with_pos = case
    model, feeds = _rope_ai_model(
        B=B, S=S, N=N, hd=hd, max_seq=S + 8, interleaved=interleaved,
        rot_dim=rot_dim, layout=layout, with_pos=with_pos,
    )
    if not _cpu_can_run(model, feeds):
        pytest.skip("ORT CPU cannot run this ai.onnx RotaryEmbedding form")
    check(model, feeds)


# name, B, S, N, hd, interleaved, rot_dim(0=full), layout
ROPE_AI_NONCONTIG_CASES = [
    ("4d", 2, 3, 4, 8, 0, 0, "4d"),
    ("4d-interleaved", 2, 3, 4, 8, 1, 0, "4d"),
    ("4d-partial", 2, 3, 4, 8, 0, 4, "4d"),
    ("3d", 2, 3, 4, 8, 0, 0, "3d"),
]


@pytest.mark.parametrize("case", ROPE_AI_NONCONTIG_CASES, ids=[c[0] for c in ROPE_AI_NONCONTIG_CASES])
def test_rotary_embedding_ai_noncontiguous_pos(case: tuple) -> None:
    """Regression: [B,S] position_ids may carry arbitrary (non-contiguous) positions.

    The fused mlx_fast_rope path derives each position as ``offset + s`` (contiguous from the
    row's first id), which is wrong when position_ids are scrambled; the handler must fall back to
    the per-position cache gather. The stock ai.onnx backend cases only use ``arange(S)`` positions,
    so this case exercises the non-contiguous path explicitly.
    """
    name, B, S, N, hd, interleaved, rot_dim, layout = case
    max_seq = 50
    model, feeds = _rope_ai_model(
        B=B, S=S, N=N, hd=hd, max_seq=max_seq, interleaved=interleaved,
        rot_dim=rot_dim, layout=layout, with_pos=True,
    )
    rng = np.random.default_rng(1234 + interleaved * 7 + rot_dim)
    feeds["pos"] = np.stack([rng.permutation(max_seq)[:S] for _ in range(B)]).astype(np.int64)
    if not _cpu_can_run(model, feeds):
        pytest.skip("ORT CPU cannot run this ai.onnx RotaryEmbedding form")
    check(model, feeds)
ROPE_MS_CASES = [
    ("3d-gather", 1, 6, 4, 16, 0, 0, "3d", None),
    ("3d-interleaved", 1, 6, 4, 16, 1, 0, "3d", None),
    ("4d-gather", 1, 5, 3, 16, 0, 0, "4d", None),
    ("3d-partial", 1, 4, 2, 16, 0, 8, "3d", None),
    ("3d-offset", 1, 4, 4, 16, 0, 0, "3d", 3),
    ("4d-offset", 1, 4, 3, 16, 0, 0, "4d", 5),
]


@pytest.mark.parametrize("case", ROPE_MS_CASES, ids=[c[0] for c in ROPE_MS_CASES])
def test_rotary_embedding_ms(case: tuple) -> None:
    name, B, S, N, hd, interleaved, rot_dim, layout, pos_offset = case
    model, feeds = _rope_ms_model(
        B=B, S=S, N=N, hd=hd, max_seq=S + 16, interleaved=interleaved,
        rot_dim=rot_dim, layout=layout, pos_offset=pos_offset,
    )
    if not _cpu_can_run(model, feeds):
        pytest.skip("ORT CPU cannot run this com.microsoft RotaryEmbedding form")
    check(model, feeds)
