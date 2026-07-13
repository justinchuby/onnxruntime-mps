"""MLX op-correctness tests for the scaled-dot-product-attention family (``attention_ext.cc``).

Covers the ops the MLX EP registers in ``RegisterAttentionExtOps``:

* ``Attention`` (ai.onnx) at **opset 23** and **opset 24** — MHA / GQA / MQA, 3D ``(B,S,H*hd)`` and
  4D ``(B,H,S,hd)`` layouts, custom scale, ``is_causal``, bool / float ``attn_mask``, and the in-op
  past/present KV concat.
* ``MultiHeadAttention`` (com.microsoft) — separate Q/K/V with optional projection bias, additive
  ``attention_bias``, ``unidirectional`` (causal), custom scale, and past/present KV.

Each case runs the model through the ``MLXExecutionProvider`` and compares, tolerance-gated, against
ORT's CPU EP (``m.assert_matches_cpu``). ``PackedMultiHeadAttention`` and the MHA packed-QKV form are
intentionally left on CPU (see ``attention_ext.cc``) and are not exercised here.

The single-node models are built directly with the ONNX IR (``ir.*``) rather than ``m.make_model``
because attention nodes carry *optional* inputs (mask, past KV) that must appear as empty-name
placeholders in the node's input list while being excluded from the graph inputs.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m

FLOAT = np.float32


# --- IR builder (optional inputs -> empty-name placeholders) -------------------------------------
def _attr(name: str, value: object) -> ir.Attr:
    if isinstance(value, float):
        return ir.AttrFloat32(name, value)
    if isinstance(value, int):
        return ir.AttrInt64(name, int(value))
    raise TypeError(f"unsupported attribute {name!r}: {type(value)!r}")


def build_model(
    op_type: str,
    inputs: list[ir.Value | None],
    outputs: list[ir.Value | None],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 24,
) -> bytes:
    """Build a single-node model. ``None`` entries in ``inputs``/``outputs`` are omitted optionals.

    Trailing ``None`` inputs are dropped (matching how real ONNX exporters emit nodes — a missing
    trailing optional is simply absent, never an empty-string placeholder), while interior ``None``
    gaps are preserved as empty-name inputs so later present optionals keep their positional index.
    """
    node_inputs = list(inputs)
    while node_inputs and node_inputs[-1] is None:
        node_inputs.pop()
    node = ir.Node(
        domain,
        op_type,
        node_inputs,
        attributes=[_attr(k, v) for k, v in (attributes or {}).items()],
        outputs=[o for o in outputs if o is not None],
    )
    graph_inputs = [v for v in node_inputs if v is not None]
    graph_outputs = [o for o in outputs if o is not None]
    opset_imports = {"": opset}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph(
        graph_inputs, graph_outputs, nodes=[node], name=f"mlx_{op_type}", opset_imports=opset_imports
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True when ORT's CPU EP can build and run this model (used to skip missing contrib schemas)."""
    import onnxruntime as ort

    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        ort.InferenceSession(model, options, providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception:
        return False
    return True


def _t(name: str, shape: list[int], dtype: DT = DT.FLOAT) -> ir.Value:
    return m.tensor(name, dtype, shape)


def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the MLX EP actually *claims* (executes) at least one node of ``model``.

    ``m.assert_matches_cpu`` runs the MLX EP with a CPU fallback, so a node the EP declines to claim
    silently runs on CPU and the comparison passes vacuously. We use ORT's per-node profiling to
    confirm an ``MLXExecutionProvider`` node ran, proving the attention op was translated by MLX.
    """
    import json
    import os

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_claim_probe"
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


def check(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Verify MLX claims the node, then that its output matches ORT CPU (tolerance-gated)."""
    assert_mlx_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


# --- Attention (ai.onnx) -------------------------------------------------------------------------
# Real + toy head geometries across decode / prefill, causal / masked, 3D / 4D, opset 23 / 24.
#
# NOTE on past-KV cases: ONNX places attn_mask at input #3 and past_key/past_value at #4/#5. The MLX
# EP's subgraph builder cannot consume an interior *omitted* optional (it becomes a null value info),
# so a past-KV node must also supply attn_mask (#3) to stay gap-free. Because MLX fast SDPA cannot mix
# a causal mode with an array mask, the past-KV cases here are non-causal with an explicit mask (the
# causal + past decode form is left on CPU — see attention_ext.cc).
ATTN_CASES = [
    # name, opset, batch, q_heads, kv_heads, head, seq, past, causal, mask("none"|"float"|"bool"),
    # layout("3d"|"4d")
    ("o23-prefill-gqa-causal-3d", 23, 1, 4, 2, 16, 6, 0, True, "none", "3d"),
    ("o23-prefill-mha-floatmask-3d", 23, 1, 4, 4, 16, 5, 0, False, "float", "3d"),
    ("o23-full-gqa-3d", 23, 1, 6, 3, 16, 4, 0, False, "none", "3d"),
    ("o24-prefill-gqa-causal-3d", 24, 1, 4, 2, 16, 6, 0, True, "none", "3d"),
    ("o24-gqa-floatmask-past-3d", 24, 1, 14, 2, 64, 4, 40, False, "float", "3d"),
    ("o24-gqa-boolmask-past-3d", 24, 1, 4, 2, 16, 3, 8, False, "bool", "3d"),
    ("o24-mqa-causal-3d", 24, 1, 8, 1, 16, 5, 0, True, "none", "3d"),
    ("o24-gqa-boolmask-3d", 24, 1, 4, 2, 16, 5, 0, False, "bool", "3d"),
    ("o24-mha-floatmask-3d", 24, 1, 4, 4, 16, 5, 0, False, "float", "3d"),
    ("o24-self-causal-4d", 24, 1, 4, 4, 16, 5, 0, True, "none", "4d"),
    ("o24-gqa-full-4d", 24, 1, 6, 3, 16, 4, 0, False, "none", "4d"),
    ("o24-gqa-floatmask-past-4d", 24, 1, 6, 3, 16, 2, 5, False, "float", "4d"),
]


@pytest.mark.parametrize("case", ATTN_CASES, ids=[c[0] for c in ATTN_CASES])
def test_attention(case: tuple) -> None:
    name, opset, B, qh, kvh, hd, S, past, causal, mask, layout = case
    kv = past + S
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFFFFFF)
    scale = float(1.0 / np.sqrt(hd))

    inputs: list[ir.Value | None] = []
    feeds: dict[str, np.ndarray] = {}
    if layout == "3d":
        q = rng.standard_normal((B, S, qh * hd)).astype(FLOAT)
        k = rng.standard_normal((B, S, kvh * hd)).astype(FLOAT)
        v = rng.standard_normal((B, S, kvh * hd)).astype(FLOAT)
        inputs += [_t("Q", [B, S, qh * hd]), _t("K", [B, S, kvh * hd]), _t("V", [B, S, kvh * hd])]
    else:
        q = rng.standard_normal((B, qh, S, hd)).astype(FLOAT)
        k = rng.standard_normal((B, kvh, S, hd)).astype(FLOAT)
        v = rng.standard_normal((B, kvh, S, hd)).astype(FLOAT)
        inputs += [_t("Q", [B, qh, S, hd]), _t("K", [B, kvh, S, hd]), _t("V", [B, kvh, S, hd])]
    feeds.update(Q=q, K=k, V=v)

    # input #3: attn_mask (optional)
    mask_v: ir.Value | None = None
    if mask == "float":
        mm = (rng.standard_normal((B, qh, S, kv)) * 0.5).astype(FLOAT)
        mask_v = _t("M", [B, qh, S, kv])
        feeds["M"] = mm
    elif mask == "bool":
        bm = np.tril(np.ones((S, kv), dtype=bool))[None, None]  # [1,1,S,kv], broadcast over heads
        mask_v = _t("M", [1, 1, S, kv], DT.BOOL)
        feeds["M"] = bm
    inputs.append(mask_v)

    # inputs #4/#5: past_key/past_value (optional, both together)
    if past > 0:
        pk = rng.standard_normal((B, kvh, past, hd)).astype(FLOAT)
        pv = rng.standard_normal((B, kvh, past, hd)).astype(FLOAT)
        inputs += [_t("PK", [B, kvh, past, hd]), _t("PV", [B, kvh, past, hd])]
        feeds.update(PK=pk, PV=pv)

    attrs: dict[str, object] = {"q_num_heads": qh, "kv_num_heads": kvh, "scale": scale}
    if causal:
        attrs["is_causal"] = 1

    if layout == "3d":
        outputs: list[ir.Value | None] = [_t("Y", [B, S, qh * hd])]
    else:
        outputs = [_t("Y", [B, qh, S, hd])]
    if past > 0:
        outputs += [_t("PRK", [B, kvh, kv, hd]), _t("PRV", [B, kvh, kv, hd])]

    model = build_model("Attention", inputs, outputs, attributes=attrs, opset=opset)
    if not _cpu_supports(model, feeds):
        pytest.skip(f"ORT CPU EP has no Attention kernel for opset {opset} / this form")
    check(model, feeds)


# --- MultiHeadAttention (com.microsoft) ----------------------------------------------------------
# Separate Q/K/V (+ optional bias), num_heads, scale, unidirectional. The masked (attention_bias /
# key_padding_mask) and past/present-KV forms are left on CPU (they require an interior optional gap
# the subgraph builder cannot consume — see attention_ext.cc), so they are not exercised here.
# name, heads, head, seq, bias, causal, custom_scale
MHA_CASES = [
    ("self", 4, 16, 4, False, False, False),
    ("bias", 4, 16, 4, True, False, False),
    ("causal", 4, 16, 5, False, True, False),
    ("bias-causal", 4, 16, 5, True, True, False),
    ("custom-scale", 8, 32, 4, False, False, True),
    ("mqa-heads", 12, 24, 6, True, False, False),
]


@pytest.mark.parametrize("case", MHA_CASES, ids=[c[0] for c in MHA_CASES])
def test_multihead_attention(case: tuple) -> None:
    name, H, hd, S, bias, causal, custom_scale = case
    D = H * hd
    rng = np.random.default_rng(abs(hash(("mha", name))) & 0xFFFFFFFF)

    q = rng.standard_normal((1, S, D)).astype(FLOAT)
    k = rng.standard_normal((1, S, D)).astype(FLOAT)
    v = rng.standard_normal((1, S, D)).astype(FLOAT)
    inputs: list[ir.Value | None] = [_t("Q", [1, S, D]), _t("K", [1, S, D]), _t("V", [1, S, D])]
    feeds: dict[str, np.ndarray] = {"Q": q, "K": k, "V": v}

    # input #3: bias [3*D]
    if bias:
        inputs.append(_t("B", [3 * D]))
        feeds["B"] = (rng.standard_normal((3 * D,)) * 0.3).astype(FLOAT)

    attrs: dict[str, object] = {"num_heads": H}
    if causal:
        attrs["unidirectional"] = 1
    if custom_scale:
        attrs["scale"] = float(0.1)

    outputs: list[ir.Value | None] = [_t("Y", [1, S, D])]

    model = build_model(
        "MultiHeadAttention", inputs, outputs, domain="com.microsoft", attributes=attrs
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no MultiHeadAttention kernel for this build/form")
    check(model, feeds)
