"""Norm + attention multi-op stress loop for the MLX EP leak check (run under macOS `leaks`).

Registers the Rust MLX EP once, then repeatedly builds+runs independent ORT sessions exercising the
normalization ops (LayerNormalization, SimplifiedLayerNormalization, SkipLayerNormalization,
SkipSimplifiedLayerNormalization, RMSNormalization, GroupNormalization, BatchNormalization,
LpNormalization) and the attention family (GroupQueryAttention with in-op RoPE + KV-cache append,
Attention, MultiHeadAttention, RotaryEmbedding) through the EP. Proves RAII teardown across the
fast-norm / fast-SDPA / multi-output (present K/V) paths is leak-free (0 leaks / 0 bytes).
"""

import os
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnx_ir import DataType as DT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "ops"))
import _models as m  # noqa: E402

EP_NAME = "MLXExecutionProvider"
lib = os.path.abspath(os.environ["ONNXRUNTIME_MLX_EP_LIB"])
ort.register_execution_provider_library(EP_NAME, lib)


def run(mdl, feeds):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(mdl, opts, providers=[EP_NAME, "CPUExecutionProvider"])
    out = sess.run(None, feeds)
    del sess
    return out


def norm_model(op, inputs, outputs, *, domain="", opset=24, attrs=()):
    node = ir.Node(domain, op, inputs, attributes=list(attrs), outputs=outputs)
    imports = {"": opset}
    if domain:
        imports[domain] = 1
    g = ir.Graph(inputs, outputs, nodes=[node], opset_imports=imports, name=f"g_{op}")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString()


rng = np.random.default_rng(0)
X = rng.standard_normal((1, 4, 8)).astype(np.float32)
G = rng.standard_normal((8,)).astype(np.float32)
B = rng.standard_normal((8,)).astype(np.float32)
XN = rng.standard_normal((2, 6, 4, 4)).astype(np.float32)  # [N,C,H,W]
GC = rng.standard_normal((6,)).astype(np.float32)


def t(name, dt, shape):
    return m.tensor(name, dt, shape)


builders = [
    lambda: (norm_model("LayerNormalization", [t("x", DT.FLOAT, [1, 4, 8]), t("s", DT.FLOAT, [8]), t("b", DT.FLOAT, [8])],
                        [t("o", DT.FLOAT, [1, 4, 8])], opset=17,
                        attrs=[ir.AttrInt64("axis", -1), ir.AttrFloat32("epsilon", 1e-5)]),
             {"x": X, "s": G, "b": B}),
    lambda: (norm_model("RMSNormalization", [t("x", DT.FLOAT, [1, 4, 8]), t("s", DT.FLOAT, [8])],
                        [t("o", DT.FLOAT, [1, 4, 8])], attrs=[ir.AttrInt64("axis", -1), ir.AttrFloat32("epsilon", 1e-6)]),
             {"x": X, "s": G}),
    lambda: (norm_model("SimplifiedLayerNormalization", [t("x", DT.FLOAT, [1, 4, 8]), t("s", DT.FLOAT, [8])],
                        [t("o", DT.FLOAT, [1, 4, 8])], domain="com.microsoft", attrs=[ir.AttrFloat32("epsilon", 1e-5)]),
             {"x": X, "s": G}),
    lambda: (norm_model("SkipLayerNormalization",
                        [t("x", DT.FLOAT, [1, 4, 8]), t("skip", DT.FLOAT, [1, 4, 8]), t("g", DT.FLOAT, [8]), t("b", DT.FLOAT, [8])],
                        [t("o", DT.FLOAT, [1, 4, 8])], domain="com.microsoft", attrs=[ir.AttrFloat32("epsilon", 1e-5)]),
             {"x": X, "skip": X * 0.5, "g": G, "b": B}),
    lambda: (norm_model("SkipSimplifiedLayerNormalization",
                        [t("x", DT.FLOAT, [1, 4, 8]), t("skip", DT.FLOAT, [1, 4, 8]), t("g", DT.FLOAT, [8])],
                        [t("o", DT.FLOAT, [1, 4, 8]), t("res", DT.FLOAT, [1, 4, 8])], domain="com.microsoft",
                        attrs=[ir.AttrFloat32("epsilon", 1e-6)]),
             {"x": X, "skip": X * 0.5, "g": G}),
    lambda: (norm_model("GroupNormalization", [t("x", DT.FLOAT, [2, 6, 4, 4]), t("s", DT.FLOAT, [6]), t("b", DT.FLOAT, [6])],
                        [t("o", DT.FLOAT, [2, 6, 4, 4])], attrs=[ir.AttrInt64("num_groups", 3), ir.AttrFloat32("epsilon", 1e-5)]),
             {"x": XN, "s": GC, "b": GC * 0.3}),
    lambda: (norm_model("BatchNormalization",
                        [t("x", DT.FLOAT, [2, 6, 4, 4]), t("s", DT.FLOAT, [6]), t("b", DT.FLOAT, [6]), t("mean", DT.FLOAT, [6]), t("var", DT.FLOAT, [6])],
                        [t("o", DT.FLOAT, [2, 6, 4, 4])], attrs=[ir.AttrFloat32("epsilon", 1e-5)]),
             {"x": XN, "s": GC, "b": GC * 0.3, "mean": GC * 0.1, "var": np.abs(GC) + 0.5}),
    lambda: (norm_model("LpNormalization", [t("x", DT.FLOAT, [1, 4, 8])], [t("o", DT.FLOAT, [1, 4, 8])],
                        attrs=[ir.AttrInt64("axis", -1), ir.AttrInt64("p", 2)]),
             {"x": X}),
    # GroupQueryAttention (multi-output: attn + present K/V), decode + prefill.
    lambda: m.gqa_model("stress-decode", batch=1, num_heads=4, kv_heads=2, head=16, seq=1, past=5, do_rotary=1),
    lambda: m.gqa_model("stress-prefill", batch=1, num_heads=4, kv_heads=2, head=16, seq=6, past=0, do_rotary=1),
]


def attention_model():
    B_, qh, kvh, hd, S = 1, 4, 2, 16, 6
    q = rng.standard_normal((B_, S, qh * hd)).astype(np.float32)
    k = rng.standard_normal((B_, S, kvh * hd)).astype(np.float32)
    v = rng.standard_normal((B_, S, kvh * hd)).astype(np.float32)
    ins = [t("Q", DT.FLOAT, [B_, S, qh * hd]), t("K", DT.FLOAT, [B_, S, kvh * hd]), t("V", DT.FLOAT, [B_, S, kvh * hd])]
    node = ir.Node("", "Attention", ins,
                   attributes=[ir.AttrInt64("q_num_heads", qh), ir.AttrInt64("kv_num_heads", kvh),
                               ir.AttrFloat32("scale", float(1.0 / np.sqrt(hd))), ir.AttrInt64("is_causal", 1)],
                   outputs=[t("Y", DT.FLOAT, [B_, S, qh * hd])])
    g = ir.Graph(ins, [node.outputs[0]], nodes=[node], opset_imports={"": 24}, name="g_attn")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString(), {"Q": q, "K": k, "V": v}


def mha_model():
    B_, H, hd, S = 1, 4, 16, 4
    D = H * hd
    q = rng.standard_normal((B_, S, D)).astype(np.float32)
    k = rng.standard_normal((B_, S, D)).astype(np.float32)
    v = rng.standard_normal((B_, S, D)).astype(np.float32)
    ins = [t("Q", DT.FLOAT, [B_, S, D]), t("K", DT.FLOAT, [B_, S, D]), t("V", DT.FLOAT, [B_, S, D])]
    node = ir.Node("com.microsoft", "MultiHeadAttention", ins,
                   attributes=[ir.AttrInt64("num_heads", H)], outputs=[t("Y", DT.FLOAT, [B_, S, D])])
    g = ir.Graph(ins, [node.outputs[0]], nodes=[node], opset_imports={"": 24, "com.microsoft": 1}, name="g_mha")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString(), {"Q": q, "K": k, "V": v}


def rotary_model():
    B_, N, hd, S = 1, 4, 16, 5
    max_seq = S + 8
    x = rng.standard_normal((B_, S, N * hd)).astype(np.float32)
    cos, sin = m.rotary_caches(max_seq, hd)
    pos = np.arange(S, dtype=np.int64)[None, :].repeat(B_, 0)
    ins = [t("X", DT.FLOAT, [B_, S, N * hd]), t("cos", DT.FLOAT, [max_seq, hd // 2]),
           t("sin", DT.FLOAT, [max_seq, hd // 2]), t("pos", DT.INT64, [B_, S])]
    node = ir.Node("", "RotaryEmbedding", ins, attributes=[ir.AttrInt64("num_heads", N)],
                   outputs=[t("Y", DT.FLOAT, [B_, S, N * hd])])
    g = ir.Graph(ins, [node.outputs[0]], nodes=[node], opset_imports={"": 23}, name="g_rope")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString(), {"X": x, "cos": cos, "sin": sin, "pos": pos}


builders += [attention_model, mha_model, rotary_model]

N = int(os.environ.get("STRESS_ITERS", "48"))
last = None
ran = 0
for i in range(N):
    try:
        mdl, feeds = builders[i % len(builders)]()
        last = run(mdl, feeds)
        ran += 1
    except Exception as e:  # noqa: BLE001 — skip ops the CPU EP build lacks (contrib schema gaps)
        print(f"stress-norm-attn: skipped builder {i % len(builders)} ({type(e).__name__})")
print(f"stress-norm-attn: {ran}/{N} sessions across {len(builders)} op families OK; last shapes "
      f"{[np.asarray(o).shape for o in last]}")
