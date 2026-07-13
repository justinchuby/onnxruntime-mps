// Copyright (c) 2026. Licensed under the MIT License.
//
// AttentionExt op handlers: the scaled-dot-product-attention family that maps onto the MLX fast SDPA
// primitive (mlx_fast_scaled_dot_product_attention). See docs/OP_ARCHITECTURE.md §2/§3.2/§6.
//
//   * Attention (ai.onnx, opset 23 and opset 24) — MHA / GQA / MQA scaled dot-product attention with
//     optional attn_mask, is_causal, custom scale, 3D (B,S,H*hd) or 4D (B,H,S,hd) layouts, and the
//     in-op past/present KV concat form. Opset 23 and 24 differ only by the trailing optional input
//     (#6 nonpad_kv_seqlen, opset 24); both share one impl and the claim rejects the nonpad form.
//   * MultiHeadAttention (com.microsoft) — separate Q/K/V (B,S,D) with optional projection bias,
//     unidirectional (causal), and custom scale.
//
// Every op honors the resolved input dtype (fp32/fp16/bf16). GQA head broadcast (q_num_heads a
// multiple of kv_num_heads) is handled inside MLX SDPA, so K/V are passed with their own head count.
//
// A structural constraint shapes what can be claimed: the ORT graph API represents an *omitted*
// optional input as a null OrtValueInfo. A node that supplies a *later* optional while leaving an
// earlier one empty therefore carries an interior null input, which the shared subgraph builder
// dereferences (it only tolerates trailing omissions). So any form that requires an interior optional
// gap is left on CPU — the trailing optionals a form uses must be contiguous from Q/K/V.
//
// Forms intentionally left on CPU (claim returns false so ORT keeps the node on CPU):
//   * Attention: qk_matmul_output (4th output), softcap, nonpad_kv_seqlen (opset-24 whole-cache
//     padding-optimization input), and the is_causal + explicit attn_mask combination (MLX fast SDPA
//     cannot take a causal mode and an array mask in one call).
//   * MultiHeadAttention: packed QKV (query as (B,S,N,3,H) / packed KV) — not expressible as a dense
//     [B,H,S,hd] SDPA and unsupported by the ORT CPU reference; key_padding_mask (#4), attention_bias
//     (#5), and past/present KV (#6/#7) — these sit behind optional slots whose real-world use leaves
//     an interior gap (see the structural constraint above), plus past_sequence_length (#8) and
//     cache_indirection (#9) beam-search plumbing.
//   * PackedMultiHeadAttention (com.microsoft) — NOT registered. Its packed variable-length layout
//     (token_offset + cumulative_sequence_length) collapses the batch of ragged sequences onto a
//     single token axis, which MLX fast SDPA (dense [B,H,S,hd]) cannot express; detecting the
//     single-sequence special case would need the runtime cumulative_sequence_length values, which
//     are not available at claim time. It is left entirely to CPU.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- shared helpers -------------------------------------------------------------------------

// A NodeDesc input slot is present when it exists and is not an omitted optional.
bool Present(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent;
}

// Claim-time: an ONNX node value slot is present when it exists, is a non-null value info, and
// carries a non-empty name. ORT can hand back a NULL OrtValueInfo for an omitted optional input, so
// the null guard MUST precede GetName() (which dereferences the handle).
bool ClaimSlotPresent(const std::vector<Ort::ConstValueInfo>& vals, size_t i) {
  if (i >= vals.size()) return false;
  if (static_cast<const OrtValueInfo*>(vals[i]) == nullptr) return false;
  return !vals[i].GetName().empty();
}

bool ClaimPresent(const std::vector<Ort::ConstValueInfo>& inputs, size_t i) {
  return ClaimSlotPresent(inputs, i);
}

// A declared output slot is present (requested) when it exists with a non-empty name.
bool ClaimOutPresent(const std::vector<Ort::ConstValueInfo>& outputs, size_t i) {
  return ClaimSlotPresent(outputs, i);
}

// [B,S,H*hd] -> [B,H,S,hd] (head-major split then transpose), matching the ONNX attention reshape.
mlx_array SplitHeads(TranslationContext& ctx, mlx_array x, int B, int S, int H, int hd) {
  return ctx.Transpose(ctx.Reshape(x, {B, S, H, hd}), {0, 2, 1, 3});
}

// Run MLX fast SDPA. mask_mode is "", "causal", or "array"; mask is used only for "array".
mlx_array Sdpa(TranslationContext& ctx, mlx_array q, mlx_array k, mlx_array v, float scale,
               const char* mask_mode, mlx_array mask) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_fast_scaled_dot_product_attention(&r, q, k, v, scale, mask_mode, mask,
                                                  /*sinks=*/mlx_array_empty, ctx.stream()));
  return ctx.Keep(r);
}

// Dispatch SDPA over the (mutually exclusive) causal / array-mask / no-mask cases. `mask` is the
// already-resolved attn_mask (only consulted when has_mask && !causal).
mlx_array SdpaDispatch(TranslationContext& ctx, mlx_array q, mlx_array k, mlx_array v, float scale,
                       bool causal, bool has_mask, mlx_array mask, mlx_dtype compute_dtype) {
  if (causal) {
    return Sdpa(ctx, q, k, v, scale, "causal", mlx_array_empty);
  }
  if (has_mask) {
    // Bool masks stay bool (True = attend); float additive masks are cast to the compute dtype.
    if (mlx_array_dtype(mask) != MLX_BOOL) mask = ctx.Astype(mask, compute_dtype);
    return Sdpa(ctx, q, k, v, scale, "array", mask);
  }
  return Sdpa(ctx, q, k, v, scale, "", mlx_array_empty);
}

// ---- Attention (ai.onnx) --------------------------------------------------------------------

// Q/K/V may be 3D (B,S,H*hd) or 4D (B,H,S,hd). Optional attn_mask (#3), past_key/past_value
// (#4/#5). Applies optional in-op KV cache concat, runs SDPA, restores the input layout.
void AttentionOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array q_in = ctx.Resolve(n.inputs[0]);
  mlx_array k_in = ctx.Resolve(n.inputs[1]);
  mlx_array v_in = ctx.Resolve(n.inputs[2]);
  const mlx_dtype dt = mlx_array_dtype(q_in);

  const std::vector<int> qs = TranslationContext::ShapeOf(q_in);
  const bool is3d = qs.size() == 3;
  const int B = qs[0];

  int qh, S, hd_v;
  mlx_array qh4, kh4, vh4;
  if (is3d) {
    qh = static_cast<int>(n.ints.at("q_num_heads"));
    const int kvh = static_cast<int>(n.ints.at("kv_num_heads"));
    S = qs[1];
    const std::vector<int> ks = TranslationContext::ShapeOf(k_in);
    const std::vector<int> vs = TranslationContext::ShapeOf(v_in);
    const int hd_q = qs[2] / qh;
    const int hd_k = ks[2] / kvh;
    hd_v = vs[2] / kvh;
    qh4 = SplitHeads(ctx, q_in, B, S, qh, hd_q);
    kh4 = SplitHeads(ctx, k_in, B, ks[1], kvh, hd_k);
    vh4 = SplitHeads(ctx, v_in, B, vs[1], kvh, hd_v);
  } else {
    qh = qs[1];
    S = qs[2];
    hd_v = TranslationContext::ShapeOf(v_in)[3];
    qh4 = q_in;
    kh4 = k_in;
    vh4 = v_in;
  }

  const bool has_past = Present(n, 4) && Present(n, 5);
  mlx_array present_k = kh4;
  mlx_array present_v = vh4;
  if (has_past) {
    present_k = ctx.Concat2(ctx.Resolve(n.inputs[4]), kh4, 2);
    present_v = ctx.Concat2(ctx.Resolve(n.inputs[5]), vh4, 2);
  }

  const int hd_q = TranslationContext::ShapeOf(qh4)[3];
  const float scale = (n.floats.count("scale") && n.floats.at("scale") != 0.0f)
                          ? n.floats.at("scale")
                          : 1.0f / std::sqrt(static_cast<float>(hd_q));
  const bool causal = n.ints.count("is_causal") && n.ints.at("is_causal") != 0;
  const bool has_mask = Present(n, 3);
  mlx_array mask = has_mask ? ctx.Resolve(n.inputs[3]) : mlx_array_empty;

  mlx_array attn = SdpaDispatch(ctx, qh4, present_k, present_v, scale, causal, has_mask, mask, dt);

  if (is3d) {
    // [B,qh,S,hd_v] -> [B,S,qh*hd_v].
    ctx.Bind(n.outputs[0], ctx.Reshape(ctx.Transpose(attn, {0, 2, 1, 3}), {B, S, qh * hd_v}));
  } else {
    ctx.Bind(n.outputs[0], attn);  // already [B,qh,S,hd_v]
  }
  if (has_past) {
    if (n.outputs.size() >= 2 && !n.outputs[1].name.empty()) ctx.Bind(n.outputs[1], present_k);
    if (n.outputs.size() >= 3 && !n.outputs[2].name.empty()) ctx.Bind(n.outputs[2], present_v);
  }
}

// ---- MultiHeadAttention (com.microsoft) -----------------------------------------------------

// Separate Q/K/V (B,S,D) with num_heads. Optional projection bias (#3, [Dq+Dk+Dv]). unidirectional
// -> causal. Mask and past/present KV forms are left on CPU (see the file header).
void MultiHeadAttentionOp(TranslationContext& ctx, const NodeDesc& n) {
  const int H = static_cast<int>(n.ints.at("num_heads"));
  mlx_array q_in = ctx.Resolve(n.inputs[0]);
  mlx_array k_in = ctx.Resolve(n.inputs[1]);
  mlx_array v_in = ctx.Resolve(n.inputs[2]);

  const std::vector<int> qs = TranslationContext::ShapeOf(q_in);
  const std::vector<int> ks = TranslationContext::ShapeOf(k_in);
  const std::vector<int> vs = TranslationContext::ShapeOf(v_in);
  const int B = qs[0], S = qs[1], Dq = qs[2];
  const int Lk = ks[1], Dk = ks[2];
  const int Lv = vs[1], Dv = vs[2];
  const int hd_q = Dq / H, hd_k = Dk / H, hd_v = Dv / H;

  if (Present(n, 3)) {
    mlx_array bias = ctx.Resolve(n.inputs[3]);  // 1D [Dq+Dk+Dv]
    q_in = ctx.AddA(q_in, ctx.Reshape(ctx.Slice(bias, {0}, {Dq}), {1, 1, Dq}));
    k_in = ctx.AddA(k_in, ctx.Reshape(ctx.Slice(bias, {Dq}, {Dq + Dk}), {1, 1, Dk}));
    v_in = ctx.AddA(v_in, ctx.Reshape(ctx.Slice(bias, {Dq + Dk}, {Dq + Dk + Dv}), {1, 1, Dv}));
  }

  mlx_array qh4 = SplitHeads(ctx, q_in, B, S, H, hd_q);
  mlx_array kh4 = SplitHeads(ctx, k_in, B, Lk, H, hd_k);
  mlx_array vh4 = SplitHeads(ctx, v_in, B, Lv, H, hd_v);

  const float scale = (n.floats.count("scale") && n.floats.at("scale") != 0.0f)
                          ? n.floats.at("scale")
                          : 1.0f / std::sqrt(static_cast<float>(hd_q));
  const bool causal = n.ints.count("unidirectional") && n.ints.at("unidirectional") != 0;

  mlx_array attn = SdpaDispatch(ctx, qh4, kh4, vh4, scale, causal, /*has_mask=*/false,
                                mlx_array_empty, mlx_array_dtype(q_in));

  // [B,H,S,hd_v] -> [B,S,H*hd_v].
  ctx.Bind(n.outputs[0], ctx.Reshape(ctx.Transpose(attn, {0, 2, 1, 3}), {B, S, H * hd_v}));
}

// ---- claim predicates -----------------------------------------------------------------------

// Shared checks for a scaled-dot-product-attention node: Q/K/V present and same MLX float dtype as
// the output, matching rank, and a valid head configuration. Fills qh/kvh (declared head counts).
bool CheckQkvFloat(const std::vector<Ort::ConstValueInfo>& inputs,
                   const std::vector<Ort::ConstValueInfo>& outputs, ONNXTensorElementDataType& qd) {
  if (inputs.size() < 3 || outputs.empty()) return false;
  ONNXTensorElementDataType kd, vd, od;
  if (!TensorInfo(inputs[0], qd) || !ClaimPresent(inputs, 1) || !ClaimPresent(inputs, 2) ||
      !TensorInfo(inputs[1], kd) || !TensorInfo(inputs[2], vd) || !TensorInfo(outputs[0], od)) {
    return false;
  }
  return IsMlxFloatType(qd) && kd == qd && vd == qd && od == qd;
}

// A past/present pair must be used together, share the query dtype, and present outputs (which the
// handler only produces from a KV concat) require the past inputs.
bool CheckKvCache(const std::vector<Ort::ConstValueInfo>& inputs,
                  const std::vector<Ort::ConstValueInfo>& outputs, size_t past_k, size_t past_v,
                  ONNXTensorElementDataType qd) {
  const bool pk = ClaimPresent(inputs, past_k);
  const bool pv = ClaimPresent(inputs, past_v);
  if (pk != pv) return false;
  if (pk) {
    ONNXTensorElementDataType a, b;
    if (!TensorInfo(inputs[past_k], a) || !TensorInfo(inputs[past_v], b) || a != qd || b != qd) {
      return false;
    }
  }
  if (!pk && (ClaimOutPresent(outputs, 1) || ClaimOutPresent(outputs, 2))) return false;
  return true;
}

// An attn/attention_bias mask must be bool or the query float dtype, and cannot co-exist with a
// causal flag (MLX fast SDPA takes either a causal mode or an array mask, not both).
bool CheckMask(const std::vector<Ort::ConstValueInfo>& inputs, size_t mask_idx, bool causal,
               ONNXTensorElementDataType qd) {
  if (!ClaimPresent(inputs, mask_idx)) return true;
  if (causal) return false;
  ONNXTensorElementDataType md;
  if (!TensorInfo(inputs[mask_idx], md)) return false;
  return md == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL || IsMlxFloatType(md);
}

// Attention (ai.onnx, opset 23 and 24). Claims the 3D (B,S,H*hd) and 4D (B,H,S,hd) SDPA forms with
// optional attn_mask and past/present KV; rejects softcap, the qk_matmul_output extra output, the
// opset-24 nonpad_kv_seqlen input, and the is_causal + attn_mask combination.
bool AttentionClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  ONNXTensorElementDataType qd;
  if (!CheckQkvFloat(inputs, outputs, qd)) return false;

  std::vector<int64_t> qshape, kshape, vshape;
  TensorInfo(inputs[0], qd, &qshape);
  ONNXTensorElementDataType tmp;
  TensorInfo(inputs[1], tmp, &kshape);
  TensorInfo(inputs[2], tmp, &vshape);
  const int rank = static_cast<int>(qshape.size());
  if (rank != 3 && rank != 4) return false;
  if (kshape.size() != qshape.size() || vshape.size() != qshape.size()) return false;

  int64_t qh, kvh;
  if (rank == 3) {
    qh = IntAttribute(node, "q_num_heads", 0);
    kvh = IntAttribute(node, "kv_num_heads", 0);
    if (qh <= 0 || kvh <= 0) return false;
    // Need static B/S/hidden for the [B,S,H,hd] reshape; hidden must divide evenly by heads.
    for (int64_t d : qshape) if (d <= 0) return false;
    if (kshape[1] <= 0 || vshape[1] <= 0 || kshape[2] <= 0 || vshape[2] <= 0) return false;
    if (qshape[2] % qh != 0 || kshape[2] % kvh != 0 || vshape[2] % kvh != 0) return false;
  } else {
    qh = qshape[1];
    kvh = kshape[1];
    if (qh <= 0 || kvh <= 0) return false;
  }
  if (qh % kvh != 0) return false;

  if (FloatAttribute(node, "softcap", 0.0f) != 0.0f) return false;  // logit soft-cap unsupported
  if (ClaimOutPresent(outputs, 3)) return false;                    // qk_matmul_output unsupported
  if (ClaimPresent(inputs, 6)) return false;                        // nonpad_kv_seqlen (opset 24)

  const bool causal = IntAttribute(node, "is_causal", 0) != 0;
  if (!CheckMask(inputs, 3, causal, qd)) return false;
  return CheckKvCache(inputs, outputs, 4, 5, qd);
}

// MultiHeadAttention (com.microsoft). Claims separate 3D Q/K/V with optional projection bias,
// num_heads, scale, and unidirectional (causal). Rejects packed QKV, key_padding_mask (#4),
// attention_bias (#5), past/present KV (#6/#7), the beam-search cache inputs (#8/#9), and any
// present_key/present_value/qk output — every masked or cached form leaves an interior optional gap
// that the subgraph builder cannot consume (see the file header), so those stay on CPU.
bool MultiHeadAttentionClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (IntAttribute(node, "num_heads", 0) <= 0) return false;
  ONNXTensorElementDataType qd;
  if (!CheckQkvFloat(inputs, outputs, qd)) return false;

  const int64_t H = IntAttribute(node, "num_heads", 0);
  std::vector<int64_t> qshape, kshape, vshape;
  ONNXTensorElementDataType tmp;
  TensorInfo(inputs[0], tmp, &qshape);
  TensorInfo(inputs[1], tmp, &kshape);
  TensorInfo(inputs[2], tmp, &vshape);
  // Separate Q/K/V must all be 3D (B,S,D); packed QKV / packed KV forms are left to CPU.
  if (qshape.size() != 3 || kshape.size() != 3 || vshape.size() != 3) return false;
  for (int64_t d : qshape) if (d <= 0) return false;
  for (int64_t d : kshape) if (d <= 0) return false;
  for (int64_t d : vshape) if (d <= 0) return false;
  if (qshape[2] % H != 0 || kshape[2] % H != 0 || vshape[2] % H != 0) return false;

  if (ClaimPresent(inputs, 3)) {  // bias: 1D [Dq+Dk+Dv]
    ONNXTensorElementDataType bd;
    std::vector<int64_t> bshape;
    if (!TensorInfo(inputs[3], bd, &bshape) || bd != qd) return false;
    if (bshape.size() != 1 || bshape[0] != qshape[2] + kshape[2] + vshape[2]) return false;
  }
  // key_padding_mask (#4), attention_bias (#5), past/present KV (#6/#7), past_sequence_length (#8),
  // cache_indirection (#9) -> CPU (they imply an interior optional gap and/or extra plumbing).
  for (size_t i = 4; i <= 9; ++i) {
    if (ClaimPresent(inputs, i)) return false;
  }
  // present_key/present_value (#1/#2) require the past inputs we do not claim; qk_matmul (#3) extra.
  if (ClaimOutPresent(outputs, 1) || ClaimOutPresent(outputs, 2) || ClaimOutPresent(outputs, 3)) {
    return false;
  }
  return true;
}

}  // namespace

void RegisterAttentionExtOps(OpRegistry& registry) {
  // Attention entered ai.onnx at opset 23; opset 24 adds the trailing optional nonpad_kv_seqlen
  // input. Both ranges share one handler/claim (the claim rejects the nonpad form), registered via
  // the opset seam per docs/OP_ARCHITECTURE.md §6.
  registry.Register({"", "Attention", 23, 23, &AttentionOp, &AttentionClaim});
  registry.Register({"", "Attention", 24, kAnyOpset, &AttentionOp, &AttentionClaim});
  registry.Register({"com.microsoft", "MultiHeadAttention", kAnyOpset, kAnyOpset,
                     &MultiHeadAttentionOp, &MultiHeadAttentionClaim});
}

}  // namespace ort_mps_mlx
