// Copyright (c) 2026. Licensed under the MIT License.
//
// Attention op handlers (GroupQueryAttention with in-op RoPE + KV-cache append). Layout matches
// com.microsoft.GroupQueryAttention (f32/f16/bf16, batch-first). The MLX path applies RoPE with the
// provided cos/sin cache, appends new K/V to the cache, and runs GQA causal SDPA.

#include <cmath>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// Rotate-half (non-interleaved) or interleaved rotary over the first 2*half head dims of [B,H,S,hd].
mlx_array Rope(TranslationContext& ctx, mlx_array x, mlx_array cos, mlx_array sin, int half,
               bool interleaved) {
  std::vector<int> xs = TranslationContext::ShapeOf(x);  // [B,H,S,hd]
  const int hd = xs[3];
  const int rot = 2 * half;
  if (ctx.RopeDynamic()) {
    // Compiled decode path: rotate-half via a constant [hd,hd] matmul (Slice has no shapeless
    // shape-inference). cos/sin here are the FULL-width [1,1,S,rot] rows. Requires rotary_dim==head_dim
    // (rot==hd); BuildCompiledClosure only compiles when that holds.
    mlx_array R = ctx.RotateHalfMatrix(hd, half);
    mlx_array xr = ctx.MatMul(x, R);  // == rotate_half(x): [-x2, x1]
    return ctx.AddA(ctx.Mul(x, cos), ctx.Mul(xr, sin));
  }
  mlx_array rotated;
  if (!interleaved) {
    mlx_array x1 = ctx.Slice(x, {0, 0, 0, 0}, {xs[0], xs[1], xs[2], half});
    mlx_array x2 = ctx.Slice(x, {0, 0, 0, half}, {xs[0], xs[1], xs[2], rot});
    mlx_array o1 = ctx.SubA(ctx.Mul(x1, cos), ctx.Mul(x2, sin));
    mlx_array o2 = ctx.AddA(ctx.Mul(x2, cos), ctx.Mul(x1, sin));
    rotated = ctx.Concat2(o1, o2, 3);
  } else {
    // interleaved: pairs (2i, 2i+1). Reshape to [...,half,2], split, recombine.
    mlx_array xr = ctx.Reshape(ctx.Slice(x, {0, 0, 0, 0}, {xs[0], xs[1], xs[2], rot}),
                               {xs[0], xs[1], xs[2], half, 2});
    mlx_array xe = ctx.Slice(xr, {0, 0, 0, 0, 0}, {xs[0], xs[1], xs[2], half, 1});
    mlx_array xo = ctx.Slice(xr, {0, 0, 0, 0, 1}, {xs[0], xs[1], xs[2], half, 2});
    mlx_array c = ctx.Reshape(cos, {xs[0], 1, xs[2], half, 1});  // broadcast on heads
    mlx_array sn = ctx.Reshape(sin, {xs[0], 1, xs[2], half, 1});
    xe = ctx.Reshape(xe, {xs[0], xs[1], xs[2], half, 1});
    xo = ctx.Reshape(xo, {xs[0], xs[1], xs[2], half, 1});
    mlx_array oe = ctx.SubA(ctx.Mul(xe, c), ctx.Mul(xo, sn));
    mlx_array oo = ctx.AddA(ctx.Mul(xo, c), ctx.Mul(xe, sn));
    rotated = ctx.Reshape(ctx.Concat2(oe, oo, 4), {xs[0], xs[1], xs[2], rot});
  }
  if (rot == hd) return rotated;
  mlx_array tail = ctx.Slice(x, {0, 0, 0, rot}, {xs[0], xs[1], xs[2], hd});
  return ctx.Concat2(rotated, tail, 3);
}

void GroupQueryAttentionOp(TranslationContext& ctx, const NodeDesc& n) {
  const int num_heads = static_cast<int>(n.ints.at("num_heads"));
  const int kv_heads = static_cast<int>(n.ints.at("kv_num_heads"));
  const bool interleaved = n.ints.count("rotary_interleaved") && n.ints.at("rotary_interleaved");
  const bool do_rotary = !n.ints.count("do_rotary") || n.ints.at("do_rotary");

  mlx_array q = ctx.Resolve(n.inputs[0]);       // [B,S,num*hd]
  mlx_array k = ctx.Resolve(n.inputs[1]);       // [B,S,kv*hd]
  mlx_array v = ctx.Resolve(n.inputs[2]);       // [B,S,kv*hd]
  mlx_array past_k = ctx.Resolve(n.inputs[3]);  // [B,kv,past,hd]
  mlx_array past_v = ctx.Resolve(n.inputs[4]);

  std::vector<int> qs = TranslationContext::ShapeOf(q);
  const int B = qs[0], S = qs[1];
  const int head = qs[2] / num_heads;
  const int past = TranslationContext::ShapeOf(past_k)[2];

  float scale = n.floats.count("scale") && n.floats.at("scale") != 0.0f
                    ? n.floats.at("scale")
                    : 1.0f / std::sqrt(static_cast<float>(head));

  // [B,S,H*hd] -> [B,H,S,hd].
  auto to_heads = [&](mlx_array x, int h) {
    return ctx.Transpose(ctx.Reshape(x, {B, S, h, head}), {0, 2, 1, 3});
  };
  mlx_array qh = to_heads(q, num_heads);
  mlx_array kh = to_heads(k, kv_heads);
  mlx_array vh = to_heads(v, kv_heads);

  if (do_rotary) {
    mlx_array cos = ctx.Resolve(n.inputs[7]);  // [max_seq, rot/2]
    mlx_array sin = ctx.Resolve(n.inputs[8]);
    const int half = TranslationContext::ShapeOf(cos)[1];  // rot/2
    // cos/sin rows for positions [past, past+S). Static slice on the eager path; runtime dynamic
    // slice on the compiled decode path (so the position offset is not baked into the compiled graph).
    mlx_array cr = ctx.CosSinRow(n.inputs[7].name, cos, past, S, half);
    mlx_array sr = ctx.CosSinRow(n.inputs[8].name, sin, past, S, half);
    qh = Rope(ctx, qh, cr, sr, half, interleaved);
    kh = Rope(ctx, kh, cr, sr, half, interleaved);
  }

  // Append to KV cache along the sequence axis.
  mlx_array present_k = ctx.Concat2(past_k, kh, 2);
  mlx_array present_v = ctx.Concat2(past_v, vh, 2);

  mlx_array attn = mlx_array_new();
  MLX_CHECK(mlx_fast_scaled_dot_product_attention(&attn, qh, present_k, present_v, scale, "causal",
                                                  /*mask=*/mlx_array_empty,
                                                  /*sinks=*/mlx_array_empty, ctx.stream()));
  ctx.Keep(attn);
  // [B,H,S,hd] -> [B,S,H*hd].
  mlx_array out = ctx.Reshape(ctx.Transpose(attn, {0, 2, 1, 3}), {B, S, num_heads * head});

  ctx.Bind(n.outputs[0], out);
  if (n.outputs.size() >= 2) ctx.Bind(n.outputs[1], present_k);
  if (n.outputs.size() >= 3) ctx.Bind(n.outputs[2], present_v);
}

// ---- claim predicate (dtype/shape/attr checks; registry already matched domain/op/opset) --------

// GroupQueryAttention (com.microsoft): MLX float Q/K/V + past/present K/V share-buffer and rotary
// caches. All floating-point inputs and outputs must use one dtype.
// We claim the standard separate-QKV, 9-input decode/prefill layout used by our Qwen graph.
bool GroupQueryAttentionClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) return false;
  ONNXTensorElementDataType out_type;
  if (!TensorInfo(outputs[0], out_type) || !IsMlxFloatType(out_type)) {
    return false;
  }
  if (inputs.size() != 9) return false;  // q,k,v,past_k,past_v,seqlens_k,total_seq,cos,sin
  const int float_inputs[] = {0, 1, 2, 3, 4, 7, 8};
  for (int idx : float_inputs) {
    ONNXTensorElementDataType t;
    if (!TensorInfo(inputs[idx], t) || t != out_type) return false;
  }
  for (size_t idx = 1; idx < outputs.size() && idx < 3; ++idx) {
    ONNXTensorElementDataType t;
    if (!TensorInfo(outputs[idx], t) || t != out_type) return false;
  }
  ONNXTensorElementDataType st, tt;
  if (!TensorInfo(inputs[5], st) || st != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) return false;
  if (!TensorInfo(inputs[6], tt) || tt != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) return false;
  const int64_t nh = IntAttribute(node, "num_heads", 0);
  const int64_t kvh = IntAttribute(node, "kv_num_heads", 0);
  if (nh <= 0 || kvh <= 0 || nh % kvh != 0) return false;
  // Unsupported (rare) variants fall back to CPU. ORT materializes schema defaults, so "disabled"
  // surfaces as smooth_softmax == -1 (not 0); only a genuine enable (== 1) is rejected.
  if (IntAttribute(node, "smooth_softmax", 0) == 1) return false;
  if (IntAttribute(node, "qk_output", 0) != 0) return false;       // QK intermediate output unsupported
  if (FloatAttribute(node, "softcap", 0.0f) != 0.0f) return false;  // attention logit soft-cap unsupported
  return true;
}

}  // namespace

void RegisterAttentionOps(OpRegistry& registry) {
  registry.Register({"com.microsoft", "GroupQueryAttention", kAnyOpset, kAnyOpset,
                     &GroupQueryAttentionOp, &GroupQueryAttentionClaim});
}

}  // namespace ort_mlx
