// Copyright (c) 2026. Licensed under the MIT License.
//
// Attention op handlers (GroupQueryAttention with in-op RoPE + KV-cache append). Layout matches
// com.microsoft.GroupQueryAttention (fp32, batch-first). The MLX path applies RoPE with the provided
// cos/sin cache, appends new K/V to the cache, and runs GQA causal SDPA.

#include <cmath>

#include "mlx_engine.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// Rotate-half (non-interleaved) or interleaved rotary over the first 2*half head dims of [B,H,S,hd].
mlx_array Rope(TranslationContext& ctx, mlx_array x, mlx_array cos, mlx_array sin, int half,
               bool interleaved) {
  std::vector<int> xs = TranslationContext::ShapeOf(x);  // [B,H,S,hd]
  const int hd = xs[3];
  const int rot = 2 * half;
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
    // cos/sin rows for positions [past, past+S).
    mlx_array cr = ctx.Reshape(ctx.Slice(cos, {past, 0}, {past + S, half}), {1, 1, S, half});
    mlx_array sr = ctx.Reshape(ctx.Slice(sin, {past, 0}, {past + S, half}), {1, 1, S, half});
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

}  // namespace

void RegisterAttentionOps(OpRegistry& registry) {
  registry.Register(
      {"com.microsoft", "GroupQueryAttention", kAnyOpset, kAnyOpset, &GroupQueryAttentionOp});
}

}  // namespace ort_mps_mlx
