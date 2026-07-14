// Copyright (c) 2026. Licensed under the MIT License.
//
// Extended normalization op handlers: LayerNormalization (ai.onnx opset 17+),
// SimplifiedLayerNormalization + SkipLayerNormalization (com.microsoft), GroupNormalization,
// LpNormalization and BatchNormalization (inference form). See docs/OP_ARCHITECTURE.md §5/§6.
//
// LayerNormalization / SkipLayerNormalization use mlx_fast_layer_norm (last-axis normalization, eps
// folded in); SimplifiedLayerNormalization reuses mlx_fast_rms_norm. GroupNormalization,
// LpNormalization and BatchNormalization are composed from the reduction/elementwise primitives.
// Every handler honors the resolved input dtype (fp32/fp16/bf16) with no per-dtype branching.

#include <cmath>
#include <cstdint>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- small local MLX helpers (each Keep()s its result) --------------------------------------

mlx_array Sqrt(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_sqrt(&r, a, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Rsqrt(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_rsqrt(&r, a, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Abs(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_abs(&r, a, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Divide(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_divide(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array SumAxis(TranslationContext& ctx, mlx_array a, int axis, bool keepdims) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_sum_axis(&r, a, axis, keepdims, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array MeanAxis(TranslationContext& ctx, mlx_array a, int axis, bool keepdims) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_mean_axis(&r, a, axis, keepdims, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array VarAxis(TranslationContext& ctx, mlx_array a, int axis, bool keepdims) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_var_axis(&r, a, axis, keepdims, /*ddof=*/0, ctx.stream()));
  return ctx.Keep(r);
}
// A 0-d scalar of dtype `dt` holding `v` (eps constant), matching the compute dtype so no unwanted
// upcast occurs.
mlx_array ScalarLike(TranslationContext& ctx, float v, mlx_dtype dt) {
  mlx_array s = ctx.Keep(mlx_array_new_data(&v, nullptr, 0, MLX_FLOAT32));
  return ctx.Astype(s, dt);
}

// Reshape a per-channel vector [C] to [1, C, 1, ..., 1] so it broadcasts over N and spatial dims.
mlx_array ChannelBroadcast(TranslationContext& ctx, mlx_array v, int rank, int channels) {
  std::vector<int> shape(rank, 1);
  if (rank >= 2) shape[1] = channels;
  return ctx.Reshape(v, shape);
}

// ---- handlers -------------------------------------------------------------------------------

// LayerNormalization (ai.onnx opset 17+, last-axis form): Y = layer_norm(X, scale, bias, eps). Only
// the single-output (Y) form is claimed; Mean/InvStdDev extra outputs are left to CPU.
void LayerNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scale = ctx.Resolve(n.inputs[1]);
  mlx_array bias = mlx_array_empty;
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) bias = ctx.Resolve(n.inputs[2]);
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-5f;
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_fast_layer_norm(&r, x, scale, bias, eps, ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// SimplifiedLayerNormalization (com.microsoft): RMS normalization over the last axis (no mean
// subtraction): Y = rms_norm(X) * scale.
void SimplifiedLayerNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scale = ctx.Resolve(n.inputs[1]);
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-5f;
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_fast_rms_norm(&r, x, scale, eps, ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// SkipLayerNormalization (com.microsoft): residual = input + skip (+ bias); Y = layer_norm(residual,
// gamma, beta, eps). out[0]=Y; optional out[3]=residual sum. Mean/inv-std (out[1]/out[2]) unclaimed.
void SkipLayerNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array input = ctx.Resolve(n.inputs[0]);
  mlx_array skip = ctx.Resolve(n.inputs[1]);
  mlx_array gamma = ctx.Resolve(n.inputs[2]);
  mlx_array beta = mlx_array_empty;
  if (n.inputs.size() >= 4 && n.inputs[3].source != Src::Absent) beta = ctx.Resolve(n.inputs[3]);
  mlx_array residual = ctx.AddA(input, skip);
  if (n.inputs.size() >= 5 && n.inputs[4].source != Src::Absent) {
    residual = ctx.AddA(residual, ctx.Resolve(n.inputs[4]));
  }
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-5f;
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_fast_layer_norm(&r, residual, gamma, beta, eps, ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
  if (n.outputs.size() >= 4 && !n.outputs[3].name.empty()) ctx.Bind(n.outputs[3], residual);
}

// GroupNormalization (ai.onnx opset 21 form): normalize within each of `num_groups` channel groups,
// then apply per-channel scale/bias. X=[N,C,*S], scale/bias=[C].
void GroupNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scale = ctx.Resolve(n.inputs[1]);
  mlx_array bias = ctx.Resolve(n.inputs[2]);
  std::vector<int> shape = TranslationContext::ShapeOf(x);
  int rank = static_cast<int>(shape.size());
  int N = shape[0], C = shape[1];
  int groups = static_cast<int>(n.ints.at("num_groups"));
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-5f;

  int per_group = 1;
  for (int i = 1; i < rank; ++i) per_group *= shape[i];
  per_group /= groups;  // (C/groups) * prod(spatial)

  mlx_array grp = ctx.Reshape(x, {N, groups, per_group});
  mlx_array mean = MeanAxis(ctx, grp, /*axis=*/2, /*keepdims=*/true);
  mlx_array var = VarAxis(ctx, grp, /*axis=*/2, /*keepdims=*/true);
  mlx_array inv = Rsqrt(ctx, ctx.AddA(var, ScalarLike(ctx, eps, mlx_array_dtype(x))));
  mlx_array normed = ctx.Mul(ctx.SubA(grp, mean), inv);
  normed = ctx.Reshape(normed, shape);

  mlx_array sb = ChannelBroadcast(ctx, scale, rank, C);
  mlx_array bb = ChannelBroadcast(ctx, bias, rank, C);
  ctx.Bind(n.outputs[0], ctx.AddA(ctx.Mul(normed, sb), bb));
}

// LpNormalization (ai.onnx): Y = X / ||X||_p along `axis` (p in {1,2}, default 2; axis default -1).
void LpNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(x));
  int64_t axis = n.ints.count("axis") ? n.ints.at("axis") : -1;
  if (axis < 0) axis += rank;
  int64_t p = n.ints.count("p") ? n.ints.at("p") : 2;

  mlx_array norm;
  if (p == 1) {
    norm = SumAxis(ctx, Abs(ctx, x), static_cast<int>(axis), /*keepdims=*/true);
  } else {
    mlx_array sq = ctx.Mul(x, x);
    norm = Sqrt(ctx, SumAxis(ctx, sq, static_cast<int>(axis), /*keepdims=*/true));
  }
  ctx.Bind(n.outputs[0], Divide(ctx, x, norm));
}

// BatchNormalization (ai.onnx, inference/spatial form): Y = (X - mean)/sqrt(var+eps) * scale + B,
// per channel. Only the single-output inference form is claimed.
void BatchNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scale = ctx.Resolve(n.inputs[1]);
  mlx_array b = ctx.Resolve(n.inputs[2]);
  mlx_array mean = ctx.Resolve(n.inputs[3]);
  mlx_array var = ctx.Resolve(n.inputs[4]);
  std::vector<int> shape = TranslationContext::ShapeOf(x);
  int rank = static_cast<int>(shape.size());
  int C = rank >= 2 ? shape[1] : shape[0];
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-5f;

  mlx_array inv = Rsqrt(ctx, ctx.AddA(var, ScalarLike(ctx, eps, mlx_array_dtype(x))));
  mlx_array a = ctx.Mul(scale, inv);                // [C]
  mlx_array shift = ctx.SubA(b, ctx.Mul(mean, a));  // [C]
  mlx_array ab = ChannelBroadcast(ctx, a, rank, C);
  mlx_array shiftb = ChannelBroadcast(ctx, shift, rank, C);
  ctx.Bind(n.outputs[0], ctx.AddA(ctx.Mul(x, ab), shiftb));
}

// ---- claim predicates -----------------------------------------------------------------------

// LayerNormalization: fp32/fp16/bf16 X + scale (+ optional bias), last-axis (axis == -1 or rank-1),
// single output (Y). Extra Mean/InvStdDev outputs → CPU.
bool LayerNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, scale, out;
  std::vector<int64_t> xshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(inputs[1], scale) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  if (!IsMlxFloatType(x) || scale != x || out != x) return false;
  if (inputs.size() == 3 && SlotPresent(inputs, 2)) {
    ONNXTensorElementDataType bias;
    if (!TensorInfo(inputs[2], bias) || bias != x) return false;
  }
  int rank = static_cast<int>(xshape.size());
  int64_t axis = IntAttribute(node, "axis", -1);
  return rank > 0 && (axis == -1 || axis == rank - 1);
}

// SimplifiedLayerNormalization: X + scale, fp32/fp16/bf16, last-axis, single output.
bool SimplifiedLayerNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, scale, out;
  std::vector<int64_t> xshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(inputs[1], scale) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  if (!IsMlxFloatType(x) || scale != x || out != x) return false;
  int rank = static_cast<int>(xshape.size());
  int64_t axis = IntAttribute(node, "axis", -1);
  return rank > 0 && (axis == -1 || axis == rank - 1);
}

// SkipLayerNormalization: input, skip, gamma (+ optional beta, bias), all same float dtype. Only
// out[0] (Y) and optional out[3] (residual sum) are produced; mean/inv-std outputs → CPU.
bool SkipLayerNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 3 || inputs.size() > 5 || outputs.empty()) return false;
  ONNXTensorElementDataType x, out;
  if (!TensorInfo(inputs[0], x) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMlxFloatType(x) || out != x) return false;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!SlotPresent(inputs, i)) continue;  // omitted optional
    ONNXTensorElementDataType t;
    if (!TensorInfo(inputs[i], t) || t != x) return false;
  }
  // Reject if mean (out[1]) or inv-std (out[2]) are requested — we do not compute them.
  if (SlotPresent(outputs, 1)) return false;
  if (SlotPresent(outputs, 2)) return false;
  return true;
}

// GroupNormalization: X=[N,C,*S] float, scale/bias=[C], static C divisible by num_groups.
bool GroupNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, scale, bias, out;
  std::vector<int64_t> xshape, sshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(inputs[1], scale, &sshape) ||
      !TensorInfo(inputs[2], bias) || !TensorInfo(outputs[0], out)) {
    return false;
  }
  if (!IsMlxFloatType(x) || scale != x || bias != x || out != x) return false;
  if (xshape.size() < 2) return false;
  int64_t C = xshape[1];
  if (C <= 0) return false;
  for (int64_t d : xshape) {
    if (d <= 0) return false;  // need static dims to build the group reshape
  }
  int64_t groups = IntAttribute(node, "num_groups", 0);
  if (groups <= 0 || C % groups != 0) return false;
  // opset-21 per-channel scale/bias: shape [C].
  return sshape.size() == 1 && sshape[0] == C;
}

// LpNormalization: single float input/output, p in {1,2}.
bool LpNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> xshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMlxFloatType(x) || out != x || xshape.empty()) return false;
  int64_t p = IntAttribute(node, "p", 2);
  return p == 1 || p == 2;
}

// BatchNormalization: inference (single-output) form, 5 float inputs sharing X's dtype.
bool BatchNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 5 || outputs.size() != 1) return false;  // training outputs → CPU
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> xshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMlxFloatType(x) || out != x || xshape.size() < 2) return false;
  for (size_t i = 1; i < 5; ++i) {
    ONNXTensorElementDataType t;
    if (!TensorInfo(inputs[i], t) || t != x) return false;
  }
  if (IntAttribute(node, "training_mode", 0) != 0) return false;
  return true;
}

}  // namespace

void RegisterNormExtOps(OpRegistry& registry) {
  // LayerNormalization entered ai.onnx at opset 17.
  registry.Register({"", "LayerNormalization", 17, kAnyOpset, &LayerNormOp, &LayerNormClaim});
  registry.Register({"", "GroupNormalization", kAnyOpset, kAnyOpset, &GroupNormOp, &GroupNormClaim});
  registry.Register({"", "LpNormalization", kAnyOpset, kAnyOpset, &LpNormOp, &LpNormClaim});
  registry.Register({"", "BatchNormalization", kAnyOpset, kAnyOpset, &BatchNormOp, &BatchNormClaim});
  registry.Register({"com.microsoft", "SimplifiedLayerNormalization", kAnyOpset, kAnyOpset,
                     &SimplifiedLayerNormOp, &SimplifiedLayerNormClaim});
  registry.Register({"com.microsoft", "SkipLayerNormalization", kAnyOpset, kAnyOpset,
                     &SkipLayerNormOp, &SkipLayerNormClaim});
}

}  // namespace ort_mps_mlx
