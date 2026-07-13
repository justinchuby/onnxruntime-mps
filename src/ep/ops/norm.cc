// Copyright (c) 2026. Licensed under the MIT License.
//
// Normalization op handlers (RMSNormalization, SkipSimplifiedLayerNormalization). Dtype-generic:
// mlx_fast_rms_norm runs in whatever float dtype the resolved input carries (fp32/fp16/bf16), so no
// per-dtype branches are needed.

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// RMSNormalization (ai.onnx, opset 23+): out = rms_norm(x) * scale over the last axis.
void RmsNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array g = ctx.Resolve(n.inputs[1]);
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-6f;
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_fast_rms_norm(&r, x, g, eps, ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// SkipSimplifiedLayerNormalization (com.microsoft): residual = input + skip;
// out = rms_norm(residual) * gamma. out[0]=normalized, out[last]=residual.
void SkipRmsNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array input = ctx.Resolve(n.inputs[0]);
  mlx_array skip = ctx.Resolve(n.inputs[1]);
  mlx_array gamma = ctx.Resolve(n.inputs[2]);
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-6f;
  mlx_array residual = ctx.AddA(input, skip);
  mlx_array norm = mlx_array_new();
  MLX_CHECK(mlx_fast_rms_norm(&norm, residual, gamma, eps, ctx.stream()));
  ctx.Keep(norm);
  ctx.Bind(n.outputs[0], norm);
  if (n.outputs.size() >= 2) ctx.Bind(n.outputs.back(), residual);
}

// ---- claim predicates (dtype/shape/attr checks; registry already matched domain/op/opset) -------

// RMSNormalization (ai.onnx): X, scale, axis == -1. fp32/fp16/bf16 (mlx_fast_rms_norm is generic).
bool RmsNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.empty()) return false;
  ONNXTensorElementDataType x, g, out;
  if (!TensorInfo(outputs[0], out) || !TensorInfo(inputs[0], x) || !TensorInfo(inputs[1], g)) {
    return false;
  }
  if (!IsMlxFloatType(x) || g != x || out != x) return false;
  return IntAttribute(node, "axis", -1) == -1;
}

// SkipSimplifiedLayerNormalization (com.microsoft): input, skip, gamma. fp32/fp16/bf16. No optional
// bias/beta in our graph (3-input form).
bool SkipRmsNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.empty()) return false;
  ONNXTensorElementDataType i0, i1, i2, out;
  if (!TensorInfo(outputs[0], out) || !TensorInfo(inputs[0], i0) || !TensorInfo(inputs[1], i1) ||
      !TensorInfo(inputs[2], i2)) {
    return false;
  }
  return IsMlxFloatType(i0) && i1 == i0 && i2 == i0 && out == i0;
}

}  // namespace

void RegisterNormOps(OpRegistry& registry) {
  // RMSNormalization entered ai.onnx at opset 23; register [23, kAnyOpset]. (This is the opset seam
  // in action — a future opset-N revision with different semantics registers a second handler for
  // [N, kAnyOpset] and narrows this one to [23, N-1].)
  registry.Register({"", "RMSNormalization", 23, kAnyOpset, &RmsNormOp, &RmsNormClaim});
  registry.Register({"com.microsoft", "SkipSimplifiedLayerNormalization", kAnyOpset, kAnyOpset,
                     &SkipRmsNormOp, &SkipRmsNormClaim});
}

}  // namespace ort_mps_mlx
