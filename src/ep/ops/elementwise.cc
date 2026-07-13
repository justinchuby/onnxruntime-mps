// Copyright (c) 2026. Licensed under the MIT License.
//
// Elementwise + activation + cast op handlers. Every handler is dtype-generic: it resolves each
// input to an MLX array wrapped with its ACTUAL dtype (MlxDtypeFromOnnx) and MLX carries that dtype
// through add/mul/sub/sigmoid/softmax, so fp32, fp16 AND bf16 all work with no per-dtype code. Cast
// materializes the requested target dtype (incl. bf16).

#include "mlx_engine.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

void AddOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.AddA(ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1])));
}

void MulOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.Mul(ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1])));
}

// Integer/float Sub. Used (among others) by the seqlens-prep chain (seqlens_k = ReduceSum(mask) - 1);
// MLX-GQA does not consume seqlens, so that instance is dead in the MLX graph but must still map.
void SubOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.SubA(ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1])));
}

void SigmoidOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_sigmoid(&r, ctx.Resolve(n.inputs[0]), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

void SoftmaxOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_softmax_axis(&r, ctx.Resolve(n.inputs[0]), /*axis=*/-1, /*precise=*/true,
                             ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

void CastOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.Astype(ctx.Resolve(n.inputs[0]), MlxDtypeFromOnnx(n.outputs[0].type)));
}

}  // namespace

void RegisterElementwiseOps(OpRegistry& registry) {
  // ai.onnx elementwise/activation/cast (version-insensitive: kAnyOpset matches all opsets).
  registry.Register({"", "Add", kAnyOpset, kAnyOpset, &AddOp});
  registry.Register({"", "Mul", kAnyOpset, kAnyOpset, &MulOp});
  registry.Register({"", "Sub", kAnyOpset, kAnyOpset, &SubOp});
  registry.Register({"", "Sigmoid", kAnyOpset, kAnyOpset, &SigmoidOp});
  registry.Register({"", "Softmax", kAnyOpset, kAnyOpset, &SoftmaxOp});
  registry.Register({"", "Cast", kAnyOpset, kAnyOpset, &CastOp});
  // Sigmoid is also claimed in the com.microsoft domain (fused activation).
  registry.Register({"com.microsoft", "Sigmoid", kAnyOpset, kAnyOpset, &SigmoidOp});
}

}  // namespace ort_mps_mlx
