// Copyright (c) 2026. Licensed under the MIT License.
//
// Elementwise + activation + cast op handlers. Every handler is dtype-generic: it resolves each
// input to an MLX array wrapped with its ACTUAL dtype (MlxDtypeFromOnnx) and MLX carries that dtype
// through add/mul/sub/sigmoid/softmax, so fp32, fp16 AND bf16 all work with no per-dtype code. Cast
// materializes the requested target dtype (incl. bf16).

#include "mlx_engine.h"
#include "op_claim.h"
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

// ---- claim predicates (dtype/shape/attr checks; registry already matched domain/op/opset) -------

// The standard ai.onnx float32 Add (bias add / residual): equal shapes or trailing-suffix broadcast
// in fp32; plus the fp16/bf16 activation/residual add (same dtype, scalar-or-suffix broadcast). Div
// is NOT translated to MLX and is left to ORT's CPU EP.
bool AddClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(outputs[0], out)) {
    return false;
  }
  // fp32 residual/bias add: equal-or-suffix broadcast (rejects scalar operands).
  if (a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && b == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      out == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return SuffixBroadcast(inputs[0], inputs[1]);
  }
  // fp16/bf16 activation/residual add: same dtype, scalar-or-suffix broadcast.
  if (a == b && b == out && (a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                             a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    return ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
  }
  return false;
}

// Mul: fp32/fp16/bf16, same dtype in/out, scalar-or-suffix broadcast.
bool MulClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(outputs[0], out) ||
      a != b || b != out || !ScalarOrSuffixBroadcast(inputs[0], inputs[1])) {
    return false;
  }
  return IsMlxFloatType(a);
}

// Sub: fp32/fp16/bf16 or int64 (the seqlens-prep chain), same dtype in/out, scalar-or-suffix
// broadcast.
bool SubClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(outputs[0], out) ||
      a != b || b != out || !ScalarOrSuffixBroadcast(inputs[0], inputs[1])) {
    return false;
  }
  return a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || IsMlxFloatType(a);
}

// Sigmoid (ai.onnx or com.microsoft): single fp32/fp16/bf16 input, same dtype out. SiLU/Swish/Gelu
// are NOT claimed (left to CPU).
bool SigmoidClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out)) return false;
  return in == out && IsMlxFloatType(in);
}

// Softmax (ai.onnx): single input, softmax over the last axis. fp32/fp16/bf16.
bool SoftmaxClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.empty()) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> shape;
  if (!TensorInfo(outputs[0], out) || !TensorInfo(inputs[0], x, &shape) || !IsMlxFloatType(x) ||
      out != x) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(shape.size());
  const int64_t axis = IntAttribute(node, "axis", -1);
  return rank > 0 && (axis == -1 || axis == rank - 1);
}

// Cast (ai.onnx): float<->float among fp32/fp16/bf16 (any distinct pair) plus the int64->int32 index
// cast. Other casts remain on CPU.
bool CastClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out)) return false;
  const bool in_float = IsMlxFloatType(in);
  const bool out_float = IsMlxFloatType(out);
  if (in_float && out_float && in != out) return true;
  return in == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && out == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

}  // namespace

void RegisterElementwiseOps(OpRegistry& registry) {
  // ai.onnx elementwise/activation/cast (version-insensitive: kAnyOpset matches all opsets).
  registry.Register({"", "Add", kAnyOpset, kAnyOpset, &AddOp, &AddClaim});
  registry.Register({"", "Mul", kAnyOpset, kAnyOpset, &MulOp, &MulClaim});
  registry.Register({"", "Sub", kAnyOpset, kAnyOpset, &SubOp, &SubClaim});
  registry.Register({"", "Sigmoid", kAnyOpset, kAnyOpset, &SigmoidOp, &SigmoidClaim});
  registry.Register({"", "Softmax", kAnyOpset, kAnyOpset, &SoftmaxOp, &SoftmaxClaim});
  registry.Register({"", "Cast", kAnyOpset, kAnyOpset, &CastOp, &CastClaim});
  // Sigmoid is also claimed in the com.microsoft domain (fused activation).
  registry.Register({"com.microsoft", "Sigmoid", kAnyOpset, kAnyOpset, &SigmoidOp, &SigmoidClaim});
}

}  // namespace ort_mps_mlx
