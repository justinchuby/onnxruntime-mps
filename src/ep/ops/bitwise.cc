// Copyright (c) 2026. Licensed under the MIT License.
//
// Bitwise op handlers (ai.onnx opset-17+ coverage expansion). See docs/OP_ARCHITECTURE.md.
//
// Families: BitwiseAnd/Or/Xor/Not (integer bit ops), BitShift (LEFT/RIGHT logical shift), and the
// logical Xor (boolean). MLX carries every integer/bool dtype through these kernels with no
// per-dtype code, so the claim predicates take the broadest dtype set each op's translation
// supports (integers for the bitwise/shift ops, bool for logical Xor).

#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

using UnaryMlxOp = int (*)(mlx_array*, mlx_array, mlx_stream);
using BinaryMlxOp = int (*)(mlx_array*, mlx_array, mlx_array, mlx_stream);

mlx_array NewResult(TranslationContext& ctx) {
  return ctx.Keep(mlx_array_new());
}

mlx_array ApplyUnary(TranslationContext& ctx, mlx_array x, UnaryMlxOp op) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(op(&out, x, ctx.stream()));
  return out;
}

mlx_array ApplyBinary(TranslationContext& ctx, mlx_array a, mlx_array b, BinaryMlxOp op) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(op(&out, a, b, ctx.stream()));
  return out;
}

// ---- handlers ---------------------------------------------------------------------------------

void BitwiseAndOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]),
                                     mlx_bitwise_and));
}

void BitwiseOrOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]),
                                     mlx_bitwise_or));
}

void BitwiseXorOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]),
                                     mlx_bitwise_xor));
}

void BitwiseNotOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_bitwise_invert));
}

void BitShiftOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array a = ctx.Resolve(n.inputs[0]);
  mlx_array b = ctx.Resolve(n.inputs[1]);
  const bool left =
      n.strings.count("direction") == 0 || n.strings.at("direction") == "LEFT";
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, a, b, left ? mlx_left_shift : mlx_right_shift));
}

// ONNX Xor is logical exclusive-or on booleans. MLX has no logical_xor, but for bool operands
// `a != b` is exactly xor; mlx_not_equal broadcasts and returns a bool result.
void XorOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]),
                                     mlx_not_equal));
}

// ---- claim predicates -------------------------------------------------------------------------

bool IsIntegerType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return true;
    default:
      return false;
  }
}

bool BitwiseBinaryClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(outputs[0], out) ||
      a != b || b != out || !IsIntegerType(a)) {
    return false;
  }
  return ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool BitwiseNotClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  return TensorInfo(inputs[0], in) && TensorInfo(outputs[0], out) && in == out &&
         IsIntegerType(in);
}

bool BitShiftClaim(Ort::ConstNode node) {
  if (!BitwiseBinaryClaim(node)) return false;
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName("direction", attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return false;
  }
  std::string direction;
  if (!attr.GetValue(direction).IsOK()) return false;
  // ONNX BitShift requires an explicit direction; only LEFT/RIGHT are translatable.
  return direction == "LEFT" || direction == "RIGHT";
}

bool XorClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  return TensorInfo(inputs[0], a) && TensorInfo(inputs[1], b) && TensorInfo(outputs[0], out) &&
         a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL && b == a && out == a &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

}  // namespace

void RegisterBitwiseOps(OpRegistry& registry) {
  registry.Register({"", "BitwiseAnd", 18, kAnyOpset, &BitwiseAndOp, &BitwiseBinaryClaim});
  registry.Register({"", "BitwiseOr", 18, kAnyOpset, &BitwiseOrOp, &BitwiseBinaryClaim});
  registry.Register({"", "BitwiseXor", 18, kAnyOpset, &BitwiseXorOp, &BitwiseBinaryClaim});
  registry.Register({"", "BitwiseNot", 18, kAnyOpset, &BitwiseNotOp, &BitwiseNotClaim});
  registry.Register({"", "BitShift", 11, kAnyOpset, &BitShiftOp, &BitShiftClaim});
  registry.Register({"", "Xor", kAnyOpset, kAnyOpset, &XorOp, &XorClaim});
}

}  // namespace ort_mps_mlx
