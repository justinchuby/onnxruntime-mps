// Copyright (c) 2026. Licensed under the MIT License.
//
// Math / activation / logical op handlers (unary + binary elementwise beyond the core set).

#include <cmath>
#include <cstdint>
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

mlx_array ScalarLike(TranslationContext& ctx, float value, mlx_array like) {
  mlx_array scalar = ctx.Keep(mlx_array_new_float32(value));
  const mlx_dtype dtype = mlx_array_dtype(like);
  return dtype == MLX_FLOAT32 ? scalar : ctx.Astype(scalar, dtype);
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

void DivOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0],
           ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]), mlx_divide));
}

void ModOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array a = ctx.Resolve(n.inputs[0]);
  mlx_array b = ctx.Resolve(n.inputs[1]);
  const bool fmod = n.ints.count("fmod") && n.ints.at("fmod") != 0;
  if (!fmod) {
    ctx.Bind(n.outputs[0], ApplyBinary(ctx, a, b, mlx_remainder));
    return;
  }
  mlx_array magnitude =
      ApplyBinary(ctx, ApplyUnary(ctx, a, mlx_abs), ApplyUnary(ctx, b, mlx_abs), mlx_remainder);
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, ApplyUnary(ctx, a, mlx_sign), magnitude, mlx_multiply));
}

void ReluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array zero = NewResult(ctx);
  MLX_CHECK(mlx_zeros_like(&zero, x, ctx.stream()));
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, zero, mlx_maximum));
}

void TanhOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_tanh));
}

void SoftplusOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array zero = NewResult(ctx);
  MLX_CHECK(mlx_zeros_like(&zero, x, ctx.stream()));
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, zero, mlx_logaddexp));
}

void ClipOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[1]), mlx_maximum);
  } else if (n.floats.count("min")) {
    out = ApplyBinary(ctx, out, ScalarLike(ctx, n.floats.at("min"), out), mlx_maximum);
  }
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[2]), mlx_minimum);
  } else if (n.floats.count("max")) {
    out = ApplyBinary(ctx, out, ScalarLike(ctx, n.floats.at("max"), out), mlx_minimum);
  }
  ctx.Bind(n.outputs[0], out);
}

void GeluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scaled = ApplyBinary(ctx, x, ScalarLike(ctx, std::sqrt(0.5f), x), mlx_multiply);
  mlx_array erf = ApplyUnary(ctx, scaled, mlx_erf);
  mlx_array shifted = ApplyBinary(ctx, erf, ScalarLike(ctx, 1.0f, x), mlx_add);
  mlx_array half_x = ApplyBinary(ctx, x, ScalarLike(ctx, 0.5f, x), mlx_multiply);
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, half_x, shifted, mlx_multiply));
}

void EluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array zero = ScalarLike(ctx, 0.0f, x);
  mlx_array positive = ApplyBinary(ctx, x, zero, mlx_greater_equal);
  mlx_array negative = ApplyUnary(ctx, x, mlx_expm1);
  const float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 1.0f;
  negative = ApplyBinary(ctx, negative, ScalarLike(ctx, alpha, x), mlx_multiply);
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_where(&out, positive, x, negative, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

void SwishOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, ApplyUnary(ctx, x, mlx_sigmoid), mlx_multiply));
}

void LogSoftmaxOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  // mlx_logsumexp aborts at construction on a zero-size array. LogSoftmax is shape-preserving, so an
  // empty input yields an empty output — emit it directly on MLX without touching the reduce kernel.
  if (mlx_array_size(x) == 0) {
    mlx_array empty = NewResult(ctx);
    MLX_CHECK(mlx_zeros_like(&empty, x, ctx.stream()));
    ctx.Bind(n.outputs[0], empty);
    return;
  }
  mlx_array normalizer = NewResult(ctx);
  MLX_CHECK(mlx_logsumexp_axis(&normalizer, x, /*axis=*/-1, /*keepdims=*/true, ctx.stream()));
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, normalizer, mlx_subtract));
}

void RoundOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_round(&out, ctx.Resolve(n.inputs[0]), /*decimals=*/0, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

using ArgMlxOp = int (*)(mlx_array*, const mlx_array, int, bool, const mlx_stream);

mlx_array ScalarI64(TranslationContext& ctx, int64_t value) {
  return ctx.Keep(mlx_array_new_data(&value, nullptr, 0, MLX_INT64));
}

mlx_array ScalarIntegerLike(TranslationContext& ctx, int64_t value, mlx_array like) {
  if (mlx_array_dtype(like) == MLX_INT32) {
    const int32_t narrowed = static_cast<int32_t>(value);
    return ctx.Keep(mlx_array_new_data(&narrowed, nullptr, 0, MLX_INT32));
  }
  return ScalarI64(ctx, value);
}

void ArgMinMaxOp(TranslationContext& ctx, const NodeDesc& n, ArgMlxOp op) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const int rank = static_cast<int>(mlx_array_ndim(x));
  int axis = static_cast<int>(n.ints.count("axis") ? n.ints.at("axis") : 0);
  if (axis < 0) axis += rank;
  const bool keepdims = !n.ints.count("keepdims") || n.ints.at("keepdims") != 0;
  const bool select_last =
      n.ints.count("select_last_index") && n.ints.at("select_last_index") != 0;

  mlx_array arg_input = x;
  const int dim = mlx_array_dim(x, axis);
  if (select_last) {
    mlx_array reverse_indices = NewResult(ctx);
    MLX_CHECK(mlx_arange(&reverse_indices, dim - 1, -1, -1, MLX_INT32, ctx.stream()));
    arg_input = NewResult(ctx);
    MLX_CHECK(mlx_take_axis(&arg_input, x, reverse_indices, axis, ctx.stream()));
  }

  mlx_array result = NewResult(ctx);
  MLX_CHECK(op(&result, arg_input, axis, keepdims, ctx.stream()));
  result = ctx.Astype(result, MLX_INT64);
  if (select_last) {
    result = ApplyBinary(ctx, ScalarI64(ctx, dim - 1), result, mlx_subtract);
  }
  ctx.Bind(n.outputs[0], result);
}

void ArgMinOp(TranslationContext& ctx, const NodeDesc& n) {
  ArgMinMaxOp(ctx, n, mlx_argmin_axis);
}

void ArgMaxOp(TranslationContext& ctx, const NodeDesc& n) {
  ArgMinMaxOp(ctx, n, mlx_argmax_axis);
}

int64_t ReadConstI64(TranslationContext& ctx, const TensorRef& ref) {
  HostBytes host = ctx.RawHost(ref);
  if (host.count != 1) throw MlxError("MLX expected a scalar int64 initializer");
  return *static_cast<const int64_t*>(host.data);
}

void OneHotOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array indices = ctx.Resolve(n.inputs[0]);
  const int depth = static_cast<int>(ReadConstI64(ctx, n.inputs[1]));
  mlx_array values = ctx.Resolve(n.inputs[2]);
  const int output_rank = static_cast<int>(mlx_array_ndim(indices)) + 1;
  int axis = static_cast<int>(n.ints.count("axis") ? n.ints.at("axis") : -1);
  if (axis < 0) axis += output_rank;

  mlx_array categories = NewResult(ctx);
  MLX_CHECK(mlx_arange(&categories, 0, depth, 1, mlx_array_dtype(indices), ctx.stream()));
  std::vector<int> category_shape(output_rank, 1);
  category_shape[axis] = depth;
  categories = ctx.Reshape(categories, category_shape);

  mlx_array zero = ScalarIntegerLike(ctx, 0, indices);
  mlx_array negative = ApplyBinary(ctx, indices, zero, mlx_less);
  mlx_array wrapped =
      ApplyBinary(ctx, indices, ScalarIntegerLike(ctx, depth, indices), mlx_add);
  mlx_array normalized = NewResult(ctx);
  MLX_CHECK(mlx_where(&normalized, negative, wrapped, indices, ctx.stream()));

  mlx_array expanded_indices = NewResult(ctx);
  MLX_CHECK(mlx_expand_dims(&expanded_indices, normalized, axis, ctx.stream()));
  mlx_array selected = ApplyBinary(ctx, expanded_indices, categories, mlx_equal);

  mlx_array zero_index = ScalarI64(ctx, 0);
  mlx_array one_index = ScalarI64(ctx, 1);
  mlx_array off = NewResult(ctx);
  MLX_CHECK(mlx_take(&off, values, zero_index, ctx.stream()));
  mlx_array on = NewResult(ctx);
  MLX_CHECK(mlx_take(&on, values, one_index, ctx.stream()));
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_where(&out, selected, on, off, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

void TriluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const int k = n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent
                    ? static_cast<int>(ReadConstI64(ctx, n.inputs[1]))
                    : 0;
  const bool upper = !n.ints.count("upper") || n.ints.at("upper") != 0;
  mlx_array out = NewResult(ctx);
  MLX_CHECK((upper ? mlx_triu : mlx_tril)(&out, x, k, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

#define DEFINE_UNARY_HANDLER(name, mlx_op)                                      \
  void name(TranslationContext& ctx, const NodeDesc& n) {                       \
    ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_op));   \
  }

DEFINE_UNARY_HANDLER(ExpOp, mlx_exp)
DEFINE_UNARY_HANDLER(LogOp, mlx_log)
DEFINE_UNARY_HANDLER(SqrtOp, mlx_sqrt)
DEFINE_UNARY_HANDLER(ReciprocalOp, mlx_reciprocal)
DEFINE_UNARY_HANDLER(NegOp, mlx_negative)
DEFINE_UNARY_HANDLER(AbsOp, mlx_abs)
DEFINE_UNARY_HANDLER(FloorOp, mlx_floor)
DEFINE_UNARY_HANDLER(SignOp, mlx_sign)
DEFINE_UNARY_HANDLER(ErfOp, mlx_erf)
DEFINE_UNARY_HANDLER(SinOp, mlx_sin)
DEFINE_UNARY_HANDLER(CosOp, mlx_cos)
DEFINE_UNARY_HANDLER(NotOp, mlx_logical_not)

#undef DEFINE_UNARY_HANDLER

void MinOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[i]), mlx_minimum);
  }
  ctx.Bind(n.outputs[0], out);
}

void MaxOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[i]), mlx_maximum);
  }
  ctx.Bind(n.outputs[0], out);
}

void PowOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0],
           ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]), mlx_power));
}

void CastLikeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array like = ctx.Resolve(n.inputs[1]);
  ctx.Bind(n.outputs[0], ctx.Astype(ctx.Resolve(n.inputs[0]), mlx_array_dtype(like)));
}

void WhereOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_where(&out, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]),
                      ctx.Resolve(n.inputs[2]), ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

#define DEFINE_BINARY_HANDLER(name, mlx_op)                                            \
  void name(TranslationContext& ctx, const NodeDesc& n) {                              \
    ctx.Bind(n.outputs[0], ApplyBinary(ctx, ctx.Resolve(n.inputs[0]),                   \
                                       ctx.Resolve(n.inputs[1]), mlx_op));               \
  }

DEFINE_BINARY_HANDLER(EqualOp, mlx_equal)
DEFINE_BINARY_HANDLER(LessOp, mlx_less)
DEFINE_BINARY_HANDLER(LessOrEqualOp, mlx_less_equal)
DEFINE_BINARY_HANDLER(GreaterOp, mlx_greater)
DEFINE_BINARY_HANDLER(GreaterOrEqualOp, mlx_greater_equal)
DEFINE_BINARY_HANDLER(AndOp, mlx_logical_and)
DEFINE_BINARY_HANDLER(OrOp, mlx_logical_or)

#undef DEFINE_BINARY_HANDLER

bool IsSignedIntegerType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsUnsignedIntegerType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
}

bool IsMlxNumericType(ONNXTensorElementDataType type) {
  return IsMlxFloatType(type) || IsSignedIntegerType(type) || IsUnsignedIntegerType(type);
}

bool UnarySameTypeClaim(Ort::ConstNode node, bool allow_signed_integer) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out) || in != out) return false;
  return IsMlxFloatType(in) || (allow_signed_integer && IsSignedIntegerType(in));
}

bool FloatUnaryClaim(Ort::ConstNode node) {
  return UnarySameTypeClaim(node, false);
}

bool SignedNumericUnaryClaim(Ort::ConstNode node) {
  return UnarySameTypeClaim(node, true);
}

bool BinarySameTypeClaim(Ort::ConstNode node, bool floats_only) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(outputs[0], out) ||
      a != b || b != out || !ScalarOrSuffixBroadcast(inputs[0], inputs[1])) {
    return false;
  }
  return floats_only ? IsMlxFloatType(a) : IsMlxNumericType(a);
}

bool DivClaim(Ort::ConstNode node) {
  return BinarySameTypeClaim(node, true);
}

bool ModClaim(Ort::ConstNode node) {
  if (!BinarySameTypeClaim(node, false)) return false;
  const int64_t fmod = IntAttribute(node, "fmod", 0);
  if (fmod != 0 && fmod != 1) return false;
  ONNXTensorElementDataType type;
  if (!TensorInfo(node.GetInputs()[0], type) ||
      type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
    return false;
  }
  return fmod == 1 ? IsMlxFloatType(type) : !IsMlxFloatType(type);
}

bool ReluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node);
}

bool TanhClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node);
}

bool SoftplusClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node);
}

bool ClipClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  if (!TensorInfo(inputs[0], x) || !TensorInfo(outputs[0], out) || !IsMlxFloatType(x) ||
      out != x) {
    return false;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!SlotPresent(inputs, i)) continue;  // omitted optional min/max (NULL value info)
    ONNXTensorElementDataType bound;
    if (!TensorInfo(inputs[i], bound) || bound != x ||
        !ScalarOrSuffixBroadcast(inputs[0], inputs[i])) {
      return false;
    }
  }
  return true;
}

std::string StringAttribute(Ort::ConstNode node, const char* name, const char* default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return default_value;
  }
  std::string value;
  return attr.GetValue(value).IsOK() ? value : default_value;
}

bool GeluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && StringAttribute(node, "approximate", "none") == "none";
}

bool EluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && std::isfinite(FloatAttribute(node, "alpha", 1.0f));
}

bool LogSoftmaxClaim(Ort::ConstNode node) {
  if (!FloatUnaryClaim(node)) return false;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type;
  if (!TensorInfo(node.GetInputs()[0], type, &shape) || shape.empty()) return false;
  int64_t axis = IntAttribute(node, "axis", -1);
  if (axis < 0) axis += static_cast<int64_t>(shape.size());
  return axis == static_cast<int64_t>(shape.size()) - 1;
}

bool ArgMinMaxClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  std::vector<int64_t> shape;
  if (!TensorInfo(inputs[0], in, &shape) || !TensorInfo(outputs[0], out) || shape.empty() ||
      !IsMlxNumericType(in) || in == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 ||
      out != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return false;
  }
  int64_t axis = IntAttribute(node, "axis", 0);
  if (axis < 0) axis += static_cast<int64_t>(shape.size());
  const int64_t keepdims = IntAttribute(node, "keepdims", 1);
  const int64_t select_last = IntAttribute(node, "select_last_index", 0);
  return axis >= 0 && axis < static_cast<int64_t>(shape.size()) &&
         (keepdims == 0 || keepdims == 1) && (select_last == 0 || select_last == 1);
}

bool IsConstScalarI64(Ort::ConstValueInfo value, int64_t* result = nullptr) {
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
  if (!TensorInfo(value, type, &shape) || type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      !(shape.empty() || (shape.size() == 1 && shape[0] == 1)) ||
      !value.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstValue initializer{nullptr};
  if (!value.GetInitializer(initializer).IsOK() ||
      static_cast<const OrtValue*>(initializer) == nullptr ||
      initializer.GetTensorTypeAndShapeInfo().GetElementCount() != 1) {
    return false;
  }
  const auto* data = static_cast<const int64_t*>(initializer.GetTensorRawData());
  if (data == nullptr) return false;
  if (result != nullptr) *result = *data;
  return true;
}

bool IsBoundaryValueType(ONNXTensorElementDataType type) {
  return IsMlxFloatType(type) ||  // float64 excluded: aborts on the Metal GPU (→ CPU fallback)
         IsSignedIntegerType(type) ||
         (IsUnsignedIntegerType(type) && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool OneHotClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType indices, depth_type, values, out;
  std::vector<int64_t> indices_shape, depth_shape, values_shape, out_shape;
  if (!TensorInfo(inputs[0], indices, &indices_shape) ||
      !TensorInfo(inputs[1], depth_type, &depth_shape) ||
      !TensorInfo(inputs[2], values, &values_shape) ||
      !TensorInfo(outputs[0], out, &out_shape) ||
      (indices != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       indices != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      depth_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      values_shape != std::vector<int64_t>{2} || values != out ||
      !IsBoundaryValueType(values)) {
    return false;
  }
  int64_t depth = 0;
  if (!IsConstScalarI64(inputs[1], &depth) || depth <= 0) return false;
  const int64_t output_rank = static_cast<int64_t>(indices_shape.size()) + 1;
  int64_t axis = IntAttribute(node, "axis", -1);
  if (axis < 0) axis += output_rank;
  if (axis < 0 || axis >= output_rank || out_shape.size() != static_cast<size_t>(output_rank)) {
    return false;
  }
  std::vector<int64_t> expected = indices_shape;
  expected.insert(expected.begin() + axis, depth);
  return expected == out_shape;
}

bool TriluClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  std::vector<int64_t> shape;
  if (!TensorInfo(inputs[0], in, &shape) || !TensorInfo(outputs[0], out) || shape.size() < 2 ||
      in != out || !IsBoundaryValueType(in)) {
    return false;
  }
  if (inputs.size() == 2 && SlotPresent(inputs, 1) && !IsConstScalarI64(inputs[1])) {
    return false;
  }
  const int64_t upper = IntAttribute(node, "upper", 1);
  return upper == 0 || upper == 1;
}

bool MinMaxClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType type, out;
  if (!TensorInfo(inputs[0], type) || !TensorInfo(outputs[0], out) || type != out ||
      !IsMlxNumericType(type)) {
    return false;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    ONNXTensorElementDataType other;
    if (!TensorInfo(inputs[i], other) || other != type ||
        !ScalarOrSuffixBroadcast(inputs[0], inputs[i])) {
      return false;
    }
  }
  return true;
}

bool PowClaim(Ort::ConstNode node) {
  return BinarySameTypeClaim(node, true);
}

bool CastLikeClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, like, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(inputs[1], like) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  return IsMlxFloatType(in) && IsMlxFloatType(like) && out == like;
}

bool WhereClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType condition, x, y, out;
  if (!TensorInfo(inputs[0], condition) || !TensorInfo(inputs[1], x) ||
      !TensorInfo(inputs[2], y) || !TensorInfo(outputs[0], out) ||
      condition != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL || x != y || y != out ||
      !(IsMlxNumericType(x) || x == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)) {
    return false;
  }
  return ScalarOrSuffixBroadcast(inputs[0], outputs[0]) &&
         ScalarOrSuffixBroadcast(inputs[1], outputs[0]) &&
         ScalarOrSuffixBroadcast(inputs[2], outputs[0]);
}

bool ComparisonClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  return a == b && IsMlxNumericType(a) && out == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool EqualClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  return a == b && (IsMlxNumericType(a) || a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) &&
         out == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool LogicalBinaryClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  return TensorInfo(inputs[0], a) && TensorInfo(inputs[1], b) && TensorInfo(outputs[0], out) &&
         a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL && b == a && out == a &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool NotClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  return TensorInfo(inputs[0], in) && TensorInfo(outputs[0], out) &&
         in == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL && out == in;
}

}  // namespace

void RegisterMathOps(OpRegistry& registry) {
  registry.Register({"", "Div", kAnyOpset, kAnyOpset, &DivOp, &DivClaim});
  registry.Register({"", "Mod", 10, kAnyOpset, &ModOp, &ModClaim});
  registry.Register({"", "Relu", kAnyOpset, kAnyOpset, &ReluOp, &ReluClaim});
  registry.Register({"", "Tanh", kAnyOpset, kAnyOpset, &TanhOp, &TanhClaim});
  registry.Register({"", "Softplus", kAnyOpset, kAnyOpset, &SoftplusOp, &SoftplusClaim});
  registry.Register({"", "Clip", kAnyOpset, kAnyOpset, &ClipOp, &ClipClaim});
  registry.Register({"", "Gelu", 20, kAnyOpset, &GeluOp, &GeluClaim});
  registry.Register({"", "Elu", kAnyOpset, kAnyOpset, &EluOp, &EluClaim});
  registry.Register({"com.microsoft", "Swish", kAnyOpset, kAnyOpset, &SwishOp,
                     &FloatUnaryClaim});
  registry.Register(
      {"", "LogSoftmax", kAnyOpset, kAnyOpset, &LogSoftmaxOp, &LogSoftmaxClaim});
  registry.Register({"", "Exp", kAnyOpset, kAnyOpset, &ExpOp, &FloatUnaryClaim});
  registry.Register({"", "Log", kAnyOpset, kAnyOpset, &LogOp, &FloatUnaryClaim});
  registry.Register({"", "Sqrt", kAnyOpset, kAnyOpset, &SqrtOp, &FloatUnaryClaim});
  registry.Register({"", "Reciprocal", kAnyOpset, kAnyOpset, &ReciprocalOp, &FloatUnaryClaim});
  registry.Register({"", "Neg", kAnyOpset, kAnyOpset, &NegOp, &SignedNumericUnaryClaim});
  registry.Register({"", "Abs", kAnyOpset, kAnyOpset, &AbsOp, &SignedNumericUnaryClaim});
  registry.Register({"", "Floor", kAnyOpset, kAnyOpset, &FloorOp, &FloatUnaryClaim});
  registry.Register({"", "Sign", kAnyOpset, kAnyOpset, &SignOp, &SignedNumericUnaryClaim});
  registry.Register({"", "Erf", kAnyOpset, kAnyOpset, &ErfOp, &FloatUnaryClaim});
  registry.Register({"", "Sin", kAnyOpset, kAnyOpset, &SinOp, &FloatUnaryClaim});
  registry.Register({"", "Cos", kAnyOpset, kAnyOpset, &CosOp, &FloatUnaryClaim});
  registry.Register({"", "Round", 11, kAnyOpset, &RoundOp, &FloatUnaryClaim});
  registry.Register({"", "ArgMin", kAnyOpset, kAnyOpset, &ArgMinOp, &ArgMinMaxClaim});
  registry.Register({"", "ArgMax", kAnyOpset, kAnyOpset, &ArgMaxOp, &ArgMinMaxClaim});
  registry.Register({"", "OneHot", 9, kAnyOpset, &OneHotOp, &OneHotClaim});
  registry.Register({"", "Trilu", 14, kAnyOpset, &TriluOp, &TriluClaim});
  registry.Register({"", "Min", kAnyOpset, kAnyOpset, &MinOp, &MinMaxClaim});
  registry.Register({"", "Max", kAnyOpset, kAnyOpset, &MaxOp, &MinMaxClaim});
  registry.Register({"", "Pow", kAnyOpset, kAnyOpset, &PowOp, &PowClaim});
  registry.Register({"", "CastLike", 15, kAnyOpset, &CastLikeOp, &CastLikeClaim});
  registry.Register({"", "Where", kAnyOpset, kAnyOpset, &WhereOp, &WhereClaim});
  registry.Register({"", "Equal", kAnyOpset, kAnyOpset, &EqualOp, &EqualClaim});
  registry.Register({"", "Less", kAnyOpset, kAnyOpset, &LessOp, &ComparisonClaim});
  registry.Register({"", "LessOrEqual", 12, kAnyOpset, &LessOrEqualOp, &ComparisonClaim});
  registry.Register({"", "Greater", kAnyOpset, kAnyOpset, &GreaterOp, &ComparisonClaim});
  registry.Register(
      {"", "GreaterOrEqual", 12, kAnyOpset, &GreaterOrEqualOp, &ComparisonClaim});
  registry.Register({"", "And", kAnyOpset, kAnyOpset, &AndOp, &LogicalBinaryClaim});
  registry.Register({"", "Or", kAnyOpset, kAnyOpset, &OrOp, &LogicalBinaryClaim});
  registry.Register({"", "Not", kAnyOpset, kAnyOpset, &NotOp, &NotClaim});
}

}  // namespace ort_mps_mlx
