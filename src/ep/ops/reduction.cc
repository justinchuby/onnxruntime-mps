// Copyright (c) 2026. Licensed under the MIT License.
//
// Reduction op handlers (ReduceSum/Max/Mean/Min/SumSquare, CumSum, TopK).

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

enum class ReductionKind { Sum, Max, Mean, Min };

mlx_array NewResult(TranslationContext& ctx) {
  return ctx.Keep(mlx_array_new());
}

bool HasAxesInput(const NodeDesc& n) {
  return n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent;
}

std::vector<int64_t> ReadAxes(TranslationContext& ctx, const NodeDesc& n) {
  if (HasAxesInput(n)) {
    HostBytes host = ctx.RawHost(n.inputs[1]);
    const int64_t* data = static_cast<const int64_t*>(host.data);
    return std::vector<int64_t>(data, data + host.count);
  }
  auto it = n.int_arrays.find("axes");
  return it == n.int_arrays.end() ? std::vector<int64_t>{} : it->second;
}

std::vector<int> NormalizeAxes(const std::vector<int64_t>& axes, int rank) {
  std::vector<int> normalized;
  normalized.reserve(axes.size());
  for (int64_t raw : axes) {
    int64_t axis = raw < 0 ? raw + rank : raw;
    if (axis < 0 || axis >= rank) throw MlxError("MLX reduction axis is out of range");
    int value = static_cast<int>(axis);
    if (std::find(normalized.begin(), normalized.end(), value) != normalized.end()) {
      throw MlxError("MLX reduction axes contain a duplicate");
    }
    normalized.push_back(value);
  }
  return normalized;
}

mlx_array ApplyReduction(TranslationContext& ctx, mlx_array x, const std::vector<int>& axes,
                         bool reduce_all, bool keepdims, ReductionKind kind) {
  mlx_array out = NewResult(ctx);
  int status = 0;
  if (reduce_all) {
    switch (kind) {
      case ReductionKind::Sum: status = mlx_sum(&out, x, keepdims, ctx.stream()); break;
      case ReductionKind::Max: status = mlx_max(&out, x, keepdims, ctx.stream()); break;
      case ReductionKind::Mean: status = mlx_mean(&out, x, keepdims, ctx.stream()); break;
      case ReductionKind::Min: status = mlx_min(&out, x, keepdims, ctx.stream()); break;
    }
  } else {
    switch (kind) {
      case ReductionKind::Sum:
        status = mlx_sum_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
      case ReductionKind::Max:
        status = mlx_max_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
      case ReductionKind::Mean:
        status = mlx_mean_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
      case ReductionKind::Min:
        status = mlx_min_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
    }
  }
  if (status != 0) throw MlxError("mlx reduction call failed");
  return out;
}

void Reduce(TranslationContext& ctx, const NodeDesc& n, ReductionKind kind, bool square_first,
            bool sqrt_after = false) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array reduced_input = x;
  if (square_first) {
    reduced_input = NewResult(ctx);
    MLX_CHECK(mlx_square(&reduced_input, x, ctx.stream()));
  }
  const bool has_axes = HasAxesInput(n) || n.int_arrays.count("axes") != 0;
  const std::vector<int64_t> raw_axes = ReadAxes(ctx, n);
  const bool noop = n.ints.count("noop_with_empty_axes") &&
                    n.ints.at("noop_with_empty_axes") != 0;
  if (has_axes && raw_axes.empty() && noop) {
    if (sqrt_after) {
      mlx_array rooted = NewResult(ctx);
      MLX_CHECK(mlx_sqrt(&rooted, reduced_input, ctx.stream()));
      reduced_input = rooted;
    }
    ctx.Bind(n.outputs[0], reduced_input);
    return;
  }

  const std::vector<int> axes =
      raw_axes.empty() ? std::vector<int>{}
                       : NormalizeAxes(raw_axes, static_cast<int>(mlx_array_ndim(x)));
  const bool keepdims = !n.ints.count("keepdims") || n.ints.at("keepdims") != 0;
  mlx_array result =
      ApplyReduction(ctx, reduced_input, axes, raw_axes.empty(), keepdims, kind);
  if (sqrt_after) {
    mlx_array rooted = NewResult(ctx);
    MLX_CHECK(mlx_sqrt(&rooted, result, ctx.stream()));
    result = rooted;
  }
  ctx.Bind(n.outputs[0], result);
}

void ReduceSumOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReductionKind::Sum, false);
}

void ReduceMaxOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReductionKind::Max, false);
}

void ReduceMeanOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReductionKind::Mean, false);
}

void ReduceMinOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReductionKind::Min, false);
}

void ReduceSumSquareOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReductionKind::Sum, true);
}

void ReduceL2Op(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReductionKind::Sum, true, true);
}

int64_t ReadScalarInteger(TranslationContext& ctx, const TensorRef& ref) {
  mlx_array value = ctx.Resolve(ref);
  HostBytes host = ctx.RawHost(ref);
  if (host.count != 1) throw MlxError("MLX expected a scalar integer input");
  if (mlx_array_dtype(value) == MLX_INT32) {
    return *static_cast<const int32_t*>(host.data);
  }
  if (mlx_array_dtype(value) == MLX_INT64) {
    return *static_cast<const int64_t*>(host.data);
  }
  throw MlxError("MLX expected an int32 or int64 scalar input");
}

void CumSumOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const int rank = static_cast<int>(mlx_array_ndim(x));
  int64_t axis = ReadScalarInteger(ctx, n.inputs[1]);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw MlxError("MLX CumSum axis is out of range");
  const bool reverse = n.ints.count("reverse") && n.ints.at("reverse") != 0;
  const bool inclusive = !n.ints.count("exclusive") || n.ints.at("exclusive") == 0;
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_cumsum(&out, x, static_cast<int>(axis), reverse, inclusive, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

void TopKOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const std::vector<int> shape = TranslationContext::ShapeOf(x);
  int axis = static_cast<int>(n.ints.count("axis") ? n.ints.at("axis") : -1);
  if (axis < 0) axis += static_cast<int>(shape.size());
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    throw MlxError("MLX TopK axis is out of range");
  }
  const int64_t k64 = ReadScalarInteger(ctx, n.inputs[1]);
  if (k64 <= 0 || k64 > shape[axis]) throw MlxError("MLX TopK K is out of range");
  const int k = static_cast<int>(k64);
  const bool largest = !n.ints.count("largest") || n.ints.at("largest") != 0;

  mlx_array sort_input = x;
  if (largest) {
    sort_input = NewResult(ctx);
    MLX_CHECK(mlx_negative(&sort_input, x, ctx.stream()));
  }
  mlx_array sorted_indices = NewResult(ctx);
  MLX_CHECK(mlx_argsort_axis(&sorted_indices, sort_input, axis, ctx.stream()));

  mlx_array selector = NewResult(ctx);
  MLX_CHECK(mlx_arange(&selector, 0, k, 1, MLX_INT32, ctx.stream()));

  mlx_array top_indices = NewResult(ctx);
  MLX_CHECK(mlx_take_axis(&top_indices, sorted_indices, selector, axis, ctx.stream()));
  mlx_array values = NewResult(ctx);
  MLX_CHECK(mlx_take_along_axis(&values, x, top_indices, axis, ctx.stream()));
  mlx_array contiguous_values = NewResult(ctx);
  MLX_CHECK(mlx_contiguous(&contiguous_values, values, false, ctx.stream()));
  mlx_array contiguous_indices = NewResult(ctx);
  MLX_CHECK(mlx_contiguous(&contiguous_indices, top_indices, false, ctx.stream()));
  ctx.Bind(n.outputs[0], contiguous_values);
  ctx.Bind(n.outputs[1], ctx.Astype(contiguous_indices, MLX_INT64));
}

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

bool IntsAttribute(Ort::ConstNode node, const char* name, std::vector<int64_t>& values,
                   bool& present) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() == ORT_OP_ATTR_UNDEFINED) {
    present = false;
    values.clear();
    return true;
  }
  present = true;
  return attr.GetType() == ORT_OP_ATTR_INTS && attr.GetValueArray(values).IsOK();
}

bool AxesAreValid(const std::vector<int64_t>& axes, int64_t rank) {
  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  for (int64_t axis : axes) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank ||
        std::find(normalized.begin(), normalized.end(), axis) != normalized.end()) {
      return false;
    }
    normalized.push_back(axis);
  }
  return true;
}

bool ReductionClaim(Ort::ConstNode node, bool float_only) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> shape;
  if (!TensorInfo(inputs[0], x, &shape) || !TensorInfo(outputs[0], out) || shape.empty() ||
      x != out || (float_only ? !IsMlxFloatType(x) : !IsMlxNumericType(x))) {
    return false;
  }
  if (inputs.size() == 2 && !inputs[1].GetName().empty()) {
    ONNXTensorElementDataType axes_type;
    std::vector<int64_t> axes_shape;
    if (!TensorInfo(inputs[1], axes_type, &axes_shape) ||
        axes_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || axes_shape.size() > 1) {
      return false;
    }
  }
  std::vector<int64_t> axes;
  bool present = false;
  if (!IntsAttribute(node, "axes", axes, present) ||
      (present && !AxesAreValid(axes, static_cast<int64_t>(shape.size())))) {
    return false;
  }
  const int64_t keepdims = IntAttribute(node, "keepdims", 1);
  const int64_t noop = IntAttribute(node, "noop_with_empty_axes", 0);
  return (keepdims == 0 || keepdims == 1) && (noop == 0 || noop == 1);
}

bool ReduceNumericClaim(Ort::ConstNode node) {
  return ReductionClaim(node, false);
}

bool ReduceMeanClaim(Ort::ConstNode node) {
  return ReductionClaim(node, true);
}

bool CumSumClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, axis, out;
  std::vector<int64_t> x_shape, axis_shape;
  if (!TensorInfo(inputs[0], x, &x_shape) || !TensorInfo(inputs[1], axis, &axis_shape) ||
      !TensorInfo(outputs[0], out) || x_shape.empty() || x != out || !IsMlxNumericType(x) ||
      (axis != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       axis != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  if (!(axis_shape.empty() || (axis_shape.size() == 1 && axis_shape[0] == 1))) return false;
  const int64_t exclusive = IntAttribute(node, "exclusive", 0);
  const int64_t reverse = IntAttribute(node, "reverse", 0);
  return (exclusive == 0 || exclusive == 1) && (reverse == 0 || reverse == 1);
}

bool TopKClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 2) return false;
  ONNXTensorElementDataType x, k, values, indices;
  std::vector<int64_t> x_shape, k_shape;
  if (!TensorInfo(inputs[0], x, &x_shape) || !TensorInfo(inputs[1], k, &k_shape) ||
      !TensorInfo(outputs[0], values) || !TensorInfo(outputs[1], indices) || x_shape.empty() ||
      !IsMlxFloatType(x) || values != x || k != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      indices != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      !(k_shape.empty() || (k_shape.size() == 1 && k_shape[0] == 1))) {
    return false;
  }
  int64_t axis = IntAttribute(node, "axis", -1);
  if (axis < 0) axis += static_cast<int64_t>(x_shape.size());
  if (axis < 0 || axis >= static_cast<int64_t>(x_shape.size())) return false;
  const int64_t largest = IntAttribute(node, "largest", 1);
  const int64_t sorted = IntAttribute(node, "sorted", 1);
  return (largest == 0 || largest == 1) && sorted == 1;
}

}  // namespace

void RegisterReductionOps(OpRegistry& registry) {
  registry.Register({"", "ReduceSum", kAnyOpset, kAnyOpset, &ReduceSumOp, &ReduceNumericClaim});
  registry.Register({"", "ReduceMax", kAnyOpset, kAnyOpset, &ReduceMaxOp, &ReduceNumericClaim});
  registry.Register({"", "ReduceMean", kAnyOpset, kAnyOpset, &ReduceMeanOp, &ReduceMeanClaim});
  registry.Register({"", "ReduceMin", kAnyOpset, kAnyOpset, &ReduceMinOp, &ReduceNumericClaim});
  registry.Register(
      {"", "ReduceSumSquare", kAnyOpset, kAnyOpset, &ReduceSumSquareOp, &ReduceNumericClaim});
  registry.Register({"", "ReduceL2", kAnyOpset, kAnyOpset, &ReduceL2Op, &ReduceMeanClaim});
  registry.Register({"", "CumSum", 11, kAnyOpset, &CumSumOp, &CumSumClaim});
  registry.Register({"", "TopK", 10, kAnyOpset, &TopKOp, &TopKClaim});
}

}  // namespace ort_mps_mlx
