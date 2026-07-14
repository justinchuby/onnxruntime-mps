// Copyright (c) 2026. Licensed under the MIT License.
//
// Reduction2 op handlers (ai.onnx opset-17+ coverage expansion). See docs/OP_ARCHITECTURE.md.
//
// Families:
//   * Axis reductions:  ReduceL1 (sum|x|), ReduceLogSum (log sum), ReduceLogSumExp (logsumexp),
//                       ReduceProd (prod). Axes come from the opset-18 `axes` input OR the legacy
//                       `axes` attribute; keepdims / noop_with_empty_axes honored.
//   * Hardmax:          one-hot of the argmax along `axis`.
//   * CumProd:          cumulative product along an axis input (mirrors reduction.cc CumSum).
//   * Variadic:         Sum / Mean (elementwise add of N broadcastable float operands).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

enum class ReduceKind { Sum, Prod, LogSumExp };
enum class PreOp { None, Abs };
enum class PostOp { None, Log };

mlx_array NewResult(TranslationContext& ctx) {
  return ctx.Keep(mlx_array_new());
}

mlx_array ScalarLike(TranslationContext& ctx, float value, mlx_array like) {
  mlx_array scalar = ctx.Keep(mlx_array_new_float32(value));
  const mlx_dtype dtype = mlx_array_dtype(like);
  return dtype == MLX_FLOAT32 ? scalar : ctx.Astype(scalar, dtype);
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

// LogSumExp over a zero-size array aborts inside mlx at construction (same failure mode as
// mlx_max/mlx_min). Synthesise the correctly-shaped result filled with the LogSumExp identity
// (-inf) directly on MLX instead of calling the aborting kernel.
mlx_array EmptyLogSumExp(TranslationContext& ctx, mlx_array x, const std::vector<int>& axes,
                         bool reduce_all, bool keepdims) {
  const std::vector<int> in = TranslationContext::ShapeOf(x);
  const int rank = static_cast<int>(in.size());
  std::vector<bool> reduced(rank, false);
  if (reduce_all) {
    std::fill(reduced.begin(), reduced.end(), true);
  } else {
    for (int a : axes) {
      if (a >= 0 && a < rank) reduced[a] = true;
    }
  }
  std::vector<int> out_shape;
  for (int i = 0; i < rank; ++i) {
    if (reduced[i]) {
      if (keepdims) out_shape.push_back(1);
    } else {
      out_shape.push_back(in[i]);
    }
  }
  const mlx_dtype dt = mlx_array_dtype(x);
  mlx_array scalar = ctx.Keep(mlx_array_new_float32(-INFINITY));
  if (dt != MLX_FLOAT32) scalar = ctx.Astype(scalar, dt);
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_full(&out, out_shape.data(), out_shape.size(), scalar, dt, ctx.stream()));
  return out;
}

mlx_array ApplyReduce(TranslationContext& ctx, mlx_array x, const std::vector<int>& axes,
                      bool reduce_all, bool keepdims, ReduceKind kind) {
  if (kind == ReduceKind::LogSumExp && mlx_array_size(x) == 0) {
    return EmptyLogSumExp(ctx, x, axes, reduce_all, keepdims);
  }
  mlx_array out = NewResult(ctx);
  int status = 0;
  if (reduce_all) {
    switch (kind) {
      case ReduceKind::Sum: status = mlx_sum(&out, x, keepdims, ctx.stream()); break;
      case ReduceKind::Prod: status = mlx_prod(&out, x, keepdims, ctx.stream()); break;
      case ReduceKind::LogSumExp:
        status = mlx_logsumexp(&out, x, keepdims, ctx.stream());
        break;
    }
  } else {
    switch (kind) {
      case ReduceKind::Sum:
        status = mlx_sum_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
      case ReduceKind::Prod:
        status = mlx_prod_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
      case ReduceKind::LogSumExp:
        status = mlx_logsumexp_axes(&out, x, axes.data(), axes.size(), keepdims, ctx.stream());
        break;
    }
  }
  if (status != 0) throw MlxError("mlx reduction call failed");
  return out;
}

void Reduce(TranslationContext& ctx, const NodeDesc& n, ReduceKind kind, PreOp pre, PostOp post) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array body = x;
  if (pre == PreOp::Abs) {
    body = NewResult(ctx);
    MLX_CHECK(mlx_abs(&body, x, ctx.stream()));
  }

  const bool has_axes = HasAxesInput(n) || n.int_arrays.count("axes") != 0;
  const std::vector<int64_t> raw_axes = ReadAxes(ctx, n);
  const bool noop =
      n.ints.count("noop_with_empty_axes") && n.ints.at("noop_with_empty_axes") != 0;

  auto apply_post = [&](mlx_array v) {
    if (post == PostOp::Log) {
      mlx_array logged = NewResult(ctx);
      MLX_CHECK(mlx_log(&logged, v, ctx.stream()));
      return logged;
    }
    return v;
  };

  if (has_axes && raw_axes.empty() && noop) {
    ctx.Bind(n.outputs[0], apply_post(body));
    return;
  }

  const std::vector<int> axes =
      raw_axes.empty() ? std::vector<int>{}
                       : NormalizeAxes(raw_axes, static_cast<int>(mlx_array_ndim(x)));
  const bool keepdims = !n.ints.count("keepdims") || n.ints.at("keepdims") != 0;
  mlx_array reduced = ApplyReduce(ctx, body, axes, raw_axes.empty(), keepdims, kind);
  ctx.Bind(n.outputs[0], apply_post(reduced));
}

void ReduceL1Op(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReduceKind::Sum, PreOp::Abs, PostOp::None);
}

void ReduceLogSumOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReduceKind::Sum, PreOp::None, PostOp::Log);
}

void ReduceLogSumExpOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReduceKind::LogSumExp, PreOp::None, PostOp::None);
}

void ReduceProdOp(TranslationContext& ctx, const NodeDesc& n) {
  Reduce(ctx, n, ReduceKind::Prod, PreOp::None, PostOp::None);
}

void HardmaxOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const int rank = static_cast<int>(mlx_array_ndim(x));
  int axis = static_cast<int>(n.ints.count("axis") ? n.ints.at("axis") : -1);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw MlxError("MLX Hardmax axis is out of range");

  if (mlx_array_size(x) == 0) {
    mlx_array empty = NewResult(ctx);
    MLX_CHECK(mlx_zeros_like(&empty, x, ctx.stream()));
    ctx.Bind(n.outputs[0], empty);
    return;
  }

  const int dim = mlx_array_dim(x, axis);
  mlx_array argmax = NewResult(ctx);
  MLX_CHECK(mlx_argmax_axis(&argmax, x, axis, /*keepdims=*/true, ctx.stream()));
  argmax = ctx.Astype(argmax, MLX_INT32);

  mlx_array iota = NewResult(ctx);
  MLX_CHECK(mlx_arange(&iota, 0, dim, 1, MLX_INT32, ctx.stream()));
  std::vector<int> iota_shape(rank, 1);
  iota_shape[axis] = dim;
  iota = ctx.Reshape(iota, iota_shape);

  mlx_array selected = NewResult(ctx);
  MLX_CHECK(mlx_equal(&selected, iota, argmax, ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Astype(selected, mlx_array_dtype(x)));
}

int64_t ReadScalarInteger(TranslationContext& ctx, const TensorRef& ref) {
  HostBytes host = ctx.RawHost(ref);
  if (host.count != 1) throw MlxError("MLX expected a scalar integer input");
  mlx_array value = ctx.Resolve(ref);
  if (mlx_array_dtype(value) == MLX_INT32) {
    return *static_cast<const int32_t*>(host.data);
  }
  if (mlx_array_dtype(value) == MLX_INT64) {
    return *static_cast<const int64_t*>(host.data);
  }
  throw MlxError("MLX expected an int32 or int64 scalar input");
}

void CumProdOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const int rank = static_cast<int>(mlx_array_ndim(x));
  int64_t axis = ReadScalarInteger(ctx, n.inputs[1]);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw MlxError("MLX CumProd axis is out of range");
  const bool reverse = n.ints.count("reverse") && n.ints.at("reverse") != 0;
  const bool inclusive = !n.ints.count("exclusive") || n.ints.at("exclusive") == 0;
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_cumprod(&out, x, static_cast<int>(axis), reverse, inclusive, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

void SumOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    mlx_array acc = NewResult(ctx);
    MLX_CHECK(mlx_add(&acc, out, ctx.Resolve(n.inputs[i]), ctx.stream()));
    out = acc;
  }
  ctx.Bind(n.outputs[0], out);
}

void MeanOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    mlx_array acc = NewResult(ctx);
    MLX_CHECK(mlx_add(&acc, out, ctx.Resolve(n.inputs[i]), ctx.stream()));
    out = acc;
  }
  mlx_array divisor = ScalarLike(ctx, static_cast<float>(n.inputs.size()), out);
  mlx_array mean = NewResult(ctx);
  MLX_CHECK(mlx_divide(&mean, out, divisor, ctx.stream()));
  ctx.Bind(n.outputs[0], mean);
}

// ---- claim predicates -------------------------------------------------------------------------

bool IsNonBoolNumeric(ONNXTensorElementDataType type) {
  return IsMlxSupportedType(type) && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
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
      x != out || (float_only ? !IsMlxFloatType(x) : !IsNonBoolNumeric(x))) {
    return false;
  }
  if (inputs.size() == 2 && SlotPresent(inputs, 1)) {
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

bool ReduceFloatClaim(Ort::ConstNode node) {
  return ReductionClaim(node, true);
}

bool HardmaxClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  std::vector<int64_t> shape;
  if (!TensorInfo(inputs[0], in, &shape) || !TensorInfo(outputs[0], out) || shape.empty() ||
      in != out || !IsMlxFloatType(in)) {
    return false;
  }
  int64_t axis = IntAttribute(node, "axis", -1);
  if (axis < 0) axis += static_cast<int64_t>(shape.size());
  return axis >= 0 && axis < static_cast<int64_t>(shape.size());
}

bool CumProdClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, axis, out;
  std::vector<int64_t> x_shape, axis_shape;
  if (!TensorInfo(inputs[0], x, &x_shape) || !TensorInfo(inputs[1], axis, &axis_shape) ||
      !TensorInfo(outputs[0], out) || x_shape.empty() || x != out || !IsNonBoolNumeric(x) ||
      (axis != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       axis != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    return false;
  }
  if (!(axis_shape.empty() || (axis_shape.size() == 1 && axis_shape[0] == 1))) return false;
  const int64_t exclusive = IntAttribute(node, "exclusive", 0);
  const int64_t reverse = IntAttribute(node, "reverse", 0);
  return (exclusive == 0 || exclusive == 1) && (reverse == 0 || reverse == 1);
}

bool VariadicFloatClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType type, out;
  if (!TensorInfo(inputs[0], type, nullptr) || !TensorInfo(outputs[0], out) || type != out ||
      !IsMlxFloatType(type)) {
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

}  // namespace

void RegisterReduction2Ops(OpRegistry& registry) {
  registry.Register({"", "ReduceL1", kAnyOpset, kAnyOpset, &ReduceL1Op, &ReduceNumericClaim});
  registry.Register(
      {"", "ReduceLogSum", kAnyOpset, kAnyOpset, &ReduceLogSumOp, &ReduceFloatClaim});
  registry.Register(
      {"", "ReduceLogSumExp", kAnyOpset, kAnyOpset, &ReduceLogSumExpOp, &ReduceFloatClaim});
  registry.Register({"", "ReduceProd", kAnyOpset, kAnyOpset, &ReduceProdOp, &ReduceNumericClaim});
  registry.Register({"", "Hardmax", 13, kAnyOpset, &HardmaxOp, &HardmaxClaim});
  registry.Register({"", "CumProd", 26, kAnyOpset, &CumProdOp, &CumProdClaim});
  registry.Register({"", "Mean", kAnyOpset, kAnyOpset, &MeanOp, &VariadicFloatClaim});
  registry.Register({"", "Sum", kAnyOpset, kAnyOpset, &SumOp, &VariadicFloatClaim});
}

}  // namespace ort_mps_mlx
