// Copyright (c) 2026. Licensed under the MIT License.
//
// NormPool op handlers (ai.onnx opset-17+ coverage expansion): the normalization + pooling +
// vision-sampling family. See docs/OP_ARCHITECTURE.md §5/§6.
//
// Registered (translatable on MLX, most-relaxed float dtypes fp16/bf16/fp32):
//   * InstanceNormalization      — per-(N,C) normalize over spatial dims, per-channel scale/bias.
//   * MeanVarianceNormalization  — normalize over `axes` (default [0,2,3]); eps (1e-9) added post-sqrt.
//   * LRN                        — cross-channel local response norm (size/alpha/beta/bias); the
//                                  clamped channel window is a zero-padded cumsum difference.
//   * LpPool / GlobalLpPool      — p-norm pooling: (sum |x|^p)^(1/p) over a window / all spatial dims.
//
// Left to ORT CPU (documented, NOT force-fit — see the bottom of this file for the reasons):
//   * MaxUnpool, RoiAlign, MaxRoiPool, GridSample — index-scatter / per-ROI bilinear-sampling vision
//     ops with no clean mlx-c primitive; a claimed-but-untranslatable node is a HARD failure, so
//     these stay unclaimed and run on ORT CPU.
//
// Every handler honors the resolved input dtype (fp32/fp16/bf16) with no per-dtype branching.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- small local MLX helpers (each Keep()s its result) --------------------------------------

mlx_array NewArray(TranslationContext& ctx) { return ctx.Keep(mlx_array_new()); }

mlx_array Square(TranslationContext& ctx, mlx_array a) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_square(&r, a, ctx.stream()));
  return r;
}
mlx_array Sqrt(TranslationContext& ctx, mlx_array a) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_sqrt(&r, a, ctx.stream()));
  return r;
}
mlx_array Rsqrt(TranslationContext& ctx, mlx_array a) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_rsqrt(&r, a, ctx.stream()));
  return r;
}
mlx_array Abs(TranslationContext& ctx, mlx_array a) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_abs(&r, a, ctx.stream()));
  return r;
}
mlx_array Divide(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_divide(&r, a, b, ctx.stream()));
  return r;
}
mlx_array Power(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_power(&r, a, b, ctx.stream()));
  return r;
}
mlx_array Cumsum(TranslationContext& ctx, mlx_array a, int axis) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_cumsum(&r, a, axis, /*reverse=*/false, /*inclusive=*/true, ctx.stream()));
  return r;
}
mlx_array MeanAxes(TranslationContext& ctx, mlx_array a, const std::vector<int>& axes,
                   bool keepdims) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_mean_axes(&r, a, axes.data(), axes.size(), keepdims, ctx.stream()));
  return r;
}
mlx_array SumAxes(TranslationContext& ctx, mlx_array a, const std::vector<int>& axes,
                  bool keepdims) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_sum_axes(&r, a, axes.data(), axes.size(), keepdims, ctx.stream()));
  return r;
}
mlx_array MeanAxis(TranslationContext& ctx, mlx_array a, int axis, bool keepdims) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_mean_axis(&r, a, axis, keepdims, ctx.stream()));
  return r;
}
mlx_array VarAxis(TranslationContext& ctx, mlx_array a, int axis, bool keepdims) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_var_axis(&r, a, axis, keepdims, /*ddof=*/0, ctx.stream()));
  return r;
}
mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return r;
}

// A 0-d scalar of dtype `dt` holding `v`, matching the compute dtype so no unwanted upcast occurs.
mlx_array ScalarLike(TranslationContext& ctx, float v, mlx_dtype dt) {
  mlx_array s = ctx.Keep(mlx_array_new_float32(v));
  return ctx.Astype(s, dt);
}

// Reshape a per-channel vector [C] to [1, C, 1, ..., 1] so it broadcasts over N and spatial dims.
mlx_array ChannelBroadcast(TranslationContext& ctx, mlx_array v, int rank, int channels) {
  std::vector<int> shape(rank, 1);
  if (rank >= 2) shape[1] = channels;
  return ctx.Reshape(v, shape);
}

// Pad a single axis with `low`/`high` copies of `value` (used for the LRN channel window).
mlx_array PadAxis(TranslationContext& ctx, mlx_array a, int axis, int low, int high,
                  mlx_array value) {
  const int axes[1] = {axis};
  const int low_pad[1] = {low};
  const int high_pad[1] = {high};
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_pad(&r, a, axes, 1, low_pad, 1, high_pad, 1, value, "constant", ctx.stream()));
  return Contiguous(ctx, r);
}

// ---- shared attribute readers ---------------------------------------------------------------

std::string StringAttribute(Ort::ConstNode node, const char* name, const std::string& def) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return def;
  }
  std::string value;
  return attr.GetValue(value).IsOK() ? value : def;
}

bool ReadSpatialAttribute(Ort::ConstNode node, const char* name, size_t spatial_rank,
                          int64_t default_value, std::vector<int64_t>& values) {
  bool present = false;
  if (!IntsAttribute(node, name, values, present)) return false;
  if (!present) values.assign(spatial_rank, default_value);
  if (values.size() != spatial_rank) return false;
  return std::all_of(values.begin(), values.end(), [](int64_t v) { return v > 0; });
}

bool ReadPads(Ort::ConstNode node, size_t spatial_rank, std::vector<int64_t>& pads) {
  bool present = false;
  if (!IntsAttribute(node, "pads", pads, present)) return false;
  if (!present) pads.assign(2 * spatial_rank, 0);
  if (pads.size() != 2 * spatial_rank) return false;
  return std::all_of(pads.begin(), pads.end(), [](int64_t v) { return v >= 0; });
}

bool StaticPositiveShape(const std::vector<int64_t>& shape, size_t rank) {
  return shape.size() == rank &&
         std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d > 0; });
}

bool SameKnownShape(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected) {
  if (actual.size() != expected.size()) return false;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] > 0 && actual[i] != expected[i]) return false;
  }
  return true;
}

// ---- pooling window helpers (channels-last NHWC, mirrors conv.cc) ---------------------------

mlx_array ToChannelsLast2d(TranslationContext& ctx, mlx_array x) {
  return Contiguous(ctx, ctx.Transpose(x, {0, 2, 3, 1}));
}
mlx_array FromChannelsLast2d(TranslationContext& ctx, mlx_array x) {
  return Contiguous(ctx, ctx.Transpose(x, {0, 3, 1, 2}));
}

mlx_array PadSpatial2d(TranslationContext& ctx, mlx_array x, const std::vector<int64_t>& pads,
                       mlx_array value) {
  if (std::all_of(pads.begin(), pads.end(), [](int64_t p) { return p == 0; })) return x;
  const int axes[2] = {1, 2};
  const int low[2] = {static_cast<int>(pads[0]), static_cast<int>(pads[1])};
  const int high[2] = {static_cast<int>(pads[2]), static_cast<int>(pads[3])};
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_pad(&r, x, axes, 2, low, 2, high, 2, value, "constant", ctx.stream()));
  return Contiguous(ctx, r);
}

mlx_array SlidingWindows2d(TranslationContext& ctx, mlx_array x, const std::vector<int64_t>& kernel,
                           const std::vector<int64_t>& strides) {
  const std::vector<int> shape = ctx.ShapeOf(x);
  const int n = shape[0], h = shape[1], w = shape[2], c = shape[3];
  const int out_h = (h - static_cast<int>(kernel[0])) / static_cast<int>(strides[0]) + 1;
  const int out_w = (w - static_cast<int>(kernel[1])) / static_cast<int>(strides[1]) + 1;
  const std::vector<int> window_shape{
      n, out_h, out_w, static_cast<int>(kernel[0]), static_cast<int>(kernel[1]), c};
  const int64_t row_stride = static_cast<int64_t>(w) * c;
  const std::vector<int64_t> window_strides{
      static_cast<int64_t>(h) * row_stride,
      static_cast<int64_t>(strides[0]) * row_stride,
      static_cast<int64_t>(strides[1]) * c,
      row_stride,
      c,
      1,
  };
  mlx_array r = NewArray(ctx);
  MLX_CHECK(mlx_as_strided(&r, x, window_shape.data(), window_shape.size(), window_strides.data(),
                           window_strides.size(), 0, ctx.stream()));
  return r;
}

// ---- handlers -------------------------------------------------------------------------------

// InstanceNormalization (ai.onnx): per (N,C) normalize over spatial dims using the biased variance,
// then apply per-channel scale/bias. X=[N,C,*S], scale/bias=[C]. Y = scale*(X-mean)/sqrt(var+eps)+B.
void InstanceNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scale = ctx.Resolve(n.inputs[1]);
  mlx_array bias = ctx.Resolve(n.inputs[2]);
  std::vector<int> shape = TranslationContext::ShapeOf(x);
  int rank = static_cast<int>(shape.size());
  int N = shape[0], C = shape[1];
  float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-5f;

  int spatial = 1;
  for (int i = 2; i < rank; ++i) spatial *= shape[i];

  mlx_array grp = ctx.Reshape(x, {N, C, spatial});
  mlx_array mean = MeanAxis(ctx, grp, /*axis=*/2, /*keepdims=*/true);
  mlx_array var = VarAxis(ctx, grp, /*axis=*/2, /*keepdims=*/true);
  mlx_array inv = Rsqrt(ctx, ctx.AddA(var, ScalarLike(ctx, eps, mlx_array_dtype(x))));
  mlx_array normed = ctx.Mul(ctx.SubA(grp, mean), inv);
  normed = ctx.Reshape(normed, shape);

  mlx_array sb = ChannelBroadcast(ctx, scale, rank, C);
  mlx_array bb = ChannelBroadcast(ctx, bias, rank, C);
  ctx.Bind(n.outputs[0], ctx.AddA(ctx.Mul(normed, sb), bb));
}

// MeanVarianceNormalization (ai.onnx): Y = (X - mean) / (sqrt(var) + 1e-9) reduced over `axes`
// (default [0,2,3]). Epsilon is a fixed 1e-9 added AFTER the sqrt (matches the ONNX function def).
void MeanVarianceNormOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(x));
  std::vector<int> axes;
  if (n.int_arrays.count("axes")) {
    for (int64_t a : n.int_arrays.at("axes")) axes.push_back(static_cast<int>(a < 0 ? a + rank : a));
  } else {
    for (int a : {0, 2, 3}) {
      if (a < rank) axes.push_back(a);
    }
  }
  std::sort(axes.begin(), axes.end());

  mlx_array mean = MeanAxes(ctx, x, axes, /*keepdims=*/true);
  mlx_array mean_sq = MeanAxes(ctx, Square(ctx, x), axes, /*keepdims=*/true);
  mlx_array var = ctx.SubA(mean_sq, Square(ctx, mean));
  mlx_array denom = ctx.AddA(Sqrt(ctx, var), ScalarLike(ctx, 1e-9f, mlx_array_dtype(x)));
  ctx.Bind(n.outputs[0], Divide(ctx, ctx.SubA(x, mean), denom));
}

// LRN (ai.onnx): cross-channel local response normalization. For each element,
//   Y = X / (bias + (alpha/size) * sum_{j in window} X[j]^2)^beta
// where the window is the `size` channels centered on the current one, clamped to [0, C-1]. Because
// the clamp only ever DROPS out-of-range channels (they add nothing to a SUM), the clamped window
// sum equals a zero-padded fixed-width window sum, which we compute as a cumsum difference.
void LRNOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(x);
  int rank = static_cast<int>(shape.size());
  int C = shape[1];
  int size = static_cast<int>(n.ints.at("size"));
  float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 1e-4f;
  float beta = n.floats.count("beta") ? n.floats.at("beta") : 0.75f;
  float bias = n.floats.count("bias") ? n.floats.at("bias") : 1.0f;
  mlx_dtype dt = mlx_array_dtype(x);

  int pad_left = (size - 1) / 2;
  int pad_right = size - 1 - pad_left;
  mlx_array zero = ScalarLike(ctx, 0.0f, dt);

  mlx_array sq = Square(ctx, x);
  mlx_array sqp = PadAxis(ctx, sq, /*axis=*/1, pad_left, pad_right, zero);  // [N, C+size-1, *S]
  mlx_array cs = Cumsum(ctx, sqp, /*axis=*/1);
  mlx_array cz = PadAxis(ctx, cs, /*axis=*/1, /*low=*/1, /*high=*/0, zero);  // [N, C+size, *S]

  std::vector<int> lo_start(rank, 0), lo_stop = shape;
  std::vector<int> hi_start(rank, 0), hi_stop = shape;
  lo_stop[1] = C;      // czero[0 : C]
  hi_start[1] = size;  // czero[size : size+C]
  hi_stop[1] = size + C;
  mlx_array sqsum = ctx.SubA(ctx.Slice(cz, hi_start, hi_stop), ctx.Slice(cz, lo_start, lo_stop));

  mlx_array base =
      ctx.AddA(ScalarLike(ctx, bias, dt), ctx.Mul(ScalarLike(ctx, alpha / size, dt), sqsum));
  mlx_array denom = Power(ctx, base, ScalarLike(ctx, beta, dt));
  ctx.Bind(n.outputs[0], Divide(ctx, x, denom));
}

// LpPool (ai.onnx, 2D form): Y = (sum_{window} |X|^p)^(1/p). p-th powers are formed elementwise on
// the zero-padded input (|0|^p = 0, so padding contributes nothing) before the window reduction.
void LpPoolOp(TranslationContext& ctx, const NodeDesc& n) {
  const std::vector<int64_t>& kernel = n.int_arrays.at("kernel_shape");
  std::vector<int64_t> strides =
      n.int_arrays.count("strides") ? n.int_arrays.at("strides") : std::vector<int64_t>{1, 1};
  std::vector<int64_t> pads =
      n.int_arrays.count("pads") ? n.int_arrays.at("pads") : std::vector<int64_t>{0, 0, 0, 0};
  float p = static_cast<float>(n.ints.count("p") ? n.ints.at("p") : 2);

  mlx_array x = ToChannelsLast2d(ctx, ctx.Resolve(n.inputs[0]));
  mlx_dtype dt = mlx_array_dtype(x);
  mlx_array padded = PadSpatial2d(ctx, x, pads, ScalarLike(ctx, 0.0f, dt));
  mlx_array powered = Power(ctx, Abs(ctx, padded), ScalarLike(ctx, p, dt));
  mlx_array windows = SlidingWindows2d(ctx, powered, kernel, strides);
  mlx_array summed = SumAxes(ctx, windows, {3, 4}, /*keepdims=*/false);
  mlx_array out = Power(ctx, summed, ScalarLike(ctx, 1.0f / p, dt));
  ctx.Bind(n.outputs[0], FromChannelsLast2d(ctx, out));
}

// GlobalLpPool (ai.onnx): p-norm over ALL spatial dims -> [N, C, 1, ..., 1].
void GlobalLpPoolOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(x));
  float p = static_cast<float>(n.ints.count("p") ? n.ints.at("p") : 2);
  mlx_dtype dt = mlx_array_dtype(x);
  std::vector<int> axes;
  for (int i = 2; i < rank; ++i) axes.push_back(i);

  mlx_array powered = Power(ctx, Abs(ctx, x), ScalarLike(ctx, p, dt));
  mlx_array summed = SumAxes(ctx, powered, axes, /*keepdims=*/true);
  ctx.Bind(n.outputs[0], Power(ctx, summed, ScalarLike(ctx, 1.0f / p, dt)));
}

// ---- claim predicates -----------------------------------------------------------------------

// InstanceNormalization: X=[N,C,*S] float (static shape for the group reshape), scale/bias=[C].
bool InstanceNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, scale, bias, out;
  std::vector<int64_t> xshape, sshape, bshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(inputs[1], scale, &sshape) ||
      !TensorInfo(inputs[2], bias, &bshape) || !TensorInfo(outputs[0], out)) {
    return false;
  }
  if (!IsMlxFloatType(x) || scale != x || bias != x || out != x) return false;
  if (xshape.size() < 2) return false;
  for (int64_t d : xshape) {
    if (d <= 0) return false;  // need static dims to build the per-instance reshape
  }
  int64_t C = xshape[1];
  return sshape.size() == 1 && sshape[0] == C && bshape.size() == 1 && bshape[0] == C;
}

// MeanVarianceNormalization: single float input/output; `axes` (if present) in-range for the rank.
bool MeanVarianceNormClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> xshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMlxFloatType(x) || out != x || xshape.empty()) return false;
  int rank = static_cast<int>(xshape.size());
  std::vector<int64_t> axes;
  bool present = false;
  if (!IntsAttribute(node, "axes", axes, present)) return false;
  if (present) {
    for (int64_t a : axes) {
      if (a < -rank || a >= rank) return false;
    }
  } else if (rank < 2) {
    return false;  // default axes [0,2,3] need a batch+channel layout
  }
  return true;
}

// LRN: single float input/output, static shape (rank>=2) so the channel window is statically sized;
// `size` present and positive.
bool LRNClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> xshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMlxFloatType(x) || out != x || xshape.size() < 2) return false;
  for (int64_t d : xshape) {
    if (d <= 0) return false;
  }
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName("size", attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_INT) {
    return false;  // size is required
  }
  return IntAttribute(node, "size", 0) > 0;
}

// LpPool: 2D form (4D static input), kernel of length 2, p>0, NOTSET auto_pad, ceil_mode 0,
// unit dilations. Output shape must match the floor formula.
bool LpPoolClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> xshape, oshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(outputs[0], out, &oshape)) return false;
  if (!IsMlxFloatType(x) || out != x || !StaticPositiveShape(xshape, 4) || oshape.size() != 4)
    return false;
  if (StringAttribute(node, "auto_pad", "NOTSET") != "NOTSET" ||
      IntAttribute(node, "ceil_mode", 0) != 0 || IntAttribute(node, "p", 2) <= 0) {
    return false;
  }
  std::vector<int64_t> kernel, strides, pads, dilations;
  bool kernel_present = false;
  if (!IntsAttribute(node, "kernel_shape", kernel, kernel_present) || !kernel_present ||
      kernel.size() != 2 || kernel[0] <= 0 || kernel[1] <= 0 ||
      !ReadSpatialAttribute(node, "strides", 2, 1, strides) || !ReadPads(node, 2, pads) ||
      !ReadSpatialAttribute(node, "dilations", 2, 1, dilations) || dilations[0] != 1 ||
      dilations[1] != 1) {
    return false;
  }
  const int64_t padded_h = xshape[2] + pads[0] + pads[2];
  const int64_t padded_w = xshape[3] + pads[1] + pads[3];
  if (padded_h < kernel[0] || padded_w < kernel[1]) return false;
  const std::vector<int64_t> expected{
      xshape[0],
      xshape[1],
      (padded_h - kernel[0]) / strides[0] + 1,
      (padded_w - kernel[1]) / strides[1] + 1,
  };
  return SameKnownShape(oshape, expected);
}

// GlobalLpPool: 4D static float input, p>0, output [N, C, 1, 1].
bool GlobalLpPoolClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  std::vector<int64_t> xshape, oshape;
  if (!TensorInfo(inputs[0], x, &xshape) || !TensorInfo(outputs[0], out, &oshape)) return false;
  if (!IsMlxFloatType(x) || out != x || !StaticPositiveShape(xshape, 4)) return false;
  if (IntAttribute(node, "p", 2) <= 0) return false;
  return SameKnownShape(oshape, {xshape[0], xshape[1], 1, 1});
}

}  // namespace

// MaxUnpool, RoiAlign, MaxRoiPool and GridSample are deliberately NOT registered: they need either a
// scatter from per-element flattened argmax indices (MaxUnpool) or per-ROI bilinear gather/sampling
// (RoiAlign / MaxRoiPool / GridSample) that has no clean mlx-c primitive. Force-fitting them would
// risk a claimed-but-untranslatable node (a HARD failure), so they stay unclaimed and run on ORT CPU.
void RegisterNormPoolOps(OpRegistry& registry) {
  registry.Register(
      {"", "InstanceNormalization", kAnyOpset, kAnyOpset, &InstanceNormOp, &InstanceNormClaim});
  registry.Register({"", "MeanVarianceNormalization", kAnyOpset, kAnyOpset, &MeanVarianceNormOp,
                     &MeanVarianceNormClaim});
  registry.Register({"", "LRN", kAnyOpset, kAnyOpset, &LRNOp, &LRNClaim});
  registry.Register({"", "LpPool", kAnyOpset, kAnyOpset, &LpPoolOp, &LpPoolClaim});
  registry.Register({"", "GlobalLpPool", kAnyOpset, kAnyOpset, &GlobalLpPoolOp, &GlobalLpPoolClaim});
}

}  // namespace ort_mps_mlx
