// Copyright (c) 2026. Licensed under the MIT License.
//
// Convolution and pooling op handlers.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

mlx_array NewArray(TranslationContext &ctx) {
  return ctx.Keep(mlx_array_new());
}

mlx_array Contiguous(TranslationContext &ctx, mlx_array a) {
  mlx_array out = NewArray(ctx);
  MLX_CHECK(mlx_contiguous(&out, a, /*allow_col_major=*/false, ctx.stream()));
  return out;
}

std::vector<int64_t> AttrOr(const NodeDesc &n, const char *name, size_t size,
                            int64_t value) {
  auto it = n.int_arrays.find(name);
  return it == n.int_arrays.end() ? std::vector<int64_t>(size, value)
                                  : it->second;
}

bool IntsAttribute(Ort::ConstNode node, const char *name,
                   std::vector<int64_t> &values, bool &present) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr *>(attr) == nullptr ||
      attr.GetType() == ORT_OP_ATTR_UNDEFINED) {
    present = false;
    values.clear();
    return true;
  }
  present = true;
  return attr.GetType() == ORT_OP_ATTR_INTS &&
         attr.GetValueArray(values).IsOK();
}

std::string StringAttribute(Ort::ConstNode node, const char *name,
                            const std::string &def) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr *>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return def;
  }
  std::string value;
  return attr.GetValue(value).IsOK() ? value : def;
}

bool ReadSpatialAttribute(Ort::ConstNode node, const char *name,
                          size_t spatial_rank, int64_t default_value,
                          std::vector<int64_t> &values) {
  bool present = false;
  if (!IntsAttribute(node, name, values, present))
    return false;
  if (!present)
    values.assign(spatial_rank, default_value);
  if (values.size() != spatial_rank)
    return false;
  return std::all_of(values.begin(), values.end(),
                     [](int64_t v) { return v > 0; });
}

bool ReadPads(Ort::ConstNode node, size_t spatial_rank,
              std::vector<int64_t> &pads) {
  bool present = false;
  if (!IntsAttribute(node, "pads", pads, present))
    return false;
  if (!present)
    pads.assign(2 * spatial_rank, 0);
  if (pads.size() != 2 * spatial_rank)
    return false;
  return std::all_of(pads.begin(), pads.end(),
                     [](int64_t v) { return v >= 0; });
}

bool StaticPositiveShape(const std::vector<int64_t> &shape, size_t rank) {
  return shape.size() == rank &&
         std::all_of(shape.begin(), shape.end(),
                     [](int64_t dim) { return dim > 0; });
}

bool SameKnownShape(const std::vector<int64_t> &actual,
                    const std::vector<int64_t> &expected) {
  if (actual.size() != expected.size())
    return false;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] > 0 && actual[i] != expected[i])
      return false;
  }
  return true;
}

bool OptionalBiasIsValid(const std::vector<Ort::ConstValueInfo> &inputs,
                         ONNXTensorElementDataType dtype, int64_t channels) {
  if (!SlotPresent(inputs, 2))
    return true;  // omitted optional bias (absent or NULL value info)
  ONNXTensorElementDataType bias_type;
  std::vector<int64_t> bias_shape;
  return TensorInfo(inputs[2], bias_type, &bias_shape) && bias_type == dtype &&
         bias_shape == std::vector<int64_t>{channels};
}

mlx_array ToChannelsLast(TranslationContext &ctx, mlx_array x,
                         int spatial_rank) {
  if (spatial_rank == 1)
    return Contiguous(ctx, ctx.Transpose(x, {0, 2, 1}));
  return Contiguous(ctx, ctx.Transpose(x, {0, 2, 3, 1}));
}

mlx_array FromChannelsLast(TranslationContext &ctx, mlx_array x,
                           int spatial_rank) {
  if (spatial_rank == 1)
    return Contiguous(ctx, ctx.Transpose(x, {0, 2, 1}));
  return Contiguous(ctx, ctx.Transpose(x, {0, 3, 1, 2}));
}

mlx_array ConvWeightToMlx(TranslationContext &ctx, mlx_array weight,
                          int spatial_rank) {
  if (spatial_rank == 1)
    return Contiguous(ctx, ctx.Transpose(weight, {0, 2, 1}));
  return Contiguous(ctx, ctx.Transpose(weight, {0, 2, 3, 1}));
}

void ConvOp(TranslationContext &ctx, const NodeDesc &n) {
  const int spatial_rank =
      static_cast<int>(ctx.ShapeOf(ctx.Resolve(n.inputs[0])).size()) - 2;
  const std::vector<int64_t> strides = AttrOr(n, "strides", spatial_rank, 1);
  const std::vector<int64_t> pads = AttrOr(n, "pads", 2 * spatial_rank, 0);
  const std::vector<int64_t> dilations =
      AttrOr(n, "dilations", spatial_rank, 1);
  const int group =
      static_cast<int>(n.ints.count("group") ? n.ints.at("group") : 1);

  mlx_array x = ToChannelsLast(ctx, ctx.Resolve(n.inputs[0]), spatial_rank);
  mlx_array weight =
      ConvWeightToMlx(ctx, ctx.Resolve(n.inputs[1]), spatial_rank);
  mlx_array out = NewArray(ctx);
  if (spatial_rank == 1) {
    MLX_CHECK(mlx_conv1d(&out, x, weight, static_cast<int>(strides[0]),
                         static_cast<int>(pads[0]),
                         static_cast<int>(dilations[0]), group, ctx.stream()));
  } else {
    MLX_CHECK(mlx_conv2d(&out, x, weight, static_cast<int>(strides[0]),
                         static_cast<int>(strides[1]),
                         static_cast<int>(pads[0]), static_cast<int>(pads[1]),
                         static_cast<int>(dilations[0]),
                         static_cast<int>(dilations[1]), group, ctx.stream()));
  }
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    out = ctx.AddA(out, ctx.Resolve(n.inputs[2]));
  }
  ctx.Bind(n.outputs[0], FromChannelsLast(ctx, out, spatial_rank));
}

bool ConvClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
    return false;

  ONNXTensorElementDataType x_type, weight_type, out_type;
  std::vector<int64_t> x_shape, weight_shape, out_shape;
  if (!TensorInfo(inputs[0], x_type, &x_shape) ||
      !TensorInfo(inputs[1], weight_type, &weight_shape) ||
      !TensorInfo(outputs[0], out_type, &out_shape) ||
      !IsMlxFloatType(x_type) || weight_type != x_type || out_type != x_type) {
    return false;
  }
  if (x_shape.size() != 3 && x_shape.size() != 4)
    return false;
  const size_t spatial_rank = x_shape.size() - 2;
  if (!StaticPositiveShape(x_shape, spatial_rank + 2) ||
      !StaticPositiveShape(weight_shape, spatial_rank + 2) ||
      out_shape.size() != spatial_rank + 2) {
    return false;
  }
  if (StringAttribute(node, "auto_pad", "NOTSET") != "NOTSET")
    return false;

  std::vector<int64_t> strides, pads, dilations, kernel_shape;
  if (!ReadSpatialAttribute(node, "strides", spatial_rank, 1, strides) ||
      !ReadSpatialAttribute(node, "dilations", spatial_rank, 1, dilations) ||
      !ReadPads(node, spatial_rank, pads)) {
    return false;
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    if (pads[i] != pads[i + spatial_rank])
      return false;
  }

  bool kernel_present = false;
  if (!IntsAttribute(node, "kernel_shape", kernel_shape, kernel_present))
    return false;
  if (kernel_present) {
    if (kernel_shape.size() != spatial_rank)
      return false;
    for (size_t i = 0; i < spatial_rank; ++i) {
      if (kernel_shape[i] != weight_shape[i + 2])
        return false;
    }
  }

  const int64_t group = IntAttribute(node, "group", 1);
  const int64_t channels = x_shape[1];
  const int64_t out_channels = weight_shape[0];
  if (group <= 0 || channels % group != 0 || out_channels % group != 0 ||
      weight_shape[1] != channels / group ||
      !OptionalBiasIsValid(inputs, x_type, out_channels)) {
    return false;
  }

  std::vector<int64_t> expected{out_shape[0] > 0 ? out_shape[0] : x_shape[0],
                                out_channels};
  for (size_t i = 0; i < spatial_rank; ++i) {
    const int64_t effective_kernel =
        dilations[i] * (weight_shape[i + 2] - 1) + 1;
    const int64_t padded = x_shape[i + 2] + pads[i] + pads[i + spatial_rank];
    if (padded < effective_kernel)
      return false;
    expected.push_back((padded - effective_kernel) / strides[i] + 1);
  }
  expected[0] = x_shape[0];
  return SameKnownShape(out_shape, expected);
}

void ConvTransposeOp(TranslationContext &ctx, const NodeDesc &n) {
  const std::vector<int64_t> strides = AttrOr(n, "strides", 2, 1);
  const std::vector<int64_t> pads = AttrOr(n, "pads", 4, 0);
  const std::vector<int64_t> output_padding = AttrOr(n, "output_padding", 2, 0);

  mlx_array x = ToChannelsLast(ctx, ctx.Resolve(n.inputs[0]), 2);
  mlx_array weight =
      Contiguous(ctx, ctx.Transpose(ctx.Resolve(n.inputs[1]), {1, 2, 3, 0}));
  mlx_array out = NewArray(ctx);
  MLX_CHECK(mlx_conv_transpose2d(
      &out, x, weight, static_cast<int>(strides[0]),
      static_cast<int>(strides[1]), static_cast<int>(pads[0]),
      static_cast<int>(pads[1]), 1, 1, static_cast<int>(output_padding[0]),
      static_cast<int>(output_padding[1]), 1, ctx.stream()));
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    out = ctx.AddA(out, ctx.Resolve(n.inputs[2]));
  }
  ctx.Bind(n.outputs[0], FromChannelsLast(ctx, out, 2));
}

bool ConvTransposeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
    return false;

  ONNXTensorElementDataType x_type, weight_type, out_type;
  std::vector<int64_t> x_shape, weight_shape, out_shape;
  if (!TensorInfo(inputs[0], x_type, &x_shape) ||
      !TensorInfo(inputs[1], weight_type, &weight_shape) ||
      !TensorInfo(outputs[0], out_type, &out_shape) ||
      !IsMlxFloatType(x_type) || weight_type != x_type || out_type != x_type ||
      !StaticPositiveShape(x_shape, 4) ||
      !StaticPositiveShape(weight_shape, 4) || out_shape.size() != 4) {
    return false;
  }
  if (StringAttribute(node, "auto_pad", "NOTSET") != "NOTSET" ||
      IntAttribute(node, "group", 1) != 1 || weight_shape[0] != x_shape[1]) {
    return false;
  }

  std::vector<int64_t> strides, pads, dilations, output_padding, kernel_shape,
      output_shape;
  if (!ReadSpatialAttribute(node, "strides", 2, 1, strides) ||
      !ReadSpatialAttribute(node, "dilations", 2, 1, dilations) ||
      dilations[0] != 1 || dilations[1] != 1 || !ReadPads(node, 2, pads) ||
      pads[0] != pads[2] || pads[1] != pads[3]) {
    return false;
  }
  bool output_padding_present = false;
  if (!IntsAttribute(node, "output_padding", output_padding,
                     output_padding_present))
    return false;
  if (!output_padding_present)
    output_padding = {0, 0};
  if (output_padding.size() != 2 || output_padding[0] < 0 ||
      output_padding[1] < 0 || output_padding[0] >= strides[0] ||
      output_padding[1] >= strides[1]) {
    return false;
  }
  bool output_shape_present = false;
  if (!IntsAttribute(node, "output_shape", output_shape,
                     output_shape_present) ||
      output_shape_present) {
    return false;
  }
  bool kernel_present = false;
  if (!IntsAttribute(node, "kernel_shape", kernel_shape, kernel_present))
    return false;
  if (kernel_present &&
      (kernel_shape !=
       std::vector<int64_t>{weight_shape[2], weight_shape[3]})) {
    return false;
  }

  const int64_t out_channels = weight_shape[1];
  if (!OptionalBiasIsValid(inputs, x_type, out_channels))
    return false;
  std::vector<int64_t> expected{
      x_shape[0],
      out_channels,
      strides[0] * (x_shape[2] - 1) + output_padding[0] + weight_shape[2] -
          pads[0] - pads[2],
      strides[1] * (x_shape[3] - 1) + output_padding[1] + weight_shape[3] -
          pads[1] - pads[3],
  };
  return expected[2] > 0 && expected[3] > 0 &&
         SameKnownShape(out_shape, expected);
}

mlx_array ScalarForDtype(TranslationContext &ctx, float value,
                         mlx_dtype dtype) {
  mlx_array scalar = ctx.Keep(mlx_array_new_float32(value));
  return ctx.Astype(scalar, dtype);
}

mlx_array PadSpatial(TranslationContext &ctx, mlx_array x,
                     const std::vector<int64_t> &pads, mlx_array value) {
  if (std::all_of(pads.begin(), pads.end(),
                  [](int64_t pad) { return pad == 0; }))
    return x;
  const int axes[2] = {1, 2};
  const int low[2] = {static_cast<int>(pads[0]), static_cast<int>(pads[1])};
  const int high[2] = {static_cast<int>(pads[2]), static_cast<int>(pads[3])};
  mlx_array out = NewArray(ctx);
  MLX_CHECK(mlx_pad(&out, x, axes, 2, low, 2, high, 2, value, "constant",
                    ctx.stream()));
  return Contiguous(ctx, out);
}

mlx_array SlidingWindows2d(TranslationContext &ctx, mlx_array x,
                           const std::vector<int64_t> &kernel,
                           const std::vector<int64_t> &strides) {
  const std::vector<int> shape = ctx.ShapeOf(x);
  const int n = shape[0];
  const int h = shape[1];
  const int w = shape[2];
  const int c = shape[3];
  const int out_h =
      (h - static_cast<int>(kernel[0])) / static_cast<int>(strides[0]) + 1;
  const int out_w =
      (w - static_cast<int>(kernel[1])) / static_cast<int>(strides[1]) + 1;
  const std::vector<int> window_shape{
      n, out_h, out_w, static_cast<int>(kernel[0]), static_cast<int>(kernel[1]),
      c};
  const int64_t row_stride = static_cast<int64_t>(w) * c;
  const std::vector<int64_t> window_strides{
      static_cast<int64_t>(h) * row_stride,
      static_cast<int64_t>(strides[0]) * row_stride,
      static_cast<int64_t>(strides[1]) * c,
      row_stride,
      c,
      1,
  };
  mlx_array out = NewArray(ctx);
  MLX_CHECK(mlx_as_strided(&out, x, window_shape.data(), window_shape.size(),
                           window_strides.data(), window_strides.size(), 0,
                           ctx.stream()));
  return out;
}

mlx_array ReducePoolWindows(TranslationContext &ctx, mlx_array windows,
                            bool average) {
  const int axes[2] = {3, 4};
  mlx_array out = NewArray(ctx);
  if (average) {
    MLX_CHECK(mlx_mean_axes(&out, windows, axes, 2, false, ctx.stream()));
  } else {
    MLX_CHECK(mlx_max_axes(&out, windows, axes, 2, false, ctx.stream()));
  }
  return out;
}

void PoolOp(TranslationContext &ctx, const NodeDesc &n, bool average) {
  const std::vector<int64_t> &kernel = n.int_arrays.at("kernel_shape");
  const std::vector<int64_t> strides = AttrOr(n, "strides", 2, 1);
  const std::vector<int64_t> pads = AttrOr(n, "pads", 4, 0);
  const bool count_include_pad = average && n.ints.count("count_include_pad") &&
                                 n.ints.at("count_include_pad") != 0;

  mlx_array x = ToChannelsLast(ctx, ctx.Resolve(n.inputs[0]), 2);
  const float pad_value =
      average ? 0.0f : -std::numeric_limits<float>::infinity();
  mlx_array padded = PadSpatial(
      ctx, x, pads, ScalarForDtype(ctx, pad_value, mlx_array_dtype(x)));
  mlx_array windows = SlidingWindows2d(ctx, padded, kernel, strides);

  mlx_array out;
  const bool has_padding = std::any_of(pads.begin(), pads.end(),
                                       [](int64_t pad) { return pad != 0; });
  if (!average || count_include_pad || !has_padding) {
    out = ReducePoolWindows(ctx, windows, average);
  } else {
    const int axes[2] = {3, 4};
    mlx_array sums = NewArray(ctx);
    MLX_CHECK(mlx_sum_axes(&sums, windows, axes, 2, false, ctx.stream()));

    const std::vector<int> x_shape = ctx.ShapeOf(x);
    const int mask_shape[4] = {x_shape[0], x_shape[1], x_shape[2], 1};
    mlx_array mask = NewArray(ctx);
    MLX_CHECK(mlx_ones(&mask, mask_shape, 4, mlx_array_dtype(x), ctx.stream()));
    mlx_array zero = ScalarForDtype(ctx, 0.0f, mlx_array_dtype(x));
    mlx_array padded_mask = PadSpatial(ctx, mask, pads, zero);
    mlx_array mask_windows =
        SlidingWindows2d(ctx, padded_mask, kernel, strides);
    mlx_array counts = NewArray(ctx);
    MLX_CHECK(
        mlx_sum_axes(&counts, mask_windows, axes, 2, false, ctx.stream()));
    out = NewArray(ctx);
    MLX_CHECK(mlx_divide(&out, sums, counts, ctx.stream()));
  }
  ctx.Bind(n.outputs[0], FromChannelsLast(ctx, out, 2));
}

void AveragePoolOp(TranslationContext &ctx, const NodeDesc &n) {
  PoolOp(ctx, n, true);
}

void MaxPoolOp(TranslationContext &ctx, const NodeDesc &n) {
  PoolOp(ctx, n, false);
}

bool PoolClaim(Ort::ConstNode node, bool average) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  ONNXTensorElementDataType x_type, out_type;
  std::vector<int64_t> x_shape, out_shape;
  if (!TensorInfo(inputs[0], x_type, &x_shape) ||
      !TensorInfo(outputs[0], out_type, &out_shape) ||
      !IsMlxFloatType(x_type) || out_type != x_type ||
      !StaticPositiveShape(x_shape, 4) || out_shape.size() != 4 ||
      StringAttribute(node, "auto_pad", "NOTSET") != "NOTSET" ||
      IntAttribute(node, "ceil_mode", 0) != 0) {
    return false;
  }

  std::vector<int64_t> kernel, strides, pads, dilations;
  bool kernel_present = false;
  if (!IntsAttribute(node, "kernel_shape", kernel, kernel_present) ||
      !kernel_present || kernel.size() != 2 || kernel[0] <= 0 ||
      kernel[1] <= 0 || !ReadSpatialAttribute(node, "strides", 2, 1, strides) ||
      !ReadPads(node, 2, pads) ||
      !ReadSpatialAttribute(node, "dilations", 2, 1, dilations) ||
      dilations[0] != 1 || dilations[1] != 1) {
    return false;
  }
  if (average) {
    const int64_t count_include_pad =
        IntAttribute(node, "count_include_pad", 0);
    if (count_include_pad != 0 && count_include_pad != 1)
      return false;
  } else if (IntAttribute(node, "storage_order", 0) != 0) {
    return false;
  }

  const int64_t padded_h = x_shape[2] + pads[0] + pads[2];
  const int64_t padded_w = x_shape[3] + pads[1] + pads[3];
  if (padded_h < kernel[0] || padded_w < kernel[1])
    return false;
  const std::vector<int64_t> expected{
      x_shape[0],
      x_shape[1],
      (padded_h - kernel[0]) / strides[0] + 1,
      (padded_w - kernel[1]) / strides[1] + 1,
  };
  return SameKnownShape(out_shape, expected);
}

bool AveragePoolClaim(Ort::ConstNode node) { return PoolClaim(node, true); }

bool MaxPoolClaim(Ort::ConstNode node) { return PoolClaim(node, false); }

void GlobalPoolOp(TranslationContext &ctx, const NodeDesc &n, bool average) {
  mlx_array out = NewArray(ctx);
  const int axes[2] = {2, 3};
  if (average) {
    MLX_CHECK(mlx_mean_axes(&out, ctx.Resolve(n.inputs[0]), axes, 2, true,
                            ctx.stream()));
  } else {
    MLX_CHECK(mlx_max_axes(&out, ctx.Resolve(n.inputs[0]), axes, 2, true,
                           ctx.stream()));
  }
  ctx.Bind(n.outputs[0], out);
}

void GlobalAveragePoolOp(TranslationContext &ctx, const NodeDesc &n) {
  GlobalPoolOp(ctx, n, true);
}

void GlobalMaxPoolOp(TranslationContext &ctx, const NodeDesc &n) {
  GlobalPoolOp(ctx, n, false);
}

bool GlobalPoolClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  ONNXTensorElementDataType x_type, out_type;
  std::vector<int64_t> x_shape, out_shape;
  return TensorInfo(inputs[0], x_type, &x_shape) &&
         TensorInfo(outputs[0], out_type, &out_shape) &&
         IsMlxFloatType(x_type) && out_type == x_type &&
         StaticPositiveShape(x_shape, 4) &&
         SameKnownShape(out_shape, {x_shape[0], x_shape[1], 1, 1});
}

} // namespace

void RegisterConvOps(OpRegistry &registry) {
  registry.Register({"", "Conv", 1, kAnyOpset, &ConvOp, &ConvClaim});
  registry.Register({"", "ConvTranspose", 1, kAnyOpset, &ConvTransposeOp,
                     &ConvTransposeClaim});
  registry.Register(
      {"", "AveragePool", 1, kAnyOpset, &AveragePoolOp, &AveragePoolClaim});
  registry.Register({"", "GlobalAveragePool", 1, kAnyOpset,
                     &GlobalAveragePoolOp, &GlobalPoolClaim});
  registry.Register({"", "MaxPool", 1, kAnyOpset, &MaxPoolOp, &MaxPoolClaim});
  registry.Register(
      {"", "GlobalMaxPool", 1, kAnyOpset, &GlobalMaxPoolOp, &GlobalPoolClaim});
}

} // namespace ort_mps_mlx
