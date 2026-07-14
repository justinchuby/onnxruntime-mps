// Copyright (c) 2026. Licensed under the MIT License.
//
// Vision2 op handlers (ai.onnx opset-17+ coverage — VISION / SPATIAL-TRANSFORM family). See
// docs/OP_ARCHITECTURE.md §5/§6. This module adds the spatial-transform ops that resample or fold
// an [N,C,H,W] feature map through a coordinate transform:
//   GridSample, AffineGrid, Col2Im.
//
// All three reduce to the primitives the shape2.cc family already uses — host-computed gather/scatter
// index maps + take/take_along_axis/scatter_add + arithmetic blend — so they translate exactly to
// MLX with no bespoke kernel:
//   * GridSample : denormalize the grid coordinates on-device, gather the nearest / 4 bilinear
//                  neighbours with take_along_axis, and blend by the fractional weights. 2D form.
//   * AffineGrid : build the normalized base grid host-side from the constant `size`, then batched
//                  matmul by `theta`. 2D form.
//   * Col2Im     : fold columns back into the image by scatter-add of every block element at its
//                  host-computed output position (overlap-add). Static (constant image_shape /
//                  block_shape) form.
//
// Forms deliberately left to ORT CPU (unclaimed -> CPU fallback), because they are not expressible
// exactly / cheaply with these primitives:
//   * GridSample: padding_mode="reflection", mode="cubic"/"bicubic", and the 5-D (volumetric) form.
//   * AffineGrid: the 3-D form (theta [N,3,4], size length 5).
//   * Col2Im:     dynamic (non-constant) image_shape/block_shape, and non-float payloads (the MLX
//                 GPU scatter kernels abort on integer payloads — mirrors ScatterND in shape2.cc).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// ---- claim-time helpers ---------------------------------------------------------------------

// Read a STRING attribute at claim time, falling back to `def` when absent/other type.
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

// True iff every dim of `shape` is statically known (>= 0).
bool AllStatic(const std::vector<int64_t>& shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= 0; });
}

// True iff `vi` is a tensor(int32|int64) constant initializer (a shape/size parameter we can read at
// translate time).
bool IsConstIntTensor(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t) || !vi.IsConstantInitializer()) return false;
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// Read the int32/int64 values of a constant-initializer value info AT CLAIM TIME (widened to int64).
bool ReadConstIntAtClaim(Ort::ConstValueInfo vi, std::vector<int64_t>& out) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t) || !vi.IsConstantInitializer()) return false;
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const void* p = value.GetTensorRawData();
  if (p == nullptr && count != 0) return false;
  out.clear();
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const auto* q = static_cast<const int64_t*>(p);
    out.assign(q, q + count);
    return true;
  }
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const auto* q = static_cast<const int32_t*>(p);
    out.assign(q, q + count);
    return true;
  }
  return false;
}

// ---- translate-time helpers -----------------------------------------------------------------

mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array ScalarF(TranslationContext& ctx, float v) {
  return ctx.Keep(mlx_array_new_float32(v));
}

mlx_array ScalarI(TranslationContext& ctx, int v) { return ctx.Keep(mlx_array_new_int(v)); }

mlx_array Floor(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_floor(&r, a, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array Round(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_round(&r, a, /*decimals=*/0, ctx.stream()));
  return ctx.Keep(r);
}

// Clamp `a` to [lo, hi] (scalar bounds).
mlx_array Clip(TranslationContext& ctx, mlx_array a, float lo, float hi) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_clip(&r, a, ScalarF(ctx, lo), ScalarF(ctx, hi), ctx.stream()));
  return ctx.Keep(r);
}

// 1.0 where lo <= a <= hi, else 0.0 (as float32) — the zeros-padding validity mask for a coordinate.
mlx_array InRangeMask(TranslationContext& ctx, mlx_array a, float lo, float hi) {
  mlx_array ge = mlx_array_new();
  MLX_CHECK(mlx_greater_equal(&ge, a, ScalarF(ctx, lo), ctx.stream()));
  ctx.Keep(ge);
  mlx_array le = mlx_array_new();
  MLX_CHECK(mlx_less_equal(&le, a, ScalarF(ctx, hi), ctx.stream()));
  ctx.Keep(le);
  mlx_array both = mlx_array_new();
  MLX_CHECK(mlx_logical_and(&both, ge, le, ctx.stream()));
  ctx.Keep(both);
  return ctx.Astype(both, MLX_FLOAT32);
}

mlx_array BroadcastTo(TranslationContext& ctx, mlx_array a, const std::vector<int>& shape) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_broadcast_to(&r, a, shape.data(), shape.size(), ctx.stream()));
  return ctx.Keep(r);
}

mlx_array TakeAlongAxis(TranslationContext& ctx, mlx_array a, mlx_array idx, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_along_axis(&r, a, idx, axis, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array TakeAxis(TranslationContext& ctx, mlx_array a, mlx_array idx, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_axis(&r, a, idx, axis, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array Matmul(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_matmul(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}

// Read a constant int64/int32 parameter INPUT (image_shape / block_shape / size) at translate time.
std::vector<int64_t> ReadConstInts(TranslationContext& ctx, const TensorRef& ref) {
  mlx_array value = ctx.Resolve(ref);
  HostBytes h = ctx.RawHost(ref);
  std::vector<int64_t> out;
  out.reserve(h.count);
  if (mlx_array_dtype(value) == MLX_INT64) {
    const auto* p = static_cast<const int64_t*>(h.data);
    out.assign(p, p + h.count);
  } else {
    const auto* p = static_cast<const int32_t*>(h.data);
    for (size_t i = 0; i < h.count; ++i) out.push_back(p[i]);
  }
  return out;
}

// =============================================================================================
// GridSample (2D): X[N,C,H,W] sampled at grid[N,Hout,Wout,2] -> Y[N,C,Hout,Wout].
// grid[...,0] indexes W (x), grid[...,1] indexes H (y), both normalized to [-1,1]. Coordinates are
// denormalized on-device (align_corners variant), then the nearest / 4 bilinear neighbours are
// gathered with take_along_axis on the flattened spatial axis and blended by the fractional weights.
// padding_mode zeros (out-of-range neighbours contribute 0) and border (clamp neighbour index) only.
// =============================================================================================
void GridSampleOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Astype(Contiguous(ctx, ctx.Resolve(n.inputs[0])), MLX_FLOAT32);
  std::vector<int> xs = TranslationContext::ShapeOf(x);
  const int N = xs[0], C = xs[1], H = xs[2], W = xs[3];

  mlx_array grid = ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_FLOAT32);
  std::vector<int> gs = TranslationContext::ShapeOf(grid);
  const int Hout = gs[1], Wout = gs[2];
  const int P = Hout * Wout;

  const std::string mode = n.strings.count("mode") ? n.strings.at("mode") : "linear";
  const std::string padding =
      n.strings.count("padding_mode") ? n.strings.at("padding_mode") : "zeros";
  const bool align = n.ints.count("align_corners") ? n.ints.at("align_corners") != 0 : false;
  const bool nearest = mode == "nearest";
  const bool zeros = padding == "zeros";

  // Denormalize affine: coord = g * a + b (per axis of length L).
  //   align_corners : (g+1)/2 * (L-1)          -> a=(L-1)/2,  b=(L-1)/2
  //   else          : ((g+1)*L - 1)/2          -> a=L/2,      b=(L-1)/2
  const float ax = align ? (W - 1) / 2.0f : W / 2.0f;
  const float bx = (W - 1) / 2.0f;
  const float ay = align ? (H - 1) / 2.0f : H / 2.0f;
  const float by = (H - 1) / 2.0f;

  mlx_array xf = ctx.Reshape(x, {N, C, H * W});
  mlx_array gflat = ctx.Reshape(grid, {N, P, 2});
  mlx_array gx = ctx.Reshape(ctx.Slice(gflat, {0, 0, 0}, {N, P, 1}), {N, P});
  mlx_array gy = ctx.Reshape(ctx.Slice(gflat, {0, 0, 1}, {N, P, 2}), {N, P});

  mlx_array ix = ctx.AddA(ctx.Mul(gx, ScalarF(ctx, ax)), ScalarF(ctx, bx));  // [N,P]
  mlx_array iy = ctx.AddA(ctx.Mul(gy, ScalarF(ctx, ay)), ScalarF(ctx, by));  // [N,P]

  // Gather X at (integer) coordinates (xf_coord, yf_coord) weighted by w, accumulating into `acc`.
  // `xf_coord`/`yf_coord` are float coordinate arrays [N,P]; validity (zeros padding) is derived from
  // them before clamping the gather index into range.
  mlx_array acc = mlx_array_new();  // set on first corner
  bool have_acc = false;
  auto sample = [&](mlx_array xf_coord, mlx_array yf_coord, mlx_array w) {
    mlx_array weight = w;
    if (zeros) {
      weight = ctx.Mul(weight, InRangeMask(ctx, xf_coord, 0.0f, W - 1.0f));
      weight = ctx.Mul(weight, InRangeMask(ctx, yf_coord, 0.0f, H - 1.0f));
    }
    mlx_array xi = ctx.Astype(Clip(ctx, xf_coord, 0.0f, W - 1.0f), MLX_INT32);
    mlx_array yi = ctx.Astype(Clip(ctx, yf_coord, 0.0f, H - 1.0f), MLX_INT32);
    mlx_array flat = ctx.AddA(ctx.Mul(yi, ScalarI(ctx, W)), xi);  // [N,P] int32
    mlx_array idx = BroadcastTo(ctx, ctx.Reshape(flat, {N, 1, P}), {N, C, P});
    mlx_array g = TakeAlongAxis(ctx, xf, idx, /*axis=*/2);  // [N,C,P]
    mlx_array w3 = ctx.Reshape(weight, {N, 1, P});
    mlx_array contrib = ctx.Mul(g, w3);
    if (!have_acc) {
      acc = contrib;
      have_acc = true;
    } else {
      acc = ctx.AddA(acc, contrib);
    }
  };

  if (nearest) {
    mlx_array w1 = BroadcastTo(ctx, ScalarF(ctx, 1.0f), {N, P});
    sample(Round(ctx, ix), Round(ctx, iy), w1);
  } else {
    mlx_array x0 = Floor(ctx, ix);
    mlx_array y0 = Floor(ctx, iy);
    mlx_array x1 = ctx.AddA(x0, ScalarF(ctx, 1.0f));
    mlx_array y1 = ctx.AddA(y0, ScalarF(ctx, 1.0f));
    mlx_array wx1 = ctx.SubA(ix, x0);  // frac
    mlx_array wx0 = ctx.SubA(ScalarF(ctx, 1.0f), wx1);
    mlx_array wy1 = ctx.SubA(iy, y0);
    mlx_array wy0 = ctx.SubA(ScalarF(ctx, 1.0f), wy1);
    sample(x0, y0, ctx.Mul(wx0, wy0));
    sample(x1, y0, ctx.Mul(wx1, wy0));
    sample(x0, y1, ctx.Mul(wx0, wy1));
    sample(x1, y1, ctx.Mul(wx1, wy1));
  }

  mlx_array out = ctx.Reshape(acc, {N, C, Hout, Wout});
  out = ctx.Astype(out, MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, out));
}

bool GridSampleClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType xt, gt, ot;
  std::vector<int64_t> xshape, gshape;
  if (!TensorInfo(inputs[0], xt, &xshape) || !TensorInfo(inputs[1], gt, &gshape) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsMlxFloatType(xt) || ot != xt || !IsMlxFloatType(gt)) return false;
  // 2D form only: X[N,C,H,W], grid[N,Hout,Wout,2]. (5D volumetric left to CPU.)
  if (xshape.size() != 4 || gshape.size() != 4 || gshape.back() != 2) return false;
  if (!AllStatic(xshape) || !AllStatic(gshape)) return false;

  const std::string mode = StringAttribute(node, "mode", "linear");
  if (mode != "linear" && mode != "bilinear" && mode != "nearest") return false;  // no cubic
  const std::string padding = StringAttribute(node, "padding_mode", "zeros");
  if (padding != "zeros" && padding != "border") return false;  // no reflection
  return true;
}

// =============================================================================================
// AffineGrid (2D): theta[N,2,3] + size[4]=(N,C,H,W) -> grid[N,H,W,2].
// Build the normalized homogeneous base grid host-side ([H*W,3], rows (x_w, y_h, 1)), then batched
// matmul base @ theta^T -> [N,H*W,2] -> reshape [N,H,W,2].
// =============================================================================================
void AffineGridOp(TranslationContext& ctx, const NodeDesc& n) {
  std::vector<int64_t> size = ReadConstInts(ctx, n.inputs[1]);  // (N, C, H, W)
  const int N = static_cast<int>(size[0]);
  const int Hd = static_cast<int>(size[2]);
  const int Wd = static_cast<int>(size[3]);
  const bool align = n.ints.count("align_corners") ? n.ints.at("align_corners") != 0 : false;

  // Per-axis normalized coordinate (matches onnx reference construct_original_grid):
  //   align_corners : a[i] = -1 + i*2/(L-1)        (endpoints -1 .. 1)
  //   else          : a[i] = (2i+1)/L - 1          (cell centers)
  auto coord = [align](int i, int L) -> float {
    if (align) return L <= 1 ? -1.0f : -1.0f + static_cast<float>(i) * (2.0f / (L - 1));
    return static_cast<float>((2.0 * i + 1.0) / L - 1.0);
  };
  std::vector<float> base(static_cast<size_t>(Hd) * Wd * 3);
  for (int h = 0; h < Hd; ++h) {
    const float yh = coord(h, Hd);
    for (int w = 0; w < Wd; ++w) {
      const size_t r = (static_cast<size_t>(h) * Wd + w) * 3;
      base[r + 0] = coord(w, Wd);  // x
      base[r + 1] = yh;            // y
      base[r + 2] = 1.0f;
    }
  }
  int bshape[] = {1, Hd * Wd, 3};
  mlx_array base_arr = ctx.Keep(mlx_array_new_data(base.data(), bshape, 3, MLX_FLOAT32));

  mlx_array theta = ctx.Astype(ctx.Resolve(n.inputs[0]), MLX_FLOAT32);  // [N,2,3]
  mlx_array theta_t = ctx.Transpose(theta, {0, 2, 1});                  // [N,3,2]
  mlx_array grid = Matmul(ctx, base_arr, theta_t);                      // [N,H*W,2]
  grid = ctx.Reshape(grid, {N, Hd, Wd, 2});
  grid = ctx.Astype(grid, MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, grid));
}

bool AffineGridClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType tt, ot;
  std::vector<int64_t> tshape;
  if (!TensorInfo(inputs[0], tt, &tshape) || !TensorInfo(outputs[0], ot)) return false;
  if (!IsMlxFloatType(tt) || !IsMlxFloatType(ot)) return false;
  if (!IsConstIntTensor(inputs[1])) return false;
  std::vector<int64_t> size;
  if (!ReadConstIntAtClaim(inputs[1], size)) return false;
  // 2D form only: theta [N,2,3], size length 4. (3D form theta[N,3,4]/size len 5 left to CPU.)
  if (size.size() != 4) return false;
  if (tshape.size() != 3 || tshape[1] != 2 || tshape[2] != 3 || !AllStatic(tshape)) return false;
  for (int64_t d : size) {
    if (d < 1) return false;
  }
  return true;
}

// =============================================================================================
// Col2Im: input[N, C*prod(block), L] folded into image[N, C, image_shape...] by overlap-add.
// Each column (block position l, block element k) is added into the output at spatial position
// out_pos[d] = l_d*stride[d] + k_d*dilation[d] - pad_begin[d] (dropped if outside the image). The
// (col, out_pos) map is computed host-side; the payload is gathered from the columns and scattered
// with scatter_add. Float-only (the MLX GPU scatter kernels abort on integer payloads).
// =============================================================================================
void Col2ImOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array input = Contiguous(ctx, ctx.Resolve(n.inputs[0]));
  std::vector<int> is = TranslationContext::ShapeOf(input);
  const int Nn = is[0];
  const int L = is[2];

  std::vector<int64_t> image_shape = ReadConstInts(ctx, n.inputs[1]);
  std::vector<int64_t> block_shape = ReadConstInts(ctx, n.inputs[2]);
  const int R = static_cast<int>(image_shape.size());

  std::vector<int64_t> dil = n.int_arrays.count("dilations")
                                 ? n.int_arrays.at("dilations")
                                 : std::vector<int64_t>(R, 1);
  std::vector<int64_t> stride = n.int_arrays.count("strides") ? n.int_arrays.at("strides")
                                                              : std::vector<int64_t>(R, 1);
  std::vector<int64_t> pads = n.int_arrays.count("pads") ? n.int_arrays.at("pads")
                                                         : std::vector<int64_t>(2 * R, 0);

  int64_t K = 1;
  for (int d = 0; d < R; ++d) K *= block_shape[d];
  const int C = is[1] / static_cast<int>(K);
  int64_t S = 1;
  for (int d = 0; d < R; ++d) S *= image_shape[d];

  // Number of block positions along each spatial dim, and their (row-major) strides.
  std::vector<int64_t> npos(R), img_stride(R, 1);
  for (int d = 0; d < R; ++d) {
    npos[d] = (image_shape[d] + pads[d] + pads[R + d] - dil[d] * (block_shape[d] - 1) - 1) /
                  stride[d] +
              1;
  }
  for (int d = R - 2; d >= 0; --d) img_stride[d] = img_stride[d + 1] * image_shape[d + 1];

  // Host-compute, for every (k, l) column, the output spatial position (or drop if out of range).
  std::vector<int32_t> col_of;   // selected column index k*L + l
  std::vector<int32_t> spatial;  // target spatial flat index in [0, S)
  col_of.reserve(static_cast<size_t>(K) * L);
  spatial.reserve(static_cast<size_t>(K) * L);
  std::vector<int64_t> kc(R), lc(R);
  for (int64_t k = 0; k < K; ++k) {
    int64_t rem = k;
    for (int d = R - 1; d >= 0; --d) {
      kc[d] = rem % block_shape[d];
      rem /= block_shape[d];
    }
    for (int64_t l = 0; l < L; ++l) {
      int64_t r2 = l;
      for (int d = R - 1; d >= 0; --d) {
        lc[d] = r2 % npos[d];
        r2 /= npos[d];
      }
      int64_t p = 0;
      bool ok = true;
      for (int d = 0; d < R; ++d) {
        int64_t op = lc[d] * stride[d] + kc[d] * dil[d] - pads[d];
        if (op < 0 || op >= image_shape[d]) {
          ok = false;
          break;
        }
        p += op * img_stride[d];
      }
      if (ok) {
        col_of.push_back(static_cast<int32_t>(k * L + l));
        spatial.push_back(static_cast<int32_t>(p));
      }
    }
  }
  const int M = static_cast<int>(col_of.size());

  // Zero accumulator, flattened to [N*C*S].
  const int total = Nn * C * static_cast<int>(S);
  mlx_array out_acc = mlx_array_new();
  int oz_shape[] = {total};
  MLX_CHECK(mlx_zeros(&out_acc, oz_shape, 1, MLX_FLOAT32, ctx.stream()));
  ctx.Keep(out_acc);

  if (M > 0) {
    // Gather the M contributing columns: input[N, C*K, L] -> [N, C, K*L] -> take columns -> [N,C,M].
    mlx_array in_cols =
        ctx.Astype(ctx.Reshape(input, {Nn, C, static_cast<int>(K) * L}), MLX_FLOAT32);
    int cshape[] = {M};
    mlx_array col_idx = ctx.Keep(mlx_array_new_data(col_of.data(), cshape, 1, MLX_INT32));
    mlx_array gathered = TakeAxis(ctx, in_cols, col_idx, /*axis=*/2);  // [N,C,M]

    // Flat scatter index [N*C*M]: element (n,c,j) -> (n*C+c)*S + spatial[j].
    std::vector<int32_t> scatter_idx(static_cast<size_t>(Nn) * C * M);
    size_t w = 0;
    for (int nn = 0; nn < Nn; ++nn) {
      for (int c = 0; c < C; ++c) {
        const int32_t base = static_cast<int32_t>((static_cast<int64_t>(nn) * C + c) * S);
        for (int j = 0; j < M; ++j) scatter_idx[w++] = base + spatial[j];
      }
    }
    int ishape[] = {static_cast<int>(scatter_idx.size())};
    mlx_array idx = ctx.Keep(mlx_array_new_data(scatter_idx.data(), ishape, 1, MLX_INT32));
    mlx_array updates = ctx.Reshape(gathered, {static_cast<int>(scatter_idx.size()), 1});

    mlx_vector_array vec = mlx_vector_array_new();
    mlx_vector_array_append_value(vec, idx);
    const int axes0 = 0;
    mlx_array scattered = mlx_array_new();
    int rc = mlx_scatter_add(&scattered, out_acc, vec, updates, &axes0, 1, ctx.stream());
    mlx_vector_array_free(vec);
    MLX_CHECK(rc);
    out_acc = ctx.Keep(scattered);
  }

  std::vector<int> out_shape = {Nn, C};
  for (int d = 0; d < R; ++d) out_shape.push_back(static_cast<int>(image_shape[d]));
  mlx_array out = ctx.Reshape(out_acc, out_shape);
  out = ctx.Astype(out, MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, out));
}

bool Col2ImClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType it, ot;
  std::vector<int64_t> ishape;
  if (!TensorInfo(inputs[0], it, &ishape) || !TensorInfo(outputs[0], ot)) return false;
  // GPU scatter kernels abort on integer payloads -> keep Col2Im to MLX float types.
  if (!IsMlxFloatType(it) || ot != it) return false;
  if (ishape.size() != 3 || !AllStatic(ishape)) return false;
  if (!IsConstIntTensor(inputs[1]) || !IsConstIntTensor(inputs[2])) return false;

  std::vector<int64_t> image_shape, block_shape;
  if (!ReadConstIntAtClaim(inputs[1], image_shape) ||
      !ReadConstIntAtClaim(inputs[2], block_shape)) {
    return false;
  }
  const int R = static_cast<int>(image_shape.size());
  if (R < 1 || R > 3 || static_cast<int>(block_shape.size()) != R) return false;
  for (int64_t d : image_shape) {
    if (d < 1) return false;
  }
  int64_t K = 1;
  for (int64_t b : block_shape) {
    if (b < 1) return false;
    K *= b;
  }
  // input channel dim must be C*K for an integer C >= 1.
  if (K == 0 || ishape[1] % K != 0 || ishape[1] / K < 1) return false;

  // Attribute lengths, when present, must match the spatial rank.
  auto ints_ok = [&](const char* name, int want) {
    Ort::ConstOpAttr attr;
    Ort::Status status = node.GetAttributeByName(name, attr);
    if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
        attr.GetType() == ORT_OP_ATTR_UNDEFINED) {
      return true;  // absent -> default
    }
    std::vector<int64_t> v;
    if (attr.GetType() != ORT_OP_ATTR_INTS || !attr.GetValueArray(v).IsOK()) return false;
    return static_cast<int>(v.size()) == want;
  };
  return ints_ok("dilations", R) && ints_ok("strides", R) && ints_ok("pads", 2 * R);
}

}  // namespace

void RegisterVision2Ops(OpRegistry& registry) {
  registry.Register({"", "GridSample", kAnyOpset, kAnyOpset, &GridSampleOp, &GridSampleClaim});
  registry.Register({"", "AffineGrid", kAnyOpset, kAnyOpset, &AffineGridOp, &AffineGridClaim});
  registry.Register({"", "Col2Im", kAnyOpset, kAnyOpset, &Col2ImOp, &Col2ImClaim});
}

}  // namespace ort_mlx
