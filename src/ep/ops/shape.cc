// Copyright (c) 2026. Licensed under the MIT License.
//
// Shape / data-movement op handlers (Gather, GatherElements, ScatterElements, Concat, Reshape,
// Transpose, Unsqueeze, Squeeze, Flatten, Expand, Slice, Split, Tile, Pad, Identity, Range, Shape,
// Size, SpaceToDepth, Compress, Constant, ConstantOfShape). See docs/OP_ARCHITECTURE.md §5/§6.
//
// These ops are dtype-agnostic (pure data movement): the handler resolves each data input to an MLX
// array carrying its ACTUAL dtype (fp32/fp16/bf16 AND int/uint/bool) and MLX moves the bytes through
// take/reshape/transpose/concat/... unchanged, so a single implementation covers every dtype.
//
// Many ONNX shape "attributes" (shape, axes, starts/ends/steps, pads, repeats) arrive as runtime
// INPUT tensors, not attrs. We claim ONLY the forms where those params are CONSTANT INITIALIZERS, so
// the handler can read them at translate time (ctx.RawHost); genuinely dynamic-shape forms are left
// unclaimed and run on ORT CPU (correct, just not accelerated).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- dtype gating ---------------------------------------------------------------------------

// Dtypes the pure data-movement ops can carry end-to-end (every case CopyOut can memcpy at the
// subgraph boundary). uint64 is excluded (no CopyOut case); everything else MLX maps flows through.
bool IsMovableType(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return true;
    default:
      return false;
  }
}

bool IsIntIndexType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsRangeType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
         t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// ---- claim-time constant-initializer helpers ------------------------------------------------

// True iff `vi` is a tensor(int64) constant initializer (the shape/axes/starts/ends/steps/pads/
// repeats/split parameter form we can read at translate time).
bool IsConstInt64(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t)) return false;
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && vi.IsConstantInitializer();
}

// Read the int64 values of a constant-initializer value info AT CLAIM TIME (used when the claim must
// inspect the actual numbers, e.g. Slice step signs / Pad non-negativity). Returns false (→ node
// left to CPU) if the value is not a readable int64 constant initializer.
bool ReadConstInt64AtClaim(Ort::ConstValueInfo vi, std::vector<int64_t>& out) {
  if (!IsConstInt64(vi)) return false;
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const auto* p = static_cast<const int64_t*>(value.GetTensorRawData());
  if (p == nullptr) return false;
  out.assign(p, p + count);
  return true;
}

// Read the float32 values of a constant-initializer value info AT CLAIM TIME (the Resize `scales`
// input form). Returns false (→ node left to CPU) if the value is not a readable float32 constant
// initializer.
bool ReadConstFloat32AtClaim(Ort::ConstValueInfo vi, std::vector<float>& out) {
  ONNXTensorElementDataType t;
  std::vector<int64_t> shape;
  if (!TensorInfo(vi, t, &shape) || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      !vi.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const auto* p = static_cast<const float*>(value.GetTensorRawData());
  if (p == nullptr && count != 0) return false;
  out.clear();
  if (count != 0) out.assign(p, p + count);
  return true;
}

bool ReadConstBoolAtClaim(Ort::ConstValueInfo vi, std::vector<bool>& out) {
  ONNXTensorElementDataType t;
  std::vector<int64_t> shape;
  if (!TensorInfo(vi, t, &shape) || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
      shape.size() != 1 || !vi.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const auto* p = static_cast<const bool*>(value.GetTensorRawData());
  if (p == nullptr && count != 0) return false;
  out.clear();
  if (count != 0) out.assign(p, p + count);
  return true;
}

bool StaticTensorInfo(Ort::ConstValueInfo vi, ONNXTensorElementDataType& type,
                      std::vector<int64_t>& shape) {
  if (!TensorInfo(vi, type, &shape)) return false;
  return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= 0; });
}

bool ReadConstRangeScalarAtClaim(Ort::ConstValueInfo vi, ONNXTensorElementDataType expected,
                                 double& out) {
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
  if (!TensorInfo(vi, type, &shape) || type != expected || !shape.empty() ||
      !vi.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  const void* raw = value.GetTensorRawData();
  if (raw == nullptr) return false;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      out = *static_cast<const int16_t*>(raw);
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      out = *static_cast<const int32_t*>(raw);
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      int64_t v = *static_cast<const int64_t*>(raw);
      constexpr int64_t kMaxExactDoubleInteger = int64_t{1} << 53;
      if (v < -kMaxExactDoubleInteger || v > kMaxExactDoubleInteger) return false;
      out = static_cast<double>(v);
      return true;
    }
    default:
      return false;
  }
}

// True iff the node carries an attribute named `name` (any genuine type). Used to reject
// ConstantOfShape with an explicit `value` TENSOR attribute, which NodeDesc does not carry.
bool HasAttribute(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  return status.IsOK() && static_cast<const OrtOpAttr*>(attr) != nullptr &&
         attr.GetType() != ORT_OP_ATTR_UNDEFINED;
}

// A node input slot is "present" iff ORT handed back a non-null value info with a non-empty name.
// ORT surfaces an omitted optional input (e.g. Resize's roi, or an unused scales-or-sizes slot) as
// a NULL OrtValueInfo, so the pointer MUST be checked before any ValueInfo method is called.
bool InputPresent(Ort::ConstValueInfo vi) {
  return static_cast<const OrtValueInfo*>(vi) != nullptr && !vi.GetName().empty();
}

bool TensorScalarIsInt64(Ort::ConstNode node, const char* name, int64_t expected) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(name, attr).IsOK() ||
      static_cast<const OrtOpAttr*>(attr) == nullptr || attr.GetType() != ORT_OP_ATTR_TENSOR) {
    return false;
  }
  Ort::Value value{nullptr};
  if (!attr.GetTensorAttributeAsOrtValue(value).IsOK() ||
      static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  return info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
         info.GetElementCount() == 1 &&
         *static_cast<const int64_t*>(value.GetTensorRawData()) == expected;
}

bool TensorScalarIsFloat32(Ort::ConstNode node, const char* name, float expected) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(name, attr).IsOK() ||
      static_cast<const OrtOpAttr*>(attr) == nullptr || attr.GetType() != ORT_OP_ATTR_TENSOR) {
    return false;
  }
  Ort::Value value{nullptr};
  if (!attr.GetTensorAttributeAsOrtValue(value).IsOK() ||
      static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  return info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
         info.GetElementCount() == 1 &&
         *static_cast<const float*>(value.GetTensorRawData()) == expected;
}

// Read a STRING attribute at claim time, falling back to `default_value` when absent/other type.
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

// ---- translate-time helpers -----------------------------------------------------------------

// Read a constant int64 parameter input (shape/axes/starts/...) at translate time. The claim already
// verified it is a tensor(int64) constant initializer, so RawHost yields live bytes.
std::vector<int64_t> ReadInts(TranslationContext& ctx, const TensorRef& ref) {
  HostBytes h = ctx.RawHost(ref);
  const auto* p = static_cast<const int64_t*>(h.data);
  return std::vector<int64_t>(p, p + h.count);
}

double ReadRangeScalar(TranslationContext& ctx, const TensorRef& ref) {
  mlx_array value = ctx.Resolve(ref);
  HostBytes h = ctx.RawHost(ref);
  if (h.count != 1 || h.data == nullptr) throw MlxError("Range expected a scalar initializer");
  switch (mlx_array_dtype(value)) {
    case MLX_INT16:
      return *static_cast<const int16_t*>(h.data);
    case MLX_INT32:
      return *static_cast<const int32_t*>(h.data);
    case MLX_INT64:
      return static_cast<double>(*static_cast<const int64_t*>(h.data));
    default:
      throw MlxError("Range initializer dtype is not supported");
  }
}

// A 0-d int32 scalar array (kept for teardown).
mlx_array ScalarI32(TranslationContext& ctx, int32_t v) {
  return ctx.Keep(mlx_array_new_data(&v, nullptr, 0, MLX_INT32));
}

int64_t Clamp(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Force a (possibly strided/offset/broadcast) MLX view to row-major contiguous. The shared CopyOut
// does a raw memcpy of the array's data buffer, so a boundary output produced by a view op
// (transpose / slice / expand / split) MUST be materialized contiguous first, otherwise the copied
// bytes are the wrong ones.
mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return ctx.Keep(r);
}

int NormAxis(int64_t axis, int rank) {
  if (axis < 0) axis += rank;
  return static_cast<int>(axis);
}

std::pair<int, int> ShapeInterval(int rank, int64_t start, int64_t end) {
  if (start < 0) start += rank;
  if (end < 0) end += rank;
  start = Clamp(start, 0, rank);
  end = Clamp(end, 0, rank);
  if (end < start) end = start;
  return {static_cast<int>(start), static_cast<int>(end)};
}

// Normalize + adjust negative gather indices into [0, dim) and return them as int32 (the index dtype
// MLX gather/take consume), so ONNX negative indexing is honored for take / take_along_axis.
mlx_array NormalizeIndices(TranslationContext& ctx, mlx_array indices, int dim) {
  mlx_array idx = ctx.Astype(indices, MLX_INT32);
  mlx_array dim_s = ScalarI32(ctx, dim);
  mlx_array zero_s = ScalarI32(ctx, 0);
  mlx_array neg = mlx_array_new();
  MLX_CHECK(mlx_less(&neg, idx, zero_s, ctx.stream()));
  ctx.Keep(neg);
  mlx_array wrapped = ctx.AddA(idx, dim_s);
  mlx_array out = mlx_array_new();
  MLX_CHECK(mlx_where(&out, neg, wrapped, idx, ctx.stream()));
  return ctx.Keep(out);
}

// ---- Resize coordinate-transform helpers (host-side, ONNX/ORT-CPU exact) --------------------
// The output coordinate `oj` (in the resized axis) is mapped back to a source coordinate in the
// input axis exactly as ORT's CPU Resize does. `scale` is the per-axis scale actually used (the
// provided `scales` value, or output_len/input_len when `sizes` is given).
double ResizeSrcCoord(const std::string& mode, int64_t oj, double scale, int64_t in_len,
                      int64_t out_len) {
  if (mode == "align_corners") {
    return out_len == 1 ? 0.0
                        : static_cast<double>(oj) * (in_len - 1) / static_cast<double>(out_len - 1);
  }
  if (mode == "asymmetric") {
    return static_cast<double>(oj) / scale;
  }
  if (mode == "pytorch_half_pixel") {
    return out_len > 1 ? (oj + 0.5) / scale - 0.5 : 0.0;
  }
  // half_pixel (default)
  return (oj + 0.5) / scale - 0.5;
}

// Apply nearest_mode rounding then clamp to a valid input index [0, in_len-1].
int32_t ResizeNearestIndex(const std::string& nmode, double src, int64_t in_len) {
  double v;
  if (nmode == "floor") {
    v = std::floor(src);
  } else if (nmode == "ceil") {
    v = std::ceil(src);
  } else if (nmode == "round_prefer_ceil") {
    v = std::floor(src + 0.5);  // ties -> up
  } else {                      // round_prefer_floor (default)
    v = std::ceil(src - 0.5);   // ties -> down
  }
  int64_t i = static_cast<int64_t>(v);
  return static_cast<int32_t>(Clamp(i, 0, in_len - 1));
}

// ---- handlers -------------------------------------------------------------------------------

// Gather (ai.onnx): out = take(data, indices, axis). ONNX negative indices wrap; multi-dim indices
// produce out rank = data.rank-1 + indices.rank (native take_axis semantics).
void GatherOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  mlx_array indices = ctx.Resolve(n.inputs[1]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  int dim = mlx_array_dim(data, axis);
  mlx_array idx = NormalizeIndices(ctx, indices, dim);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_axis(&r, data, idx, axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// GatherElements (ai.onnx): out[i..] = data[.., indices[i..], ..] along axis (take_along_axis).
void GatherElementsOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  mlx_array indices = ctx.Resolve(n.inputs[1]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  int dim = mlx_array_dim(data, axis);
  mlx_array idx = NormalizeIndices(ctx, indices, dim);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_along_axis(&r, data, idx, axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// ScatterElements (ai.onnx, reduction=none): inverse of GatherElements via put_along_axis.
void ScatterElementsOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  mlx_array indices = ctx.Resolve(n.inputs[1]);
  mlx_array updates = ctx.Resolve(n.inputs[2]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  mlx_array idx = NormalizeIndices(ctx, indices, mlx_array_dim(data, axis));
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_put_along_axis(&r, data, idx, updates, axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Concat (ai.onnx): concatenate all inputs along axis.
void ConcatOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_vector_array vec = mlx_vector_array_new();
  int rank = 0;
  for (size_t i = 0; i < n.inputs.size(); ++i) {
    mlx_array a = ctx.Resolve(n.inputs[i]);
    if (i == 0) rank = static_cast<int>(mlx_array_ndim(a));
    mlx_vector_array_append_value(vec, a);
  }
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  mlx_array r = mlx_array_new();
  int rc = mlx_concatenate_axis(&r, vec, axis, ctx.stream());
  mlx_vector_array_free(vec);
  MLX_CHECK(rc);
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Reshape (ai.onnx, allowzero=0): shape read from the constant `shape` input. A 0 entry copies the
// corresponding input dim (allowzero=0 semantics); a single -1 is inferred by MLX.
void ReshapeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> shape = ReadInts(ctx, n.inputs[1]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(data);
  std::vector<int> target(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0 && i < in_shape.size()) {
      target[i] = in_shape[i];  // allowzero=0: copy the input dim
    } else {
      target[i] = static_cast<int>(shape[i]);
    }
  }
  ctx.Bind(n.outputs[0], ctx.Reshape(data, target));
}

// Transpose (ai.onnx): perm from int_arrays["perm"], defaulting to a full reversal.
void TransposeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  std::vector<int> perm;
  if (n.int_arrays.count("perm")) {
    for (int64_t p : n.int_arrays.at("perm")) perm.push_back(NormAxis(p, rank));
  } else {
    for (int i = rank - 1; i >= 0; --i) perm.push_back(i);
  }
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Transpose(data, perm)));
}

// Unsqueeze (ai.onnx): insert size-1 dims at `axes` (opset-13 input form; opset<13 attr form). Axes
// index the OUTPUT tensor (mlx_expand_dims_axes semantics).
void UnsqueezeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> axes;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[1]);
  } else if (n.int_arrays.count("axes")) {
    axes = n.int_arrays.at("axes");
  }
  int out_rank = static_cast<int>(mlx_array_ndim(data)) + static_cast<int>(axes.size());
  std::vector<int> a;
  for (int64_t ax : axes) a.push_back(NormAxis(ax, out_rank));
  std::sort(a.begin(), a.end());
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_expand_dims_axes(&r, data, a.data(), a.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Squeeze (ai.onnx): remove size-1 dims at `axes`, or all size-1 dims when axes absent.
void SqueezeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  std::vector<int64_t> axes;
  bool have = false;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[1]);
    have = true;
  } else if (n.int_arrays.count("axes")) {
    axes = n.int_arrays.at("axes");
    have = true;
  }
  mlx_array r = mlx_array_new();
  if (have) {
    std::vector<int> a;
    for (int64_t ax : axes) a.push_back(NormAxis(ax, rank));
    MLX_CHECK(mlx_squeeze_axes(&r, data, a.data(), a.size(), ctx.stream()));
  } else {
    MLX_CHECK(mlx_squeeze(&r, data, ctx.stream()));
  }
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Flatten (ai.onnx): reshape to [d0*..*d(axis-1), d(axis)*..*d(n-1)] (axis default 1).
void FlattenOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(data);
  int rank = static_cast<int>(shape.size());
  int64_t axis = n.ints.count("axis") ? n.ints.at("axis") : 1;
  if (axis < 0) axis += rank;
  int outer = 1, inner = 1;
  for (int i = 0; i < rank; ++i) (static_cast<int64_t>(i) < axis ? outer : inner) *= shape[i];
  ctx.Bind(n.outputs[0], ctx.Reshape(data, {outer, inner}));
}

// Expand (ai.onnx): broadcast data to broadcast(data.shape, shape-input) (bidirectional).
void ExpandOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> target = ReadInts(ctx, n.inputs[1]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(data);
  size_t out_rank = std::max(in_shape.size(), target.size());
  std::vector<int> result(out_rank);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t d_in = i < out_rank - in_shape.size() ? 1 : in_shape[i - (out_rank - in_shape.size())];
    int64_t d_t = i < out_rank - target.size() ? 1 : target[i - (out_rank - target.size())];
    result[i] = static_cast<int>(std::max(d_in, d_t));
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_broadcast_to(&r, data, result.data(), result.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Slice (ai.onnx, opset-10 input form): starts/ends and optional axes/steps are constant inputs; only
// positive steps are claimed. Builds full-rank start/stop/stride vectors with ONNX clamping.
void SliceOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(data);
  int rank = static_cast<int>(shape.size());
  std::vector<int64_t> starts = ReadInts(ctx, n.inputs[1]);
  std::vector<int64_t> ends = ReadInts(ctx, n.inputs[2]);
  std::vector<int64_t> axes;
  if (n.inputs.size() >= 4 && n.inputs[3].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[3]);
  } else {
    for (size_t i = 0; i < starts.size(); ++i) axes.push_back(static_cast<int64_t>(i));
  }
  std::vector<int64_t> steps;
  if (n.inputs.size() >= 5 && n.inputs[4].source != Src::Absent) {
    steps = ReadInts(ctx, n.inputs[4]);
  } else {
    steps.assign(starts.size(), 1);
  }

  std::vector<int> start(rank, 0), stop(shape), stride(rank, 1);
  for (size_t i = 0; i < starts.size(); ++i) {
    int ax = NormAxis(axes[i], rank);
    int dim = shape[ax];
    int64_t s = starts[i] < 0 ? starts[i] + dim : starts[i];
    int64_t e = ends[i] < 0 ? ends[i] + dim : ends[i];
    start[ax] = static_cast<int>(Clamp(s, 0, dim));
    stop[ax] = static_cast<int>(Clamp(e, 0, dim));
    stride[ax] = static_cast<int>(steps[i]);
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_slice(&r, data, start.data(), start.size(), stop.data(), stop.size(), stride.data(),
                      stride.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Split (ai.onnx): split along axis into equal chunks (num_outputs) or explicit `split` sizes.
void SplitOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  size_t num_out = n.outputs.size();

  std::vector<int64_t> sizes;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    sizes = ReadInts(ctx, n.inputs[1]);
  } else if (n.int_arrays.count("split")) {
    sizes = n.int_arrays.at("split");
  }

  mlx_vector_array parts = mlx_vector_array_new();
  int rc;
  if (!sizes.empty()) {
    // Cumulative boundary indices (exclusive of the final section) for mlx_split_sections.
    std::vector<int> indices;
    int acc = 0;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      acc += static_cast<int>(sizes[i]);
      indices.push_back(acc);
    }
    rc = mlx_split_sections(&parts, data, indices.data(), indices.size(), axis, ctx.stream());
  } else {
    rc = mlx_split(&parts, data, static_cast<int>(num_out), axis, ctx.stream());
  }
  if (rc != 0) {
    mlx_vector_array_free(parts);
    MLX_CHECK(rc);
  }
  size_t count = mlx_vector_array_size(parts);
  for (size_t i = 0; i < count && i < num_out; ++i) {
    mlx_array part = mlx_array_new();
    mlx_vector_array_get(&part, parts, i);
    ctx.Bind(n.outputs[i], Contiguous(ctx, ctx.Keep(part)));
  }
  mlx_vector_array_free(parts);
}

// Tile (ai.onnx): repeat data `repeats[i]` times along each axis.
void TileOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> repeats = ReadInts(ctx, n.inputs[1]);
  std::vector<int> reps(repeats.begin(), repeats.end());
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_tile(&r, data, reps.data(), reps.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Pad (ai.onnx, mode=constant): pads is a constant input of 2*naxes non-negative entries; optional
// constant_value / axes inputs.
void PadOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  std::vector<int64_t> pads = ReadInts(ctx, n.inputs[1]);

  std::vector<int64_t> axes;
  if (n.inputs.size() >= 4 && n.inputs[3].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[3]);
  } else {
    for (int i = 0; i < rank; ++i) axes.push_back(i);
  }
  size_t naxes = axes.size();
  std::vector<int> ax(naxes), low(naxes), high(naxes);
  for (size_t i = 0; i < naxes; ++i) {
    ax[i] = NormAxis(axes[i], rank);
    low[i] = static_cast<int>(pads[i]);
    high[i] = static_cast<int>(pads[i + naxes]);
  }

  mlx_array pad_value;
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    pad_value = ctx.Astype(ctx.Resolve(n.inputs[2]), mlx_array_dtype(data));
  } else {
    int64_t zero = 0;
    pad_value = ctx.Astype(ctx.Keep(mlx_array_new_data(&zero, nullptr, 0, MLX_INT64)),
                           mlx_array_dtype(data));
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_pad(&r, data, ax.data(), ax.size(), low.data(), low.size(), high.data(), high.size(),
                    pad_value, "constant", ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Identity (ai.onnx): alias the input array to the output name (no copy; freed once via its owner).
void IdentityOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.Resolve(n.inputs[0]));
}

// Range (ai.onnx): constant scalar integer start/limit/delta inputs mapped to mlx_arange.
void RangeOp(TranslationContext& ctx, const NodeDesc& n) {
  double start = ReadRangeScalar(ctx, n.inputs[0]);
  double limit = ReadRangeScalar(ctx, n.inputs[1]);
  double delta = ReadRangeScalar(ctx, n.inputs[2]);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_arange(&r, start, limit, delta, MlxDtypeFromOnnx(n.outputs[0].type), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Shape (ai.onnx): materialize the known input shape (optionally sliced by start/end) as int64.
void ShapeOp(TranslationContext& ctx, const NodeDesc& n) {
  std::vector<int> input_shape = TranslationContext::ShapeOf(ctx.Resolve(n.inputs[0]));
  int rank = static_cast<int>(input_shape.size());
  int64_t start_attr = n.ints.count("start") ? n.ints.at("start") : 0;
  int64_t end_attr = n.ints.count("end") ? n.ints.at("end") : rank;
  auto [start, end] = ShapeInterval(rank, start_attr, end_attr);
  std::vector<int64_t> result;
  result.reserve(static_cast<size_t>(end - start));
  for (int i = start; i < end; ++i) result.push_back(input_shape[i]);
  int shape[] = {static_cast<int>(result.size())};
  ctx.Bind(n.outputs[0],
           ctx.Keep(mlx_array_new_data(result.data(), shape, 1, MLX_INT64)));
}

// Size (ai.onnx): materialize the total element count as an int64 scalar.
void SizeOp(TranslationContext& ctx, const NodeDesc& n) {
  int64_t size = static_cast<int64_t>(mlx_array_size(ctx.Resolve(n.inputs[0])));
  ctx.Bind(n.outputs[0], ctx.Keep(mlx_array_new_data(&size, nullptr, 0, MLX_INT64)));
}

// SpaceToDepth (ai.onnx): [N,C,H,W] -> reshape, transpose block axes before C, reshape.
void SpaceToDepthOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(data);
  int block = static_cast<int>(n.ints.at("blocksize"));
  int n_batch = shape[0], channels = shape[1], height = shape[2], width = shape[3];
  mlx_array blocked =
      ctx.Reshape(data, {n_batch, channels, height / block, block, width / block, block});
  mlx_array moved = ctx.Transpose(blocked, {0, 3, 5, 1, 2, 4});
  mlx_array result =
      ctx.Reshape(moved, {n_batch, channels * block * block, height / block, width / block});
  ctx.Bind(n.outputs[0], Contiguous(ctx, result));
}

// Compress (ai.onnx): the condition is a constant initializer, converted once to take indices.
void CompressOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  HostBytes condition = ctx.RawHost(n.inputs[1]);
  const bool* mask = static_cast<const bool*>(condition.data);
  std::vector<int32_t> selected;
  for (size_t i = 0; i < condition.count; ++i) {
    if (mask[i]) selected.push_back(static_cast<int32_t>(i));
  }
  int index_shape[] = {static_cast<int>(selected.size())};
  mlx_array indices =
      ctx.Keep(mlx_array_new_data(selected.data(), index_shape, 1, MLX_INT32));
  int axis = 0;
  if (n.ints.count("axis")) {
    axis = NormAxis(n.ints.at("axis"), static_cast<int>(mlx_array_ndim(data)));
  } else {
    data = ctx.Reshape(data, {static_cast<int>(mlx_array_size(data))});
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_axis(&r, data, indices, axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Constant scalar/list attribute forms. TENSOR-valued `value` is deliberately left to CPU because
// NodeDesc does not carry TENSOR attributes.
void ConstantOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array result;
  if (n.ints.count("value_int")) {
    int64_t value = n.ints.at("value_int");
    result = mlx_array_new_data(&value, nullptr, 0, MLX_INT64);
  } else if (n.floats.count("value_float")) {
    float value = n.floats.at("value_float");
    result = mlx_array_new_data(&value, nullptr, 0, MLX_FLOAT32);
  } else if (n.int_arrays.count("value_ints")) {
    const std::vector<int64_t>& values = n.int_arrays.at("value_ints");
    int shape[] = {static_cast<int>(values.size())};
    result = mlx_array_new_data(values.data(), shape, 1, MLX_INT64);
  } else if (n.float_arrays.count("value_floats")) {
    const std::vector<float>& values = n.float_arrays.at("value_floats");
    int shape[] = {static_cast<int>(values.size())};
    result = mlx_array_new_data(values.data(), shape, 1, MLX_FLOAT32);
  } else {
    throw MlxError("Constant attribute form is not supported");
  }
  ctx.Bind(n.outputs[0], ctx.Keep(result));
}

// ConstantOfShape: default/explicit float32 zero, plus Mobius's int64 -1 fill. NodeDesc does not
// carry TENSOR attrs, so the claim restricts explicit values to forms the handler can infer from the
// output dtype: float32 always uses zero, while int64 is accepted only when claim verified -1.
void ConstantOfShapeOp(TranslationContext& ctx, const NodeDesc& n) {
  std::vector<int64_t> shape = ReadInts(ctx, n.inputs[0]);
  std::vector<int> s(shape.begin(), shape.end());
  mlx_array r = mlx_array_new();
  if (n.outputs[0].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    int64_t minus_one = -1;
    mlx_array value = ctx.Keep(mlx_array_new_data(&minus_one, nullptr, 0, MLX_INT64));
    MLX_CHECK(mlx_full(&r, s.data(), s.size(), value, MLX_INT64, ctx.stream()));
  } else {
    MLX_CHECK(mlx_zeros(&r, s.data(), s.size(), MLX_FLOAT32, ctx.stream()));
  }
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Resize (ai.onnx): nearest + (bi)linear sampling of the spatial axes. The claim guarantees a
// constant `scales` (input 2) or `sizes` (input 3), a static input/output shape, an MLX float dtype,
// and a supported (mode, coordinate_transformation_mode, nearest_mode) combination; unsupported
// forms (cubic, roi/tf_crop, exclude_outside, antialias, `axes`, dynamic params, >4D) are left to
// ORT CPU. Sampling is SEPARABLE: each resized axis is handled independently by gathering along it
// (mlx_take_axis) with source indices computed exactly as ORT CPU does; linear additionally gathers
// the two integer neighbors per axis and blends them by the fractional weight. All coordinate math
// is host-side (input/output lengths and scales are known at translate time) so the numbers match
// ORT CPU up to float rounding.
void ResizeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(data);
  int rank = static_cast<int>(in_shape.size());

  const std::string mode = n.strings.count("mode") ? n.strings.at("mode") : "nearest";
  const std::string ctm = n.strings.count("coordinate_transformation_mode")
                              ? n.strings.at("coordinate_transformation_mode")
                              : "half_pixel";
  const std::string nmode =
      n.strings.count("nearest_mode") ? n.strings.at("nearest_mode") : "round_prefer_floor";

  // Per-axis output length and the scale used by the coordinate transform.
  std::vector<int> out_len(rank);
  std::vector<double> scale(rank);
  const bool have_sizes = n.inputs.size() > 3 && n.inputs[3].source != Src::Absent;
  if (have_sizes) {
    std::vector<int64_t> sizes = ReadInts(ctx, n.inputs[3]);
    for (int i = 0; i < rank; ++i) {
      out_len[i] = static_cast<int>(sizes[i]);
      scale[i] = static_cast<double>(out_len[i]) / in_shape[i];
    }
  } else {
    HostBytes h = ctx.RawHost(n.inputs[2]);
    const auto* sc = static_cast<const float*>(h.data);
    for (int i = 0; i < rank; ++i) {
      scale[i] = sc[i];
      out_len[i] = static_cast<int>(std::floor(scale[i] * in_shape[i]));
    }
  }

  const bool linear = mode == "linear";
  if (linear) data = ctx.Astype(data, MLX_FLOAT32);  // blend in fp32, restore dtype at the end

  for (int ax = 0; ax < rank; ++ax) {
    const int64_t li = in_shape[ax];
    const int64_t lo = out_len[ax];
    if (lo == li) continue;  // scale==1 axis (e.g. N,C) is left untouched

    if (!linear) {
      std::vector<int32_t> idx(lo);
      for (int64_t j = 0; j < lo; ++j) {
        idx[j] = ResizeNearestIndex(nmode, ResizeSrcCoord(ctm, j, scale[ax], li, lo), li);
      }
      int ishape[] = {static_cast<int>(lo)};
      mlx_array idx_arr = ctx.Keep(mlx_array_new_data(idx.data(), ishape, 1, MLX_INT32));
      mlx_array r = mlx_array_new();
      MLX_CHECK(mlx_take_axis(&r, data, idx_arr, ax, ctx.stream()));
      data = ctx.Keep(r);
      continue;
    }

    // Linear: gather the two integer neighbors and blend by the fractional weight.
    std::vector<int32_t> idx_lo(lo), idx_hi(lo);
    std::vector<float> w_lo(lo), w_hi(lo);
    for (int64_t j = 0; j < lo; ++j) {
      double src = ResizeSrcCoord(ctm, j, scale[ax], li, lo);
      double x0 = std::floor(src);
      double frac = src - x0;
      int64_t i0 = static_cast<int64_t>(x0);
      idx_lo[j] = static_cast<int32_t>(Clamp(i0, 0, li - 1));
      idx_hi[j] = static_cast<int32_t>(Clamp(i0 + 1, 0, li - 1));
      w_hi[j] = static_cast<float>(frac);
      w_lo[j] = static_cast<float>(1.0 - frac);
    }
    int ishape[] = {static_cast<int>(lo)};
    mlx_array lo_idx = ctx.Keep(mlx_array_new_data(idx_lo.data(), ishape, 1, MLX_INT32));
    mlx_array hi_idx = ctx.Keep(mlx_array_new_data(idx_hi.data(), ishape, 1, MLX_INT32));
    // Weights broadcast along the resized axis: shape is 1 everywhere except `lo` at `ax`.
    std::vector<int> wshape(rank, 1);
    wshape[ax] = static_cast<int>(lo);
    mlx_array w_lo_arr =
        ctx.Keep(mlx_array_new_data(w_lo.data(), wshape.data(), rank, MLX_FLOAT32));
    mlx_array w_hi_arr =
        ctx.Keep(mlx_array_new_data(w_hi.data(), wshape.data(), rank, MLX_FLOAT32));

    mlx_array lo_g = mlx_array_new();
    MLX_CHECK(mlx_take_axis(&lo_g, data, lo_idx, ax, ctx.stream()));
    mlx_array hi_g = mlx_array_new();
    MLX_CHECK(mlx_take_axis(&hi_g, data, hi_idx, ax, ctx.stream()));
    mlx_array blended =
        ctx.AddA(ctx.Mul(ctx.Keep(lo_g), w_lo_arr), ctx.Mul(ctx.Keep(hi_g), w_hi_arr));
    data = blended;
  }

  if (linear) data = ctx.Astype(data, MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, data));
}

// ---- claim predicates -----------------------------------------------------------------------

// Gather / GatherElements: movable data, matching output dtype, int32/int64 (dynamic) indices.
bool GatherLikeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, idx, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(inputs[1], idx) || !TensorInfo(outputs[0], out)) {
    return false;
  }
  return IsMovableType(data) && out == data && IsIntIndexType(idx);
}

bool ScatterElementsClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1 ||
      StringAttribute(node, "reduction", "none") != "none") {
    return false;
  }
  ONNXTensorElementDataType data, indices, updates, out;
  std::vector<int64_t> data_shape, index_shape, update_shape, out_shape;
  if (!StaticTensorInfo(inputs[0], data, data_shape) ||
      !StaticTensorInfo(inputs[1], indices, index_shape) ||
      !StaticTensorInfo(inputs[2], updates, update_shape) ||
      !StaticTensorInfo(outputs[0], out, out_shape)) {
    return false;
  }
  // mlx_put_along_axis's GPU kernel does not support int64 payloads (it aborts instead of returning
  // an error), so keep the claim to MLX floating types used by Mobius routing/logit scatters.
  if (!IsMlxFloatType(data) || !IsIntIndexType(indices) || updates != data || out != data ||
      data_shape.empty() || index_shape != update_shape || index_shape.size() != data_shape.size() ||
      out_shape != data_shape) {
    return false;
  }
  int rank = static_cast<int>(data_shape.size());
  int64_t axis = IntAttribute(node, "axis", 0);
  if (axis < -rank || axis >= rank) return false;
  int ax = NormAxis(axis, rank);
  for (int i = 0; i < rank; ++i) {
    if (i != ax && index_shape[i] > data_shape[i]) return false;
  }
  return true;
}

bool ConcatClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType out;
  if (!TensorInfo(outputs[0], out) || !IsMovableType(out)) return false;
  for (const auto& in : inputs) {
    ONNXTensorElementDataType t;
    if (!TensorInfo(in, t) || t != out) return false;
  }
  return true;
}

bool ReshapeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data || !IsConstInt64(inputs[1])) return false;
  return IntAttribute(node, "allowzero", 0) == 0;  // allowzero=1 (literal-0 dims) left to CPU
}

bool TransposeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data;
}

// Unsqueeze: axes as a constant int64 input (opset-13) or an INTS attr (opset<13).
bool UnsqueezeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (inputs.size() == 2) return IsConstInt64(inputs[1]);
  return inputs.size() == 1;  // opset<13 attr form (axes read from int_arrays at translate)
}

// Squeeze: like Unsqueeze but the no-axes form (squeeze all size-1 dims) is also allowed.
bool SqueezeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (inputs.size() == 2) return IsConstInt64(inputs[1]);
  return inputs.size() == 1;
}

bool FlattenClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data;
}

bool ExpandClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data && IsConstInt64(inputs[1]);
}

bool SliceClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (!IsConstInt64(inputs[1]) || !IsConstInt64(inputs[2])) return false;
  if (inputs.size() >= 4 && !inputs[3].GetName().empty() && !IsConstInt64(inputs[3])) return false;
  if (inputs.size() >= 5 && !inputs[4].GetName().empty()) {
    std::vector<int64_t> steps;
    if (!ReadConstInt64AtClaim(inputs[4], steps)) return false;
    for (int64_t st : steps) {
      if (st < 1) return false;  // negative / zero strides left to CPU
    }
  }
  return true;
}

bool SplitClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.empty()) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !IsMovableType(data)) return false;
  for (const auto& o : outputs) {
    if (!TensorInfo(o, out) || out != data) return false;
  }
  if (inputs.size() == 2 && !inputs[1].GetName().empty()) return IsConstInt64(inputs[1]);
  return true;
}

bool TileClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data && IsConstInt64(inputs[1]);
}

bool PadClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (StringAttribute(node, "mode", "constant") != "constant") return false;
  std::vector<int64_t> pads;
  if (!ReadConstInt64AtClaim(inputs[1], pads)) return false;
  for (int64_t p : pads) {
    if (p < 0) return false;  // negative pads (cropping) left to CPU
  }
  if (inputs.size() >= 3 && !inputs[2].GetName().empty()) {
    ONNXTensorElementDataType cv;
    if (!TensorInfo(inputs[2], cv) || cv != data) return false;
  }
  if (inputs.size() >= 4 && !inputs[3].GetName().empty() && !IsConstInt64(inputs[3])) return false;
  return true;
}

bool IdentityClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data;
}

bool RangeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType type, out;
  std::vector<int64_t> out_shape;
  if (!TensorInfo(inputs[0], type) || !IsRangeType(type) ||
      !StaticTensorInfo(outputs[0], out, out_shape) || out != type || out_shape.size() != 1) {
    return false;
  }
  double start, limit, delta;
  if (!ReadConstRangeScalarAtClaim(inputs[0], type, start) ||
      !ReadConstRangeScalarAtClaim(inputs[1], type, limit) ||
      !ReadConstRangeScalarAtClaim(inputs[2], type, delta) || delta == 0.0) {
    return false;
  }
  double count = std::max(std::ceil((limit - start) / delta), 0.0);
  return std::isfinite(count) && count <= std::numeric_limits<int>::max() &&
         out_shape[0] == static_cast<int64_t>(count);
}

bool ShapeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  std::vector<int64_t> input_shape, output_shape;
  if (!StaticTensorInfo(inputs[0], data, input_shape) ||
      !StaticTensorInfo(outputs[0], out, output_shape) || !IsMovableType(data) ||
      out != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || output_shape.size() != 1) {
    return false;
  }
  int rank = static_cast<int>(input_shape.size());
  auto interval =
      ShapeInterval(rank, IntAttribute(node, "start", 0), IntAttribute(node, "end", rank));
  return output_shape[0] == interval.second - interval.first;
}

bool SizeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  std::vector<int64_t> input_shape, output_shape;
  return StaticTensorInfo(inputs[0], data, input_shape) &&
         StaticTensorInfo(outputs[0], out, output_shape) && IsMovableType(data) &&
         out == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && output_shape.empty();
}

bool SpaceToDepthClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  std::vector<int64_t> input_shape, output_shape;
  if (!StaticTensorInfo(inputs[0], data, input_shape) ||
      !StaticTensorInfo(outputs[0], out, output_shape) || !IsMovableType(data) || out != data ||
      input_shape.size() != 4 || output_shape.size() != 4) {
    return false;
  }
  int64_t block = IntAttribute(node, "blocksize", 0);
  if (block <= 0 || input_shape[2] % block != 0 || input_shape[3] % block != 0) return false;
  std::vector<int64_t> expected = {input_shape[0], input_shape[1] * block * block,
                                   input_shape[2] / block, input_shape[3] / block};
  return output_shape == expected;
}

bool CompressClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  std::vector<int64_t> input_shape, output_shape;
  std::vector<bool> condition;
  if (!StaticTensorInfo(inputs[0], data, input_shape) ||
      !StaticTensorInfo(outputs[0], out, output_shape) || !IsMovableType(data) || out != data ||
      input_shape.empty() || !ReadConstBoolAtClaim(inputs[1], condition)) {
    return false;
  }
  int64_t true_count = static_cast<int64_t>(std::count(condition.begin(), condition.end(), true));
  if (HasAttribute(node, "axis")) {
    int rank = static_cast<int>(input_shape.size());
    int64_t axis = IntAttribute(node, "axis", 0);
    if (axis < -rank || axis >= rank) return false;
    int ax = NormAxis(axis, rank);
    if (condition.size() > static_cast<size_t>(input_shape[ax]) ||
        output_shape.size() != input_shape.size()) {
      return false;
    }
    std::vector<int64_t> expected = input_shape;
    expected[ax] = true_count;
    return output_shape == expected;
  }
  int64_t size = 1;
  for (int64_t dim : input_shape) size *= dim;
  return condition.size() <= static_cast<size_t>(size) &&
         output_shape == std::vector<int64_t>{true_count};
}

bool ConstantClaim(Ort::ConstNode node) {
  if (!node.GetInputs().empty() || node.GetOutputs().size() != 1) return false;
  ONNXTensorElementDataType out;
  std::vector<int64_t> shape;
  if (!StaticTensorInfo(node.GetOutputs()[0], out, shape)) return false;

  struct Form {
    const char* name;
    OrtOpAttrType attr_type;
    ONNXTensorElementDataType output_type;
    bool scalar;
  };
  const Form forms[] = {
      {"value_int", ORT_OP_ATTR_INT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, true},
      {"value_float", ORT_OP_ATTR_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true},
      {"value_ints", ORT_OP_ATTR_INTS, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, false},
      {"value_floats", ORT_OP_ATTR_FLOATS, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, false},
  };
  int matched = 0;
  for (const Form& form : forms) {
    Ort::ConstOpAttr attr;
    Ort::Status status = node.GetAttributeByName(form.name, attr);
    if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
        attr.GetType() == ORT_OP_ATTR_UNDEFINED) {
      continue;
    }
    if (attr.GetType() != form.attr_type || out != form.output_type) return false;
    if (form.scalar) {
      if (!shape.empty()) return false;
    } else {
      size_t count = 0;
      if (form.attr_type == ORT_OP_ATTR_INTS) {
        std::vector<int64_t> values;
        if (!attr.GetValueArray(values).IsOK()) return false;
        count = values.size();
      } else {
        std::vector<float> values;
        if (!attr.GetValueArray(values).IsOK()) return false;
        count = values.size();
      }
      if (shape != std::vector<int64_t>{static_cast<int64_t>(count)}) return false;
    }
    ++matched;
  }
  return matched == 1 && !HasAttribute(node, "value") && !HasAttribute(node, "sparse_value");
}

bool ConstantOfShapeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType out;
  if (!TensorInfo(outputs[0], out) || !IsMovableType(out)) return false;
  std::vector<int64_t> shape;
  if (!ReadConstInt64AtClaim(inputs[0], shape)) return false;
  for (int64_t dim : shape) {
    if (dim < 0 || dim > std::numeric_limits<int>::max()) return false;
  }
  if (!HasAttribute(node, "value")) {
    return out == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  if (out == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return TensorScalarIsFloat32(node, "value", 0.0f);
  }
  return out == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
         TensorScalarIsInt64(node, "value", -1);
}

// Resize: claim ONLY the exactly-translatable forms. Requires a constant `scales`/`sizes`, a static
// input+output shape, an MLX float dtype, spatial-only resize (N,C unchanged for rank>=3), and a
// supported (mode, coordinate_transformation_mode, nearest_mode). Everything else — cubic, roi /
// tf_crop_and_resize, exclude_outside, antialias, the `axes` attribute, non-"stretch"
// keep_aspect_ratio_policy, dynamic params and >4D — is left to ORT CPU.
bool ResizeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1) return false;

  ONNXTensorElementDataType data_t, out_t;
  std::vector<int64_t> in_shape, out_shape;
  if (!StaticTensorInfo(inputs[0], data_t, in_shape) ||
      !StaticTensorInfo(outputs[0], out_t, out_shape)) {
    return false;
  }
  if (!IsMlxFloatType(data_t) || out_t != data_t) return false;
  const int rank = static_cast<int>(in_shape.size());
  if (rank < 1 || rank > 4 || out_shape.size() != static_cast<size_t>(rank)) return false;

  const std::string mode = StringAttribute(node, "mode", "nearest");
  if (mode != "nearest" && mode != "linear") return false;
  const std::string ctm = StringAttribute(node, "coordinate_transformation_mode", "half_pixel");
  if (ctm != "half_pixel" && ctm != "asymmetric" && ctm != "align_corners" &&
      ctm != "pytorch_half_pixel") {
    return false;
  }
  if (mode == "nearest") {
    const std::string nmode = StringAttribute(node, "nearest_mode", "round_prefer_floor");
    if (nmode != "round_prefer_floor" && nmode != "round_prefer_ceil" && nmode != "floor" &&
        nmode != "ceil") {
      return false;
    }
  }
  if (IntAttribute(node, "exclude_outside", 0) != 0) return false;
  if (IntAttribute(node, "antialias", 0) != 0) return false;
  if (HasAttribute(node, "axes")) return false;
  if (StringAttribute(node, "keep_aspect_ratio_policy", "stretch") != "stretch") return false;

  // roi (input 1) must be omitted (tf_crop_and_resize is left to CPU).
  if (inputs.size() >= 2 && InputPresent(inputs[1])) return false;

  const bool has_scales = inputs.size() >= 3 && InputPresent(inputs[2]);
  const bool has_sizes = inputs.size() >= 4 && InputPresent(inputs[3]);
  if (has_scales == has_sizes) return false;  // exactly one of scales/sizes

  std::vector<int64_t> computed(rank);
  if (has_sizes) {
    std::vector<int64_t> sizes;
    if (!ReadConstInt64AtClaim(inputs[3], sizes) || sizes.size() != static_cast<size_t>(rank)) {
      return false;
    }
    computed = sizes;
  } else {
    std::vector<float> scales;
    if (!ReadConstFloat32AtClaim(inputs[2], scales) || scales.size() != static_cast<size_t>(rank)) {
      return false;
    }
    for (int i = 0; i < rank; ++i) {
      if (!(scales[i] > 0.0f)) return false;
      computed[i] = static_cast<int64_t>(std::floor(static_cast<double>(scales[i]) * in_shape[i]));
    }
  }
  for (int i = 0; i < rank; ++i) {
    if (computed[i] < 1 || computed[i] != out_shape[i]) return false;
  }
  // Spatial-only: outer batch/channel axes (N,C) must be unchanged for rank>=3.
  if (rank >= 3 && (computed[0] != in_shape[0] || computed[1] != in_shape[1])) return false;
  return true;
}

}  // namespace

void RegisterShapeOps(OpRegistry& registry) {
  registry.Register({"", "Gather", kAnyOpset, kAnyOpset, &GatherOp, &GatherLikeClaim});
  registry.Register(
      {"", "GatherElements", kAnyOpset, kAnyOpset, &GatherElementsOp, &GatherLikeClaim});
  registry.Register(
      {"", "ScatterElements", kAnyOpset, kAnyOpset, &ScatterElementsOp, &ScatterElementsClaim});
  registry.Register({"", "Concat", kAnyOpset, kAnyOpset, &ConcatOp, &ConcatClaim});
  registry.Register({"", "Reshape", kAnyOpset, kAnyOpset, &ReshapeOp, &ReshapeClaim});
  registry.Register({"", "Transpose", kAnyOpset, kAnyOpset, &TransposeOp, &TransposeClaim});
  registry.Register({"", "Unsqueeze", kAnyOpset, kAnyOpset, &UnsqueezeOp, &UnsqueezeClaim});
  registry.Register({"", "Squeeze", kAnyOpset, kAnyOpset, &SqueezeOp, &SqueezeClaim});
  registry.Register({"", "Flatten", kAnyOpset, kAnyOpset, &FlattenOp, &FlattenClaim});
  registry.Register({"", "Expand", kAnyOpset, kAnyOpset, &ExpandOp, &ExpandClaim});
  registry.Register({"", "Slice", kAnyOpset, kAnyOpset, &SliceOp, &SliceClaim});
  registry.Register({"", "Split", kAnyOpset, kAnyOpset, &SplitOp, &SplitClaim});
  registry.Register({"", "Tile", kAnyOpset, kAnyOpset, &TileOp, &TileClaim});
  registry.Register({"", "Pad", kAnyOpset, kAnyOpset, &PadOp, &PadClaim});
  registry.Register({"", "Identity", kAnyOpset, kAnyOpset, &IdentityOp, &IdentityClaim});
  registry.Register({"", "Range", kAnyOpset, kAnyOpset, &RangeOp, &RangeClaim});
  registry.Register({"", "Shape", kAnyOpset, kAnyOpset, &ShapeOp, &ShapeClaim});
  registry.Register({"", "Size", kAnyOpset, kAnyOpset, &SizeOp, &SizeClaim});
  registry.Register(
      {"", "SpaceToDepth", kAnyOpset, kAnyOpset, &SpaceToDepthOp, &SpaceToDepthClaim});
  registry.Register({"", "Compress", kAnyOpset, kAnyOpset, &CompressOp, &CompressClaim});
  registry.Register({"", "Constant", kAnyOpset, kAnyOpset, &ConstantOp, &ConstantClaim});
  registry.Register(
      {"", "ConstantOfShape", kAnyOpset, kAnyOpset, &ConstantOfShapeOp, &ConstantOfShapeClaim});
  registry.Register({"", "Resize", kAnyOpset, kAnyOpset, &ResizeOp, &ResizeClaim});
}

}  // namespace ort_mps_mlx
