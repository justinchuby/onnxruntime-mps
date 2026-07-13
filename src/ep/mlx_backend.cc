// Copyright (c) 2026. Licensed under the MIT License.
//
// The MLX (mlx-c) translation ENGINE — the SOLE compute path of the MLX-native ORT execution
// provider. This file owns the plan/eval machinery (Plan lifetime, per-run TranslationContext,
// Resolve/Bind, boundary eval + copy-out) and the BuildPlan/RunPlan/DestroyPlan API. The actual
// ONNX->MLX op translations live in the modular registry (src/ep/ops/*.cc, dispatched via
// op_registry.h); this engine just resolves inputs, invokes the registered handler for each node,
// runs ONE mlx_eval at the subgraph boundary, and copies the boundary tensors across the ORT
// boundary. There are no hand-tuned .metal kernels and no fallback (mlx-c is a hard dependency).
//
// See mlx_engine.h (Plan + TranslationContext), op_registry.h (the (domain,op,opset) registry), and
// docs/OP_ARCHITECTURE.md.

#include "mlx_backend.h"

#include <cstdlib>
#include <cstring>

#include "mlx/c/mlx.h"
#include "mlx_engine.h"
#include "op_registry.h"

namespace ort_mps_mlx {

mlx_dtype MlxDtypeFromOnnx(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return MLX_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return MLX_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return MLX_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return MLX_FLOAT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return MLX_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return MLX_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return MLX_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return MLX_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return MLX_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return MLX_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return MLX_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return MLX_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return MLX_BOOL;
    default: return MLX_FLOAT32;
  }
}

// ---- TranslationContext: shape/array bookkeeping helpers -------------------------------------

std::vector<int> TranslationContext::ToInt(const std::vector<int64_t>& v) {
  std::vector<int> r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = static_cast<int>(v[i]);
  return r;
}

std::vector<int> TranslationContext::ShapeOf(mlx_array a) {
  size_t nd = mlx_array_ndim(a);
  const int* sh = mlx_array_shape(a);
  return std::vector<int>(sh, sh + nd);
}

// Raw host bytes for a weight/scale tensor. Constant initializers are surfaced by ORT either as
// compile-time initializers (init_data) or, with drop_constant_initializers=false, as runtime
// context inputs. Handle both so weight repack works regardless of how ORT hoisted them. The
// returned pointer is valid for the current run; MatMulNBits repacks once and caches, so reading at
// the first run is sufficient.
HostBytes TranslationContext::RawHost(const TensorRef& ref) {
  HostBytes h;
  if (ref.source == Src::Initializer) {
    h.data = ref.init_data;
    h.shape = ref.init_shape;
    h.count = ref.init_count;
  } else if (ref.source == Src::CtxInput) {
    Ort::ConstValue v = ctx_.GetInput(ref.ctx_index);
    auto info = v.GetTensorTypeAndShapeInfo();
    h.data = v.GetTensorRawData();
    h.shape = info.GetShape();
    h.count = info.GetElementCount();
  } else {
    throw MlxError("MLX: RawHost on non-constant input " + ref.name);
  }
  return h;
}

// ---- input resolution -----------------------------------------------------------------------
// Intermediate -> produced env; CtxInput -> wrap ORT input (per-run); Initializer -> wrap raw once
// and cache persistently on the plan (gammas, biases, cos/sin, embedding table, ...). Each tensor is
// wrapped with its ACTUAL dtype via MlxDtypeFromOnnx, so fp16/bf16/int graphs flow through unchanged.
mlx_array TranslationContext::Resolve(const TensorRef& ref) {
  switch (ref.source) {
    case Src::Intermediate: {
      auto it = env_.find(ref.name);
      if (it == env_.end()) throw MlxError("MLX: missing intermediate " + ref.name);
      return it->second;
    }
    case Src::CtxInput: {
      // Constant ctx inputs (hoisted initializers) are wrapped once and cached persistently on the
      // plan; genuinely dynamic inputs (ids, position, KV cache) are wrapped per-run in env_.
      if (ref.constant) {
        auto ci = plan_.cache.find(ref.name);
        if (ci != plan_.cache.end()) return ci->second;
      } else {
        auto it = env_.find(ref.name);
        if (it != env_.end()) return it->second;
      }
      Ort::ConstValue v = ctx_.GetInput(ref.ctx_index);
      auto info = v.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> shp = info.GetShape();
      std::vector<int> ishp = ToInt(shp);
      mlx_array raw = mlx_array_new_data(v.GetTensorRawData(), ishp.data(),
                                         static_cast<int>(ishp.size()),
                                         MlxDtypeFromOnnx(info.GetElementType()));
      if (ref.constant) {
        plan_.cache[ref.name] = raw;  // persistent copy; ctx data is read only on the first run
        return raw;
      }
      mlx_array a = Keep(raw);
      env_[ref.name] = a;
      return a;
    }
    case Src::Initializer: {
      auto it = plan_.cache.find(ref.name);
      if (it != plan_.cache.end()) return it->second;
      std::vector<int> ishp = ToInt(ref.init_shape);
      mlx_array a = mlx_array_new_data(ref.init_data, ishp.data(), static_cast<int>(ishp.size()),
                                       MlxDtypeFromOnnx(ref.init_type));
      plan_.cache[ref.name] = a;  // persistent
      return a;
    }
    default:
      throw MlxError("MLX: absent input");
  }
}

// ---- MLX op helpers (each Keep()s and returns the result) -----------------------------------

mlx_array TranslationContext::Reshape(mlx_array a, const std::vector<int>& shape) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_reshape(&r, a, shape.data(), shape.size(), s_));
  return Keep(r);
}
mlx_array TranslationContext::Transpose(mlx_array a, const std::vector<int>& axes) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_transpose_axes(&r, a, axes.data(), axes.size(), s_));
  return Keep(r);
}
mlx_array TranslationContext::Astype(mlx_array a, mlx_dtype t) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_astype(&r, a, t, s_));
  return Keep(r);
}
mlx_array TranslationContext::Mul(mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_multiply(&r, a, b, s_));
  return Keep(r);
}
mlx_array TranslationContext::AddA(mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_add(&r, a, b, s_));
  return Keep(r);
}
mlx_array TranslationContext::SubA(mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_subtract(&r, a, b, s_));
  return Keep(r);
}
mlx_array TranslationContext::Slice(mlx_array a, const std::vector<int>& start,
                                    const std::vector<int>& stop) {
  std::vector<int> strides(start.size(), 1);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_slice(&r, a, start.data(), start.size(), stop.data(), stop.size(), strides.data(),
                      strides.size(), s_));
  return Keep(r);
}
mlx_array TranslationContext::Concat2(mlx_array a, mlx_array b, int axis) {
  mlx_vector_array v = mlx_vector_array_new();
  mlx_vector_array_append_value(v, a);
  mlx_vector_array_append_value(v, b);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_concatenate_axis(&r, v, axis, s_));
  mlx_vector_array_free(v);
  return Keep(r);
}
mlx_array TranslationContext::ScalarU32(uint32_t val) {
  mlx_array r = mlx_array_new_data(&val, nullptr, 0, MLX_UINT32);
  return Keep(r);
}

// ---- dispatch + boundary eval / copy-out ----------------------------------------------------

// Dispatch one node through the modular op registry (keyed by domain, op_type, opset range). A node
// with no registered handler is a hard error — GetCapability claims only registered ops.
void TranslationContext::Translate(const NodeDesc& n) {
  OpHandler handler = OpRegistry::Instance().Find(n.domain, n.op_type, n.since_version);
  if (!handler) {
    throw MlxError("MLX: no translation for op " +
                   (n.domain.empty() ? std::string("ai.onnx") : n.domain) + "::" + n.op_type);
  }
  handler(*this, n);
}

void TranslationContext::CopyOut(const OutRef& o) {
  mlx_array a = env_.at(o.name);  // already cast to MlxDtypeFromOnnx(o.type) and evaluated
  std::vector<int> sh = ShapeOf(a);
  std::vector<int64_t> shp(sh.begin(), sh.end());
  size_t count = 1;
  for (int d : sh) count *= static_cast<size_t>(d);
  Ort::UnownedValue out = ctx_.GetOutput(o.ctx_index, shp);
  void* dst = out.GetTensorMutableRawData();
  // Typed memcpy matching the ORT output element type (must mirror MlxDtypeFromOnnx()).
  switch (o.type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      std::memcpy(dst, mlx_array_data_float32(a), count * sizeof(float));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      std::memcpy(dst, mlx_array_data_float16(a), count * sizeof(uint16_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      std::memcpy(dst, mlx_array_data_bfloat16(a), count * sizeof(uint16_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      std::memcpy(dst, mlx_array_data_float64(a), count * sizeof(double));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      std::memcpy(dst, mlx_array_data_int64(a), count * sizeof(int64_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      std::memcpy(dst, mlx_array_data_int32(a), count * sizeof(int32_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      std::memcpy(dst, mlx_array_data_int16(a), count * sizeof(int16_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      std::memcpy(dst, mlx_array_data_int8(a), count * sizeof(int8_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      std::memcpy(dst, mlx_array_data_uint32(a), count * sizeof(uint32_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      std::memcpy(dst, mlx_array_data_uint16(a), count * sizeof(uint16_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      std::memcpy(dst, mlx_array_data_uint8(a), count * sizeof(uint8_t));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      std::memcpy(dst, mlx_array_data_bool(a), count * sizeof(bool));
      break;
    default:
      throw MlxError("MLX: unsupported boundary output dtype for " + o.name);
  }
}

void TranslationContext::Execute() {
  for (const NodeDesc& node : plan_.nodes) Translate(node);

  // Collect boundary outputs and evaluate the whole graph in one shot. Each boundary array is cast
  // to its ORT output dtype here (before eval) so CopyOut can do a straight typed memcpy — the fused
  // subgraph can emit non-fp32 boundary tensors (e.g. int64 index math, fp16/bf16 activations).
  mlx_vector_array outs = mlx_vector_array_new();
  std::vector<const OutRef*> ext;
  for (const NodeDesc& node : plan_.nodes) {
    for (const OutRef& o : node.outputs) {
      if (o.external && env_.count(o.name)) {
        mlx_array casted = Astype(env_.at(o.name), MlxDtypeFromOnnx(o.type));
        env_[o.name] = casted;  // rebind so CopyOut reads the cast (and evaluated) array
        ext.push_back(&o);
        mlx_vector_array_append_value(outs, casted);
      }
    }
  }
  MLX_CHECK(mlx_eval(outs));
  mlx_vector_array_free(outs);

  // Copy each boundary output back across the ORT boundary (accepted boundary copy).
  for (const OutRef* o : ext) CopyOut(*o);
}

// ---- public plan API ------------------------------------------------------------------------

Plan* BuildPlan(std::vector<NodeDesc> nodes, std::string& error) {
  for (const NodeDesc& n : nodes) {
    // Consult the SAME registry the translator dispatches through — a claimed subgraph containing an
    // unregistered op is a hard error (there is no hand-kernel fallback).
    if (!OpRegistry::Instance().Find(n.domain, n.op_type, n.since_version)) {
      error = "MLX backend cannot translate op '" +
              (n.domain.empty() ? std::string("ai.onnx") : n.domain) + "::" + n.op_type + "'";
      return nullptr;
    }
  }
  // Bound MLX's caching allocator so it coexists with our MTLBuffer pool (memory-safety note).
  size_t prev = 0;
  mlx_set_cache_limit(&prev, static_cast<size_t>(512) << 20);  // 512 MB cache cap
  mlx_set_wired_limit(&prev, static_cast<size_t>(1) << 30);    // 1 GB wired cap
  auto* plan = new Plan();
  plan->nodes = std::move(nodes);
  return plan;
}

void DestroyPlan(Plan* plan) { delete plan; }

bool RunPlan(Plan& plan, Ort::KernelContext& ctx, std::string& error) {
  try {
    TranslationContext run(plan, ctx);
    run.Execute();
    return true;
  } catch (const std::exception& ex) {
    error = ex.what();
    return false;
  }
}

}  // namespace ort_mps_mlx
