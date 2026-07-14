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

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "mlx/c/mlx.h"
#include "mlx_engine.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {
inline double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

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
    Ort::ConstValue v = ctx_->GetInput(ref.ctx_index);
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
      Ort::ConstValue v = ctx_->GetInput(ref.ctx_index);
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
mlx_array TranslationContext::MatMul(mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_matmul(&r, a, b, s_));
  return Keep(r);
}
// Constant [hd,hd] matrix M such that (x @ M) == rotate_half(x) for non-interleaved RoPE, i.e.
// rotate_half([x1,x2]) = [-x2, x1] with half = hd/2. Built once and cached on the plan (freed in
// ~Plan). This lets the compiled decode graph do rotate-half with a matmul instead of a Slice
// (which shapeless mlx_compile cannot shape-infer).
mlx_array TranslationContext::RotateHalfMatrix(int hd, int half) {
  const std::string key = "__rope_rotate_half_" + std::to_string(hd);
  auto it = plan_.cache.find(key);
  if (it != plan_.cache.end()) return it->second;
  std::vector<float> m(static_cast<size_t>(hd) * hd, 0.0f);
  for (int i = 0; i < half; ++i) {
    m[static_cast<size_t>(i + half) * hd + i] = -1.0f;  // col i (<half) picks -x[i+half]
    m[static_cast<size_t>(i) * hd + (i + half)] = 1.0f;  // col i+half picks  x[i]
  }
  int shp[2] = {hd, hd};
  mlx_array a = mlx_array_new_data(m.data(), shp, 2, MLX_FLOAT32);
  plan_.cache[key] = a;  // persistent
  return a;
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
  CopyOutArray(o, env_.at(o.name));  // already cast to MlxDtypeFromOnnx(o.type) and evaluated
}

void TranslationContext::CopyOutArray(const OutRef& o, mlx_array a) {
  std::vector<int> sh = ShapeOf(a);
  std::vector<int64_t> shp(sh.begin(), sh.end());
  size_t count = 1;
  for (int d : sh) count *= static_cast<size_t>(d);
  Ort::UnownedValue out = ctx_->GetOutput(o.ctx_index, shp);
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

// cos/sin rows for absolute positions [start_pos, start_pos+S). Eager path (prefill / compile
// disabled) uses a static slice with the concrete offset; the compiled decode trace uses a runtime
// dynamic slice keyed on the [past] placeholder so the position offset is NOT baked into the
// compiled graph (KV length grows every token). Returns the row reshaped to [1,1,S,half].
mlx_array TranslationContext::CosSinRow(const std::string& cache_name, mlx_array full, int start_pos,
                                       int S, int half) {
  if (rope_dynamic_) {
    // Compiled decode: the placeholder is the FULL-width [S, 2*half] row (cos duplicated across both
    // halves), fed as a closure input. Reshape for broadcast over [B,H,S,2*half].
    mlx_array row = env_.at(RopeRowKey(cache_name));
    return Reshape(row, {1, 1, S, 2 * half});
  }
  mlx_array row = Slice(full, {start_pos, 0}, {start_pos + S, half});
  return Reshape(row, {1, 1, S, half});
}

int TranslationContext::DetectSeqLen() {
  // Query sequence length S = trailing dim of the input_ids dynamic ctx input (decode => 1). Scans
  // the plan nodes directly so it works before the compiled closure's dyn_inputs list is built.
  for (const NodeDesc& node : plan_.nodes) {
    for (const TensorRef& in : node.inputs) {
      if (in.source == Src::CtxInput && !in.constant && in.name == "input_ids") {
        Ort::ConstValue v = ctx_->GetInput(in.ctx_index);
        std::vector<int64_t> shp = v.GetTensorTypeAndShapeInfo().GetShape();
        if (!shp.empty()) return static_cast<int>(shp.back());
      }
    }
  }
  return -1;
}

void TranslationContext::Execute() {
  MlxProfile& prof = plan_.prof;
  const bool profiling = prof.enabled;

  if (profiling) prof.cache_miss_at_start = plan_.cache_misses;

  // Dispatch: the compiled decode fast-path handles S==1 forwards (falling back to eager if the plan
  // is not compile-eligible); prefill (S>1) always uses the eager per-forward build.
  bool used_compiled = false;
  if (plan_.compile_enabled && DetectSeqLen() == 1) {
    used_compiled = ExecuteCompiledDecode();
  }
  if (!used_compiled) ExecuteEager();

  if (profiling) {
    prof.cache_misses += (plan_.cache_misses - prof.cache_miss_at_start);
    ++prof.forwards;
    if (prof.forwards % MlxProfile::kProfileInterval == 0) {
      double n = static_cast<double>(MlxProfile::kProfileInterval);
      double wb = (prof.build_us - prof.win_build_us) / n;
      double we = (prof.eval_us - prof.win_eval_us) / n;
      double wc = (prof.copy_us - prof.win_copy_us) / n;
      std::fprintf(stderr,
                   "[MLX_PROFILE] fwd=%ld window us/fwd: build=%.1f eval=%.1f copy=%.1f total=%.1f "
                   "| compiled=%d repack_misses_total=%ld (this fwd=%ld)\n",
                   prof.forwards, wb, we, wc, wb + we + wc, used_compiled ? 1 : 0,
                   prof.cache_misses, plan_.cache_misses - prof.cache_miss_at_start);
      prof.win_build_us = prof.build_us;
      prof.win_eval_us = prof.eval_us;
      prof.win_copy_us = prof.copy_us;
    }
  }
}

// Eager per-forward path: translate the whole subgraph, one mlx_eval at the boundary, copy out.
void TranslationContext::ExecuteEager() {
  MlxProfile& prof = plan_.prof;
  const bool profiling = prof.enabled;
  double t_build0 = profiling ? NowUs() : 0;

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
  double t_eval0 = profiling ? NowUs() : 0;
  MLX_CHECK(mlx_eval(outs));
  mlx_vector_array_free(outs);

  double t_copy0 = profiling ? NowUs() : 0;
  // Copy each boundary output back across the ORT boundary (accepted boundary copy).
  for (const OutRef* o : ext) CopyOut(*o);

  if (profiling) {
    double t_end = NowUs();
    prof.build_us += (t_eval0 - t_build0);
    prof.eval_us += (t_copy0 - t_eval0);
    prof.copy_us += (t_end - t_copy0);
  }
}

// mlx_closure trace thunk: seeds each dynamic ctx input as a placeholder (from the closure's input
// vector) + the pre-sliced RoPE cos/sin row inputs, translates the whole subgraph (RoPE in its
// slice-free matmul form), and returns the cast external boundary outputs. Constants (weights /
// cos-sin caches) come from the already-populated plan cache (prefill runs before any decode step,
// so they are all resident). Invoked lazily by mlx on the first mlx_closure_apply.
int TranslationContext::TraceThunk(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
  Plan* plan = static_cast<Plan*>(payload);
  try {
    TranslationContext tctx(*plan, *plan->trace_ctx);  // ctx valid: constant host reads work
    tctx.rope_dynamic_ = true;
    tctx.retain_transient_ = true;  // graph outlives this trace; Plan frees the handles
    const size_t ndyn = plan->dyn_inputs.size();
    for (size_t i = 0; i < ndyn; ++i) {
      mlx_array a = mlx_array_new();
      mlx_vector_array_get(&a, in, i);
      tctx.env_[plan->dyn_inputs[i].name] = a;
      tctx.Keep(a);  // free our handle at trace teardown (graph holds its own ref)
    }
    // Pre-sliced RoPE cos/sin row placeholders follow the dynamic ctx inputs, in synth_ropes order.
    for (size_t j = 0; j < plan->synth_ropes.size(); ++j) {
      mlx_array a = mlx_array_new();
      mlx_vector_array_get(&a, in, ndyn + j);
      tctx.env_[plan->synth_ropes[j].key] = a;
      tctx.Keep(a);
    }

    for (const NodeDesc& node : plan->nodes) tctx.Translate(node);

    mlx_vector_array res = mlx_vector_array_new();
    for (const OutRef* o : plan->ext_outputs) {
      mlx_array casted = tctx.Astype(tctx.env_.at(o->name), MlxDtypeFromOnnx(o->type));
      mlx_vector_array_append_value(res, casted);
    }
    *out = res;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] compiled-decode trace failed (%s); falling back to eager\n", e.what());
    return 1;
  }
}

// One-time discovery + compile of the decode closure. Populates plan_.dyn_inputs (ordered dynamic ctx
// inputs), plan_.ext_outputs (boundary outputs in append order), the RoPE-start source, and compiles
// the closure shapeless. Returns false (=> caller falls back to eager) if the plan is not eligible.
bool TranslationContext::BuildCompiledClosure() {
  // Ordered, de-duplicated dynamic (non-constant) ctx inputs = the closure input vector.
  std::unordered_map<std::string, bool> seen;
  for (const NodeDesc& node : plan_.nodes) {
    for (const TensorRef& in : node.inputs) {
      if (in.source == Src::CtxInput && !in.constant && !seen.count(in.name)) {
        seen[in.name] = true;
        plan_.dyn_inputs.push_back({in.name, in.ctx_index});
        // Read the RoPE start (past length) from a past-KV key input's sequence axis.
        if (plan_.rope_past_ctx_index < 0 && in.name.find(".key") != std::string::npos) {
          plan_.rope_past_ctx_index = static_cast<int>(in.ctx_index);
        }
      }
    }
  }
  // External boundary outputs, in stable node/output order (the closure appends in this order).
  for (const NodeDesc& node : plan_.nodes) {
    for (const OutRef& o : node.outputs) {
      if (o.external) plan_.ext_outputs.push_back(&o);
    }
  }
  // Distinct RoPE cos/sin caches (GQA inputs[7]/[8] when do_rotary). Their per-step [S,half] rows are
  // fed as synthetic closure inputs so the compiled graph never slices a cache at a runtime position.
  std::unordered_map<std::string, bool> synth_seen;
  for (const NodeDesc& node : plan_.nodes) {
    if (node.op_type != "GroupQueryAttention" || node.inputs.size() < 9) continue;
    const bool do_rotary = !node.ints.count("do_rotary") || node.ints.at("do_rotary");
    if (!do_rotary) continue;
    for (int idx : {7, 8}) {
      const std::string& nm = node.inputs[idx].name;
      if (synth_seen.count(nm)) continue;
      synth_seen[nm] = true;
      plan_.synth_ropes.push_back({RopeRowKey(nm), nm});
    }
  }
  if (plan_.dyn_inputs.empty() || plan_.ext_outputs.empty() || plan_.rope_past_ctx_index < 0 ||
      plan_.synth_ropes.empty()) {
    return false;  // not the expected decoder shape; stay on the eager path
  }
  // The compiled RoPE uses a [hd,hd] rotate-half matmul, which requires rotary_dim == head_dim
  // (rot == hd). Validate from the live ctx (head dim = last axis of the past-KV cache) vs the cos
  // cache width (half); fall back to eager if the model rotates only part of the head.
  {
    int hd = 0, half = 0;
    if (ctx_) {
      Ort::ConstValue v = ctx_->GetInput(static_cast<size_t>(plan_.rope_past_ctx_index));
      std::vector<int64_t> shp = v.GetTensorTypeAndShapeInfo().GetShape();
      if (!shp.empty()) hd = static_cast<int>(shp.back());
    }
    auto ci = plan_.cache.find(plan_.synth_ropes.front().cache_name);
    if (ci != plan_.cache.end()) half = ShapeOf(ci->second)[1];
    if (hd == 0 || half == 0 || 2 * half != hd) {
      return false;  // partial-rotary head not supported by the compiled path
    }
  }

  mlx_closure base = mlx_closure_new_func_payload(&TranslationContext::TraceThunk, &plan_,
                                                  /*dtor=*/nullptr);
  // Shapeless so the growing KV length never triggers a recompile.
  if (mlx_compile(&plan_.compiled, base, /*shapeless=*/true) != 0) {
    mlx_closure_free(base);
    return false;
  }
  mlx_closure_free(base);
  plan_.compiled_valid = true;
  return true;
}

// Compiled decode fast-path: apply the compiled closure to the current dynamic inputs, one eval,
// copy out. Returns false if the plan is not compile-eligible (caller falls back to eager).
bool TranslationContext::ExecuteCompiledDecode() {
  MlxProfile& prof = plan_.prof;
  const bool profiling = prof.enabled;
  double t_build0 = profiling ? NowUs() : 0;

  if (!plan_.compile_attempted) {
    plan_.compile_attempted = true;
    BuildCompiledClosure();  // sets compiled_valid on success
  }
  if (!plan_.compiled_valid) return false;

  // Gather the dynamic inputs (wrap live ORT ctx data), in closure order.
  std::vector<mlx_array> in_tmp;
  mlx_vector_array in = mlx_vector_array_new();
  for (const auto& di : plan_.dyn_inputs) {
    Ort::ConstValue v = ctx_->GetInput(di.ctx_index);
    auto info = v.GetTensorTypeAndShapeInfo();
    std::vector<int> ishp = ToInt(info.GetShape());
    mlx_array a = mlx_array_new_data(v.GetTensorRawData(), ishp.data(),
                                     static_cast<int>(ishp.size()),
                                     MlxDtypeFromOnnx(info.GetElementType()));
    in_tmp.push_back(a);
    mlx_vector_array_append_value(in, a);
  }
  int past = 0;
  {
    Ort::ConstValue v = ctx_->GetInput(static_cast<size_t>(plan_.rope_past_ctx_index));
    std::vector<int64_t> shp = v.GetTensorTypeAndShapeInfo().GetShape();
    if (plan_.rope_past_axis < static_cast<int>(shp.size())) {
      past = static_cast<int>(shp[plan_.rope_past_axis]);
    }
  }
  // Pre-slice each RoPE cos/sin cache at the current position [past, past+1) and feed the FULL-width
  // row (the half-width slice duplicated across both halves) in. Doing the slice + duplicate here,
  // outside the compiled graph, keeps the graph static-shaped (shapeless compile) and free of any
  // Slice primitive (which has no shapeless shape-inference).
  const int S = 1;  // compiled path only runs for decode
  for (const auto& sr : plan_.synth_ropes) {
    auto it = plan_.cache.find(sr.cache_name);
    if (it == plan_.cache.end()) {  // cache not resident yet -> fall back to eager
      mlx_vector_array_free(in);
      for (mlx_array a : in_tmp) mlx_array_free(a);
      return false;
    }
    const int half = ShapeOf(it->second)[1];
    mlx_array row = Slice(it->second, {past, 0}, {past + S, half});  // [S,half], Keep()'d
    mlx_array full = Concat2(row, row, 1);                           // [S,2*half], Keep()'d
    mlx_vector_array_append_value(in, full);  // evaluated lazily inside the single main eval
  }

  mlx_vector_array outs = mlx_vector_array_new();
  plan_.trace_ctx = ctx_;  // valid only for the (synchronous) first-apply trace
  int rc = mlx_closure_apply(&outs, plan_.compiled, in);
  plan_.trace_ctx = nullptr;
  double t_eval0 = profiling ? NowUs() : 0;
  if (rc == 0) rc = mlx_eval(outs);
  double t_copy0 = profiling ? NowUs() : 0;

  bool ok = (rc == 0) && (mlx_vector_array_size(outs) == plan_.ext_outputs.size());
  if (ok) {
    for (size_t i = 0; i < plan_.ext_outputs.size(); ++i) {
      mlx_array a = mlx_array_new();
      mlx_vector_array_get(&a, outs, i);
      CopyOutArray(*plan_.ext_outputs[i], a);
      mlx_array_free(a);
    }
  }
  mlx_vector_array_free(outs);
  for (mlx_array a : in_tmp) mlx_array_free(a);
  mlx_vector_array_free(in);

  if (!ok) throw MlxError("MLX: compiled decode apply failed");

  if (profiling) {
    double t_end = NowUs();
    prof.build_us += (t_eval0 - t_build0);
    prof.eval_us += (t_copy0 - t_eval0);
    prof.copy_us += (t_end - t_copy0);
  }
  return true;
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
  plan->prof.enabled = std::getenv("ONNX_GENAI_MLX_PROFILE") != nullptr;
  // Compiled decode fast-path on by default; ONNX_GENAI_MLX_COMPILE=0 forces the eager path (A/B).
  const char* ce = std::getenv("ONNX_GENAI_MLX_COMPILE");
  plan->compile_enabled = !(ce && ce[0] == '0');
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

}  // namespace ort_mlx
