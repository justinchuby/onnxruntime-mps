// Copyright (c) 2026. Licensed under the MIT License.
//
// The MLX translation engine shared by all op handler modules. Defines:
//   * MlxDtypeFromOnnx  — the ONNX-dtype -> MLX-dtype mapping (fp32/fp16/bf16/ints/uint/bool).
//   * Plan              — the persistent per-subgraph MLX state (stream + repacked-weight cache).
//   * TranslationContext — the object a handler uses to Resolve inputs, Bind outputs, and emit MLX
//                          ops. Handlers are free functions (see op_registry.h) that take a
//                          TranslationContext& and a NodeDesc&; the context owns Resolve/Bind, the
//                          per-run env, the transient-array bookkeeping, and the MLX op helpers.
//
// mlx_backend.cc defines the non-inline methods and the BuildPlan/RunPlan/DestroyPlan engine; the
// src/ep/ops/*.cc modules include this header to implement handlers.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/c/mlx.h"
#include "mlx_backend.h"
#include "onnxruntime_cxx_api.h"

namespace ort_mlx {

// ONNX tensor element type -> MLX dtype. Covers every dtype mlx-c exposes that the decoder graph can
// carry: fp32/fp16/bf16, the signed/unsigned integer widths, and bool. Unknown types fall back to
// fp32 (the historical default) so a stray dtype never crashes the wrap. Used by Resolve/Bind,
// constant materialization, the boundary cast, and CopyOut so every tensor honors its actual dtype.
mlx_dtype MlxDtypeFromOnnx(ONNXTensorElementDataType t);

// Thrown by MLX_CHECK / handlers on an MLX-C failure or an unmappable op; caught in RunPlan.
struct MlxError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define MLX_CHECK(expr)                                                         \
  do {                                                                          \
    if ((expr) != 0) throw ort_mlx::MlxError(std::string("mlx call failed: ") + #expr); \
  } while (0)

// Lightweight per-forward profiling accumulator (env-gated via ONNX_GENAI_MLX_PROFILE=1). Splits the
// per-forward cost into graph BUILD (NodeDesc->MLX translate), mlx_eval, and boundary CopyOut, and
// counts weight-repack cache misses to confirm they are one-time (not per token). Prints a running
// average every kProfileInterval forwards to stderr.
struct MlxProfile {
  bool enabled = false;
  long forwards = 0;
  double build_us = 0, eval_us = 0, copy_us = 0;
  double win_build_us = 0, win_eval_us = 0, win_copy_us = 0;
  long cache_miss_at_start = 0, cache_misses = 0;
  static constexpr long kProfileInterval = 16;
};

// Persistent, per-subgraph MLX state: the stream, tuned memory bounds, and the cache of
// repacked-weight / wrapped-initializer MLX arrays keyed by initializer name (reused every step so
// weights are repacked exactly once, not per token).
struct Plan {
  std::vector<NodeDesc> nodes;
  mlx_stream stream;
  std::unordered_map<std::string, mlx_array> cache;  // persistent (freed in ~Plan)
  long cache_misses = 0;                              // total repack/wrap misses (profiling)
  MlxProfile prof;

  // ---- compiled decode fast-path (mlx_compile) ------------------------------------------------
  // For decode (query seq-len S==1) the graph STRUCTURE is invariant across steps: only input DATA
  // and the KV length grow. We translate the subgraph into an mlx_closure over its dynamic inputs
  // ONCE, compile it shapeless (so growing KV length needs no recompile), cache the compiled closure
  // here, and on each decode step just apply the compiled closure to the new inputs. This fuses the
  // ~1.7k primitives into far fewer kernel launches, collapsing the per-token mlx_eval cost.
  struct DynInput {
    std::string name;
    size_t ctx_index;
  };
  bool compile_enabled = false;    // env ONNX_GENAI_MLX_COMPILE (default on); false disables entirely
  bool compile_attempted = false;  // have we tried to build the compiled closure yet?
  bool compiled_valid = false;     // is `compiled` usable?
  mlx_closure compiled{nullptr};   // compiled decode closure (freed in ~Plan)
  // Transient MLX handles created during the one-time closure trace. The compiled graph is walked by
  // mlx AFTER the trace thunk returns, so these handles must outlive the trace context; we hand them
  // to the Plan (freed once in ~Plan) instead of freeing them at trace teardown.
  std::vector<mlx_array> trace_transient_;
  std::vector<DynInput> dyn_inputs;    // ordered dynamic ctx inputs = closure inputs [0..n)
  // Pre-sliced RoPE cos/sin row inputs (one per distinct cos/sin cache), appended to the closure
  // input vector AFTER dyn_inputs. Each step we slice cache[past : past+S] on the host-side eager
  // context and feed the [S,half] row in, so the position offset is data (not baked) and the
  // compiled graph stays static-shaped (no dynamic slice for shapeless compile to shape-infer).
  struct SynthRope {
    std::string key;         // env placeholder key (RopeRowKey(cache_name))
    std::string cache_name;  // the cos/sin cache to slice from plan cache each step
  };
  std::vector<SynthRope> synth_ropes;
  std::vector<const OutRef*> ext_outputs;  // external boundary outputs, in closure append order
  int rope_past_ctx_index = -1;        // ctx input to read the RoPE start (past KV length) from
  int rope_past_axis = 2;              // sequence axis of that KV input
  // Live ORT ctx for the (synchronous) first-apply trace only, so constant host reads (axes/shapes
  // via RawHost) still work while building the closure. Dynamic inputs are seeded as placeholders,
  // so they are never read from here. Null except during the trace call.
  Ort::KernelContext* trace_ctx = nullptr;

  Plan() { stream = mlx_default_gpu_stream_new(); }
  ~Plan() {
    if (compiled.ctx) mlx_closure_free(compiled);
    for (mlx_array a : trace_transient_) mlx_array_free(a);
    for (auto& kv : cache) mlx_array_free(kv.second);
    mlx_stream_free(stream);
  }
};

// Raw host bytes for a constant weight/scale tensor (surfaced either as a compile-time initializer
// or, with drop_constant_initializers=false, as a runtime context input).
struct HostBytes {
  const void* data = nullptr;
  std::vector<int64_t> shape;
  size_t count = 0;
};

// Per-Compute execution context: builds the MLX graph for one forward pass, evals once, copies out.
// Handlers receive this by reference and use its public API (Resolve/Bind/Cached + the MLX op
// helpers) to translate a node.
class TranslationContext {
 public:
  TranslationContext(Plan& plan, Ort::KernelContext& ctx)
      : plan_(plan), ctx_(&ctx), s_(plan.stream) {}

  ~TranslationContext() {
    if (retain_transient_) {
      // Hand the trace's transient handles to the Plan; the compiled graph still references them.
      auto& sink = plan_.trace_transient_;
      sink.insert(sink.end(), transient_.begin(), transient_.end());
    } else {
      for (mlx_array a : transient_) mlx_array_free(a);
    }
  }

  // Translate every node (registry dispatch), eval the whole graph once, copy boundary outputs out.
  // Dispatches to the compiled decode fast-path when eligible, else the eager per-forward build.
  void Execute();

  // ---- handler-facing API -------------------------------------------------------------------
  // Register a freshly-created MLX array for teardown at end of this run; returns it for chaining.
  mlx_array Keep(mlx_array a) {
    transient_.push_back(a);
    return a;
  }

  // Resolve a node input to an MLX array (intermediate env / wrapped ctx input / cached initializer).
  mlx_array Resolve(const TensorRef& ref);

  // Bind a node output name to a produced MLX array (visible to downstream nodes and CopyOut).
  void Bind(const OutRef& o, mlx_array a) { env_[o.name] = a; }

  // Fetch-or-build a persistent cached array under `key` using `build` (for repacked weights).
  template <typename F>
  mlx_array Cached(const std::string& key, F&& build) {
    auto it = plan_.cache.find(key);
    if (it != plan_.cache.end()) return it->second;
    mlx_array a = build();
    plan_.cache[key] = a;
    ++plan_.cache_misses;
    return a;
  }

  // Raw host bytes of a constant weight/scale tensor.
  HostBytes RawHost(const TensorRef& ref);

  mlx_stream stream() const { return s_; }

  static std::vector<int> ToInt(const std::vector<int64_t>& v);
  static std::vector<int> ShapeOf(mlx_array a);

  // ---- compiled decode support --------------------------------------------------------------
  // True while translating inside the compiled decode closure trace. In this mode RoPE uses the
  // pre-sliced cos/sin ROW placeholders fed as extra closure inputs (see Plan::synth_ropes) instead
  // of slicing the full cache at a baked position -- so no position offset is baked into the graph
  // and no dynamic slice (which shapeless compile cannot shape-infer) appears in the graph.
  bool RopeDynamic() const { return rope_dynamic_; }
  // Placeholder env key for the pre-sliced cos/sin row of the cache named `cache_name`.
  static std::string RopeRowKey(const std::string& cache_name) { return "__rope_row__" + cache_name; }
  // cos/sin rows for absolute positions [start_pos, start_pos+S). On the eager path (prefill /
  // compile-disabled) this static-slices `full` (the whole cache). On the compiled decode path it
  // returns the pre-sliced row placeholder for `cache_name` (fed as a closure input each step).
  // `half` is the rot/2 width.
  mlx_array CosSinRow(const std::string& cache_name, mlx_array full, int start_pos, int S, int half);

  // ---- MLX op helpers (each Keep()s and returns the result) ---------------------------------
  mlx_array Reshape(mlx_array a, const std::vector<int>& shape);
  mlx_array Transpose(mlx_array a, const std::vector<int>& axes);
  mlx_array Astype(mlx_array a, mlx_dtype t);
  mlx_array Mul(mlx_array a, mlx_array b);
  mlx_array AddA(mlx_array a, mlx_array b);
  mlx_array SubA(mlx_array a, mlx_array b);
  mlx_array Slice(mlx_array a, const std::vector<int>& start, const std::vector<int>& stop);
  mlx_array Concat2(mlx_array a, mlx_array b, int axis);
  mlx_array ScalarU32(uint32_t val);
  mlx_array MatMul(mlx_array a, mlx_array b);
  // Constant [hd,hd] rotate-half matrix (x @ M == rotate_half(x)); cached on the plan.
  mlx_array RotateHalfMatrix(int hd, int half);

 private:
  void Translate(const NodeDesc& n);
  void CopyOut(const OutRef& o);
  void CopyOutArray(const OutRef& o, mlx_array a);
  // Eager per-forward path: build the whole graph, one mlx_eval, copy boundary outputs out.
  void ExecuteEager();
  // Compiled decode fast-path: build+compile the closure once, then apply it each step. Returns
  // false (so the caller falls back to ExecuteEager) if the plan is not compile-eligible.
  bool ExecuteCompiledDecode();
  // One-time: discover the dynamic ctx inputs + external outputs and compile the decode closure.
  bool BuildCompiledClosure();
  // mlx_closure trace thunk (payload = Plan*): seeds dynamic-input placeholders, translates the whole
  // subgraph with dynamic RoPE, and returns the cast external boundary outputs.
  static int TraceThunk(mlx_vector_array* out, const mlx_vector_array in, void* payload);
  // Detect the query sequence length S (from the input_ids dynamic ctx input). Returns -1 if unknown.
  int DetectSeqLen();

  Plan& plan_;
  Ort::KernelContext* ctx_;  // live ORT kernel context (also valid during the compiled-decode trace)
  mlx_stream s_;
  std::unordered_map<std::string, mlx_array> env_;
  std::vector<mlx_array> transient_;
  bool rope_dynamic_ = false;
  bool retain_transient_ = false;  // trace path: hand transient handles to Plan instead of freeing
};

}  // namespace ort_mlx
