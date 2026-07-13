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

namespace ort_mps_mlx {

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
    if ((expr) != 0) throw ort_mps_mlx::MlxError(std::string("mlx call failed: ") + #expr); \
  } while (0)

// Persistent, per-subgraph MLX state: the stream, tuned memory bounds, and the cache of
// repacked-weight / wrapped-initializer MLX arrays keyed by initializer name (reused every step so
// weights are repacked exactly once, not per token).
struct Plan {
  std::vector<NodeDesc> nodes;
  mlx_stream stream;
  std::unordered_map<std::string, mlx_array> cache;  // persistent (freed in ~Plan)

  Plan() { stream = mlx_default_gpu_stream_new(); }
  ~Plan() {
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
      : plan_(plan), ctx_(ctx), s_(plan.stream) {}

  ~TranslationContext() {
    for (mlx_array a : transient_) mlx_array_free(a);
  }

  // Translate every node (registry dispatch), eval the whole graph once, copy boundary outputs out.
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
    return a;
  }

  // Raw host bytes of a constant weight/scale tensor.
  HostBytes RawHost(const TensorRef& ref);

  mlx_stream stream() const { return s_; }

  static std::vector<int> ToInt(const std::vector<int64_t>& v);
  static std::vector<int> ShapeOf(mlx_array a);

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

 private:
  void Translate(const NodeDesc& n);
  void CopyOut(const OutRef& o);

  Plan& plan_;
  Ort::KernelContext& ctx_;
  mlx_stream s_;
  std::unordered_map<std::string, mlx_array> env_;
  std::vector<mlx_array> transient_;
};

}  // namespace ort_mps_mlx
