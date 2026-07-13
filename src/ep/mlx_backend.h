// Copyright (c) 2026. Licensed under the MIT License.
//
// MLX (mlx-c) backend for the fused decoder subgraph — Phase-0 GO/NO-GO prototype (Nabil).
//
// This is a FLAG-GATED, ISOLATED alternative execution path for a single fused subgraph. When the
// env var ONNX_GENAI_METAL_EP_MLX=1 is set AND the plugin was built with -DORT_MPS_ENABLE_MLX=ON,
// the EP hands the WHOLE fused decoder subgraph to MLX as one unit: the ONNX op graph is translated
// into an MLX lazy graph (mlx_quantized_matmul for MatMulNBits, fast_scaled_dot_product_attention +
// rope for GroupQueryAttention, fast_rms_norm, elementwise, ...), evaluated with a SINGLE mlx_eval
// at the subgraph boundary, and the boundary inputs/outputs are copied across the ORT boundary. When
// the flag is off (or MLX was not compiled in), the default hand-kernel Metal path runs unchanged.
//
// The plan description below (NodeDesc/TensorRef/OutRef) is a self-contained, kernel-agnostic view
// of the subgraph built in ep.cc::CompileImpl, so this translator has no dependency on the EP's
// internal kernel objects.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ort_mps_mlx {

// Where a node input resolves from. Mirrors ep.cc's Source but kept independent so the MLX backend
// stays decoupled from the EP-internal (anonymous-namespace) plan structures.
enum class Src { CtxInput, Initializer, Intermediate, Absent };

// A single node input reference.
struct TensorRef {
  std::string name;
  Src source = Src::Absent;
  size_t ctx_index = 0;  // valid when source == CtxInput
  // True when a CtxInput is actually a hoisted constant initializer (weights/scales/caches). The
  // translator wraps/repacks it once from live ctx data and caches it persistently on the plan.
  bool constant = false;
  // Initializer payload (session-owned; valid for the session lifetime).
  const void* init_data = nullptr;
  std::vector<int64_t> init_shape;
  ONNXTensorElementDataType init_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t init_count = 0;
};

// A single node output reference.
struct OutRef {
  std::string name;
  bool external = false;  // a subgraph boundary output routed to ctx.GetOutput(ctx_index)
  size_t ctx_index = 0;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

// One ONNX node with just the metadata the MLX translator needs.
struct NodeDesc {
  std::string op_type;
  std::string domain;
  std::unordered_map<std::string, int64_t> ints;
  std::unordered_map<std::string, float> floats;
  std::vector<TensorRef> inputs;
  std::vector<OutRef> outputs;
};

// Opaque compiled MLX plan (owns the persistent repacked-weight / cos-sin cache MLX arrays).
struct Plan;

// True if MLX support was compiled into this build (-DORT_MPS_ENABLE_MLX=ON).
bool Available();

// True if the MLX path is requested at runtime (env ONNX_GENAI_METAL_EP_MLX set) AND Available().
bool Enabled();

// Build a runnable MLX plan from the node descriptors (topological order). Returns nullptr and sets
// `error` if an op in the subgraph has no MLX translation (the caller then keeps the hand path).
// Ownership transfers to the caller (wrap with PlanDeleter / DestroyPlan).
Plan* BuildPlan(std::vector<NodeDesc> nodes, std::string& error);

// Destroys a plan (frees cached MLX arrays). Declared so ep.cc can hold a unique_ptr<Plan>.
void DestroyPlan(Plan* plan);

// Run the whole subgraph through MLX: read the ORT ctx inputs, build the MLX graph, one mlx_eval at
// the boundary, copy results into the ORT ctx outputs. Returns false + `error` on failure.
bool RunPlan(Plan& plan, Ort::KernelContext& ctx, std::string& error);

// Deleter so callers can `std::unique_ptr<Plan, PlanDeleter>`.
struct PlanDeleter {
  void operator()(Plan* p) const { DestroyPlan(p); }
};

}  // namespace ort_mps_mlx
