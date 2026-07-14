// Copyright (c) 2026. Licensed under the MIT License.
//
// MLX (mlx-c) backend for the fused decoder subgraph — the SOLE compute path of the MLX-native ORT
// execution provider.
//
// The EP hands the WHOLE fused decoder subgraph to MLX as one unit: the ONNX op graph is translated
// into an MLX lazy graph (mlx_quantized_matmul for MatMulNBits, fast_scaled_dot_product_attention +
// rope for GroupQueryAttention, fast_rms_norm, elementwise, ...), evaluated with a SINGLE mlx_eval
// at the subgraph boundary, and the boundary inputs/outputs are copied across the ORT boundary. This
// runs for BOTH prefill and decode; there are no hand-tuned .metal kernels and no fallback path
// (mlx-c is a hard build dependency).
//
// The plan description below (NodeDesc/TensorRef/OutRef) is a self-contained, kernel-agnostic view
// of the subgraph built in ep.cc::CompileImpl, so this translator has no dependency on the EP's
// internal structures.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ort_mlx {

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
  // Initializer payload. `init_data` points either at session-owned memory (fused-graph
  // initializers, valid for the session lifetime) or into `init_owned` when set. Control-flow body
  // initializers are reached through a subgraph handle that ORT releases once the body has been
  // walked, so their bytes are COPIED into `init_owned` (a shared_ptr keeps the buffer alive across
  // NodeDesc copies for the plan's lifetime).
  const void* init_data = nullptr;
  std::shared_ptr<std::vector<uint8_t>> init_owned;
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

struct NodeDesc;  // forward-declared: a control-flow body carries its own NodeDesc list.

// A control-flow node's body subgraph (If then/else branch, Scan/Loop body), captured so the
// translator can realize the control flow by translating the body inline. `input_names` and
// `output_names` are the body graph's FORMAL inputs/outputs (positional); the control-flow handler
// binds inputs before translating the body and reads the outputs afterward. `nodes` are the body's
// nodes in topological order, whose input TensorRefs already resolve against the body scope with a
// fall-through to the enclosing scope (implicit inputs). See ep.cc::BuildSubgraphs and
// ops/controlflow.cc.
struct SubgraphDesc {
  std::string attr_name;                    // "then_branch" / "else_branch" / "body"
  std::vector<std::string> input_names;     // formal body inputs (positional)
  std::vector<std::string> output_names;    // formal body outputs (positional)
  std::vector<NodeDesc> nodes;              // topo-ordered body nodes
};

// One ONNX node with just the metadata the MLX translator needs.
struct NodeDesc {
  std::string op_type;
  std::string domain;
  // The opset version in which this node's op was first defined (Ort::ConstNode::GetSinceVersion),
  // threaded from ep.cc so the registry can dispatch opset-23 vs opset-24 variants of an op to
  // different handlers. 0 when unknown (matches any registration).
  int since_version = 0;
  // Every ONNX attribute on the node, copied generically by ep.cc (no fixed per-op list). Keyed by
  // attribute name and split by ONNX attribute type so a handler reads e.g. n.ints.at("axis"),
  // n.int_arrays.at("axes"), n.strings.at("mode"). Absent attributes are simply not in the map, so
  // handlers use n.<map>.count(name) ? n.<map>.at(name) : <default> (or .at() for required attrs).
  // GRAPH and TENSOR attributes are not carried today (no claimed op reads them; see docs §2/§6).
  std::unordered_map<std::string, int64_t> ints;                       // ORT_OP_ATTR_INT
  std::unordered_map<std::string, float> floats;                       // ORT_OP_ATTR_FLOAT
  std::unordered_map<std::string, std::vector<int64_t>> int_arrays;    // ORT_OP_ATTR_INTS
  std::unordered_map<std::string, std::vector<float>> float_arrays;    // ORT_OP_ATTR_FLOATS
  std::unordered_map<std::string, std::string> strings;               // ORT_OP_ATTR_STRING
  std::vector<TensorRef> inputs;
  std::vector<OutRef> outputs;
  // Body subgraphs for a control-flow node (If/Scan/Loop). Empty for every ordinary op. Populated
  // recursively in ep.cc::BuildSubgraphs so the control-flow handler can translate the body inline.
  std::vector<SubgraphDesc> subgraphs;
};

// Claim-time membership check: can the MLX backend translate (domain, op_type) at this opset? Backed
// by the SAME registry the run-time translator uses, so a claimed op is always translatable. Called
// from BuildPlan; also underpins Claimable().
bool Supported(const std::string& domain, const std::string& op_type, int since_version);

// Claim-time node predicate consulted from ep.cc::GetCapability. True iff the registry has a matching
// (domain, op, opset) entry AND that entry's registered claim predicate accepts this concrete node
// (its dtypes/shapes/attributes/input form). This is the single hook the EP uses to decide claims —
// all per-op claim logic lives beside the translate handlers in ops/*.cc, so adding an op requires
// no ep.cc edits.
bool Claimable(Ort::ConstNode node);

// Opaque compiled MLX plan (owns the persistent repacked-weight / cos-sin cache MLX arrays).
struct Plan;

// Build a runnable MLX plan from the node descriptors (topological order). Returns nullptr and sets
// `error` if an op in the subgraph has no MLX translation (a hard error — there is no fallback).
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

}  // namespace ort_mlx
