// Copyright (c) 2026. Licensed under the MIT License.
//
// Control-flow op handlers: If, Scan, Loop. Unlike ordinary ops these carry their computation as a
// nested subgraph (GraphProto) ATTRIBUTE — a body ORT surfaces to a plugin EP via Node_GetSubgraphs
// (ep.cc::BuildSubgraphs captures each body as a NodeDesc::subgraphs entry). The MLX-native EP owns
// the control-flow node WHOLE (its body is declined for independent offload in ep.cc::GetCapability)
// and realizes the control flow by translating the body inline through TranslationContext::RunSubgraph:
//
//   * If   — read the runtime `cond` host-side each forward and translate the taken branch only. Both
//            branches must be translatable (either can be taken across forwards); the graph is rebuilt
//            per forward (eager path), so per-forward branch selection is exact. A runtime `cond` needs
//            no mlx_where because we pick the branch at graph-build time.
//   * Scan — STATIC trip count (scan axis length known from the input shape). Unroll the body over the
//            axis, carrying state and stacking scan outputs. Forward direction, axis 0 only (MVP).
//   * Loop — CONSTANT trip count M with a cond that is a pass-through of the loop cond input (the
//            canonical `for i in range(M)` idiom, provably a fixed M-iteration loop). Unroll M times.
//            Carried-state only (no per-iteration scan outputs) in this MVP. Dynamic / cond-dependent
//            trip counts are NOT statically unrollable and are left to ORT CPU (unclaimed).
//
// Everything claimed here is translated correctly; anything outside these static/foldable forms is
// left unclaimed and runs on ORT's CPU control-flow kernels (with the body ops still offloaded to MLX
// via the ordinary flat path — see ep.cc::GetCapability).

#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// Every node in a control-flow body must be MLX-translatable (recursively, so a nested If/Scan/Loop
// is checked through its own claim predicate). Consulted at claim time so a claimed control-flow node
// can never contain an untranslatable op.
bool BodyClaimable(Ort::ConstGraph body) {
  for (Ort::ConstNode node : body.GetNodes()) {
    if (!Claimable(node)) return false;
  }
  return true;
}

// Find a body subgraph by attribute name.
const SubgraphDesc* FindBody(const NodeDesc& n, const char* attr) {
  for (const SubgraphDesc& sg : n.subgraphs) {
    if (sg.attr_name == attr) return &sg;
  }
  return nullptr;
}

// Read a scalar bool from a foldable (initializer / ctx) node input.
bool ReadHostBool(TranslationContext& ctx, const TensorRef& ref) {
  HostBytes h = ctx.RawHost(ref);
  return h.data != nullptr && *reinterpret_cast<const uint8_t*>(h.data) != 0;
}

// Stack a list of same-shaped MLX arrays along a new leading axis (axis 0), like ONNX scan-output
// accumulation. Keeps + returns the result.
mlx_array StackAxis0(TranslationContext& ctx, const std::vector<mlx_array>& parts) {
  mlx_vector_array vec = mlx_vector_array_new();
  for (mlx_array a : parts) mlx_vector_array_append_value(vec, a);
  mlx_array r = mlx_array_new();
  int rc = mlx_stack_axis(&r, vec, 0, ctx.stream());
  mlx_vector_array_free(vec);
  if (rc != 0) throw MlxError("MLX Scan/Loop: mlx_stack_axis failed");
  return ctx.Keep(r);
}

// ---- If ---------------------------------------------------------------------------------------

void IfOp(TranslationContext& ctx, const NodeDesc& n) {
  const bool cond = ReadHostBool(ctx, n.inputs[0]);
  const SubgraphDesc* branch = FindBody(n, cond ? "then_branch" : "else_branch");
  if (branch == nullptr) throw MlxError("MLX If: missing branch subgraph");
  std::vector<mlx_array> outs = ctx.RunSubgraph(*branch, {});
  if (outs.size() != n.outputs.size()) throw MlxError("MLX If: branch output arity mismatch");
  for (size_t i = 0; i < n.outputs.size(); ++i) ctx.Bind(n.outputs[i], outs[i]);
}

bool IfClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.empty()) return false;
  ONNXTensorElementDataType cond_type;
  if (!TensorInfo(inputs[0], cond_type) || cond_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return false;
  }
  std::vector<Ort::AttrNameSubgraph> subs = node.GetSubgraphs();
  if (subs.size() != 2) return false;
  bool have_then = false, have_else = false;
  for (Ort::AttrNameSubgraph& as : subs) {
    if (as.attr_name == "then_branch") have_then = true;
    else if (as.attr_name == "else_branch") have_else = true;
    else return false;
    Ort::ConstGraph body = as.sub_graph;
    if (!body.GetInputs().empty()) return false;                 // If branches take no formal inputs
    if (body.GetOutputs().size() != outputs.size()) return false;
    if (!BodyClaimable(body)) return false;
  }
  return have_then && have_else;
}

// ---- Scan -------------------------------------------------------------------------------------

// Reject non-default (non-forward / non-axis-0) direction/axis attributes — MVP supports forward
// iteration over axis 0 only.
bool AllZeroIntsAttr(Ort::ConstNode node, const char* name) {
  std::vector<int64_t> v;
  bool present = false;
  IntsAttribute(node, name, v, present);
  if (!present) return true;
  for (int64_t x : v) {
    if (x != 0) return false;
  }
  return true;
}

void ScanOp(TranslationContext& ctx, const NodeDesc& n) {
  const int64_t num_scan = n.ints.at("num_scan_inputs");
  const int num_state = static_cast<int>(n.inputs.size()) - static_cast<int>(num_scan);
  const SubgraphDesc* body = FindBody(n, "body");
  if (body == nullptr) throw MlxError("MLX Scan: missing body subgraph");

  std::vector<mlx_array> state;
  for (int i = 0; i < num_state; ++i) state.push_back(ctx.Resolve(n.inputs[i]));
  std::vector<mlx_array> scans;
  for (int i = 0; i < static_cast<int>(num_scan); ++i) {
    scans.push_back(ctx.Resolve(n.inputs[num_state + i]));
  }

  const std::vector<int> s0 = TranslationContext::ShapeOf(scans[0]);
  if (s0.empty()) throw MlxError("MLX Scan: scan input is a scalar");
  const int trip = s0[0];

  const int num_scan_out = static_cast<int>(body->output_names.size()) - num_state;
  if (num_scan_out < 0) throw MlxError("MLX Scan: body output arity");
  std::vector<std::vector<mlx_array>> collected(num_scan_out);

  for (int t = 0; t < trip; ++t) {
    std::vector<mlx_array> bin;
    bin.reserve(num_state + num_scan);
    for (int i = 0; i < num_state; ++i) bin.push_back(state[i]);
    for (int i = 0; i < static_cast<int>(num_scan); ++i) {
      const std::vector<int> shp = TranslationContext::ShapeOf(scans[i]);
      std::vector<int> start(shp.size(), 0), stop = shp;
      start[0] = t;
      stop[0] = t + 1;
      mlx_array sl = ctx.Slice(scans[i], start, stop);  // [1, ...]
      mlx_array sq = mlx_array_new();
      if (mlx_squeeze_axis(&sq, sl, 0, ctx.stream()) != 0) {
        throw MlxError("MLX Scan: mlx_squeeze_axis failed");
      }
      bin.push_back(ctx.Keep(sq));
    }
    std::vector<mlx_array> bout = ctx.RunSubgraph(*body, bin);
    for (int i = 0; i < num_state; ++i) state[i] = bout[i];
    for (int i = 0; i < num_scan_out; ++i) collected[i].push_back(bout[num_state + i]);
  }

  for (int i = 0; i < num_state; ++i) ctx.Bind(n.outputs[i], state[i]);
  for (int i = 0; i < num_scan_out; ++i) {
    ctx.Bind(n.outputs[num_state + i], StackAxis0(ctx, collected[i]));
  }
}

bool ScanClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  const int64_t num_scan = IntAttribute(node, "num_scan_inputs", -1);
  if (num_scan <= 0 || static_cast<int64_t>(inputs.size()) < num_scan) return false;
  const int64_t num_state = static_cast<int64_t>(inputs.size()) - num_scan;
  if (num_state < 0) return false;

  if (!AllZeroIntsAttr(node, "scan_input_directions")) return false;
  if (!AllZeroIntsAttr(node, "scan_output_directions")) return false;
  if (!AllZeroIntsAttr(node, "scan_input_axes")) return false;
  if (!AllZeroIntsAttr(node, "scan_output_axes")) return false;

  // Every scan input must have a statically-known, non-empty scan axis (axis 0) for a static unroll.
  for (int64_t i = num_state; i < static_cast<int64_t>(inputs.size()); ++i) {
    ONNXTensorElementDataType t;
    std::vector<int64_t> shp;
    if (!TensorInfo(inputs[i], t, &shp)) return false;
    if (shp.empty() || shp[0] < 1) return false;
  }

  std::vector<Ort::AttrNameSubgraph> subs = node.GetSubgraphs();
  if (subs.size() != 1 || subs[0].attr_name != "body") return false;
  Ort::ConstGraph body = subs[0].sub_graph;
  if (static_cast<int64_t>(body.GetInputs().size()) != num_state + num_scan) return false;
  if (static_cast<int64_t>(body.GetOutputs().size()) < num_state) return false;
  // The node exposes carried state + every scan output (= all body outputs beyond the state).
  if (static_cast<int64_t>(outputs.size()) != static_cast<int64_t>(body.GetOutputs().size())) {
    return false;
  }
  return BodyClaimable(body);
}

// ---- Loop -------------------------------------------------------------------------------------

// True iff the body's cond output (body output 0) is a pass-through of the body's cond input (body
// input 1): either a direct graph-output alias, or an Identity node copying it. With an initial cond
// of true this makes the loop run EXACTLY M iterations (no data-dependent early exit), so a constant M
// is a static trip count we can unroll.
bool LoopCondIsPassthrough(Ort::ConstGraph body) {
  const std::vector<Ort::ConstValueInfo> bin = body.GetInputs();
  const std::vector<Ort::ConstValueInfo> bout = body.GetOutputs();
  if (bin.size() < 2 || bout.empty()) return false;
  const std::string cond_in = bin[1].GetName();
  const std::string cond_out = bout[0].GetName();
  if (cond_in.empty() || cond_out.empty()) return false;
  if (cond_in == cond_out) return true;  // graph output directly aliases the cond input
  for (Ort::ConstNode node : body.GetNodes()) {
    if (node.GetOperatorType() != "Identity") continue;
    const std::vector<Ort::ConstValueInfo> ins = node.GetInputs();
    const std::vector<Ort::ConstValueInfo> outs = node.GetOutputs();
    if (ins.size() == 1 && outs.size() == 1 && ins[0].GetName() == cond_in &&
        outs[0].GetName() == cond_out) {
      return true;
    }
  }
  return false;
}

void LoopOp(TranslationContext& ctx, const NodeDesc& n) {
  const SubgraphDesc* body = FindBody(n, "body");
  if (body == nullptr) throw MlxError("MLX Loop: missing body subgraph");
  const int num_state = static_cast<int>(n.inputs.size()) - 2;  // inputs = [M, cond, state...]

  int64_t trip_count = 0;
  {
    HostBytes h = ctx.RawHost(n.inputs[0]);  // M (int64 scalar)
    if (h.data == nullptr) throw MlxError("MLX Loop: null trip count");
    trip_count = *reinterpret_cast<const int64_t*>(h.data);
  }
  const bool cond0 = ReadHostBool(ctx, n.inputs[1]);
  const int trip = cond0 ? static_cast<int>(trip_count) : 0;

  std::vector<mlx_array> state;
  for (int i = 0; i < num_state; ++i) state.push_back(ctx.Resolve(n.inputs[2 + i]));

  const bool cond_true = true;
  for (int t = 0; t < trip; ++t) {
    const int64_t iter_val = t;
    mlx_array iter = ctx.Keep(mlx_array_new_data(&iter_val, nullptr, 0, MLX_INT64));
    mlx_array condin = ctx.Keep(mlx_array_new_data(&cond_true, nullptr, 0, MLX_BOOL));
    std::vector<mlx_array> bin;
    bin.reserve(2 + num_state);
    bin.push_back(iter);
    bin.push_back(condin);
    for (int i = 0; i < num_state; ++i) bin.push_back(state[i]);
    std::vector<mlx_array> bout = ctx.RunSubgraph(*body, bin);
    // bout[0] = cond_out (pass-through, guaranteed true by the claim); bout[1..N] = carried state.
    for (int i = 0; i < num_state; ++i) state[i] = bout[1 + i];
  }

  for (int i = 0; i < num_state; ++i) ctx.Bind(n.outputs[i], state[i]);
}

bool LoopClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2) return false;  // require explicit M and cond (MVP)
  const int num_state = static_cast<int>(inputs.size()) - 2;
  // M (int64 scalar) and cond (bool) must be present so we can read them host-side.
  ONNXTensorElementDataType mt, ct;
  if (!TensorInfo(inputs[0], mt) || mt != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return false;
  if (!TensorInfo(inputs[1], ct) || ct != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) return false;

  std::vector<Ort::AttrNameSubgraph> subs = node.GetSubgraphs();
  if (subs.size() != 1 || subs[0].attr_name != "body") return false;
  Ort::ConstGraph body = subs[0].sub_graph;
  // body inputs = [iter_num, cond_in, state...]; carried-state-only (no scan outputs) in this MVP,
  // so body outputs = [cond_out, state...] and node outputs = [state...].
  if (static_cast<int>(body.GetInputs().size()) != 2 + num_state) return false;
  if (static_cast<int>(body.GetOutputs().size()) != 1 + num_state) return false;
  if (static_cast<int>(outputs.size()) != num_state) return false;
  if (!LoopCondIsPassthrough(body)) return false;
  return BodyClaimable(body);
}

}  // namespace

void RegisterControlFlowOps(OpRegistry& registry) {
  // ai.onnx control flow. Version-insensitive registration: the claim predicates gate the concrete
  // static/foldable forms; any node outside them is left to ORT CPU.
  registry.Register({"", "If", kAnyOpset, kAnyOpset, &IfOp, &IfClaim});
  registry.Register({"", "Scan", kAnyOpset, kAnyOpset, &ScanOp, &ScanClaim});
  registry.Register({"", "Loop", kAnyOpset, kAnyOpset, &LoopOp, &LoopClaim});
}

}  // namespace ort_mlx
