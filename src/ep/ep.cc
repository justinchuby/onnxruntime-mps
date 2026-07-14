// Copyright (c) 2026. Licensed under the MIT License.

#include "ep.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ep_factory.h"
#include "mlx_backend.h"

namespace {

// Claim-time node predicate. The per-op dtype/shape/attribute claim logic now lives beside each
// translate handler in src/ep/ops/*.cc (registered as an OpRegistration::claimable). ort_mps_mlx::
// Claimable consults the SAME (domain, op, opset) registry the run-time translator dispatches
// through, so "claimed" can never outrun "translatable" and adding an op needs ZERO edits here.
// config.claim_enabled is the single kill-switch (ONNX_GENAI_METAL_EP_CLAIM=none -> claim nothing).
bool NodeClaimable(Ort::ConstNode node, const MetalEp::Config& config) {
  if (!config.claim_enabled) return false;
  return ort_mps_mlx::Claimable(node);
}

// ORT surfaces an OMITTED interior optional node input (e.g. Resize's `roi`, Clip's `min`) as a
// NULL OrtValueInfo. Dereferencing it via GetName() segfaults, so treat a null info as an unnamed
// (absent) input; callers keep positional indices intact by still visiting the slot.
std::string InputName(const Ort::ConstValueInfo& vi) {
  return static_cast<const OrtValueInfo*>(vi) == nullptr ? std::string() : vi.GetName();
}

}  // namespace

// ---------------------------------------------------------------------------
// Subgraph execution plan + executor
// ---------------------------------------------------------------------------

// The concrete SubgraphPlan (forward-declared in ep.h). Owns the compiled MLX plan for this fused
// subgraph (the persistent repacked-weight / cos-sin cache MLX arrays live inside it).
struct SubgraphPlan {
  MetalEp* ep = nullptr;

  // The whole fused decoder subgraph translated into an MLX graph. Built once in Compile and run
  // for BOTH prefill and decode (Phase-0's full-MLX path, promoted to the sole compute path). Both
  // prefill and decode read past K/V from the SAME ORT ctx inputs and write present K/V to the
  // SAME ORT ctx outputs in the identical [B, kv_heads, total_seq, head] fp32 layout with RoPE
  // applied to stored K at absolute positions [past, past+M), so the KV cache handoff across the
  // prefill->decode boundary (and every decode step) is layout- and position-continuous.
  std::unique_ptr<ort_mps_mlx::Plan, ort_mps_mlx::PlanDeleter> mlx_plan;
};

namespace {

// Runs an entire fused subgraph through MLX: build the MLX graph for this forward, one mlx_eval at
// the subgraph boundary, copy the boundary outputs back across the ORT boundary. Used for both
// prefill and decode — MLX is the sole compute path (no hand-kernel fallback).
OrtStatus* RunSubgraph(SubgraphPlan& plan, OrtKernelContext* kernel_context) {
  MetalEp* ep = plan.ep;
  const OrtApi& ort_api_ = ep->ort_api;
  try {
    Ort::KernelContext ctx(kernel_context);
    if (!plan.mlx_plan) {
      return ort_api_.CreateStatus(ORT_EP_FAIL, "MetalEP: fused subgraph has no MLX plan");
    }
    std::string mlx_err;
    if (!ort_mps_mlx::RunPlan(*plan.mlx_plan, ctx, mlx_err)) {
      return ort_api_.CreateStatus(ORT_EP_FAIL,
                                   ("MetalEP MLX subgraph failed: " + mlx_err).c_str());
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

// ---- Convex subgraph clustering (GetCapability) ----

using Bitset = std::vector<uint64_t>;

inline void BitSet(Bitset& b, size_t i) { b[i >> 6] |= (uint64_t{1} << (i & 63)); }
inline bool BitTest(const Bitset& b, size_t i) { return (b[i >> 6] >> (i & 63)) & 1u; }
inline void BitOrInto(Bitset& dst, const Bitset& src) {
  for (size_t i = 0; i < dst.size(); ++i) dst[i] |= src[i];
}
inline bool BitIntersects(const Bitset& a, const Bitset& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] & b[i]) return true;
  }
  return false;
}

struct UnionFind {
  std::vector<size_t> parent;
  explicit UnionFind(size_t n) : parent(n) {
    for (size_t i = 0; i < n; ++i) parent[i] = i;
  }
  size_t Find(size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
};

// Groups supported nodes into maximal, convex, connected clusters. A set S is convex (a valid
// single fused node) iff no node x outside S lies on a path between two members of S; contracting
// a non-convex set would create a cycle that ORT rejects. Returns one node-index vector per
// cluster. When `fuse` is false, each supported node becomes its own singleton cluster.
std::vector<std::vector<size_t>> BuildConvexClusters(const std::vector<Ort::ConstNode>& nodes,
                                                     const std::vector<char>& supported,
                                                     bool fuse) {
  const size_t n = nodes.size();
  const size_t words = (n + 63) / 64;

  if (!fuse) {
    std::vector<std::vector<size_t>> clusters;
    for (size_t i = 0; i < n; ++i) {
      if (supported[i]) clusters.push_back({i});
    }
    return clusters;
  }

  // Map tensor name -> producing node index.
  std::unordered_map<std::string, size_t> producer;
  producer.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    for (const auto& out : nodes[i].GetOutputs()) {
      std::string name = out.GetName();
      if (!name.empty()) producer.emplace(std::move(name), i);
    }
  }

  // Direct successors and predecessors within the graph.
  std::vector<std::vector<size_t>> succ(n), pred(n);
  for (size_t j = 0; j < n; ++j) {
    std::unordered_set<size_t> seen;
    for (const auto& in : nodes[j].GetInputs()) {
      std::string name = InputName(in);
      if (name.empty()) continue;
      auto it = producer.find(name);
      if (it == producer.end() || it->second == j) continue;
      if (seen.insert(it->second).second) {
        succ[it->second].push_back(j);
        pred[j].push_back(it->second);
      }
    }
  }

  // Topological order (Kahn) for reachability accumulation.
  std::vector<size_t> indeg(n, 0);
  for (size_t j = 0; j < n; ++j) indeg[j] = pred[j].size();
  std::vector<size_t> order;
  order.reserve(n);
  std::vector<size_t> stack;
  for (size_t i = 0; i < n; ++i) {
    if (indeg[i] == 0) stack.push_back(i);
  }
  while (!stack.empty()) {
    size_t u = stack.back();
    stack.pop_back();
    order.push_back(u);
    for (size_t v : succ[u]) {
      if (--indeg[v] == 0) stack.push_back(v);
    }
  }
  if (order.size() != n) {
    // Cyclic or unexpected; fall back to the node order we were given.
    order.clear();
    for (size_t i = 0; i < n; ++i) order.push_back(i);
  }

  // reach[i] = set of nodes reachable from i (transitive successors, excluding i).
  std::vector<Bitset> reach(n, Bitset(words, 0));
  for (size_t idx = order.size(); idx-- > 0;) {
    const size_t u = order[idx];
    for (size_t v : succ[u]) {
      BitSet(reach[u], v);
      BitOrInto(reach[u], reach[v]);
    }
  }

  // Cluster state keyed by union-find root.
  UnionFind uf(n);
  std::vector<Bitset> cluster_bits(n, Bitset(words, 0));
  std::vector<Bitset> reach_bits(n, Bitset(words, 0));
  for (size_t i = 0; i < n; ++i) {
    if (!supported[i]) continue;
    BitSet(cluster_bits[i], i);
    reach_bits[i] = reach[i];
  }

  // Candidate merge edges: direct data edges between two supported nodes.
  std::vector<std::pair<size_t, size_t>> edges;
  for (size_t u = 0; u < n; ++u) {
    if (!supported[u]) continue;
    for (size_t v : succ[u]) {
      if (supported[v]) edges.emplace_back(u, v);
    }
  }

  auto is_convex = [&](const Bitset& s_bits, const Bitset& reach_s) -> bool {
    for (size_t x = 0; x < n; ++x) {
      if (BitTest(s_bits, x)) continue;
      if (!BitTest(reach_s, x)) continue;      // S cannot reach x
      if (BitIntersects(reach[x], s_bits)) {   // x can reach back into S
        return false;
      }
    }
    return true;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& e : edges) {
      const size_t ra = uf.Find(e.first);
      const size_t rb = uf.Find(e.second);
      if (ra == rb) continue;
      Bitset merged = cluster_bits[ra];
      BitOrInto(merged, cluster_bits[rb]);
      Bitset merged_reach = reach_bits[ra];
      BitOrInto(merged_reach, reach_bits[rb]);
      if (!is_convex(merged, merged_reach)) continue;
      uf.parent[rb] = ra;
      cluster_bits[ra] = std::move(merged);
      reach_bits[ra] = std::move(merged_reach);
      changed = true;
    }
  }

  std::unordered_map<size_t, std::vector<size_t>> grouped;
  for (size_t i = 0; i < n; ++i) {
    if (!supported[i]) continue;
    grouped[uf.Find(i)].push_back(i);
  }
  std::vector<std::vector<size_t>> clusters;
  clusters.reserve(grouped.size());
  for (auto& kv : grouped) {
    std::sort(kv.second.begin(), kv.second.end());
    clusters.push_back(std::move(kv.second));
  }
  return clusters;
}

// Base with a virtual dtor so ReleaseNodeComputeInfos can delete polymorphically.
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

// One OrtNodeComputeInfo per fused subgraph. CreateState resolves the plan by fused-node name;
// Compute runs the whole subgraph into a single command buffer.
struct SubgraphNodeComputeInfo : NodeComputeInfoBase {
  explicit SubgraphNodeComputeInfo(MetalEp& ep) : ep_(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* self = static_cast<SubgraphNodeComputeInfo*>(this_ptr);
    MetalEp& ep = self->ep_;
    std::string fused_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto it = ep.Plans().find(fused_name);
    if (it == ep.Plans().end()) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                     ("No subgraph plan for fused node " + fused_name).c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state,
                                             OrtKernelContext* kernel_context) {
    return RunSubgraph(*static_cast<SubgraphPlan*>(compute_state), kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) {
    // The plan is owned by MetalEp::plans_; nothing to free here.
  }

  MetalEp& ep_;
};

}  // namespace

// ---------------------------------------------------------------------------
// MetalEp
// ---------------------------------------------------------------------------

MetalEp::MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
                 std::shared_ptr<ort_mlx::MetalContext> metal, const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      metal_{std::move(metal)},
      logger_{&logger} {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  GetDefaultMemoryDevice = GetDefaultMemoryDeviceImpl;
}

MetalEp::~MetalEp() = default;

/*static*/
const char* ORT_API_CALL MetalEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const MetalEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  const OrtApi& ort_api_ = ep->ort_api;  // for MPS_LOG
  const OrtLogger* logger_ = ep->logger_;
  try {
    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    const size_t total = nodes.size();

    std::vector<char> supported(total, 0);
    for (size_t i = 0; i < total; ++i) {
      supported[i] = NodeClaimable(nodes[i], ep->config_) ? 1 : 0;
    }

    const bool fuse = std::getenv("ONNX_GENAI_METAL_EP_NOFUSE") == nullptr;
    std::vector<std::vector<size_t>> clusters = BuildConvexClusters(nodes, supported, fuse);

    size_t claimed = 0;
    for (const auto& cluster : clusters) {
      std::vector<const OrtNode*> group;
      group.reserve(cluster.size());
      for (size_t idx : cluster) {
        group.push_back(static_cast<const OrtNode*>(nodes[idx]));
      }
      OrtNodeFusionOptions fusion_options = {};
      fusion_options.ort_version_supported = ORT_API_VERSION;
      fusion_options.drop_constant_initializers = false;  // ORT supplies initializers at runtime
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info, group.data(), group.size(), &fusion_options));
      claimed += cluster.size();
    }

    MPS_LOG(INFO, "MetalEP GetCapability: claimed "
                      << claimed << " of " << total << " nodes for Metal across " << clusters.size()
                      << " fused subgraph(s); remaining fall back to CPU");
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** /*ep_context_nodes*/) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  const OrtApi& ort_api_ = ep->ort_api;  // for MPS_LOG
  const OrtLogger* logger_ = ep->logger_;
  try {
    for (size_t i = 0; i < count; ++i) {
      Ort::ConstGraph graph{graphs[i]};
      Ort::ConstNode fused_node{fused_nodes[i]};
      const std::string fused_name = fused_node.GetName();

      auto plan = std::make_unique<SubgraphPlan>();
      plan->ep = ep;
      std::vector<ort_mps_mlx::NodeDesc> mlx_nodes;  // the whole subgraph, translated to MLX

      // Fused-node input/output name -> OrtKernelContext index (the runtime I/O boundary).
      std::unordered_map<std::string, size_t> ctx_input_index;
      {
        std::vector<Ort::ConstValueInfo> ins = fused_node.GetInputs();
        for (size_t k = 0; k < ins.size(); ++k) {
          std::string name = InputName(ins[k]);
          if (!name.empty()) ctx_input_index.emplace(std::move(name), k);
        }
      }
      std::unordered_map<std::string, size_t> ctx_output_index;
      {
        std::vector<Ort::ConstValueInfo> outs = fused_node.GetOutputs();
        for (size_t k = 0; k < outs.size(); ++k) {
          std::string name = outs[k].GetName();
          if (!name.empty()) ctx_output_index.emplace(std::move(name), k);
        }
      }

      // Constant initializers referenced by the subgraph (session-owned storage).
      struct InitData {
        const void* data = nullptr;
        std::vector<int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        size_t count = 0;
      };
      std::unordered_map<std::string, InitData> initializers;
      for (const auto& vi : graph.GetInitializers()) {
        std::string name = vi.GetName();
        if (name.empty()) continue;
        Ort::ConstValue value{nullptr};
        Ort::Status st = vi.GetInitializer(value);
        if (!st.IsOK() || static_cast<const OrtValue*>(value) == nullptr) continue;
        auto info = value.GetTensorTypeAndShapeInfo();
        InitData d;
        d.data = value.GetTensorRawData();
        d.shape = info.GetShape();
        d.type = info.GetElementType();
        d.count = info.GetElementCount();
        initializers.emplace(std::move(name), std::move(d));
      }

      std::vector<Ort::ConstNode> snodes = graph.GetNodes();

      // Producer of each intra-subgraph tensor.
      std::unordered_map<std::string, size_t> producer;
      for (size_t k = 0; k < snodes.size(); ++k) {
        for (const auto& out : snodes[k].GetOutputs()) {
          std::string name = out.GetName();
          if (!name.empty()) producer.emplace(std::move(name), k);
        }
      }

      // Topological order over the subgraph so producers run before consumers.
      std::vector<std::vector<size_t>> succ(snodes.size());
      std::vector<size_t> indeg(snodes.size(), 0);
      for (size_t j = 0; j < snodes.size(); ++j) {
        std::unordered_set<size_t> seen;
        for (const auto& in : snodes[j].GetInputs()) {
          std::string name = InputName(in);
          if (name.empty()) continue;
          auto it = producer.find(name);
          if (it == producer.end() || it->second == j) continue;
          if (seen.insert(it->second).second) {
            succ[it->second].push_back(j);
            ++indeg[j];
          }
        }
      }
      std::vector<size_t> order;
      order.reserve(snodes.size());
      std::vector<size_t> stack;
      for (size_t k = 0; k < snodes.size(); ++k) {
        if (indeg[k] == 0) stack.push_back(k);
      }
      while (!stack.empty()) {
        size_t u = stack.back();
        stack.pop_back();
        order.push_back(u);
        for (size_t v : succ[u]) {
          if (--indeg[v] == 0) stack.push_back(v);
        }
      }
      if (order.size() != snodes.size()) {
        order.clear();
        for (size_t k = 0; k < snodes.size(); ++k) order.push_back(k);
      }

      // Translate each node (topological order) into an MLX NodeDesc. The whole fused subgraph
      // becomes ONE MLX graph, run for both prefill and decode. There is no hand-kernel path.
      for (size_t idx : order) {
        Ort::ConstNode node = snodes[idx];

        ort_mps_mlx::NodeDesc mnd;
        mnd.op_type = node.GetOperatorType();
        mnd.domain = node.GetDomain();
        // Opset version the op was introduced at — threaded so the MLX registry can dispatch
        // opset-specific handler variants (e.g. Attention opset23 vs opset24) by version range.
        mnd.since_version = node.GetSinceVersion();
        // Generic attribute copy: every ONNX attribute on the node, split by type into the NodeDesc
        // maps (ints/floats/int_arrays/float_arrays/strings). Handlers read what they need
        // (n.ints.at("K"), n.int_arrays.at("axes"), n.strings.at("mode"), …) with a default for
        // absent optional attrs. No fixed per-op list, so a new op's attributes flow through with no
        // ep.cc edit. GRAPH/TENSOR/STRINGS attrs are skipped (no claimed op reads them today).
        for (const Ort::ConstOpAttr& attr : node.GetAttributes()) {
          std::string name = attr.GetName();
          switch (attr.GetType()) {
            case ORT_OP_ATTR_INT: {
              int64_t v = 0;
              if (attr.GetValue(v).IsOK()) mnd.ints[name] = v;
              break;
            }
            case ORT_OP_ATTR_FLOAT: {
              float v = 0.0f;
              if (attr.GetValue(v).IsOK()) mnd.floats[name] = v;
              break;
            }
            case ORT_OP_ATTR_INTS: {
              std::vector<int64_t> v;
              if (attr.GetValueArray(v).IsOK()) mnd.int_arrays[name] = std::move(v);
              break;
            }
            case ORT_OP_ATTR_FLOATS: {
              std::vector<float> v;
              if (attr.GetValueArray(v).IsOK()) mnd.float_arrays[name] = std::move(v);
              break;
            }
            case ORT_OP_ATTR_STRING: {
              std::string v;
              if (attr.GetValue(v).IsOK()) mnd.strings[name] = std::move(v);
              break;
            }
            default:
              break;  // ORT_OP_ATTR_STRINGS / GRAPH / TENSOR not carried (unused by claimed ops).
          }
        }

        for (const auto& in : node.GetInputs()) {
          ort_mps_mlx::TensorRef tr;
          tr.name = InputName(in);
          if (tr.name.empty()) {
            tr.source = ort_mps_mlx::Src::Absent;
          } else if (producer.count(tr.name)) {
            tr.source = ort_mps_mlx::Src::Intermediate;
          } else if (auto ci = ctx_input_index.find(tr.name); ci != ctx_input_index.end()) {
            tr.source = ort_mps_mlx::Src::CtxInput;
            tr.ctx_index = ci->second;
          } else if (auto ii = initializers.find(tr.name); ii != initializers.end()) {
            tr.source = ort_mps_mlx::Src::Initializer;
            tr.init_data = ii->second.data;
            tr.init_shape = ii->second.shape;
            tr.init_type = ii->second.type;
            tr.init_count = ii->second.count;
          } else {
            return ep->ort_api.CreateStatus(
                ORT_EP_FAIL, ("MetalEP could not resolve subgraph input " + tr.name).c_str());
          }
          // ORT hoists constant initializers (weights/scales/biases/caches) into the fused
          // subgraph's context inputs (drop_constant_initializers=false), so they are read via
          // ctx.GetInput at Run. Their compile-time init_data pointers are graph-owned and go stale
          // after Compile, so we must NOT dereference them at Run; instead we mark which ctx inputs
          // are constant so the MLX translator wraps/repacks each ONCE (from live ctx data on the
          // first Run) and caches it on the plan, avoiding a per-decode-step recopy.
          tr.constant = tr.source == ort_mps_mlx::Src::CtxInput &&
                        initializers.find(tr.name) != initializers.end();
          mnd.inputs.push_back(std::move(tr));
        }

        for (const auto& out : node.GetOutputs()) {
          ort_mps_mlx::OutRef o;
          o.name = out.GetName();
          auto tinfo = out.TypeInfo();
          if (tinfo.GetONNXType() == ONNX_TYPE_TENSOR) {
            o.type = tinfo.GetTensorTypeAndShapeInfo().GetElementType();
          }
          if (!o.name.empty()) {
            auto co = ctx_output_index.find(o.name);
            if (co != ctx_output_index.end()) {
              o.external = true;
              o.ctx_index = co->second;
            }
          }
          mnd.outputs.push_back(std::move(o));
        }

        mlx_nodes.push_back(std::move(mnd));
      }

      const size_t node_count = mlx_nodes.size();
      std::string mlx_err;
      plan->mlx_plan.reset(ort_mps_mlx::BuildPlan(std::move(mlx_nodes), mlx_err));
      if (!plan->mlx_plan) {
        // MLX is the sole compute path: an untranslatable op in a claimed subgraph is a hard error
        // (there is no hand-kernel fallback). GetCapability claims only MLX-translatable ops, so
        // reaching here indicates a claim/translation mismatch.
        return ep->ort_api.CreateStatus(
            ORT_EP_FAIL,
            ("MetalEP: could not build MLX plan for fused subgraph " + fused_name + ": " + mlx_err)
                .c_str());
      }
      MPS_LOG(INFO, "MetalEP: MLX plan built for fused subgraph "
                        << fused_name << " (" << node_count
                        << " nodes; prefill+decode via MLX)");

      ep->plans_[fused_name] = std::move(plan);
      node_compute_infos[i] = new SubgraphNodeComputeInfo(*ep);
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
void ORT_API_CALL MetalEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetDefaultMemoryDeviceImpl(const OrtEp* this_ptr,
                                                            const OrtMemoryDevice** device) noexcept {
  const auto* ep = static_cast<const MetalEp*>(this_ptr);
  *device = ep->ep_api.MemoryInfo_GetMemoryDevice(ep->factory_.GetDefaultMemoryInfo());
  return nullptr;
}
