// Copyright (c) 2026. Licensed under the MIT License.
//
// MetalEp: the per-session OrtEp for the MLX-native execution provider. GetCapability claims
// maximal, convex, connected subgraphs of MLX-translatable ops as ONE fused node each (leaving
// unsupported ops to ORT's CPU EP). Compile translates each fused subgraph into an MLX graph
// (see mlx_backend.cc); the fused node's Compute evaluates the whole subgraph with a SINGLE
// mlx_eval at the subgraph boundary. There are no hand-written Metal kernels — MLX owns all
// compute for both prefill and decode.
//
// See docs/DESIGN.md §2.4/§2.5.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "metal_context.h"
#include "plugin_ep_utils.h"

class MetalEpFactory;

// Per-subgraph compiled MLX plan built in Compile and run by the fused node's Compute. Defined in
// ep.cc; forward-declared here so MetalEp can own the plans by fused-node name.
struct SubgraphPlan;

class MetalEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    // Whether to claim any MLX-translatable op. Set to false via ONNX_GENAI_METAL_EP_CLAIM=none for
    // a pure-CPU-fallback proof. Per-op claim decisions live in the registry (ops/*.cc claim
    // predicates); this is just the global on/off switch.
    bool claim_enabled = true;
  };

  MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
          std::shared_ptr<ort_mlx::MetalContext> metal, const OrtLogger& logger);
  ~MetalEp();

  std::unordered_map<std::string, std::unique_ptr<SubgraphPlan>>& Plans() { return plans_; }
  ort_mlx::MetalContext* Metal() const { return metal_.get(); }
  const OrtLogger* Logger() const { return logger_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** ep_context_nodes) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;
  static OrtStatus* ORT_API_CALL GetDefaultMemoryDeviceImpl(const OrtEp* this_ptr,
                                                            const OrtMemoryDevice** device) noexcept;

  MetalEpFactory& factory_;
  std::string name_;
  Config config_;
  std::shared_ptr<ort_mlx::MetalContext> metal_;
  const OrtLogger* logger_;  // for MPS_LOG
  std::unordered_map<std::string, std::unique_ptr<SubgraphPlan>> plans_;
};
