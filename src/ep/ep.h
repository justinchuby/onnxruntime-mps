// Copyright (c) 2026. Licensed under the MIT License.
//
// MetalEp: the per-session OrtEp. GetCapability claims maximal, convex, connected subgraphs of
// supported ops as ONE fused node each (leaving unsupported ops to ORT's CPU EP). Compile builds
// a per-subgraph execution plan; the fused node's Compute runs every node of the subgraph into a
// SINGLE Metal command buffer (one commit / one waitUntilCompleted per subgraph), threading
// device-resident intermediate MTLBuffers between kernels with no per-node host round-trip.
//
// See docs/DESIGN.md 2.4/2.5. Kernels encode into the shared command buffer via the KernelIO
// interface, so the same kernel code serves both the fused-subgraph path and direct testing.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "metal_context.h"
#include "plugin_ep_utils.h"

class MetalEpFactory;

// A single tensor operand handed to a kernel. `data` is a read view (inputs and outputs);
// `mutable_data` is the write view (outputs). Both point at unified-memory MTLBuffer storage
// resolved by the EP: subgraph inputs/outputs come from ORT, intermediates from MetalContext.
struct IOTensor {
  const void* data = nullptr;
  void* mutable_data = nullptr;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t element_count = 0;

  template <typename T>
  const T* Data() const {
    return static_cast<const T*>(data);
  }
  template <typename T>
  T* MutableData() const {
    return static_cast<T*>(mutable_data);
  }
};

// Abstracts a single node's I/O so a kernel is agnostic to whether it runs standalone against an
// OrtKernelContext or as one node inside a fused subgraph (where inputs/outputs are the EP's
// device-resident intermediates). Output(index, shape) allocates/binds the output for `shape`
// exactly like OrtKernelContext::GetOutput, so kernels keep computing their own output shapes.
class KernelIO {
 public:
  virtual ~KernelIO() = default;
  virtual size_t InputCount() const = 0;
  virtual size_t OutputCount() const = 0;
  virtual const IOTensor& Input(size_t index) = 0;
  virtual IOTensor& Output(size_t index, const std::vector<int64_t>& shape) = 0;
};

// Common base for every Metal kernel. ComputeIO encodes the op into whatever command buffer the
// MetalContext currently has open (a shared per-subgraph buffer when a batch is active), so the
// EP controls submission boundaries.
struct KernelBase {
  KernelBase(const OrtApi& ort_api, ort_mps::MetalContext* metal)
      : ort_api_(ort_api), metal_(metal) {}
  virtual ~KernelBase() = default;
  virtual OrtStatus* ComputeIO(KernelIO& io) = 0;

  const OrtApi& ort_api_;
  ort_mps::MetalContext* metal_;
};

// Executes a single ONNX Add node on the GPU (float32 or float16).
struct AddKernel : KernelBase {
  AddKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode /*node*/)
      : KernelBase(ort_api, metal) {}
  OrtStatus* ComputeIO(KernelIO& io) override;
};

// Coco-owned data movement, quantization, and activation kernels.
struct CocoKernel : KernelBase {
  CocoKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode node);
  OrtStatus* ComputeIO(KernelIO& io) override;

  std::string op_type_;
  int64_t to_type_ = 0;
  int64_t axis_ = 0;
  int64_t block_size_ = 32;
  int64_t gather_axis_ = 0;
  int64_t quantize_axis_ = 1;
  int64_t bits_ = 4;
  int64_t num_heads_ = 0;
  int64_t rotary_embedding_dim_ = 0;
  bool interleaved_ = false;
  bool gelu_tanh_ = false;
  bool allowzero_ = false;
  std::vector<int64_t> permutation_;
};

// Mariette-owned core-compute kernels (MatMulNBits, RMSNormalization,
// SkipSimplifiedLayerNormalization, Softmax). Constant int4 weights + scales for MatMulNBits are
// copied into device buffers once (via MetalContext::Alloc) and reused across decode steps, so
// the model weights become device-resident after the first token.
struct MarietteKernel : KernelBase {
  MarietteKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode node);
  ~MarietteKernel() override;
  OrtStatus* ComputeIO(KernelIO& io) override;

  std::string op_type_;
  float epsilon_ = 1e-6f;

  // GroupQueryAttention attributes.
  int64_t num_heads_ = 0;
  int64_t kv_num_heads_ = 0;
  float scale_ = 0.0f;
  int64_t do_rotary_ = 0;
  int64_t rotary_interleaved_ = 0;
  int64_t local_window_size_ = -1;

  // MatMulNBits constant-weight device cache (nullptr until the first Compute).
  void* b_dev_ = nullptr;
  void* scales_dev_ = nullptr;
  size_t b_bytes_ = 0;
  size_t scales_bytes_ = 0;
};

// Per-subgraph execution plan built in Compile and run by the fused node's Compute. Defined in
// ep.cc; forward-declared here so MetalEp can own the plans by fused-node name.
struct SubgraphPlan;

class MetalEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool claim_add = true;
    bool claim_coco = true;
    bool claim_mariette = true;
  };

  MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
          ort_mps::MetalContext* metal, const OrtLogger& logger);
  ~MetalEp();

  std::unordered_map<std::string, std::unique_ptr<SubgraphPlan>>& Plans() { return plans_; }
  ort_mps::MetalContext* Metal() const { return metal_; }
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
  ort_mps::MetalContext* metal_;
  const OrtLogger* logger_;  // for MPS_LOG
  std::unordered_map<std::string, std::unique_ptr<SubgraphPlan>> plans_;
};
