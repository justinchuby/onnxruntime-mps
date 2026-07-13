// Copyright (c) 2026. Licensed under the MIT License.
//
// Thin C++ facade over the Metal device/queue/library used by the EP. All Objective-C /
// Metal types are hidden behind a PIMPL so the rest of the EP stays portable C++
// (see docs/DESIGN.md 2.7). Implemented in metal_context.mm.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ort_mps {

enum class ScalarType {
  Float16,
  Float32,
  Int32,
  Int64,
};

enum class BinaryOp {
  Add,
  Mul,
  Sub,
  Div,
};

enum class UnaryOp {
  Sigmoid,
  SiLU,
  Gelu,
  GeluTanh,
};

struct RotaryEmbeddingParams {
  uint32_t batch_size = 0;
  uint32_t sequence_length = 0;
  uint32_t num_heads = 0;
  uint32_t head_size = 0;
  uint32_t rotary_embedding_dim = 0;
  uint32_t cache_stride = 0;
  uint32_t max_sequence_length = 0;
  bool rank3_bsh = false;
  bool interleaved = false;
};

// Shapes/attributes for a single com.microsoft.GroupQueryAttention invocation (see
// MetalContext::GroupQueryAttention). All tensors are fp32, batch-first.
struct GroupQueryAttentionParams {
  uint32_t batch_size = 0;
  uint32_t sequence_length = 0;    // S: number of new query tokens this step
  uint32_t num_heads = 0;          // query heads
  uint32_t kv_num_heads = 0;       // key/value heads (grouped)
  uint32_t head_size = 0;          // per-head dimension
  uint32_t rotary_dim = 0;         // rotary width (<= head_size); 0 disables rotary
  uint32_t past_seq = 0;           // seq dimension of the past K/V buffers
  uint32_t present_seq = 0;        // seq dimension of the present K/V buffers
  bool do_rotary = false;
  bool interleaved = false;
  int32_t local_window_size = -1;  // sliding-window left size; -1 => full causal
  float scale = 0.0f;
};

// Owns id<MTLDevice>, id<MTLCommandQueue>, the compiled id<MTLLibrary> and the
// name -> MTLComputePipelineState map. One instance per EP factory.
class MetalContext {
 public:
  // Creates the context using the system default Metal device and compiles the built-in
  // kernel library from source at runtime. Returns nullptr and sets `error` on failure.
  static std::unique_ptr<MetalContext> Create(std::string& error);

  ~MetalContext();

  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;

  // Human-readable Metal device name (e.g. "Apple M1 Max").
  const std::string& DeviceName() const { return device_name_; }

  // ---- Command-buffer batching ----
  // While a batch is open, every kernel dispatch below ENCODES into a single shared, serial
  // MTLComputeCommandEncoder/MTLCommandBuffer instead of creating its own; commit +
  // waitUntilCompleted + any host copy-backs are deferred to EndBatch. This turns one GPU
  // submission per node into one per fused subgraph. Intermediate tensors that are device
  // buffers (from Alloc) flow between encoded dispatches with no host round-trip. Serial
  // dispatch ordering guarantees each encoded kernel observes the previous kernel's writes.
  // Nesting is not supported. Not thread-safe with concurrent dispatches on other threads.
  bool BeginBatch(std::string& error);
  bool EndBatch(std::string& error);
  bool BatchActive() const;

  // Total GPU command-buffer submissions (commits) so far. One per fused subgraph while batching,
  // one per dispatch otherwise; lets callers measure host/GPU round-trips per decode token.
  uint64_t CommitCount() const;

  // ---- Device allocator (shared unified-memory MTLBuffer pool) ----
  // Allocates `bytes` of shared-storage device memory and returns a CPU-addressable
  // pointer (MTLBuffer.contents). Returns nullptr on failure. Thread-safe.
  void* Alloc(size_t bytes);
  // Frees a pointer previously returned by Alloc. Safe to call with nullptr. Thread-safe.
  void Free(void* ptr);

  // ---- Kernels ----
  // Elementwise c = a + b over `n` output elements, float32, computed on the GPU, with
  // trailing-suffix broadcast: c[i] = a[i % na] + b[i % nb]. Pass na/nb = per-operand element
  // counts and n = max(na, nb) = output element count. Equal shapes use na == nb == n; a bias
  // add [.., C] + [C] uses the smaller operand's count for its dimension. Pointers may be
  // device-allocated (from Alloc) or arbitrary host pointers; the implementation wraps or
  // copies as needed. Returns false and sets `error` on failure.
  bool AddF32(const float* a, size_t na, const float* b, size_t nb, float* c, size_t n,
              std::string& error);

  // ---- Core-compute kernels (Mariette) ----
  // All I/O is fp32, row-major, contiguous. Constant weight tensors (`b`, `scales`) are
  // cached in device buffers on first use and reused across decode steps (see .mm), so the
  // model weights become device-resident after the first token — the key perf lever.

  // MatMulNBits (com.microsoft): Y[M,N] = A[M,K] * dequant(B)^T (+ bias), int4 block-quantized
  // weights with symmetric default zero-point 8. `b` is packed uint8 [N, nblocks, block_size/2],
  // `scales` is fp32 [N, nblocks]. Requires block_size == 32 (so K == nblocks*32). `bias` may be
  // nullptr. Returns false and sets `error` on failure.
  bool MatMulNBitsF32(const float* a, const uint8_t* b, const float* scales, const float* bias,
                      float* y, size_t m, size_t n, size_t k, size_t nblocks, std::string& error);

  // RMSNormalization (ai.onnx, axis=-1): y = x * rsqrt(mean(x^2)+eps) * gamma, over `rows` rows
  // of width `d`.
  bool RmsNormF32(const float* x, const float* gamma, float* y, size_t rows, size_t d, float eps,
                  std::string& error);

  // SkipSimplifiedLayerNormalization (com.microsoft): residual = input + skip; out = residual *
  // rsqrt(mean(residual^2)+eps) * gamma. `residual` may be nullptr if that output is unused.
  bool SkipSimplifiedLayerNormF32(const float* input, const float* skip, const float* gamma,
                                  float* out, float* residual, size_t rows, size_t d, float eps,
                                  std::string& error);

  // Softmax (ai.onnx, axis=-1): numerically stable, over `rows` rows of width `d`.
  bool SoftmaxF32(const float* x, float* y, size_t rows, size_t d, std::string& error);

  // GroupQueryAttention (com.microsoft), fp32. Computes attn_output and fills the present K/V
  // cache from past K/V + new K/V (with rotary embedding on Q and new K when params.do_rotary).
  // Runs as two passes (write-KV then flash attention) encoded back-to-back so the second observes
  // the first's writes. `present_key`/`present_value` may alias `past_key`/`past_value` (the
  // in-place share-buffer path) or be distinct buffers (the copy is then performed on-GPU).
  // Layouts: query [B,S,num_heads*head], key/value [B,S,kv_num_heads*head],
  // past/present K/V [B,kv_num_heads,seq,head], seqlens_k int32[B] (= total valid keys - 1 per
  // batch), cos/sin caches [max_seq, rotary_dim/2] (may be null when do_rotary is false),
  // output [B,S,num_heads*head].
  bool GroupQueryAttention(const float* query, const float* key, const float* value,
                           const float* past_key, const float* past_value,
                           const int32_t* seqlens_k, const float* cos_cache,
                           const float* sin_cache, float* output, float* present_key,
                           float* present_value, const GroupQueryAttentionParams& params,
                           std::string& error);

  bool Binary(BinaryOp op, ScalarType type, const void* a, size_t na, const void* b, size_t nb,
              void* output, size_t n, std::string& error);
  bool Unary(UnaryOp op, ScalarType type, const void* input, void* output, size_t n,
             std::string& error);
  bool Cast(ScalarType input_type, ScalarType output_type, const void* input, void* output,
            size_t n, std::string& error);

  // ONNX RotaryEmbedding. X/cos/sin/output have `type`; position_ids is optional int64.
  bool RotaryEmbedding(ScalarType type, const void* input, const void* cos_cache,
                       const void* sin_cache, const int64_t* position_ids, void* output,
                       size_t n, const RotaryEmbeddingParams& params, std::string& error);

  // com.microsoft.GatherBlockQuantized specialization used by Qwen embedding tables:
  // uint8-packed int4 data, gather_axis=0, quantize_axis=last, block-wise scales.
  bool GatherBlockQuantized(const uint8_t* data, size_t data_bytes, const void* indices,
                            bool indices_i64, size_t indices_count, const void* scales,
                            ScalarType output_type, const uint8_t* zero_points,
                            size_t zero_points_bytes, void* output, uint32_t rows,
                            uint32_t row_width, uint32_t packed_row_width,
                            uint32_t block_size, std::string& error);

  bool CopyBytes(const void* input, void* output, size_t bytes, std::string& error);
  bool TransposeBytes(const void* input, void* output, size_t element_count,
                      uint32_t element_size, uint32_t rank, const uint32_t* output_dims,
                      const uint32_t* input_strides, const uint32_t* permutation,
                      std::string& error);
  bool ConcatSliceBytes(const void* input, size_t input_bytes, void* output,
                        size_t output_bytes, uint32_t element_size, uint32_t outer,
                        uint32_t input_axis, uint32_t output_axis, uint32_t inner,
                        uint32_t axis_offset, std::string& error);

 public:
  // Opaque implementation type. Forward-declared publicly only so file-local helpers in
  // metal_context.mm can reference MetalContext::Impl. Not part of the stable API.
  struct Impl;
  Impl* impl() { return impl_.get(); }

 private:
  MetalContext();
  std::unique_ptr<Impl> impl_;
  std::string device_name_;
};

}  // namespace ort_mps
