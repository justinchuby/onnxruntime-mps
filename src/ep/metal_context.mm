// Copyright (c) 2026. Licensed under the MIT License.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_context.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ort_mps {
namespace {

static const char* kKernelSource =
#include "metal_kernels.inc"
    ;

enum class BufferAccess {
  Input,
  Output,
  ReadWrite,
};

struct BufferArg {
  const void* data;
  size_t bytes;
  BufferAccess access;
};

struct BytesArg {
  const void* data;
  size_t bytes;
};

struct GatherBlockQuantizedParams {
  uint32_t rows;
  uint32_t row_width;
  uint32_t packed_row_width;
  uint32_t blocks_per_row;
  uint32_t block_size;
  uint32_t indices_count;
  uint32_t n;
};

struct RopeParams {
  uint32_t batch_size;
  uint32_t sequence_length;
  uint32_t num_heads;
  uint32_t head_size;
  uint32_t rotary_embedding_dim;
  uint32_t cache_stride;
  uint32_t max_sequence_length;
  uint32_t rank3_bsh;
  uint32_t interleaved;
  uint32_t n;
};

struct TransposeParams {
  uint32_t rank;
  uint32_t element_size;
  uint32_t n;
  uint32_t output_dims[8];
  uint32_t input_strides[8];
  uint32_t permutation[8];
};

struct ConcatSliceParams {
  uint32_t element_size;
  uint32_t outer;
  uint32_t input_axis;
  uint32_t output_axis;
  uint32_t inner;
  uint32_t axis_offset;
  uint32_t n;
};

size_t ScalarSize(ScalarType type) {
  switch (type) {
    case ScalarType::Float16:
      return 2;
    case ScalarType::Float32:
    case ScalarType::Int32:
      return 4;
    case ScalarType::Int64:
      return 8;
  }
  return 0;
}

bool ToU32(size_t value, uint32_t& result, const char* what, std::string& error) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    error = std::string(what) + " exceeds Metal kernel uint32 limits";
    return false;
  }
  result = static_cast<uint32_t>(value);
  return true;
}

const char* BinaryPipeline(BinaryOp op, ScalarType type) {
  if (type == ScalarType::Float32) {
    switch (op) {
      case BinaryOp::Add:
        return "mps_add_f32";
      case BinaryOp::Mul:
        return "mps_mul_f32";
      case BinaryOp::Sub:
        return "mps_sub_f32";
      case BinaryOp::Div:
        return "mps_div_f32";
    }
  }
  if (type == ScalarType::Float16) {
    switch (op) {
      case BinaryOp::Add:
        return "mps_add_f16";
      case BinaryOp::Mul:
        return "mps_mul_f16";
      case BinaryOp::Sub:
        return "mps_sub_f16";
      case BinaryOp::Div:
        return "mps_div_f16";
    }
  }
  if (type == ScalarType::Int64 && op == BinaryOp::Sub) {
    return "mps_sub_i64";
  }
  return nullptr;
}

const char* UnaryPipeline(UnaryOp op, ScalarType type) {
  const char* suffix = type == ScalarType::Float16 ? "f16" :
                       type == ScalarType::Float32 ? "f32" : nullptr;
  if (suffix == nullptr) {
    return nullptr;
  }
  switch (op) {
    case UnaryOp::Sigmoid:
      return type == ScalarType::Float16 ? "mps_sigmoid_f16" : "mps_sigmoid_f32";
    case UnaryOp::SiLU:
      return type == ScalarType::Float16 ? "mps_silu_f16" : "mps_silu_f32";
    case UnaryOp::Gelu:
      return type == ScalarType::Float16 ? "mps_gelu_f16" : "mps_gelu_f32";
    case UnaryOp::GeluTanh:
      return type == ScalarType::Float16 ? "mps_gelu_tanh_f16" : "mps_gelu_tanh_f32";
  }
  return nullptr;
}

}  // namespace

struct MetalContext::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

  std::mutex alloc_mutex;
  std::unordered_map<void*, id<MTLBuffer>> buffers;

  // Open-batch state (see MetalContext::BeginBatch). When batch_active is true, dispatches
  // encode into batch_encoder/batch_command and defer submission/copy-back.
  bool batch_active = false;
  id<MTLCommandBuffer> batch_command = nil;    // retained while the batch is open (MRR)
  id<MTLComputeCommandEncoder> batch_encoder = nil;  // retained while the batch is open (MRR)
  // Temporary buffers created to wrap foreign host pointers during the batch; released in
  // EndBatch once the GPU has finished reading/writing them.
  std::vector<id<MTLBuffer>> batch_temps;
  // Host copy-backs (for wrapped output pointers) deferred until the batch completes.
  struct DeferredCopy {
    void* host;
    id<MTLBuffer> buffer;
    size_t bytes;
  };
  std::vector<DeferredCopy> batch_copybacks;

  // Total GPU submissions (command-buffer commits). One per fused subgraph in batch mode, one per
  // dispatch otherwise. Used to measure host/GPU round-trips per token.
  uint64_t commit_count = 0;
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() {
  if (!impl_) {
    return;
  }
  if (std::getenv("ONNX_GENAI_METAL_EP_PROFILE") != nullptr) {
    fprintf(stderr, "[MetalEP] total GPU command-buffer commits this session: %llu\n",
            static_cast<unsigned long long>(impl_->commit_count));
  }
  for (auto& entry : impl_->buffers) {
    entry.second = nil;
  }
  impl_->buffers.clear();
  for (auto& entry : impl_->pipelines) {
    entry.second = nil;
  }
  impl_->pipelines.clear();
  impl_->library = nil;
  impl_->queue = nil;
  impl_->device = nil;
}

std::unique_ptr<MetalContext> MetalContext::Create(std::string& error) {
  @autoreleasepool {
    std::unique_ptr<MetalContext> ctx(new MetalContext());
    Impl& impl = *ctx->impl_;

    impl.device = MTLCreateSystemDefaultDevice();
    if (impl.device == nil) {
      error = "MTLCreateSystemDefaultDevice returned nil (no Metal-capable GPU)";
      return nullptr;
    }
    ctx->device_name_ = std::string([[impl.device name] UTF8String]);

    impl.queue = [impl.device newCommandQueue];
    if (impl.queue == nil) {
      error = "Failed to create MTLCommandQueue";
      return nullptr;
    }

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    impl.library = [impl.device newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                                            options:opts
                                              error:&err];
    if (impl.library == nil) {
      error = std::string("Failed to compile Metal kernel library: ") +
              (err ? [[err localizedDescription] UTF8String] : "unknown error");
      return nullptr;
    }

    static const char* kPipelineNames[] = {
        "mps_add_f32",          "mps_mul_f32",          "mps_sub_f32",
        "mps_div_f32",          "mps_add_f16",          "mps_mul_f16",
        "mps_sub_f16",          "mps_div_f16",          "mps_sub_i64",
        "mps_sigmoid_f32",      "mps_silu_f32",         "mps_gelu_f32",
        "mps_gelu_tanh_f32",    "mps_sigmoid_f16",      "mps_silu_f16",
        "mps_gelu_f16",         "mps_gelu_tanh_f16",    "mps_cast_f32_f16",
        "mps_cast_f16_f32",     "mps_cast_i64_i32",     "mps_rope_f32",
        "mps_rope_pos_f32",     "mps_rope_f16",         "mps_rope_pos_f16",
        "mps_gather_q4_i64_f32", "mps_gather_q4_i32_f32",
        "mps_gather_q4_i64_f16", "mps_gather_q4_i32_f16",
        "mps_gather_q4_i64_f32_zp", "mps_gather_q4_i32_f32_zp",
        "mps_gather_q4_i64_f16_zp", "mps_gather_q4_i32_f16_zp",
        "mps_copy_bytes",       "mps_transpose_bytes",  "mps_concat_slice_bytes",
        // Mariette core-compute kernels.
        "mps_matmulnbits_f32",  "mps_rmsnorm_f32",
        "mps_skip_simplified_layernorm_f32", "mps_softmax_f32",
    };

    for (const char* name : kPipelineNames) {
      id<MTLFunction> fn =
          [impl.library newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (fn == nil) {
        error = std::string("Kernel function ") + name + " not found in compiled library";
        return nullptr;
      }
      id<MTLComputePipelineState> pipeline =
          [impl.device newComputePipelineStateWithFunction:fn error:&err];
      if (pipeline == nil) {
        error = std::string("Failed to create pipeline state for ") + name + ": " +
                (err ? [[err localizedDescription] UTF8String] : "unknown error");
        return nullptr;
      }
      impl.pipelines.emplace(name, pipeline);
    }

    return ctx;
  }
}

void* MetalContext::Alloc(size_t bytes) {
  @autoreleasepool {
    if (bytes == 0) {
      bytes = 1;
    }
    id<MTLBuffer> buffer =
        [impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return nullptr;
    }
    void* ptr = [buffer contents];
    std::lock_guard<std::mutex> lock(impl_->alloc_mutex);
    impl_->buffers.emplace(ptr, buffer);
    return ptr;
  }
}

void MetalContext::Free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->alloc_mutex);
  auto it = impl_->buffers.find(ptr);
  if (it != impl_->buffers.end()) {
    it->second = nil;
    impl_->buffers.erase(it);
  }
}

static id<MTLBuffer> LookupBuffer(MetalContext::Impl& impl, const void* ptr, size_t& offset) {
  offset = 0;
  std::lock_guard<std::mutex> lock(impl.alloc_mutex);
  auto it = impl.buffers.find(const_cast<void*>(ptr));
  return it == impl.buffers.end() ? nil : it->second;
}

bool MetalContext::BatchActive() const { return impl_->batch_active; }

uint64_t MetalContext::CommitCount() const { return impl_->commit_count; }

bool MetalContext::BeginBatch(std::string& error) {
  Impl& impl = *impl_;
  if (impl.batch_active) {
    error = "MetalContext::BeginBatch called while a batch is already open";
    return false;
  }
  id<MTLCommandBuffer> command = [impl.queue commandBuffer];
  if (command == nil) {
    error = "MetalContext::BeginBatch failed to create a command buffer";
    return false;
  }
  // Default (serial) compute encoder: encoded dispatches execute in order and each observes the
  // previous dispatch's memory writes, so data-dependent kernels need no explicit barriers.
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    error = "MetalContext::BeginBatch failed to create a compute encoder";
    return false;
  }
  // command/encoder from -commandBuffer/-computeCommandEncoder are autoreleased (+0); retain them
  // so they survive across the per-dispatch autorelease pools until EndBatch (MRR, no ARC here).
  impl.batch_command = [command retain];
  impl.batch_encoder = [encoder retain];
  impl.batch_active = true;
  return true;
}

bool MetalContext::EndBatch(std::string& error) {
  Impl& impl = *impl_;
  if (!impl.batch_active) {
    error = "MetalContext::EndBatch called with no open batch";
    return false;
  }
  bool ok = true;
  @autoreleasepool {
    [impl.batch_encoder endEncoding];
    [impl.batch_command commit];
    [impl.batch_command waitUntilCompleted];
    impl.commit_count++;
    if (impl.batch_command.status == MTLCommandBufferStatusError) {
      error = std::string("Batch command buffer failed: ") +
              (impl.batch_command.error ? [[impl.batch_command.error localizedDescription] UTF8String]
                                        : "unknown");
      ok = false;
    }
    if (ok) {
      for (const Impl::DeferredCopy& c : impl.batch_copybacks) {
        if (c.host != nullptr && c.bytes != 0) {
          std::memcpy(c.host, [c.buffer contents], c.bytes);
        }
      }
    }
  }
  for (id<MTLBuffer> temp : impl.batch_temps) {
    [temp release];
  }
  impl.batch_temps.clear();
  impl.batch_copybacks.clear();
  [impl.batch_encoder release];
  [impl.batch_command release];
  impl.batch_encoder = nil;
  impl.batch_command = nil;
  impl.batch_active = false;
  return ok;
}

static bool Dispatch(MetalContext::Impl& impl, const char* pipeline_name,
                     const std::vector<BufferArg>& buffers,
                     const std::vector<BytesArg>& constants, size_t grid_size,
                     std::string& error) {
  @autoreleasepool {
    if (grid_size == 0) {
      return true;
    }
    auto pipeline_it = impl.pipelines.find(pipeline_name);
    if (pipeline_it == impl.pipelines.end()) {
      error = std::string("Metal pipeline is unavailable: ") + pipeline_name;
      return false;
    }
    id<MTLComputePipelineState> pipeline = pipeline_it->second;

    std::vector<id<MTLBuffer>> metal_buffers;
    std::vector<size_t> offsets(buffers.size(), 0);
    metal_buffers.reserve(buffers.size());
    std::vector<bool> copy_back(buffers.size(), false);
    // Temporary (+1 owned) buffers we allocate to wrap foreign host pointers; freed after the
    // command completes (immediately when we own the command buffer, else deferred to EndBatch).
    std::vector<id<MTLBuffer>> owned_temps;
    for (size_t i = 0; i < buffers.size(); ++i) {
      const BufferArg& arg = buffers[i];
      size_t offset = 0;
      id<MTLBuffer> buffer = LookupBuffer(impl, arg.data, offset);
      if (buffer == nil) {
        const size_t length = std::max<size_t>(arg.bytes, 1);
        if (arg.access == BufferAccess::Output) {
          buffer = [impl.device newBufferWithLength:length options:MTLResourceStorageModeShared];
        } else {
          buffer = [impl.device newBufferWithBytes:arg.data
                                           length:length
                                          options:MTLResourceStorageModeShared];
        }
        copy_back[i] = arg.access != BufferAccess::Input;
        if (buffer != nil) {
          owned_temps.push_back(buffer);  // +1 owned
        }
      }
      if (buffer == nil) {
        for (id<MTLBuffer> t : owned_temps) [t release];
        error = std::string("Failed to allocate Metal buffer for ") + pipeline_name;
        return false;
      }
      offsets[i] = offset;
      metal_buffers.push_back(buffer);
    }

    const bool own_command = !impl.batch_active;
    id<MTLCommandBuffer> command = own_command ? [impl.queue commandBuffer] : impl.batch_command;
    id<MTLComputeCommandEncoder> encoder =
        own_command ? [command computeCommandEncoder] : impl.batch_encoder;
    if (command == nil || encoder == nil) {
      for (id<MTLBuffer> t : owned_temps) [t release];
      error = std::string("Failed to create Metal command encoder for ") + pipeline_name;
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    for (size_t i = 0; i < metal_buffers.size(); ++i) {
      [encoder setBuffer:metal_buffers[i] offset:offsets[i] atIndex:i];
    }
    for (size_t i = 0; i < constants.size(); ++i) {
      const BytesArg& arg = constants[i];
      [encoder setBytes:arg.data length:arg.bytes atIndex:buffers.size() + i];
    }

    NSUInteger threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;
    threads_per_group = std::min<NSUInteger>(threads_per_group, grid_size);
    threads_per_group = std::max<NSUInteger>(threads_per_group, 1);
    [encoder dispatchThreads:MTLSizeMake(grid_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

    if (!own_command) {
      // Batch mode: defer submission, copy-backs, and temp-buffer release to EndBatch.
      for (size_t i = 0; i < buffers.size(); ++i) {
        if (copy_back[i] && buffers[i].bytes != 0) {
          impl.batch_copybacks.push_back(
              {const_cast<void*>(buffers[i].data), metal_buffers[i], buffers[i].bytes});
        }
      }
      for (id<MTLBuffer> t : owned_temps) impl.batch_temps.push_back(t);
      return true;
    }

    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    impl.commit_count++;

    if (command.status == MTLCommandBufferStatusError) {
      error = std::string(pipeline_name) + " command buffer failed: " +
              (command.error ? [[command.error localizedDescription] UTF8String] : "unknown");
      for (id<MTLBuffer> t : owned_temps) [t release];
      return false;
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
      if (copy_back[i] && buffers[i].bytes != 0) {
        std::memcpy(const_cast<void*>(buffers[i].data), [metal_buffers[i] contents],
                    buffers[i].bytes);
      }
    }
    for (id<MTLBuffer> t : owned_temps) [t release];
    return true;
  }
}

bool MetalContext::AddF32(const float* a, size_t na, const float* b, size_t nb, float* c,
                          size_t n, std::string& error) {
  return Binary(BinaryOp::Add, ScalarType::Float32, a, na, b, nb, c, n, error);
}

bool MetalContext::Binary(BinaryOp op, ScalarType type, const void* a, size_t na, const void* b,
                          size_t nb, void* output, size_t n, std::string& error) {
  if (na == 0 || nb == 0) {
    error = "Binary Metal kernels require non-zero operand element counts";
    return false;
  }
  const char* pipeline = BinaryPipeline(op, type);
  if (pipeline == nullptr) {
    error = "Unsupported Metal binary op/type combination";
    return false;
  }
  uint32_t na32, nb32, n32;
  if (!ToU32(na, na32, "left operand element count", error) ||
      !ToU32(nb, nb32, "right operand element count", error) ||
      !ToU32(n, n32, "output element count", error)) {
    return false;
  }
  const size_t element_size = ScalarSize(type);
  return Dispatch(*impl_, pipeline,
                  {{a, na * element_size, BufferAccess::Input},
                   {b, nb * element_size, BufferAccess::Input},
                   {output, n * element_size, BufferAccess::Output}},
                  {{&na32, sizeof(na32)}, {&nb32, sizeof(nb32)}, {&n32, sizeof(n32)}}, n,
                  error);
}

bool MetalContext::Unary(UnaryOp op, ScalarType type, const void* input, void* output, size_t n,
                         std::string& error) {
  const char* pipeline = UnaryPipeline(op, type);
  if (pipeline == nullptr) {
    error = "Unsupported Metal unary op/type combination";
    return false;
  }
  uint32_t n32;
  if (!ToU32(n, n32, "element count", error)) {
    return false;
  }
  const size_t bytes = n * ScalarSize(type);
  return Dispatch(*impl_, pipeline,
                  {{input, bytes, BufferAccess::Input}, {output, bytes, BufferAccess::Output}},
                  {{&n32, sizeof(n32)}}, n, error);
}

bool MetalContext::Cast(ScalarType input_type, ScalarType output_type, const void* input,
                        void* output, size_t n, std::string& error) {
  const char* pipeline = nullptr;
  if (input_type == ScalarType::Float32 && output_type == ScalarType::Float16) {
    pipeline = "mps_cast_f32_f16";
  } else if (input_type == ScalarType::Float16 && output_type == ScalarType::Float32) {
    pipeline = "mps_cast_f16_f32";
  } else if (input_type == ScalarType::Int64 && output_type == ScalarType::Int32) {
    pipeline = "mps_cast_i64_i32";
  }
  if (pipeline == nullptr) {
    error = "Unsupported Metal Cast type pair";
    return false;
  }
  uint32_t n32;
  if (!ToU32(n, n32, "Cast element count", error)) {
    return false;
  }
  return Dispatch(*impl_, pipeline,
                  {{input, n * ScalarSize(input_type), BufferAccess::Input},
                   {output, n * ScalarSize(output_type), BufferAccess::Output}},
                  {{&n32, sizeof(n32)}}, n, error);
}

bool MetalContext::RotaryEmbedding(ScalarType type, const void* input, const void* cos_cache,
                                   const void* sin_cache, const int64_t* position_ids,
                                   void* output, size_t n,
                                   const RotaryEmbeddingParams& params, std::string& error) {
  if (type != ScalarType::Float16 && type != ScalarType::Float32) {
    error = "RotaryEmbedding supports float16 and float32 only";
    return false;
  }
  if (params.rotary_embedding_dim == 0 ||
      (params.rotary_embedding_dim & 1) != 0 ||
      params.rotary_embedding_dim > params.head_size) {
    error = "RotaryEmbedding requires an even rotary dimension no larger than head_size";
    return false;
  }
  uint32_t n32;
  if (!ToU32(n, n32, "RotaryEmbedding element count", error)) {
    return false;
  }
  if (position_ids != nullptr) {
    const size_t position_count =
        static_cast<size_t>(params.batch_size) * params.sequence_length;
    for (size_t i = 0; i < position_count; ++i) {
      if (position_ids[i] < 0 ||
          static_cast<uint64_t>(position_ids[i]) >= params.max_sequence_length) {
        error = "RotaryEmbedding position id is outside the cos/sin cache";
        return false;
      }
    }
  }

  RopeParams p = {params.batch_size,
                  params.sequence_length,
                  params.num_heads,
                  params.head_size,
                  params.rotary_embedding_dim,
                  params.cache_stride,
                  params.max_sequence_length,
                  params.rank3_bsh ? 1u : 0u,
                  params.interleaved ? 1u : 0u,
                  n32};
  const size_t element_size = ScalarSize(type);
  const size_t cache_bytes =
      static_cast<size_t>(params.max_sequence_length) * params.cache_stride * element_size;
  const bool has_positions = position_ids != nullptr;
  const char* pipeline = nullptr;
  if (type == ScalarType::Float16) {
    pipeline = has_positions ? "mps_rope_pos_f16" : "mps_rope_f16";
  } else {
    pipeline = has_positions ? "mps_rope_pos_f32" : "mps_rope_f32";
  }
  std::vector<BufferArg> buffers = {
      {input, n * element_size, BufferAccess::Input},
      {cos_cache, cache_bytes, BufferAccess::Input},
      {sin_cache, cache_bytes, BufferAccess::Input},
  };
  if (has_positions) {
    buffers.push_back({position_ids,
                       static_cast<size_t>(params.batch_size) * params.sequence_length *
                           sizeof(int64_t),
                       BufferAccess::Input});
  }
  buffers.push_back({output, n * element_size, BufferAccess::Output});
  return Dispatch(*impl_, pipeline, buffers, {{&p, sizeof(p)}}, n, error);
}

bool MetalContext::GatherBlockQuantized(
    const uint8_t* data, size_t data_bytes, const void* indices, bool indices_i64,
    size_t indices_count, const void* scales, ScalarType output_type,
    const uint8_t* zero_points, size_t zero_points_bytes, void* output, uint32_t rows,
    uint32_t row_width, uint32_t packed_row_width, uint32_t block_size,
    std::string& error) {
  if (output_type != ScalarType::Float16 && output_type != ScalarType::Float32) {
    error = "GatherBlockQuantized supports float16 and float32 scales/output only";
    return false;
  }
  if (block_size == 0 || row_width == 0 || packed_row_width != (row_width + 1) / 2) {
    error = "GatherBlockQuantized received inconsistent packed row dimensions";
    return false;
  }
  if (data_bytes < static_cast<size_t>(rows) * packed_row_width) {
    error = "GatherBlockQuantized packed data buffer is too small";
    return false;
  }
  if (indices_i64) {
    const int64_t* values = static_cast<const int64_t*>(indices);
    for (size_t i = 0; i < indices_count; ++i) {
      if (values[i] < -static_cast<int64_t>(rows) || values[i] >= rows) {
        error = "GatherBlockQuantized index is outside the embedding table";
        return false;
      }
    }
  } else {
    const int32_t* values = static_cast<const int32_t*>(indices);
    for (size_t i = 0; i < indices_count; ++i) {
      if (values[i] < -static_cast<int32_t>(rows) ||
          values[i] >= static_cast<int32_t>(rows)) {
        error = "GatherBlockQuantized index is outside the embedding table";
        return false;
      }
    }
  }

  const uint32_t blocks_per_row = (row_width + block_size - 1) / block_size;
  const size_t n = indices_count * static_cast<size_t>(row_width);
  uint32_t indices_count32, n32;
  if (!ToU32(indices_count, indices_count32, "GatherBlockQuantized index count", error) ||
      !ToU32(n, n32, "GatherBlockQuantized output element count", error)) {
    return false;
  }
  const bool has_zero_points = zero_points != nullptr;
  const size_t expected_zp_bytes =
      static_cast<size_t>(rows) * ((blocks_per_row + 1) / 2);
  if (has_zero_points && zero_points_bytes < expected_zp_bytes) {
    error = "GatherBlockQuantized zero-point buffer is too small";
    return false;
  }

  GatherBlockQuantizedParams p = {rows,
                                  row_width,
                                  packed_row_width,
                                  blocks_per_row,
                                  block_size,
                                  indices_count32,
                                  n32};
  std::string pipeline = "mps_gather_q4_";
  pipeline += indices_i64 ? "i64_" : "i32_";
  pipeline += output_type == ScalarType::Float16 ? "f16" : "f32";
  if (has_zero_points) {
    pipeline += "_zp";
  }

  const size_t scalar_size = ScalarSize(output_type);
  std::vector<BufferArg> buffers = {
      {data, data_bytes, BufferAccess::Input},
      {indices, indices_count * (indices_i64 ? sizeof(int64_t) : sizeof(int32_t)),
       BufferAccess::Input},
      {scales, static_cast<size_t>(rows) * blocks_per_row * scalar_size,
       BufferAccess::Input},
  };
  if (has_zero_points) {
    buffers.push_back({zero_points, zero_points_bytes, BufferAccess::Input});
  }
  buffers.push_back({output, n * scalar_size, BufferAccess::Output});
  return Dispatch(*impl_, pipeline.c_str(), buffers, {{&p, sizeof(p)}}, n, error);
}

bool MetalContext::CopyBytes(const void* input, void* output, size_t bytes,
                             std::string& error) {
  uint32_t n32;
  if (!ToU32(bytes, n32, "copy byte count", error)) {
    return false;
  }
  return Dispatch(*impl_, "mps_copy_bytes",
                  {{input, bytes, BufferAccess::Input},
                   {output, bytes, BufferAccess::Output}},
                  {{&n32, sizeof(n32)}}, bytes, error);
}

bool MetalContext::TransposeBytes(const void* input, void* output, size_t element_count,
                                  uint32_t element_size, uint32_t rank,
                                  const uint32_t* output_dims,
                                  const uint32_t* input_strides,
                                  const uint32_t* permutation, std::string& error) {
  if (rank == 0 || rank > 8 || element_size == 0 || element_size > 8) {
    error = "Transpose supports ranks 1..8 and element sizes 1..8";
    return false;
  }
  uint32_t n32;
  if (!ToU32(element_count, n32, "Transpose element count", error)) {
    return false;
  }
  TransposeParams p = {};
  p.rank = rank;
  p.element_size = element_size;
  p.n = n32;
  for (uint32_t i = 0; i < rank; ++i) {
    p.output_dims[i] = output_dims[i];
    p.input_strides[i] = input_strides[i];
    p.permutation[i] = permutation[i];
  }
  const size_t bytes = element_count * element_size;
  return Dispatch(*impl_, "mps_transpose_bytes",
                  {{input, bytes, BufferAccess::Input},
                   {output, bytes, BufferAccess::Output}},
                  {{&p, sizeof(p)}}, element_count, error);
}

bool MetalContext::ConcatSliceBytes(
    const void* input, size_t input_bytes, void* output, size_t output_bytes,
    uint32_t element_size, uint32_t outer, uint32_t input_axis, uint32_t output_axis,
    uint32_t inner, uint32_t axis_offset, std::string& error) {
  const size_t element_count = static_cast<size_t>(outer) * input_axis * inner;
  uint32_t n32;
  if (!ToU32(element_count, n32, "Concat slice element count", error)) {
    return false;
  }
  ConcatSliceParams p = {element_size, outer, input_axis, output_axis,
                         inner, axis_offset, n32};
  return Dispatch(*impl_, "mps_concat_slice_bytes",
                  {{input, input_bytes, BufferAccess::Input},
                   {output, output_bytes, BufferAccess::ReadWrite}},
                  {{&p, sizeof(p)}}, element_count, error);
}

// ---------------------------------------------------------------------------
// Mariette core-compute kernels (MatMulNBits, RMSNorm, SkipSimplifiedLayerNorm, Softmax).
// These use bespoke 2-D / threadgroup-per-row dispatch (not the 1-D Dispatch helper above) and
// resolve each operand to an MTLBuffer via LookupBuffer, wrapping foreign host pointers only for
// non-cached operands. Constant weights are expected to already be device-resident (the kernel
// object copies them once via Alloc), so LookupBuffer resolves them with zero copy per step.
// ---------------------------------------------------------------------------

namespace {

struct ResolvedBuffer {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  bool copy_back = false;  // temp output buffer whose contents must be memcpy'd to `host` after
  bool owned = false;      // buffer was newly created here (+1) and must be released after use
  void* host = nullptr;
  size_t bytes = 0;
};

static ResolvedBuffer ResolveMC(MetalContext::Impl& impl, const void* ptr, size_t bytes,
                                bool is_output) {
  ResolvedBuffer r;
  r.bytes = bytes;
  size_t offset = 0;
  id<MTLBuffer> found = LookupBuffer(impl, ptr, offset);
  if (found != nil) {
    r.buffer = found;
    r.offset = offset;
    return r;
  }
  const size_t length = std::max<size_t>(bytes, 1);
  if (is_output) {
    r.buffer = [impl.device newBufferWithLength:length options:MTLResourceStorageModeShared];
    r.copy_back = true;
    r.host = const_cast<void*>(ptr);
  } else {
    r.buffer = [impl.device newBufferWithBytes:ptr length:length
                                       options:MTLResourceStorageModeShared];
  }
  r.owned = r.buffer != nil;  // +1 owned; released after the command completes (or in EndBatch)
  return r;
}

// Encodes one compute pass. `grid`/`tg` are interpreted as thread counts when `by_threadgroups`
// is false (dispatchThreads) or as threadgroup/threadgroup-size when true (dispatchThreadgroups).
static bool RunPass(MetalContext::Impl& impl, const char* pipeline_name,
                    const std::vector<ResolvedBuffer>& buffers, const std::vector<BytesArg>& constants,
                    MTLSize grid, MTLSize tg, bool by_threadgroups, std::string& error) {
  auto it = impl.pipelines.find(pipeline_name);
  if (it == impl.pipelines.end()) {
    error = std::string("Metal pipeline is unavailable: ") + pipeline_name;
    return false;
  }
  id<MTLComputePipelineState> pipeline = it->second;
  const bool own_command = !impl.batch_active;
  id<MTLCommandBuffer> command = own_command ? [impl.queue commandBuffer] : impl.batch_command;
  id<MTLComputeCommandEncoder> encoder =
      own_command ? [command computeCommandEncoder] : impl.batch_encoder;
  if (command == nil || encoder == nil) {
    error = std::string("Failed to create Metal command encoder for ") + pipeline_name;
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  NSUInteger index = 0;
  for (const ResolvedBuffer& b : buffers) {
    [encoder setBuffer:b.buffer offset:b.offset atIndex:index++];
  }
  for (const BytesArg& c : constants) {
    [encoder setBytes:c.data length:c.bytes atIndex:index++];
  }
  if (by_threadgroups) {
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  } else {
    [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
  }

  if (!own_command) {
    // Batch mode: defer submission, copy-backs, and temp-buffer release to EndBatch.
    for (const ResolvedBuffer& b : buffers) {
      if (b.copy_back && b.host != nullptr) {
        impl.batch_copybacks.push_back({b.host, b.buffer, b.bytes});
      }
    }
    for (const ResolvedBuffer& b : buffers) {
      if (b.owned) impl.batch_temps.push_back(b.buffer);
    }
    return true;
  }

  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  impl.commit_count++;
  const bool failed = command.status == MTLCommandBufferStatusError;
  if (failed) {
    error = std::string(pipeline_name) + " command buffer failed: " +
            (command.error ? [[command.error localizedDescription] UTF8String] : "unknown");
  } else {
    for (const ResolvedBuffer& b : buffers) {
      if (b.copy_back && b.host != nullptr) {
        memcpy(b.host, [b.buffer contents], b.bytes);
      }
    }
  }
  for (const ResolvedBuffer& b : buffers) {
    if (b.owned) [b.buffer release];
  }
  return !failed;
}

}  // namespace

bool MetalContext::MatMulNBitsF32(const float* a, const uint8_t* b, const float* scales,
                                  const float* bias, float* y, size_t m, size_t n, size_t k,
                                  size_t nblocks, std::string& error) {
  @autoreleasepool {
    if (m == 0 || n == 0) {
      return true;
    }
    if (k == 0 || nblocks == 0 || k != nblocks * 32) {
      error = "MetalContext::MatMulNBitsF32 requires block_size==32 (K == nblocks*32)";
      return false;
    }
    Impl& impl = *impl_;
    const size_t bytes_per_block = 16;  // 32 int4 lanes packed two-per-byte
    std::vector<ResolvedBuffer> bufs;
    bufs.push_back(ResolveMC(impl, a, m * k * sizeof(float), false));
    bufs.push_back(ResolveMC(impl, b, n * nblocks * bytes_per_block, false));
    bufs.push_back(ResolveMC(impl, scales, n * nblocks * sizeof(float), false));
    bufs.push_back(ResolveMC(impl, y, m * n * sizeof(float), true));
    float dummy_bias = 0.0f;
    const uint32_t has_bias = bias != nullptr ? 1u : 0u;
    bufs.push_back(has_bias ? ResolveMC(impl, bias, n * sizeof(float), false)
                            : ResolveMC(impl, &dummy_bias, sizeof(float), false));
    for (const ResolvedBuffer& rb : bufs) {
      if (rb.buffer == nil) {
        error = "MetalContext::MatMulNBitsF32 failed to allocate a Metal buffer";
        return false;
      }
    }
    const uint32_t M = static_cast<uint32_t>(m), N = static_cast<uint32_t>(n),
                   K = static_cast<uint32_t>(k), NB = static_cast<uint32_t>(nblocks);
    std::vector<BytesArg> consts = {{&M, sizeof(M)}, {&N, sizeof(N)}, {&K, sizeof(K)},
                                    {&NB, sizeof(NB)}, {&has_bias, sizeof(has_bias)}};
    MTLSize grid = MTLSizeMake(n * 32, m, 1);        // one simdgroup (32 lanes) per output column
    MTLSize tg = MTLSizeMake(256, 1, 1);             // 8 columns per threadgroup
    return RunPass(impl, "mps_matmulnbits_f32", bufs, consts, grid, tg, /*by_threadgroups=*/false,
                   error);
  }
}

bool MetalContext::RmsNormF32(const float* x, const float* gamma, float* y, size_t rows, size_t d,
                              float eps, std::string& error) {
  @autoreleasepool {
    if (rows == 0 || d == 0) {
      return true;
    }
    Impl& impl = *impl_;
    std::vector<ResolvedBuffer> bufs = {
        ResolveMC(impl, x, rows * d * sizeof(float), false),
        ResolveMC(impl, gamma, d * sizeof(float), false),
        ResolveMC(impl, y, rows * d * sizeof(float), true)};
    for (const ResolvedBuffer& rb : bufs) {
      if (rb.buffer == nil) { error = "RmsNormF32 buffer alloc failed"; return false; }
    }
    const uint32_t D = static_cast<uint32_t>(d);
    std::vector<BytesArg> consts = {{&D, sizeof(D)}, {&eps, sizeof(eps)}};
    const NSUInteger tg_width = std::min<NSUInteger>(256, ((d + 31) / 32) * 32);
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize tg = MTLSizeMake(std::max<NSUInteger>(tg_width, 32), 1, 1);
    return RunPass(impl, "mps_rmsnorm_f32", bufs, consts, grid, tg, /*by_threadgroups=*/true, error);
  }
}

bool MetalContext::SkipSimplifiedLayerNormF32(const float* input, const float* skip,
                                              const float* gamma, float* out, float* residual,
                                              size_t rows, size_t d, float eps,
                                              std::string& error) {
  @autoreleasepool {
    if (rows == 0 || d == 0) {
      return true;
    }
    Impl& impl = *impl_;
    const uint32_t want_res = residual != nullptr ? 1u : 0u;
    float dummy_res = 0.0f;
    std::vector<ResolvedBuffer> bufs = {
        ResolveMC(impl, input, rows * d * sizeof(float), false),
        ResolveMC(impl, skip, rows * d * sizeof(float), false),
        ResolveMC(impl, gamma, d * sizeof(float), false),
        ResolveMC(impl, out, rows * d * sizeof(float), true),
        want_res ? ResolveMC(impl, residual, rows * d * sizeof(float), true)
                 : ResolveMC(impl, &dummy_res, sizeof(float), true)};
    for (const ResolvedBuffer& rb : bufs) {
      if (rb.buffer == nil) { error = "SkipSimplifiedLayerNormF32 buffer alloc failed"; return false; }
    }
    const uint32_t D = static_cast<uint32_t>(d);
    std::vector<BytesArg> consts = {{&D, sizeof(D)}, {&eps, sizeof(eps)}, {&want_res, sizeof(want_res)}};
    const NSUInteger tg_width = std::min<NSUInteger>(256, ((d + 31) / 32) * 32);
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize tg = MTLSizeMake(std::max<NSUInteger>(tg_width, 32), 1, 1);
    return RunPass(impl, "mps_skip_simplified_layernorm_f32", bufs, consts, grid, tg,
                   /*by_threadgroups=*/true, error);
  }
}

bool MetalContext::SoftmaxF32(const float* x, float* y, size_t rows, size_t d, std::string& error) {
  @autoreleasepool {
    if (rows == 0 || d == 0) {
      return true;
    }
    Impl& impl = *impl_;
    std::vector<ResolvedBuffer> bufs = {
        ResolveMC(impl, x, rows * d * sizeof(float), false),
        ResolveMC(impl, y, rows * d * sizeof(float), true)};
    for (const ResolvedBuffer& rb : bufs) {
      if (rb.buffer == nil) { error = "SoftmaxF32 buffer alloc failed"; return false; }
    }
    const uint32_t D = static_cast<uint32_t>(d);
    std::vector<BytesArg> consts = {{&D, sizeof(D)}};
    const NSUInteger tg_width = std::min<NSUInteger>(256, ((d + 31) / 32) * 32);
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize tg = MTLSizeMake(std::max<NSUInteger>(tg_width, 32), 1, 1);
    return RunPass(impl, "mps_softmax_f32", bufs, consts, grid, tg, /*by_threadgroups=*/true, error);
  }
}

}  // namespace ort_mps
