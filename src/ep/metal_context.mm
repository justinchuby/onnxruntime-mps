// Copyright (c) 2026. Licensed under the MIT License.
//
// MetalContext implementation: a shared unified-memory MTLBuffer pool. In the MLX-native EP,
// MLX owns all compute; this file only allocates the shared-storage buffers that back the
// OrtAllocator (device I/O + KV cache). The MRR (no-ARC) [release] in Free / the destructor is
// the leak fix — assigning nil does NOT release under MRR and previously leaked every freed GPU
// buffer across sessions.

#import <Metal/Metal.h>

#include "metal_context.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace ort_mlx {

struct MetalContext::Impl {
  id<MTLDevice> device = nil;
  std::mutex alloc_mutex;
  std::unordered_map<void*, id<MTLBuffer>> buffers;
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() {
  if (!impl_) {
    return;
  }
  // MRR (no ARC): every buffer in this map holds a +1 from Alloc; assigning nil does NOT release
  // under MRR, so send -release explicitly to avoid leaking GPU memory across sessions.
  for (auto& entry : impl_->buffers) {
    [entry.second release];
  }
  impl_->buffers.clear();
  [impl_->device release];
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
    // MRR: release the +1 taken by Alloc's newBufferWithLength. Assigning nil does NOT release
    // under MRR -- that was leaking every freed GPU buffer.
    [it->second release];
    impl_->buffers.erase(it);
  }
}

}  // namespace ort_mlx
