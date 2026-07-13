// Copyright (c) 2026. Licensed under the MIT License.
//
// A minimal OrtAllocator backed by the Metal shared-storage buffer pool. Every allocation
// is a unified-memory MTLBuffer, so ORT's tensors on the EP device are directly readable by
// both CPU (memcpy in the data transfer) and GPU kernels. Phase 2 can replace this with an
// arena on top of the same pool (see ORT's ep_arena.h reference).

#pragma once

#include "metal_context.h"
#include "plugin_ep_utils.h"

struct MetalAllocator : OrtAllocator {
  MetalAllocator(const OrtMemoryInfo* memory_info, std::shared_ptr<ort_mps::MetalContext> metal)
      : memory_info_(memory_info), metal_(std::move(metal)) {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;  // no arena distinction in Phase 1
  }

  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) {
    return static_cast<MetalAllocator*>(this_ptr)->metal_->Alloc(size);
  }
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) {
    static_cast<MetalAllocator*>(this_ptr)->metal_->Free(p);
  }
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) {
    return static_cast<const MetalAllocator*>(this_ptr)->memory_info_;
  }

 private:
  const OrtMemoryInfo* memory_info_;
  // Shared ownership: the allocator keeps the MetalContext alive as long as it (and the tensors it
  // frees through it) exist, even if the factory is released first — prevents a teardown
  // use-after-free in FreeImpl -> metal_->Free().
  std::shared_ptr<ort_mps::MetalContext> metal_;
};
