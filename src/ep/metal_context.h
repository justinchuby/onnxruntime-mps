// Copyright (c) 2026. Licensed under the MIT License.
//
// Thin C++ facade over the Metal device used by the EP's unified-memory allocator. All
// Objective-C / Metal types are hidden behind a PIMPL so the rest of the EP stays portable
// C++ (see docs/DESIGN.md §2.7). Implemented in metal_context.mm.
//
// In the MLX-native EP the compute is owned entirely by MLX (see mlx_backend.cc); this context
// no longer compiles or dispatches any Metal kernels. Its sole remaining job is the
// shared-storage MTLBuffer pool that backs the OrtAllocator for ORT device I/O and KV cache
// tensors (unified memory, so the buffers are CPU-addressable and directly usable by MLX and
// ORT's memcpy data transfer).

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace ort_mlx {

// Owns id<MTLDevice> and the address -> MTLBuffer map for the shared unified-memory pool.
// One instance per EP factory.
class MetalContext {
 public:
  // Creates the context using the system default Metal device. Returns nullptr and sets `error`
  // on failure.
  static std::unique_ptr<MetalContext> Create(std::string& error);

  ~MetalContext();

  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;

  // Human-readable Metal device name (e.g. "Apple M1 Max").
  const std::string& DeviceName() const { return device_name_; }

  // ---- Device allocator (shared unified-memory MTLBuffer pool) ----
  // Allocates `bytes` of shared-storage device memory and returns a CPU-addressable pointer
  // (MTLBuffer.contents). Returns nullptr on failure. Thread-safe.
  void* Alloc(size_t bytes);
  // Frees a pointer previously returned by Alloc. Safe to call with nullptr. Thread-safe.
  void Free(void* ptr);

 private:
  MetalContext();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string device_name_;
};

}  // namespace ort_mlx
