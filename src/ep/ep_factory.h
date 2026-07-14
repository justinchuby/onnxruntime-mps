// Copyright (c) 2026. Licensed under the MIT License.
//
// MetalEpFactory: the OrtEpFactory that reports the Apple GPU as an OrtEpDevice, owns the
// shared MetalContext (MTLDevice/queue/library), and creates a MetalEp per session.
// See docs/DESIGN.md 2.3.

#pragma once

#include <memory>
#include <string>

#include "data_transfer.h"
#include "metal_context.h"
#include "plugin_ep_utils.h"

class MetalEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  MetalEpFactory(const char* registration_name, ApiPtrs apis, const OrtLogger& default_logger);

  const OrtMemoryInfo* GetDefaultMemoryInfo() const { return default_memory_info_; }
  ort_mlx::MetalContext* Metal() const { return metal_.get(); }
  const OrtLogger& DefaultLogger() const { return default_logger_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* devices,
                                              const OrtKeyValuePairs* const* ep_metadata,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger, OrtEp** ep) noexcept;
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* allocator_options,
                                                     OrtAllocator** allocator) noexcept;
  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* stream_options,
                                                               OrtSyncStreamImpl** stream) noexcept;

  const std::string ep_name_;
  const std::string vendor_;
  const uint32_t vendor_id_;
  const std::string ep_version_;
  const OrtLogger& default_logger_;

  std::shared_ptr<ort_mlx::MetalContext> metal_;

  Ort::MemoryInfo default_memory_info_{nullptr};
  Ort::MemoryInfo readonly_memory_info_{nullptr};

  // Memory device the DataTransfer copies for. DataTransfer instances are owned by ORT (created per
  // request in CreateDataTransferImpl, freed in MetalDataTransfer::ReleaseImpl) — NOT by the factory
  // — so their lifetime never outlives a freed factory (avoids a teardown use-after-free).
  const OrtMemoryDevice* dt_device_memory_ = nullptr;
};
