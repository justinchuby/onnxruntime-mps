// Copyright (c) 2026. Licensed under the MIT License.

#include "ep_factory.h"

#include <cstdlib>
#include <cstring>

#include "allocator.h"
#include "ep.h"
#include "onnxruntime_mps/version.h"

MetalEpFactory::MetalEpFactory(const char* registration_name, ApiPtrs apis,
                               const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      ep_name_{registration_name && *registration_name ? registration_name : ORT_MPS_EP_NAME},
      vendor_{ORT_MPS_EP_VENDOR},
      vendor_id_{ORT_MPS_EP_VENDOR_ID},
      ep_version_{ORT_MPS_EP_VERSION},
      default_logger_{default_logger} {
  ort_version_supported = ORT_API_VERSION;

  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Bring up Metal. If this fails the factory is still returned so the library loads, but the
  // EP will report no supported devices (ORT then runs everything on CPU).
  std::string err;
  metal_ = ort_mps::MetalContext::Create(err);
  {
    const OrtApi& ort_api_ = ort_api;
    const OrtLogger* logger_ = &default_logger_;
    if (metal_) {
      MPS_LOG(INFO, "MLXExecutionProvider: initialized Metal device '" << metal_->DeviceName() << "'");
    } else {
      MPS_LOG(WARNING, "MLXExecutionProvider: Metal init failed (" << err << "); EP will offer no devices");
    }
  }

  // Advertise the EP's memory as a GPU device. On Apple unified memory these are shared-storage
  // MTLBuffers whose contents are CPU-addressable, so the data transfer is a memcpy.
  default_memory_info_ = Ort::MemoryInfo{"MLXExecutionProvider_Buffer",
                                         OrtMemoryInfoDeviceType_GPU,
                                         ORT_MPS_EP_VENDOR_ID, /*device_id*/ 0,
                                         OrtDeviceMemoryType_DEFAULT,
                                         /*alignment*/ 0,
                                         OrtAllocatorType::OrtDeviceAllocator};

  readonly_memory_info_ = Ort::MemoryInfo{"MLXExecutionProvider_Buffer_readonly",
                                          OrtMemoryInfoDeviceType_GPU,
                                          ORT_MPS_EP_VENDOR_ID, /*device_id*/ 0,
                                          OrtDeviceMemoryType_DEFAULT,
                                          /*alignment*/ 0,
                                          OrtAllocatorType::OrtReadOnlyAllocator};

  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_ = std::make_unique<MetalDataTransfer>(apis, device);
}

/*static*/
const char* ORT_API_CALL MetalEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MetalEpFactory*>(this_ptr)->ep_name_.c_str();
}
/*static*/
const char* ORT_API_CALL MetalEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MetalEpFactory*>(this_ptr)->vendor_.c_str();
}
/*static*/
uint32_t ORT_API_CALL MetalEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MetalEpFactory*>(this_ptr)->vendor_id_;
}
/*static*/
const char* ORT_API_CALL MetalEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const MetalEpFactory*>(this_ptr)->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL MetalEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                const OrtHardwareDevice* const* devices,
                                                                size_t num_devices,
                                                                OrtEpDevice** ep_devices,
                                                                size_t max_ep_devices,
                                                                size_t* num_ep_devices) noexcept {
  auto* factory = static_cast<MetalEpFactory*>(this_ptr);
  const OrtApi& ort_api_ = factory->ort_api;  // for MPS_LOG
  const OrtLogger* logger_ = &factory->default_logger_;
  *num_ep_devices = 0;

  if (!factory->metal_) {
    return nullptr;  // Metal unavailable: offer no devices, ORT runs on CPU.
  }

  // ORT enumerates the machine's OrtHardwareDevice instances and passes them in. We do NOT
  // synthesize one via CreateHardwareDevice (open question O3): we bind to an ORT-enumerated
  // device — the GPU if ORT surfaces the Apple GPU, otherwise the CPU device (guaranteeing the
  // EP is always selectable). Either way we advertise GPU memory via the OrtMemoryInfo above.
  const OrtHardwareDevice* gpu = nullptr;
  const OrtHardwareDevice* cpu = nullptr;
  for (size_t i = 0; i < num_devices; ++i) {
    const OrtHardwareDevice* dev = devices[i];
    OrtHardwareDeviceType type = factory->ort_api.HardwareDevice_Type(dev);
    const char* vendor = factory->ort_api.HardwareDevice_Vendor(dev);
    uint32_t vendor_id = factory->ort_api.HardwareDevice_VendorId(dev);
    MPS_LOG(INFO, "MLXExecutionProvider GetSupportedDevices: hw device " << i << " type="
                  << static_cast<int>(type) << " vendor='" << (vendor ? vendor : "?")
                  << "' vendor_id=0x" << std::hex << vendor_id << std::dec);
    if (type == OrtHardwareDeviceType_GPU && gpu == nullptr) {
      gpu = dev;
    } else if (type == OrtHardwareDeviceType_CPU && cpu == nullptr) {
      cpu = dev;
    }
  }

  const OrtHardwareDevice* selected = gpu ? gpu : cpu;
  if (selected == nullptr || max_ep_devices < 1) {
    MPS_LOG(WARNING, "MLXExecutionProvider: no suitable hardware device enumerated by ORT");
    return nullptr;
  }
  MPS_LOG(INFO, "MLXExecutionProvider: binding to " << (gpu ? "GPU" : "CPU")
                << " hardware device; advertising unified-memory GPU allocator");

  OrtEpDevice* ep_device = nullptr;
  RETURN_IF_ERROR(factory->ep_api.CreateEpDevice(factory, selected, /*ep_metadata*/ nullptr,
                                                 /*ep_options*/ nullptr, &ep_device));
  RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_));

  ep_devices[0] = ep_device;
  *num_ep_devices = 1;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MetalEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                     const OrtHardwareDevice* const* /*devices*/,
                                                     const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                     size_t num_devices,
                                                     const OrtSessionOptions* /*session_options*/,
                                                     const OrtLogger* logger, OrtEp** ep) noexcept {
  auto* factory = static_cast<MetalEpFactory*>(this_ptr);
  *ep = nullptr;
  if (num_devices != 1) {
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "MLXExecutionProvider expects to be selected for exactly one device");
  }

  // Partitioning policy. Default: claim all implemented ops. Set ONNX_GENAI_METAL_EP_CLAIM=none to
  // fall everything back to CPU (pure-fallback proof).
  MetalEp::Config config;
  if (const char* claim = std::getenv("ONNX_GENAI_METAL_EP_CLAIM")) {
    config.claim_enabled = std::strcmp(claim, "none") != 0;
  }

  auto metal_ep = std::make_unique<MetalEp>(*factory, factory->ep_name_, config, factory->metal_.get(),
                                            *logger);
  *ep = metal_ep.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL MetalEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<MetalEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL MetalEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                            const OrtMemoryInfo* memory_info,
                                                            const OrtKeyValuePairs* /*allocator_options*/,
                                                            OrtAllocator** allocator) noexcept {
  auto* factory = static_cast<MetalEpFactory*>(this_ptr);
  *allocator = new MetalAllocator(memory_info, factory->metal_.get());
  return nullptr;
}

/*static*/
void ORT_API_CALL MetalEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                       OrtAllocator* allocator) noexcept {
  delete static_cast<MetalAllocator*>(allocator);
}

/*static*/
OrtStatus* ORT_API_CALL MetalEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                               OrtDataTransferImpl** data_transfer) noexcept {
  auto* factory = static_cast<MetalEpFactory*>(this_ptr);
  *data_transfer = factory->data_transfer_.get();
  return nullptr;
}

/*static*/
bool ORT_API_CALL MetalEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;  // Phase 1 is not stream-aware (see DESIGN.md 2.3).
}

/*static*/
OrtStatus* ORT_API_CALL MetalEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* /*this_ptr*/,
                                                                      const OrtMemoryDevice* /*memory_device*/,
                                                                      const OrtKeyValuePairs* /*stream_options*/,
                                                                      OrtSyncStreamImpl** stream) noexcept {
  *stream = nullptr;
  return nullptr;
}
