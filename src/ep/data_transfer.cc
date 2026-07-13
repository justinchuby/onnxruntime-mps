// Copyright (c) 2026. Licensed under the MIT License.

#include "data_transfer.h"

#include <cstring>

/*static*/
bool ORT_API_CALL MetalDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                                 const OrtMemoryDevice* src,
                                                 const OrtMemoryDevice* dst) noexcept {
  const auto& self = *static_cast<const MetalDataTransfer*>(this_ptr);
  const bool src_ours = self.ep_api.MemoryDevice_AreEqual(src, self.device_memory_);
  const bool dst_ours = self.ep_api.MemoryDevice_AreEqual(dst, self.device_memory_);

  if (src_ours && dst_ours) {
    return true;  // device <-> device
  }

  // Unified memory: we can also copy to/from CPU or host-accessible memory.
  const OrtMemoryInfoDeviceType src_type = self.ep_api.MemoryDevice_GetDeviceType(src);
  const OrtMemoryInfoDeviceType dst_type = self.ep_api.MemoryDevice_GetDeviceType(dst);
  const OrtDeviceMemoryType src_mem = self.ep_api.MemoryDevice_GetMemoryType(src);
  const OrtDeviceMemoryType dst_mem = self.ep_api.MemoryDevice_GetMemoryType(dst);

  if (src_ours) {
    return dst_type == OrtMemoryInfoDeviceType_CPU || dst_mem == OrtDeviceMemoryType_HOST_ACCESSIBLE;
  }
  if (dst_ours) {
    return src_type == OrtMemoryInfoDeviceType_CPU || src_mem == OrtDeviceMemoryType_HOST_ACCESSIBLE;
  }
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL MetalDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                           const OrtValue** src_tensors,
                                                           OrtValue** dst_tensors,
                                                           OrtSyncStream** /*streams*/,
                                                           size_t num_tensors) noexcept {
  auto& self = *static_cast<MetalDataTransfer*>(this_ptr);

  for (size_t i = 0; i < num_tensors; ++i) {
    const void* src_data = nullptr;
    void* dst_data = nullptr;
    size_t bytes = 0;
    RETURN_IF_ERROR(self.ort_api.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(self.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
    RETURN_IF_ERROR(self.ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));

    // Both endpoints are shared-storage / CPU-addressable on unified memory: memcpy suffices.
    std::memcpy(dst_data, src_data, bytes);
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL MetalDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
  // ORT owns each instance handed out by CreateDataTransferImpl; free it here (mirrors
  // ReleaseAllocatorImpl). OrtDataTransferImpl is MetalDataTransfer's first base, so the downcast is
  // offset-0 and safe.
  delete static_cast<MetalDataTransfer*>(this_ptr);
}
