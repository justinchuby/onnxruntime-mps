// Copyright (c) 2026. Licensed under the MIT License.
//
// The two exported C symbols ORT resolves via dlsym when a session calls
// RegisterExecutionProviderLibrary (see docs/DESIGN.md 2.2).

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_mlx/metal_ep.h"

#include <memory>

#include "ep_factory.h"

extern "C" {

ORT_MLX_EXPORT OrtStatus* CreateEpFactories(const char* registration_name,
                                            const OrtApiBase* ort_api_base,
                                            const OrtLogger* default_logger,
                                            OrtEpFactory** factories,
                                            size_t max_factories,
                                            size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  if (ort_api == nullptr) {
    // The host ORT is older than the version this library was compiled against.
    return ort_api_base->GetApi(1)->CreateStatus(
        ORT_INVALID_ARGUMENT, "MetalEP requires an ONNX Runtime built with ORT_API_VERSION >= 27");
  }
  const OrtEpApi* ep_api = ort_api->GetEpApi();

  // Initialize the header-only C++ API for this library.
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "MetalEP needs room for at least one OrtEpFactory");
  }

  auto factory = std::make_unique<MetalEpFactory>(registration_name, ApiPtrs{*ort_api, *ep_api},
                                                  *default_logger);
  factories[0] = factory.release();
  *num_factories = 1;
  return nullptr;
}

ORT_MLX_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<MetalEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
