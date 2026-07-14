// Copyright (c) 2026. Licensed under the MIT License.
//
// Public C entry points for the ONNX Runtime Metal/MPS plugin execution provider.
//
// A stock prebuilt ONNX Runtime (>= 1.22, targeting ORT_API_VERSION 27) loads this
// library at runtime via OrtApi::RegisterExecutionProviderLibrary and resolves the two
// exported symbols below with dlsym. They therefore MUST have default visibility.
//
// See docs/DESIGN.md sections 2.1 and 2.2 for the ABI contract.

#pragma once

#include "onnxruntime_c_api.h"
// onnxruntime_ep_c_api.h (OrtEpFactory etc.) is included transitively by onnxruntime_c_api.h;
// it has no include guard so it must not be included a second time.

#ifdef __APPLE__
#define ORT_MLX_EXPORT __attribute__((visibility("default")))
#else
#define ORT_MLX_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Creates the OrtEpFactory instances provided by this library.
// Matches CreateEpApiFactoriesFn from onnxruntime_ep_c_api.h (since ORT 1.22).
ORT_MLX_EXPORT OrtStatus* CreateEpFactories(const char* registration_name,
                                            const OrtApiBase* ort_api_base,
                                            const OrtLogger* default_logger,
                                            OrtEpFactory** factories,
                                            size_t max_factories,
                                            size_t* num_factories);

// Releases an OrtEpFactory previously returned by CreateEpFactories.
// Matches ReleaseEpApiFactoryFn from onnxruntime_ep_c_api.h (since ORT 1.22).
ORT_MLX_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);

#ifdef __cplusplus
}  // extern "C"
#endif
