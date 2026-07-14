// Copyright (c) 2026. Licensed under the MIT License.
//
// Version information for the MLX-native ONNX Runtime execution provider plugin.
//
// NOTE: ORT_MLX_EP_NAME is the EP's registered name ("MLXExecutionProvider"). onnx-genai registers
// and binds the device by this exact name (crates/onnx-genai-ort/src/session.rs REGISTRATION_NAME);
// the two must stay in sync. The vendor string carries the repo name (onnxruntime-mlx).

#pragma once

#define ORT_MLX_EP_NAME "MLXExecutionProvider"
#define ORT_MLX_EP_VENDOR "onnxruntime-mlx"

// Apple's PCI-SIG vendor id (0x106B). Used as the OrtEpFactory vendor id.
#define ORT_MLX_EP_VENDOR_ID 0x106B

#define ORT_MLX_EP_VERSION_MAJOR 0
#define ORT_MLX_EP_VERSION_MINOR 1
#define ORT_MLX_EP_VERSION_PATCH 0
#define ORT_MLX_EP_VERSION "0.1.0"
