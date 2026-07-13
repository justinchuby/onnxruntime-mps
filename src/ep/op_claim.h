// Copyright (c) 2026. Licensed under the MIT License.
//
// Shared claim-time helpers used by op handler modules' claim predicates. Each translate handler in
// the op registry is registered next to a ClaimPredicate (op_registry.h) that answers ONE question:
// can the MLX backend translate THIS specific node — right domain/op (already matched by the
// registry key), right dtypes, shapes, attributes, and input/output form? These helpers factor the
// common dtype / broadcast / attribute checks so each per-op predicate stays a handful of lines.
//
// The predicates live beside their translate handlers (ops/*.cc); this header is the reusable
// toolbox they share, replacing the old per-family *Claimable funcs that lived in ep.cc.

#pragma once

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "plugin_ep_utils.h"  // IsFloat32Tensor, ElementwiseOrSuffixBroadcast

namespace ort_mps_mlx {

// Element type (and, optionally, shape) of a value info. Returns false if the value is not a tensor.
inline bool TensorInfo(Ort::ConstValueInfo value, ONNXTensorElementDataType& type,
                       std::vector<int64_t>* shape = nullptr) {
  auto type_info = value.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  type = tensor_info.GetElementType();
  if (shape != nullptr) {
    *shape = tensor_info.GetShape();
  }
  return true;
}

// fp32/fp16 only (the historical "float" predicate used by MatMulNBits scales / GBQ scales).
inline bool IsFloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

// Float dtypes the dtype-generic MLX paths (elementwise, activation, softmax, normalization, cast)
// handle: fp32, fp16 AND bf16. MLX carries the resolved dtype through these ops with no per-dtype
// code, so claiming bf16/fp16 alongside fp32 just widens which nodes the EP takes.
inline bool IsMlxFloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

// Read a scalar INT attribute, falling back to `default_value` when absent or of another type.
inline int64_t IntAttribute(Ort::ConstNode node, const char* name, int64_t default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  // ORT may hand back a phantom attribute (type UNDEFINED) for names absent on the node; only
  // trust a genuine INT attribute, otherwise fall back to the caller's default.
  if (attr.GetType() != ORT_OP_ATTR_INT) {
    return default_value;
  }
  int64_t value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

// Read a scalar FLOAT attribute, falling back to `default_value` when absent or of another type.
inline float FloatAttribute(Ort::ConstNode node, const char* name, float default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  if (attr.GetType() != ORT_OP_ATTR_FLOAT) {
    return default_value;
  }
  float value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

// Strict elementwise-or-trailing-suffix broadcast (rejects scalar operands): the fp32 residual/bias
// add form. Wraps the shared ElementwiseOrSuffixBroadcast dim comparison.
inline bool SuffixBroadcast(Ort::ConstValueInfo a, Ort::ConstValueInfo b) {
  bool ok = false;
  ::ElementwiseOrSuffixBroadcast(a, b, ok);
  return ok;
}

// Lenient variant that also accepts a genuine scalar operand (empty shape): used by the fp16/bf16
// and integer elementwise binary forms.
inline bool ScalarOrSuffixBroadcast(Ort::ConstValueInfo a, Ort::ConstValueInfo b) {
  ONNXTensorElementDataType ta, tb;
  std::vector<int64_t> da, db;
  if (!TensorInfo(a, ta, &da) || !TensorInfo(b, tb, &db)) {
    return false;
  }
  if (da.empty() || db.empty()) {
    return true;
  }
  return SuffixBroadcast(a, b);
}

}  // namespace ort_mps_mlx
