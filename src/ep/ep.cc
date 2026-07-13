// Copyright (c) 2026. Licensed under the MIT License.

#include "ep.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ep_factory.h"

namespace {

bool TensorInfo(Ort::ConstValueInfo value, ONNXTensorElementDataType& type,
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

bool IsFloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

bool IsFixedSizeTensorType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return true;
    default:
      return false;
  }
}

size_t ElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return 1;
    default:
      return 0;
  }
}

bool ToScalarType(ONNXTensorElementDataType type, ort_mps::ScalarType& result) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      result = ort_mps::ScalarType::Float16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      result = ort_mps::ScalarType::Float32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      result = ort_mps::ScalarType::Int32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      result = ort_mps::ScalarType::Int64;
      return true;
    default:
      return false;
  }
}

int64_t IntAttribute(Ort::ConstNode node, const char* name, int64_t default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  int64_t value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

float FloatAttribute(Ort::ConstNode node, const char* name, float default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  float value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::string StringAttribute(Ort::ConstNode node, const char* name,
                            const std::string& default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  std::string value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::vector<int64_t> IntsAttribute(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return {};
  }
  std::vector<int64_t> values;
  status = attr.GetValueArray(values);
  return status.IsOK() ? values : std::vector<int64_t>{};
}

bool ScalarOrSuffixBroadcast(Ort::ConstValueInfo a, Ort::ConstValueInfo b) {
  ONNXTensorElementDataType ta, tb;
  std::vector<int64_t> da, db;
  if (!TensorInfo(a, ta, &da) || !TensorInfo(b, tb, &db)) {
    return false;
  }
  if (da.empty() || db.empty()) {
    return true;
  }
  bool result = false;
  ElementwiseOrSuffixBroadcast(a, b, result);
  return result;
}

std::vector<int64_t> BinaryOutputShape(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  return a.size() >= b.size() ? a : b;
}

bool ProductFitsU32(const std::vector<int64_t>& shape) {
  uint64_t product = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      continue;  // symbolic dimensions are validated at runtime
    }
    product *= static_cast<uint64_t>(dim);
    if (product > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool CocoClaimable(Ort::ConstNode node) {
  const std::string op = node.GetOperatorType();
  const std::string domain = node.GetDomain();
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1) {
    return false;
  }

  ONNXTensorElementDataType output_type;
  if (!TensorInfo(outputs[0], output_type)) {
    return false;
  }

  if (domain.empty() &&
      (op == "Add" || op == "Mul" || op == "Sub" || op == "Div")) {
    if (inputs.size() != 2) return false;
    ONNXTensorElementDataType a, b;
    if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) ||
        a != b || b != output_type || !ScalarOrSuffixBroadcast(inputs[0], inputs[1])) {
      return false;
    }
    if (op == "Add") {
      return a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    if (op == "Sub" && a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return true;
    }
    return IsFloatType(a);
  }

  if ((domain.empty() || domain == "com.microsoft") &&
      (op == "Sigmoid" || op == "SiLU" || op == "Swish" || op == "Gelu")) {
    if (inputs.size() != 1) return false;
    ONNXTensorElementDataType input_type;
    return TensorInfo(inputs[0], input_type) && input_type == output_type &&
           IsFloatType(input_type);
  }

  if (domain.empty() && op == "Cast" && inputs.size() == 1) {
    ONNXTensorElementDataType input_type;
    if (!TensorInfo(inputs[0], input_type)) return false;
    return (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ||
           (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
           (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
            output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  }

  if ((domain.empty() || domain == "com.microsoft") && op == "RotaryEmbedding") {
    if (inputs.size() != 3 && inputs.size() != 4) return false;
    ONNXTensorElementDataType input_type, cos_type, sin_type;
    if (!TensorInfo(inputs[0], input_type) || !TensorInfo(inputs[1], cos_type) ||
        !TensorInfo(inputs[2], sin_type) || input_type != output_type ||
        input_type != cos_type || input_type != sin_type || !IsFloatType(input_type)) {
      return false;
    }
    if (inputs.size() == 4) {
      ONNXTensorElementDataType position_type;
      if (!TensorInfo(inputs[3], position_type) ||
          position_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return false;
      }
    }
    return true;
  }

  if (domain == "com.microsoft" && op == "GatherBlockQuantized") {
    if (inputs.size() != 3 && inputs.size() != 4) return false;
    ONNXTensorElementDataType data_type, indices_type, scales_type;
    if (!TensorInfo(inputs[0], data_type) || !TensorInfo(inputs[1], indices_type) ||
        !TensorInfo(inputs[2], scales_type) || data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
        (indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
         indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
        scales_type != output_type || !IsFloatType(scales_type)) {
      return false;
    }
    if (inputs.size() == 4) {
      ONNXTensorElementDataType zero_point_type;
      if (!TensorInfo(inputs[3], zero_point_type) ||
          zero_point_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        return false;
      }
    }
    return IntAttribute(node, "bits", 4) == 4 &&
           IntAttribute(node, "gather_axis", 0) == 0 &&
           IntAttribute(node, "quantize_axis", 1) == 1 &&
           IntAttribute(node, "block_size", 128) >= 16;
  }

  if (domain.empty() && (op == "Reshape" || op == "Transpose" || op == "Concat")) {
    if (inputs.empty() || !IsFixedSizeTensorType(output_type)) return false;
    ONNXTensorElementDataType first_type;
    if (!TensorInfo(inputs[0], first_type) || first_type != output_type) return false;
    if (op == "Reshape") {
      if (inputs.size() != 2) return false;
      ONNXTensorElementDataType shape_type;
      return TensorInfo(inputs[1], shape_type) &&
             shape_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    if (op == "Transpose") {
      std::vector<int64_t> shape;
      TensorInfo(inputs[0], first_type, &shape);
      return inputs.size() == 1 && shape.size() <= 8 && ProductFitsU32(shape);
    }
    for (const auto& input : inputs) {
      ONNXTensorElementDataType type;
      if (!TensorInfo(input, type) || type != first_type) return false;
    }
    return true;
  }

  return false;
}

// True if `node` is a Mariette core-compute op we have a Metal kernel for.
bool MarietteClaimable(Ort::ConstNode node) {
  const std::string op = node.GetOperatorType();
  const std::string domain = node.GetDomain();
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) {
    return false;
  }
  ONNXTensorElementDataType out_type;
  if (!TensorInfo(outputs[0], out_type) || out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;  // fp32-only path for now (matches the cpu-recipe graph)
  }

  // MatMulNBits: A[f32], B[uint8 packed int4], scales[f32] (+ optional bias), bits=4, block=32.
  if (domain == "com.microsoft" && op == "MatMulNBits") {
    if (inputs.size() != 3 && inputs.size() != 4) return false;
    ONNXTensorElementDataType a, b, s;
    if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(inputs[2], s)) {
      return false;
    }
    if (a != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
        s != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return false;
    }
    if (inputs.size() == 4) {
      ONNXTensorElementDataType bias;
      if (!TensorInfo(inputs[3], bias) || bias != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    }
    return IntAttribute(node, "bits", 4) == 4 && IntAttribute(node, "block_size", 32) == 32;
  }

  // RMSNormalization (ai.onnx): X[f32], scale[f32], axis == -1.
  if (domain.empty() && op == "RMSNormalization") {
    if (inputs.size() != 2) return false;
    ONNXTensorElementDataType x, g;
    if (!TensorInfo(inputs[0], x) || !TensorInfo(inputs[1], g)) return false;
    if (x != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || g != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return false;
    }
    const int64_t axis = IntAttribute(node, "axis", -1);
    return axis == -1;
  }

  // SkipSimplifiedLayerNormalization (com.microsoft): input, skip, gamma (all f32).
  if (domain == "com.microsoft" && op == "SkipSimplifiedLayerNormalization") {
    if (inputs.size() != 3) return false;  // no optional bias/beta in our graph
    ONNXTensorElementDataType i0, i1, i2;
    if (!TensorInfo(inputs[0], i0) || !TensorInfo(inputs[1], i1) || !TensorInfo(inputs[2], i2)) {
      return false;
    }
    return i0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && i1 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
           i2 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  // Softmax (ai.onnx): single f32 input, softmax over the last axis.
  if (domain.empty() && op == "Softmax") {
    if (inputs.size() != 1) return false;
    ONNXTensorElementDataType x;
    std::vector<int64_t> shape;
    if (!TensorInfo(inputs[0], x, &shape) || x != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t axis = IntAttribute(node, "axis", -1);
    return rank > 0 && (axis == -1 || axis == rank - 1);
  }

  return false;
}

// The standard ai.onnx float32 Add executed by AddKernel (bias add / residual): equal shapes or
// trailing-suffix broadcast. Float16 Add is claimed via CocoClaimable but still runs on AddKernel.
bool AddClaimable(Ort::ConstNode node) {
  if (node.GetOperatorType() != "Add" || !node.GetDomain().empty()) return false;
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  bool f0 = false, f1 = false, fo = false;
  IsFloat32Tensor(inputs[0], f0);
  IsFloat32Tensor(inputs[1], f1);
  IsFloat32Tensor(outputs[0], fo);
  if (!f0 || !f1 || !fo) return false;
  bool ok = false;
  ElementwiseOrSuffixBroadcast(inputs[0], inputs[1], ok);
  return ok;
}

// Unified predicate: is `node` supported by any of the EP's kernel families (respecting config)?
bool NodeClaimable(Ort::ConstNode node, const MetalEp::Config& config) {
  if (config.claim_add && AddClaimable(node)) return true;
  if (config.claim_mariette && MarietteClaimable(node)) return true;
  if (config.claim_coco && CocoClaimable(node)) return true;
  return false;
}

// True if `node` is the standard float32/float16 ai.onnx Add routed to AddKernel.
bool IsAddNode(Ort::ConstNode node) {
  return node.GetOperatorType() == "Add" && node.GetDomain().empty();
}

}  // namespace

// ---------------------------------------------------------------------------
// AddKernel
// ---------------------------------------------------------------------------

OrtStatus* AddKernel::ComputeIO(KernelIO& io) {
  try {
    RETURN_IF(io.InputCount() != 2, ort_api_, "MetalEP Add expects 2 inputs");
    RETURN_IF(io.OutputCount() != 1, ort_api_, "MetalEP Add expects 1 output");

    const IOTensor& in0 = io.Input(0);
    const IOTensor& in1 = io.Input(1);
    const ONNXTensorElementDataType type = in0.type;
    RETURN_IF(type != in1.type ||
                  (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                   type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
              ort_api_, "MetalEP Add expects matching float32 or float16 inputs");

    const size_t na = in0.element_count;
    const size_t nb = in1.element_count;
    const std::vector<int64_t> out_shape = BinaryOutputShape(in0.shape, in1.shape);
    const size_t n = std::max(na, nb);
    RETURN_IF(na == 0 || nb == 0, ort_api_, "MetalEP Add received an empty input tensor");
    RETURN_IF((n % na) != 0 || (n % nb) != 0, ort_api_,
              "MetalEP Add operand element counts do not divide the output (unsupported broadcast)");

    IOTensor& out = io.Output(0, out_shape);

    std::string err;
    ort_mps::ScalarType scalar_type =
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? ort_mps::ScalarType::Float32
                                                    : ort_mps::ScalarType::Float16;
    if (!metal_->Binary(ort_mps::BinaryOp::Add, scalar_type, in0.data, na, in1.data, nb,
                        out.mutable_data, n, err)) {
      return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Add kernel failed: " + err).c_str());
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// CocoKernel
// ---------------------------------------------------------------------------

CocoKernel::CocoKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode node)
    : KernelBase(ort_api, metal), op_type_(node.GetOperatorType()) {
  to_type_ = IntAttribute(node, "to", 0);
  axis_ = IntAttribute(node, "axis", 0);
  block_size_ = IntAttribute(node, "block_size", 32);
  gather_axis_ = IntAttribute(node, "gather_axis", 0);
  quantize_axis_ = IntAttribute(node, "quantize_axis", 1);
  bits_ = IntAttribute(node, "bits", 4);
  num_heads_ = IntAttribute(node, "num_heads", 0);
  rotary_embedding_dim_ = IntAttribute(node, "rotary_embedding_dim", 0);
  interleaved_ = IntAttribute(node, "interleaved", 0) != 0;
  gelu_tanh_ = StringAttribute(node, "approximate", "none") == "tanh";
  allowzero_ = IntAttribute(node, "allowzero", 0) != 0;
  permutation_ = IntsAttribute(node, "perm");
}

OrtStatus* CocoKernel::ComputeIO(KernelIO& io) {
  try {
    const size_t input_count = io.InputCount();
    RETURN_IF(io.OutputCount() < 1, ort_api_, "MetalEP Coco kernel expects 1 output");
    std::string error;

    if (op_type_ == "Mul" || op_type_ == "Sub" || op_type_ == "Div") {
      RETURN_IF(input_count != 2, ort_api_, "MetalEP binary kernel expects 2 inputs");
      const IOTensor& left = io.Input(0);
      const IOTensor& right = io.Input(1);
      RETURN_IF(left.type != right.type, ort_api_,
                "MetalEP binary inputs must have the same type");
      const size_t left_count = left.element_count;
      const size_t right_count = right.element_count;
      const size_t output_count = std::max(left_count, right_count);
      RETURN_IF(left_count == 0 || right_count == 0 ||
                    output_count % left_count != 0 || output_count % right_count != 0,
                ort_api_, "MetalEP binary broadcast is unsupported");
      std::vector<int64_t> output_shape = BinaryOutputShape(left.shape, right.shape);
      IOTensor& output = io.Output(0, output_shape);
      ort_mps::ScalarType type;
      RETURN_IF(!ToScalarType(left.type, type), ort_api_,
                "MetalEP binary input type is unsupported");
      ort_mps::BinaryOp op = op_type_ == "Mul" ? ort_mps::BinaryOp::Mul
                             : op_type_ == "Sub" ? ort_mps::BinaryOp::Sub
                                                 : ort_mps::BinaryOp::Div;
      if (!metal_->Binary(op, type, left.data, left_count, right.data, right_count,
                          output.mutable_data, output_count, error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP " + op_type_ + " failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Sigmoid" || op_type_ == "SiLU" || op_type_ == "Swish" ||
        op_type_ == "Gelu") {
      RETURN_IF(input_count != 1, ort_api_, "MetalEP unary kernel expects 1 input");
      const IOTensor& input = io.Input(0);
      ort_mps::ScalarType type;
      RETURN_IF(!ToScalarType(input.type, type), ort_api_,
                "MetalEP unary input type is unsupported");
      IOTensor& output = io.Output(0, input.shape);
      ort_mps::UnaryOp op = ort_mps::UnaryOp::Sigmoid;
      if (op_type_ == "SiLU" || op_type_ == "Swish") {
        op = ort_mps::UnaryOp::SiLU;
      } else if (op_type_ == "Gelu") {
        op = gelu_tanh_ ? ort_mps::UnaryOp::GeluTanh : ort_mps::UnaryOp::Gelu;
      }
      if (!metal_->Unary(op, type, input.data, output.mutable_data, input.element_count, error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP " + op_type_ + " failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Cast") {
      RETURN_IF(input_count != 1, ort_api_, "MetalEP Cast expects 1 input");
      const IOTensor& input = io.Input(0);
      ort_mps::ScalarType input_type, output_type;
      RETURN_IF(!ToScalarType(input.type, input_type) ||
                    !ToScalarType(static_cast<ONNXTensorElementDataType>(to_type_), output_type),
                ort_api_, "MetalEP Cast type pair is unsupported");
      IOTensor& output = io.Output(0, input.shape);
      if (!metal_->Cast(input_type, output_type, input.data, output.mutable_data,
                        input.element_count, error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Cast failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "RotaryEmbedding") {
      RETURN_IF(input_count != 3 && input_count != 4, ort_api_,
                "MetalEP RotaryEmbedding expects 3 or 4 inputs");
      const IOTensor& input = io.Input(0);
      const IOTensor& cos_cache = io.Input(1);
      const IOTensor& sin_cache = io.Input(2);
      const std::vector<int64_t>& shape = input.shape;
      RETURN_IF(shape.size() != 3 && shape.size() != 4, ort_api_,
                "MetalEP RotaryEmbedding expects rank-3 or rank-4 input");
      const std::vector<int64_t>& cos_shape = cos_cache.shape;
      RETURN_IF(cos_shape.size() != 2 || sin_cache.shape != cos_shape, ort_api_,
                "MetalEP RotaryEmbedding expects matching rank-2 cos/sin caches");

      ort_mps::RotaryEmbeddingParams params;
      if (shape.size() == 3) {
        RETURN_IF(num_heads_ <= 0 || shape[2] % num_heads_ != 0, ort_api_,
                  "MetalEP RotaryEmbedding rank-3 input requires num_heads");
        params.batch_size = static_cast<uint32_t>(shape[0]);
        params.sequence_length = static_cast<uint32_t>(shape[1]);
        params.num_heads = static_cast<uint32_t>(num_heads_);
        params.head_size = static_cast<uint32_t>(shape[2] / num_heads_);
        params.rank3_bsh = true;
      } else {
        params.batch_size = static_cast<uint32_t>(shape[0]);
        params.num_heads = static_cast<uint32_t>(shape[1]);
        params.sequence_length = static_cast<uint32_t>(shape[2]);
        params.head_size = static_cast<uint32_t>(shape[3]);
      }
      params.rotary_embedding_dim =
          static_cast<uint32_t>(rotary_embedding_dim_ > 0 ? rotary_embedding_dim_
                                                          : params.head_size);
      params.cache_stride = static_cast<uint32_t>(cos_shape[1]);
      params.max_sequence_length = static_cast<uint32_t>(cos_shape[0]);
      params.interleaved = interleaved_;
      const int64_t* position_ids =
          input_count == 4 ? io.Input(3).Data<int64_t>() : nullptr;
      ort_mps::ScalarType type;
      RETURN_IF(!ToScalarType(input.type, type), ort_api_,
                "MetalEP RotaryEmbedding type is unsupported");
      IOTensor& output = io.Output(0, shape);
      if (!metal_->RotaryEmbedding(type, input.data, cos_cache.data, sin_cache.data, position_ids,
                                   output.mutable_data, input.element_count, params, error)) {
        return ort_api_.CreateStatus(
            ORT_EP_FAIL, ("MetalEP RotaryEmbedding failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "GatherBlockQuantized") {
      RETURN_IF(input_count != 3 && input_count != 4, ort_api_,
                "MetalEP GatherBlockQuantized expects 3 or 4 inputs");
      RETURN_IF(bits_ != 4 || gather_axis_ != 0 || quantize_axis_ != 1,
                ort_api_, "MetalEP GatherBlockQuantized only supports q4 axis0/last-axis");
      const IOTensor& data = io.Input(0);
      const IOTensor& indices = io.Input(1);
      const IOTensor& scales = io.Input(2);
      const std::vector<int64_t>& data_shape = data.shape;
      const std::vector<int64_t>& scale_shape = scales.shape;
      RETURN_IF(data_shape.size() != 2 || scale_shape.size() != 2 ||
                    data_shape[0] != scale_shape[0],
                ort_api_, "MetalEP GatherBlockQuantized expects rank-2 data/scales");
      const uint32_t rows = static_cast<uint32_t>(data_shape[0]);
      const uint32_t packed_width = static_cast<uint32_t>(data_shape[1]);
      const uint32_t row_width = packed_width * 2;
      RETURN_IF(scale_shape[1] !=
                    (static_cast<int64_t>(row_width) + block_size_ - 1) / block_size_,
                ort_api_, "MetalEP GatherBlockQuantized scale shape mismatch");
      std::vector<int64_t> output_shape = indices.shape;
      output_shape.push_back(row_width);
      IOTensor& output = io.Output(0, output_shape);
      ort_mps::ScalarType output_type;
      RETURN_IF(!ToScalarType(scales.type, output_type), ort_api_,
                "MetalEP GatherBlockQuantized scale type is unsupported");
      const uint8_t* zero_points = nullptr;
      size_t zero_points_bytes = 0;
      if (input_count == 4) {
        const IOTensor& zp = io.Input(3);
        zero_points = static_cast<const uint8_t*>(zp.data);
        zero_points_bytes = zp.element_count;
      }
      if (!metal_->GatherBlockQuantized(
              static_cast<const uint8_t*>(data.data), data.element_count, indices.data,
              indices.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices.element_count,
              scales.data, output_type, zero_points, zero_points_bytes, output.mutable_data, rows,
              row_width, packed_width, static_cast<uint32_t>(block_size_), error)) {
        return ort_api_.CreateStatus(
            ORT_EP_FAIL, ("MetalEP GatherBlockQuantized failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Reshape") {
      RETURN_IF(input_count != 2, ort_api_, "MetalEP Reshape expects 2 inputs");
      const IOTensor& input = io.Input(0);
      const IOTensor& requested_shape = io.Input(1);
      const int64_t* requested = requested_shape.Data<int64_t>();
      const std::vector<int64_t>& input_shape = input.shape;
      std::vector<int64_t> output_shape(requested_shape.element_count);
      int64_t infer_axis = -1;
      uint64_t known_product = 1;
      for (size_t i = 0; i < output_shape.size(); ++i) {
        int64_t dim = requested[i];
        if (dim == 0 && !allowzero_) {
          RETURN_IF(i >= input_shape.size(), ort_api_, "MetalEP Reshape zero axis is invalid");
          dim = input_shape[i];
        } else if (dim == -1) {
          RETURN_IF(infer_axis >= 0, ort_api_, "MetalEP Reshape has multiple inferred axes");
          infer_axis = static_cast<int64_t>(i);
          output_shape[i] = -1;
          continue;
        }
        RETURN_IF(dim < 0, ort_api_, "MetalEP Reshape dimension is invalid");
        output_shape[i] = dim;
        known_product *= static_cast<uint64_t>(dim);
      }
      const size_t input_elements = input.element_count;
      if (infer_axis >= 0) {
        RETURN_IF(known_product == 0 || input_elements % known_product != 0, ort_api_,
                  "MetalEP Reshape cannot infer output dimension");
        output_shape[static_cast<size_t>(infer_axis)] =
            static_cast<int64_t>(input_elements / known_product);
      } else {
        RETURN_IF(known_product != input_elements, ort_api_,
                  "MetalEP Reshape element count mismatch");
      }
      IOTensor& output = io.Output(0, output_shape);
      const size_t bytes = input_elements * ElementSize(input.type);
      if (!metal_->CopyBytes(input.data, output.mutable_data, bytes, error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Reshape failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Transpose") {
      RETURN_IF(input_count != 1, ort_api_, "MetalEP Transpose expects 1 input");
      const IOTensor& input = io.Input(0);
      const std::vector<int64_t>& input_shape = input.shape;
      if (input_shape.empty()) {
        IOTensor& output = io.Output(0, input_shape);
        const size_t bytes = input.element_count * ElementSize(input.type);
        if (!metal_->CopyBytes(input.data, output.mutable_data, bytes, error)) {
          return ort_api_.CreateStatus(ORT_EP_FAIL,
                                       ("MetalEP Transpose failed: " + error).c_str());
        }
        return nullptr;
      }
      std::vector<int64_t> permutation = permutation_;
      if (permutation.empty()) {
        permutation.resize(input_shape.size());
        std::iota(permutation.rbegin(), permutation.rend(), 0);
      }
      RETURN_IF(permutation.size() != input_shape.size() || input_shape.size() > 8,
                ort_api_, "MetalEP Transpose permutation is invalid");
      std::vector<int64_t> output_shape(input_shape.size());
      std::array<uint32_t, 8> output_dims{};
      std::array<uint32_t, 8> input_strides{};
      std::array<uint32_t, 8> perm32{};
      uint64_t stride = 1;
      for (size_t i = input_shape.size(); i-- > 0;) {
        RETURN_IF(input_shape[i] < 0 ||
                      static_cast<uint64_t>(input_shape[i]) >
                          std::numeric_limits<uint32_t>::max(),
                  ort_api_, "MetalEP Transpose runtime dimension is invalid");
        input_strides[i] = static_cast<uint32_t>(stride);
        stride *= static_cast<uint64_t>(input_shape[i]);
      }
      for (size_t i = 0; i < input_shape.size(); ++i) {
        RETURN_IF(permutation[i] < 0 ||
                      static_cast<size_t>(permutation[i]) >= input_shape.size(),
                  ort_api_, "MetalEP Transpose permutation axis is invalid");
        output_shape[i] = input_shape[static_cast<size_t>(permutation[i])];
        output_dims[i] = static_cast<uint32_t>(output_shape[i]);
        perm32[i] = static_cast<uint32_t>(permutation[i]);
      }
      IOTensor& output = io.Output(0, output_shape);
      if (!metal_->TransposeBytes(input.data, output.mutable_data, input.element_count,
                                  static_cast<uint32_t>(ElementSize(input.type)),
                                  static_cast<uint32_t>(input_shape.size()), output_dims.data(),
                                  input_strides.data(), perm32.data(), error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP Transpose failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Concat") {
      RETURN_IF(input_count == 0, ort_api_, "MetalEP Concat expects inputs");
      std::vector<const IOTensor*> inputs;
      inputs.reserve(input_count);
      for (size_t i = 0; i < input_count; ++i) inputs.push_back(&io.Input(i));
      std::vector<int64_t> output_shape = inputs[0]->shape;
      const int64_t rank = static_cast<int64_t>(output_shape.size());
      int64_t axis = axis_ < 0 ? axis_ + rank : axis_;
      RETURN_IF(axis < 0 || axis >= rank, ort_api_, "MetalEP Concat axis is invalid");
      const ONNXTensorElementDataType first_type = inputs[0]->type;
      output_shape[static_cast<size_t>(axis)] = 0;
      for (size_t i = 0; i < input_count; ++i) {
        const std::vector<int64_t>& shape = inputs[i]->shape;
        RETURN_IF(shape.size() != static_cast<size_t>(rank), ort_api_,
                  "MetalEP Concat ranks must match");
        for (int64_t d = 0; d < rank; ++d) {
          RETURN_IF(d != axis && shape[static_cast<size_t>(d)] !=
                                      output_shape[static_cast<size_t>(d)],
                    ort_api_, "MetalEP Concat non-axis dimensions must match");
        }
        output_shape[static_cast<size_t>(axis)] += shape[static_cast<size_t>(axis)];
      }
      IOTensor& output = io.Output(0, output_shape);
      uint64_t outer = 1, inner = 1;
      for (int64_t d = 0; d < axis; ++d) outer *= output_shape[static_cast<size_t>(d)];
      for (int64_t d = axis + 1; d < rank; ++d) inner *= output_shape[static_cast<size_t>(d)];
      RETURN_IF(outer > std::numeric_limits<uint32_t>::max() ||
                    inner > std::numeric_limits<uint32_t>::max() ||
                    output_shape[static_cast<size_t>(axis)] >
                        std::numeric_limits<uint32_t>::max(),
                ort_api_, "MetalEP Concat dimensions exceed uint32 limits");
      const uint32_t element_size = static_cast<uint32_t>(ElementSize(first_type));
      uint64_t output_elements = 1;
      for (int64_t dim : output_shape) {
        RETURN_IF(dim < 0, ort_api_, "MetalEP Concat runtime dimension is invalid");
        output_elements *= static_cast<uint64_t>(dim);
      }
      RETURN_IF(output_elements > std::numeric_limits<size_t>::max() / element_size,
                ort_api_, "MetalEP Concat output byte count overflows");
      const size_t output_bytes = static_cast<size_t>(output_elements) * element_size;
      uint32_t axis_offset = 0;
      for (size_t i = 0; i < input_count; ++i) {
        const std::vector<int64_t>& shape = inputs[i]->shape;
        const uint32_t input_axis = static_cast<uint32_t>(shape[static_cast<size_t>(axis)]);
        const size_t input_bytes = inputs[i]->element_count * element_size;
        if (!metal_->ConcatSliceBytes(
                inputs[i]->data, input_bytes, output.mutable_data, output_bytes, element_size,
                static_cast<uint32_t>(outer), input_axis,
                static_cast<uint32_t>(output_shape[static_cast<size_t>(axis)]),
                static_cast<uint32_t>(inner), axis_offset, error)) {
          return ort_api_.CreateStatus(ORT_EP_FAIL,
                                       ("MetalEP Concat failed: " + error).c_str());
        }
        axis_offset += input_axis;
      }
      return nullptr;
    }

    return ort_api_.CreateStatus(ORT_EP_FAIL,
                                 ("MetalEP Coco kernel does not implement " + op_type_).c_str());
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// MarietteKernel (MatMulNBits, RMSNormalization, SkipSimplifiedLayerNormalization, Softmax)
// ---------------------------------------------------------------------------

static void RowsAndLast(const std::vector<int64_t>& shape, size_t& rows, size_t& last) {
  last = shape.empty() ? 1 : static_cast<size_t>(shape.back());
  rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= static_cast<size_t>(shape[i]);
}

MarietteKernel::MarietteKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal,
                               Ort::ConstNode node)
    : KernelBase(ort_api, metal), op_type_(node.GetOperatorType()) {
  epsilon_ = FloatAttribute(node, "epsilon", 1e-6f);
}

MarietteKernel::~MarietteKernel() {
  if (b_dev_ != nullptr) metal_->Free(b_dev_);
  if (scales_dev_ != nullptr) metal_->Free(scales_dev_);
}

OrtStatus* MarietteKernel::ComputeIO(KernelIO& io) {
  try {
    std::string err;

    if (op_type_ == "MatMulNBits") {
      const size_t input_count = io.InputCount();
      RETURN_IF(input_count != 3 && input_count != 4, ort_api_,
                "MetalEP MatMulNBits expects 3 or 4 inputs");
      const IOTensor& a_val = io.Input(0);
      const IOTensor& b_val = io.Input(1);
      const IOTensor& s_val = io.Input(2);
      const std::vector<int64_t>& a_shape = a_val.shape;
      const std::vector<int64_t>& b_shape = b_val.shape;  // [N,nblocks,16]
      RETURN_IF(a_shape.empty() || b_shape.size() != 3, ort_api_,
                "MetalEP MatMulNBits unexpected input ranks");
      const size_t K = static_cast<size_t>(a_shape.back());
      size_t M = 1;
      for (size_t i = 0; i + 1 < a_shape.size(); ++i) M *= static_cast<size_t>(a_shape[i]);
      const size_t N = static_cast<size_t>(b_shape[0]);
      const size_t nblocks = static_cast<size_t>(b_shape[1]);
      RETURN_IF(K != nblocks * 32, ort_api_, "MetalEP MatMulNBits requires block_size == 32");

      // Copy the constant int4 weights + scales into device buffers once; reuse every step.
      if (b_dev_ == nullptr) {
        b_bytes_ = N * nblocks * 16;
        scales_bytes_ = N * nblocks * sizeof(float);
        b_dev_ = metal_->Alloc(b_bytes_);
        scales_dev_ = metal_->Alloc(scales_bytes_);
        RETURN_IF(b_dev_ == nullptr || scales_dev_ == nullptr, ort_api_,
                  "MetalEP MatMulNBits failed to allocate weight cache");
        std::memcpy(b_dev_, b_val.data, b_bytes_);
        std::memcpy(scales_dev_, s_val.data, scales_bytes_);
      }

      const float* bias = input_count == 4 ? io.Input(3).Data<float>() : nullptr;
      std::vector<int64_t> out_shape(a_shape);
      out_shape.back() = static_cast<int64_t>(N);
      IOTensor& out = io.Output(0, out_shape);
      if (!metal_->MatMulNBitsF32(a_val.Data<float>(), static_cast<const uint8_t*>(b_dev_),
                                  static_cast<const float*>(scales_dev_), bias,
                                  out.MutableData<float>(), M, N, K, nblocks, err)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP MatMulNBits failed: " + err).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "RMSNormalization") {
      RETURN_IF(io.InputCount() != 2, ort_api_, "MetalEP RMSNormalization expects 2 inputs");
      const IOTensor& x = io.Input(0);
      const IOTensor& g = io.Input(1);
      size_t rows = 0, d = 0;
      RowsAndLast(x.shape, rows, d);
      IOTensor& out = io.Output(0, x.shape);
      if (!metal_->RmsNormF32(x.Data<float>(), g.Data<float>(), out.MutableData<float>(), rows, d,
                              epsilon_, err)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP RMSNormalization failed: " + err).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "SkipSimplifiedLayerNormalization") {
      RETURN_IF(io.InputCount() != 3, ort_api_,
                "MetalEP SkipSimplifiedLayerNormalization expects 3 inputs");
      const IOTensor& input = io.Input(0);
      const IOTensor& skip = io.Input(1);
      const IOTensor& gamma = io.Input(2);
      size_t rows = 0, d = 0;
      RowsAndLast(input.shape, rows, d);
      IOTensor& out0 = io.Output(0, input.shape);
      // out[0] is the normalized result; the residual (input+skip) is the last boundary output
      // (ORT drops the unused mean / inv_std_var outputs when fusing the single node).
      float* residual = nullptr;
      const size_t oc = io.OutputCount();
      if (oc >= 2) {
        residual = io.Output(oc - 1, input.shape).MutableData<float>();
      }
      if (!metal_->SkipSimplifiedLayerNormF32(input.Data<float>(), skip.Data<float>(),
                                              gamma.Data<float>(), out0.MutableData<float>(),
                                              residual, rows, d, epsilon_, err)) {
        return ort_api_.CreateStatus(
            ORT_EP_FAIL, ("MetalEP SkipSimplifiedLayerNormalization failed: " + err).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Softmax") {
      RETURN_IF(io.InputCount() != 1, ort_api_, "MetalEP Softmax expects 1 input");
      const IOTensor& x = io.Input(0);
      size_t rows = 0, d = 0;
      RowsAndLast(x.shape, rows, d);
      IOTensor& out = io.Output(0, x.shape);
      if (!metal_->SoftmaxF32(x.Data<float>(), out.MutableData<float>(), rows, d, err)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Softmax failed: " + err).c_str());
      }
      return nullptr;
    }

    return ort_api_.CreateStatus(
        ORT_EP_FAIL, ("MetalEP Mariette kernel does not implement " + op_type_).c_str());
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// Subgraph execution plan + executor
// ---------------------------------------------------------------------------

namespace {

enum class Source { CtxInput, Initializer, Intermediate, Absent };

struct InputRef {
  std::string name;
  Source source = Source::Absent;
  size_t ctx_index = 0;  // valid when source == CtxInput
  // Initializer payload (session-owned; valid for the session lifetime).
  const void* init_data = nullptr;
  std::vector<int64_t> init_shape;
  ONNXTensorElementDataType init_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t init_count = 0;
};

struct OutputRef {
  std::string name;
  bool external = false;  // true -> a subgraph output routed to ctx.GetOutput(ctx_index)
  size_t ctx_index = 0;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

struct NodePlan {
  std::unique_ptr<KernelBase> kernel;
  std::vector<InputRef> inputs;
  std::vector<OutputRef> outputs;
};

}  // namespace

// The concrete SubgraphPlan (forward-declared in ep.h). Owns the per-node kernels and a pool of
// device-resident intermediate buffers reused across decode steps.
struct SubgraphPlan {
  MetalEp* ep = nullptr;
  std::vector<NodePlan> nodes;  // topological order
  std::unordered_map<std::string, void*> intermediate_pool;
  std::unordered_map<std::string, size_t> intermediate_bytes;

  ~SubgraphPlan() {
    if (ep == nullptr) return;
    for (auto& kv : intermediate_pool) {
      if (kv.second != nullptr) ep->Metal()->Free(kv.second);
    }
  }

  // Returns a device buffer of at least `bytes` for the named intermediate edge, reusing the
  // pooled buffer when its capacity is sufficient (stable shapes across decode steps).
  void* AcquireIntermediate(const std::string& name, size_t bytes) {
    auto it = intermediate_bytes.find(name);
    if (it != intermediate_bytes.end() && it->second >= bytes) {
      return intermediate_pool[name];
    }
    if (it != intermediate_bytes.end() && intermediate_pool[name] != nullptr) {
      ep->Metal()->Free(intermediate_pool[name]);
    }
    void* buffer = ep->Metal()->Alloc(std::max<size_t>(bytes, 1));
    intermediate_pool[name] = buffer;
    intermediate_bytes[name] = bytes;
    return buffer;
  }
};

namespace {

// KernelIO backed by a fused subgraph: inputs resolve to ORT ctx inputs, session initializers, or
// device-resident intermediates produced by earlier nodes; outputs are ORT ctx outputs (for
// subgraph outputs) or freshly-allocated device intermediates.
class SubgraphIO : public KernelIO {
 public:
  SubgraphIO(Ort::KernelContext& ctx, NodePlan& node, SubgraphPlan& plan,
             std::unordered_map<std::string, IOTensor>& produced)
      : ctx_(ctx), node_(node), plan_(plan), produced_(produced) {}

  size_t InputCount() const override { return node_.inputs.size(); }
  size_t OutputCount() const override { return node_.outputs.size(); }

  const IOTensor& Input(size_t index) override {
    auto cached = input_cache_.find(index);
    if (cached != input_cache_.end()) return cached->second;
    const InputRef& ref = node_.inputs[index];
    IOTensor tensor;
    switch (ref.source) {
      case Source::Intermediate: {
        auto it = produced_.find(ref.name);
        if (it != produced_.end()) tensor = it->second;
        break;
      }
      case Source::CtxInput: {
        Ort::ConstValue value = ctx_.GetInput(ref.ctx_index);
        auto info = value.GetTensorTypeAndShapeInfo();
        tensor.data = value.GetTensorRawData();
        tensor.shape = info.GetShape();
        tensor.type = info.GetElementType();
        tensor.element_count = info.GetElementCount();
        break;
      }
      case Source::Initializer: {
        tensor.data = ref.init_data;
        tensor.shape = ref.init_shape;
        tensor.type = ref.init_type;
        tensor.element_count = ref.init_count;
        break;
      }
      case Source::Absent:
        break;
    }
    auto res = input_cache_.emplace(index, std::move(tensor));
    return res.first->second;
  }

  IOTensor& Output(size_t index, const std::vector<int64_t>& shape) override {
    auto cached = output_cache_.find(index);
    if (cached != output_cache_.end()) return cached->second;
    OutputRef& ref = node_.outputs[index];
    IOTensor tensor;
    tensor.shape = shape;
    tensor.type = ref.type;
    size_t count = 1;
    for (int64_t dim : shape) count *= dim > 0 ? static_cast<size_t>(dim) : 0;
    tensor.element_count = count;
    if (ref.external) {
      Ort::UnownedValue value = ctx_.GetOutput(ref.ctx_index, shape);
      tensor.mutable_data = value.GetTensorMutableRawData();
      tensor.data = tensor.mutable_data;
    } else {
      const size_t bytes = count * ElementSize(ref.type);
      void* buffer = plan_.AcquireIntermediate(ref.name, bytes);
      tensor.mutable_data = buffer;
      tensor.data = buffer;
    }
    if (!ref.name.empty()) {
      produced_[ref.name] = tensor;  // make it visible to later consumers in the subgraph
    }
    auto res = output_cache_.emplace(index, std::move(tensor));
    return res.first->second;
  }

 private:
  Ort::KernelContext& ctx_;
  NodePlan& node_;
  SubgraphPlan& plan_;
  std::unordered_map<std::string, IOTensor>& produced_;
  std::unordered_map<size_t, IOTensor> input_cache_;
  std::unordered_map<size_t, IOTensor> output_cache_;
};

// Runs an entire fused subgraph into a single Metal command buffer: BeginBatch, encode every
// node's kernel in topological order (device-resident intermediates flow between them with no
// host round-trip), then EndBatch (one commit + one waitUntilCompleted).
OrtStatus* RunSubgraph(SubgraphPlan& plan, OrtKernelContext* kernel_context) {
  MetalEp* ep = plan.ep;
  const OrtApi& ort_api_ = ep->ort_api;
  try {
    Ort::KernelContext ctx(kernel_context);
    ort_mps::MetalContext* metal = ep->Metal();

    std::string err;
    if (!metal->BeginBatch(err)) {
      return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP BeginBatch failed: " + err).c_str());
    }

    std::unordered_map<std::string, IOTensor> produced;
    OrtStatus* node_status = nullptr;
    for (NodePlan& node : plan.nodes) {
      SubgraphIO io(ctx, node, plan, produced);
      node_status = node.kernel->ComputeIO(io);
      if (node_status != nullptr) break;
    }

    std::string end_err;
    const bool ended = metal->EndBatch(end_err);
    if (node_status != nullptr) {
      return node_status;
    }
    if (!ended) {
      return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP EndBatch failed: " + end_err).c_str());
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

// ---- Convex subgraph clustering (GetCapability) ----

using Bitset = std::vector<uint64_t>;

inline void BitSet(Bitset& b, size_t i) { b[i >> 6] |= (uint64_t{1} << (i & 63)); }
inline bool BitTest(const Bitset& b, size_t i) { return (b[i >> 6] >> (i & 63)) & 1u; }
inline void BitOrInto(Bitset& dst, const Bitset& src) {
  for (size_t i = 0; i < dst.size(); ++i) dst[i] |= src[i];
}
inline bool BitIntersects(const Bitset& a, const Bitset& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] & b[i]) return true;
  }
  return false;
}

struct UnionFind {
  std::vector<size_t> parent;
  explicit UnionFind(size_t n) : parent(n) {
    for (size_t i = 0; i < n; ++i) parent[i] = i;
  }
  size_t Find(size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
};

// Groups supported nodes into maximal, convex, connected clusters. A set S is convex (a valid
// single fused node) iff no node x outside S lies on a path between two members of S; contracting
// a non-convex set would create a cycle that ORT rejects. Returns one node-index vector per
// cluster. When `fuse` is false, each supported node becomes its own singleton cluster.
std::vector<std::vector<size_t>> BuildConvexClusters(const std::vector<Ort::ConstNode>& nodes,
                                                     const std::vector<char>& supported,
                                                     bool fuse) {
  const size_t n = nodes.size();
  const size_t words = (n + 63) / 64;

  if (!fuse) {
    std::vector<std::vector<size_t>> clusters;
    for (size_t i = 0; i < n; ++i) {
      if (supported[i]) clusters.push_back({i});
    }
    return clusters;
  }

  // Map tensor name -> producing node index.
  std::unordered_map<std::string, size_t> producer;
  producer.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    for (const auto& out : nodes[i].GetOutputs()) {
      std::string name = out.GetName();
      if (!name.empty()) producer.emplace(std::move(name), i);
    }
  }

  // Direct successors and predecessors within the graph.
  std::vector<std::vector<size_t>> succ(n), pred(n);
  for (size_t j = 0; j < n; ++j) {
    std::unordered_set<size_t> seen;
    for (const auto& in : nodes[j].GetInputs()) {
      std::string name = in.GetName();
      if (name.empty()) continue;
      auto it = producer.find(name);
      if (it == producer.end() || it->second == j) continue;
      if (seen.insert(it->second).second) {
        succ[it->second].push_back(j);
        pred[j].push_back(it->second);
      }
    }
  }

  // Topological order (Kahn) for reachability accumulation.
  std::vector<size_t> indeg(n, 0);
  for (size_t j = 0; j < n; ++j) indeg[j] = pred[j].size();
  std::vector<size_t> order;
  order.reserve(n);
  std::vector<size_t> stack;
  for (size_t i = 0; i < n; ++i) {
    if (indeg[i] == 0) stack.push_back(i);
  }
  while (!stack.empty()) {
    size_t u = stack.back();
    stack.pop_back();
    order.push_back(u);
    for (size_t v : succ[u]) {
      if (--indeg[v] == 0) stack.push_back(v);
    }
  }
  if (order.size() != n) {
    // Cyclic or unexpected; fall back to the node order we were given.
    order.clear();
    for (size_t i = 0; i < n; ++i) order.push_back(i);
  }

  // reach[i] = set of nodes reachable from i (transitive successors, excluding i).
  std::vector<Bitset> reach(n, Bitset(words, 0));
  for (size_t idx = order.size(); idx-- > 0;) {
    const size_t u = order[idx];
    for (size_t v : succ[u]) {
      BitSet(reach[u], v);
      BitOrInto(reach[u], reach[v]);
    }
  }

  // Cluster state keyed by union-find root.
  UnionFind uf(n);
  std::vector<Bitset> cluster_bits(n, Bitset(words, 0));
  std::vector<Bitset> reach_bits(n, Bitset(words, 0));
  for (size_t i = 0; i < n; ++i) {
    if (!supported[i]) continue;
    BitSet(cluster_bits[i], i);
    reach_bits[i] = reach[i];
  }

  // Candidate merge edges: direct data edges between two supported nodes.
  std::vector<std::pair<size_t, size_t>> edges;
  for (size_t u = 0; u < n; ++u) {
    if (!supported[u]) continue;
    for (size_t v : succ[u]) {
      if (supported[v]) edges.emplace_back(u, v);
    }
  }

  auto is_convex = [&](const Bitset& s_bits, const Bitset& reach_s) -> bool {
    for (size_t x = 0; x < n; ++x) {
      if (BitTest(s_bits, x)) continue;
      if (!BitTest(reach_s, x)) continue;      // S cannot reach x
      if (BitIntersects(reach[x], s_bits)) {   // x can reach back into S
        return false;
      }
    }
    return true;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& e : edges) {
      const size_t ra = uf.Find(e.first);
      const size_t rb = uf.Find(e.second);
      if (ra == rb) continue;
      Bitset merged = cluster_bits[ra];
      BitOrInto(merged, cluster_bits[rb]);
      Bitset merged_reach = reach_bits[ra];
      BitOrInto(merged_reach, reach_bits[rb]);
      if (!is_convex(merged, merged_reach)) continue;
      uf.parent[rb] = ra;
      cluster_bits[ra] = std::move(merged);
      reach_bits[ra] = std::move(merged_reach);
      changed = true;
    }
  }

  std::unordered_map<size_t, std::vector<size_t>> grouped;
  for (size_t i = 0; i < n; ++i) {
    if (!supported[i]) continue;
    grouped[uf.Find(i)].push_back(i);
  }
  std::vector<std::vector<size_t>> clusters;
  clusters.reserve(grouped.size());
  for (auto& kv : grouped) {
    std::sort(kv.second.begin(), kv.second.end());
    clusters.push_back(std::move(kv.second));
  }
  return clusters;
}

// Base with a virtual dtor so ReleaseNodeComputeInfos can delete polymorphically.
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

// One OrtNodeComputeInfo per fused subgraph. CreateState resolves the plan by fused-node name;
// Compute runs the whole subgraph into a single command buffer.
struct SubgraphNodeComputeInfo : NodeComputeInfoBase {
  explicit SubgraphNodeComputeInfo(MetalEp& ep) : ep_(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* self = static_cast<SubgraphNodeComputeInfo*>(this_ptr);
    MetalEp& ep = self->ep_;
    std::string fused_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto it = ep.Plans().find(fused_name);
    if (it == ep.Plans().end()) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                     ("No subgraph plan for fused node " + fused_name).c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state,
                                             OrtKernelContext* kernel_context) {
    return RunSubgraph(*static_cast<SubgraphPlan*>(compute_state), kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) {
    // The plan is owned by MetalEp::plans_; nothing to free here.
  }

  MetalEp& ep_;
};

}  // namespace

// ---------------------------------------------------------------------------
// MetalEp
// ---------------------------------------------------------------------------

MetalEp::MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
                 ort_mps::MetalContext* metal, const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      metal_{metal},
      logger_{&logger} {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  GetDefaultMemoryDevice = GetDefaultMemoryDeviceImpl;
}

MetalEp::~MetalEp() = default;

/*static*/
const char* ORT_API_CALL MetalEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const MetalEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  const OrtApi& ort_api_ = ep->ort_api;  // for MPS_LOG
  const OrtLogger* logger_ = ep->logger_;
  try {
    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    const size_t total = nodes.size();

    std::vector<char> supported(total, 0);
    for (size_t i = 0; i < total; ++i) {
      supported[i] = NodeClaimable(nodes[i], ep->config_) ? 1 : 0;
    }

    const bool fuse = std::getenv("ONNX_GENAI_METAL_EP_NOFUSE") == nullptr;
    std::vector<std::vector<size_t>> clusters = BuildConvexClusters(nodes, supported, fuse);

    size_t claimed = 0;
    for (const auto& cluster : clusters) {
      std::vector<const OrtNode*> group;
      group.reserve(cluster.size());
      for (size_t idx : cluster) {
        group.push_back(static_cast<const OrtNode*>(nodes[idx]));
      }
      OrtNodeFusionOptions fusion_options = {};
      fusion_options.ort_version_supported = ORT_API_VERSION;
      fusion_options.drop_constant_initializers = false;  // ORT supplies initializers at runtime
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info, group.data(), group.size(), &fusion_options));
      claimed += cluster.size();
    }

    MPS_LOG(INFO, "MetalEP GetCapability: claimed "
                      << claimed << " of " << total << " nodes for Metal across " << clusters.size()
                      << " fused subgraph(s); remaining fall back to CPU");
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** /*ep_context_nodes*/) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  try {
    for (size_t i = 0; i < count; ++i) {
      Ort::ConstGraph graph{graphs[i]};
      Ort::ConstNode fused_node{fused_nodes[i]};
      const std::string fused_name = fused_node.GetName();

      auto plan = std::make_unique<SubgraphPlan>();
      plan->ep = ep;

      // Fused-node input/output name -> OrtKernelContext index (the runtime I/O boundary).
      std::unordered_map<std::string, size_t> ctx_input_index;
      {
        std::vector<Ort::ConstValueInfo> ins = fused_node.GetInputs();
        for (size_t k = 0; k < ins.size(); ++k) {
          std::string name = ins[k].GetName();
          if (!name.empty()) ctx_input_index.emplace(std::move(name), k);
        }
      }
      std::unordered_map<std::string, size_t> ctx_output_index;
      {
        std::vector<Ort::ConstValueInfo> outs = fused_node.GetOutputs();
        for (size_t k = 0; k < outs.size(); ++k) {
          std::string name = outs[k].GetName();
          if (!name.empty()) ctx_output_index.emplace(std::move(name), k);
        }
      }

      // Constant initializers referenced by the subgraph (session-owned storage).
      struct InitData {
        const void* data = nullptr;
        std::vector<int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        size_t count = 0;
      };
      std::unordered_map<std::string, InitData> initializers;
      for (const auto& vi : graph.GetInitializers()) {
        std::string name = vi.GetName();
        if (name.empty()) continue;
        Ort::ConstValue value{nullptr};
        Ort::Status st = vi.GetInitializer(value);
        if (!st.IsOK() || static_cast<const OrtValue*>(value) == nullptr) continue;
        auto info = value.GetTensorTypeAndShapeInfo();
        InitData d;
        d.data = value.GetTensorRawData();
        d.shape = info.GetShape();
        d.type = info.GetElementType();
        d.count = info.GetElementCount();
        initializers.emplace(std::move(name), std::move(d));
      }

      std::vector<Ort::ConstNode> snodes = graph.GetNodes();

      // Producer of each intra-subgraph tensor.
      std::unordered_map<std::string, size_t> producer;
      for (size_t k = 0; k < snodes.size(); ++k) {
        for (const auto& out : snodes[k].GetOutputs()) {
          std::string name = out.GetName();
          if (!name.empty()) producer.emplace(std::move(name), k);
        }
      }

      // Topological order over the subgraph so producers run before consumers.
      std::vector<std::vector<size_t>> succ(snodes.size());
      std::vector<size_t> indeg(snodes.size(), 0);
      for (size_t j = 0; j < snodes.size(); ++j) {
        std::unordered_set<size_t> seen;
        for (const auto& in : snodes[j].GetInputs()) {
          std::string name = in.GetName();
          if (name.empty()) continue;
          auto it = producer.find(name);
          if (it == producer.end() || it->second == j) continue;
          if (seen.insert(it->second).second) {
            succ[it->second].push_back(j);
            ++indeg[j];
          }
        }
      }
      std::vector<size_t> order;
      order.reserve(snodes.size());
      std::vector<size_t> stack;
      for (size_t k = 0; k < snodes.size(); ++k) {
        if (indeg[k] == 0) stack.push_back(k);
      }
      while (!stack.empty()) {
        size_t u = stack.back();
        stack.pop_back();
        order.push_back(u);
        for (size_t v : succ[u]) {
          if (--indeg[v] == 0) stack.push_back(v);
        }
      }
      if (order.size() != snodes.size()) {
        order.clear();
        for (size_t k = 0; k < snodes.size(); ++k) order.push_back(k);
      }

      // Build the per-node execution plan.
      for (size_t idx : order) {
        Ort::ConstNode node = snodes[idx];
        NodePlan np;

        if (IsAddNode(node)) {
          np.kernel = std::make_unique<AddKernel>(ep->ort_api, ep->metal_, node);
        } else if (MarietteClaimable(node)) {
          np.kernel = std::make_unique<MarietteKernel>(ep->ort_api, ep->metal_, node);
        } else if (CocoClaimable(node)) {
          np.kernel = std::make_unique<CocoKernel>(ep->ort_api, ep->metal_, node);
        } else {
          return ep->ort_api.CreateStatus(
              ORT_EP_FAIL,
              ("MetalEP has no compile handler for claimed op " + node.GetOperatorType()).c_str());
        }

        for (const auto& in : node.GetInputs()) {
          InputRef ref;
          ref.name = in.GetName();
          if (ref.name.empty()) {
            ref.source = Source::Absent;
          } else if (producer.count(ref.name)) {
            ref.source = Source::Intermediate;
          } else if (auto ci = ctx_input_index.find(ref.name); ci != ctx_input_index.end()) {
            ref.source = Source::CtxInput;
            ref.ctx_index = ci->second;
          } else if (auto ii = initializers.find(ref.name); ii != initializers.end()) {
            ref.source = Source::Initializer;
            ref.init_data = ii->second.data;
            ref.init_shape = ii->second.shape;
            ref.init_type = ii->second.type;
            ref.init_count = ii->second.count;
          } else {
            return ep->ort_api.CreateStatus(
                ORT_EP_FAIL, ("MetalEP could not resolve subgraph input " + ref.name).c_str());
          }
          np.inputs.push_back(std::move(ref));
        }

        for (const auto& out : node.GetOutputs()) {
          OutputRef ref;
          ref.name = out.GetName();
          auto tinfo = out.TypeInfo();
          if (tinfo.GetONNXType() == ONNX_TYPE_TENSOR) {
            ref.type = tinfo.GetTensorTypeAndShapeInfo().GetElementType();
          }
          if (!ref.name.empty()) {
            auto co = ctx_output_index.find(ref.name);
            if (co != ctx_output_index.end()) {
              ref.external = true;
              ref.ctx_index = co->second;
            }
          }
          np.outputs.push_back(std::move(ref));
        }

        plan->nodes.push_back(std::move(np));
      }

      ep->plans_[fused_name] = std::move(plan);
      node_compute_infos[i] = new SubgraphNodeComputeInfo(*ep);
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
void ORT_API_CALL MetalEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetDefaultMemoryDeviceImpl(const OrtEp* this_ptr,
                                                            const OrtMemoryDevice** device) noexcept {
  const auto* ep = static_cast<const MetalEp*>(this_ptr);
  *device = ep->ep_api.MemoryInfo_GetMemoryDevice(ep->factory_.GetDefaultMemoryInfo());
  return nullptr;
}
