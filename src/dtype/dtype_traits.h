// Copyright (c) 2026. Licensed under the MIT License.
//
// dtype_traits.h — the single dtype abstraction for the Metal EP (PROTOTYPE / scaffolding).
//
// WHY THIS EXISTS
// ---------------
// Today each kernel hard-codes fp32 (some fp16), so supporting a new dtype means copy-pasting a
// kernel and its host glue. The op-architecture refactor (docs/OP_ARCHITECTURE.md) replaces that
// with ONE dtype vocabulary shared by (a) the C++ host dispatch glue and (b) the `.metal` shader
// instantiations, so a kernel's *logic* is written once and specialized per dtype mechanically.
//
// The recommended kernel dtype strategy (see OP_ARCHITECTURE.md §4) is:
//   * MSL:   template <typename T> the kernel body, accumulate in fp32, explicitly instantiate
//            f32/f16/bf16 with `[[host_name("<base>_f32|_f16|_bf16")]]`.
//   * Host:  a runtime `DType` tag + `DTypeInfo` table selects the pipeline by name suffix and
//            carries element size / MSL type name. `DTypeTraits<D>` gives the same facts at
//            compile time for host-side templated launch glue.
//
// bfloat16 is a first-class member: `bfloat` compiles at runtime via newLibraryWithSource on this
// M1 Max toolchain (macOS 14+/Metal 3.1 — verified 2026-07-13). Adding a dtype = one enum value +
// one DTypeInfo row + one MSL instantiation line. No kernel-body edits.
//
// This header is intentionally ORT-agnostic (no onnxruntime_c_api.h include) so op modules and
// unit tests can use it in isolation. The bridge to ORT's element-type enum is done with the
// STABLE ONNX TensorProto.DataType integer values (spec-frozen), not ORT's headers.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ort_mps {

// Element data types the Metal EP reasons about. Ordered; keep in sync with kAllDTypes below.
enum class DType : uint8_t {
  Undefined = 0,
  F32,   // float32
  F16,   // float16 (IEEE half)
  BF16,  // bfloat16 (Metal `bfloat`, Metal 3.1+)
  I8,    // int8
  U8,    // uint8  (packed int4 weight blobs live here)
  I32,   // int32
  I64,   // int64
  Bool,  // boolean
};

// Static per-dtype facts. Specialized below; `AccT` is the recommended reduction/accumulator type
// (fp32 for every float dtype — the numerics contract in DESIGN.md §4).
template <DType D>
struct DTypeTraits;  // primary left undefined on purpose (unknown dtype => compile error)

// clang-format off
template <> struct DTypeTraits<DType::F32>  { using T = float;    using AccT = float; static constexpr uint8_t size = 4; static constexpr bool is_float = true;  static constexpr const char* msl = "float";   static constexpr const char* suffix = "f32";  };
template <> struct DTypeTraits<DType::F16>  { using T = uint16_t; using AccT = float; static constexpr uint8_t size = 2; static constexpr bool is_float = true;  static constexpr const char* msl = "half";    static constexpr const char* suffix = "f16";  };
template <> struct DTypeTraits<DType::BF16> { using T = uint16_t; using AccT = float; static constexpr uint8_t size = 2; static constexpr bool is_float = true;  static constexpr const char* msl = "bfloat";  static constexpr const char* suffix = "bf16"; };
template <> struct DTypeTraits<DType::I8>   { using T = int8_t;   using AccT = int32_t; static constexpr uint8_t size = 1; static constexpr bool is_float = false; static constexpr const char* msl = "char";    static constexpr const char* suffix = "i8";   };
template <> struct DTypeTraits<DType::U8>   { using T = uint8_t;  using AccT = int32_t; static constexpr uint8_t size = 1; static constexpr bool is_float = false; static constexpr const char* msl = "uchar";   static constexpr const char* suffix = "u8";   };
template <> struct DTypeTraits<DType::I32>  { using T = int32_t;  using AccT = int32_t; static constexpr uint8_t size = 4; static constexpr bool is_float = false; static constexpr const char* msl = "int";     static constexpr const char* suffix = "i32";  };
template <> struct DTypeTraits<DType::I64>  { using T = int64_t;  using AccT = int64_t; static constexpr uint8_t size = 8; static constexpr bool is_float = false; static constexpr const char* msl = "long";    static constexpr const char* suffix = "i64";  };
template <> struct DTypeTraits<DType::Bool> { using T = uint8_t;  using AccT = int32_t; static constexpr uint8_t size = 1; static constexpr bool is_float = false; static constexpr const char* msl = "bool";    static constexpr const char* suffix = "bool"; };
// clang-format on

// Runtime mirror of DTypeTraits for value-level (non-templated) dispatch — the common path in
// GetCapability/Compile where the dtype is only known at runtime.
struct DTypeInfo {
  DType dtype = DType::Undefined;
  uint8_t byte_size = 0;
  bool is_float = false;
  const char* msl_name = "void";  // Metal scalar type name
  const char* suffix = "";        // kernel name suffix, e.g. "f32"
  const char* onnx_name = "undefined";
};

namespace detail {
// Stable ONNX TensorProto.DataType integer values (onnx/onnx.proto3). These are frozen by the ONNX
// spec and identical to ORT's ONNXTensorElementDataType, so we bridge without an ORT include.
enum OnnxElemType : int {
  kOnnxUndefined = 0,
  kOnnxFloat = 1,
  kOnnxUint8 = 2,
  kOnnxInt8 = 3,
  kOnnxInt32 = 6,
  kOnnxInt64 = 7,
  kOnnxBool = 9,
  kOnnxFloat16 = 10,
  kOnnxBFloat16 = 16,
};

inline constexpr DTypeInfo kInfo[] = {
    {DType::Undefined, 0, false, "void", "", "undefined"},
    {DType::F32, 4, true, "float", "f32", "float32"},
    {DType::F16, 2, true, "half", "f16", "float16"},
    {DType::BF16, 2, true, "bfloat", "bf16", "bfloat16"},
    {DType::I8, 1, false, "char", "i8", "int8"},
    {DType::U8, 1, false, "uchar", "u8", "uint8"},
    {DType::I32, 4, false, "int", "i32", "int32"},
    {DType::I64, 8, false, "long", "i64", "int64"},
    {DType::Bool, 1, false, "bool", "bool", "bool"},
};
}  // namespace detail

inline const DTypeInfo& DTypeInfoOf(DType d) {
  return detail::kInfo[static_cast<size_t>(d)];
}

// ORT element-type enum (== ONNX TensorProto.DataType) -> DType. Unknown/unsupported => Undefined.
inline DType DTypeFromOnnx(int onnx_elem_type) {
  switch (onnx_elem_type) {
    case detail::kOnnxFloat: return DType::F32;
    case detail::kOnnxFloat16: return DType::F16;
    case detail::kOnnxBFloat16: return DType::BF16;
    case detail::kOnnxInt8: return DType::I8;
    case detail::kOnnxUint8: return DType::U8;
    case detail::kOnnxInt32: return DType::I32;
    case detail::kOnnxInt64: return DType::I64;
    case detail::kOnnxBool: return DType::Bool;
    default: return DType::Undefined;
  }
}

inline int OnnxFromDType(DType d) {
  switch (d) {
    case DType::F32: return detail::kOnnxFloat;
    case DType::F16: return detail::kOnnxFloat16;
    case DType::BF16: return detail::kOnnxBFloat16;
    case DType::I8: return detail::kOnnxInt8;
    case DType::U8: return detail::kOnnxUint8;
    case DType::I32: return detail::kOnnxInt32;
    case DType::I64: return detail::kOnnxInt64;
    case DType::Bool: return detail::kOnnxBool;
    default: return detail::kOnnxUndefined;
  }
}

inline bool IsFloat(DType d) { return DTypeInfoOf(d).is_float; }

// Fully specialized Metal pipeline-state name: "<base>_<suffix>", e.g. MslKernelName("mps_add",
// DType::BF16) == "mps_add_bf16". This is the single naming contract between the host pipeline
// cache and the `[[host_name(...)]]` MSL instantiations.
inline std::string MslKernelName(std::string_view base, DType d) {
  std::string name(base);
  name.push_back('_');
  name.append(DTypeInfoOf(d).suffix);
  return name;
}

// A compact set of supported dtypes for a claim predicate (see OpHandler::claim). `Contains` is the
// one-liner a claim uses to accept fp32/fp16/bf16 while rejecting everything else.
class DTypeSet {
 public:
  constexpr DTypeSet() = default;
  constexpr DTypeSet(std::initializer_list<DType> ds) {
    for (DType d : ds) mask_ |= Bit(d);
  }
  constexpr bool Contains(DType d) const { return (mask_ & Bit(d)) != 0; }
  constexpr DTypeSet operator|(const DTypeSet& o) const { return DTypeSet(mask_ | o.mask_); }
  constexpr bool Empty() const { return mask_ == 0; }

 private:
  constexpr explicit DTypeSet(uint32_t m) : mask_(m) {}
  static constexpr uint32_t Bit(DType d) { return uint32_t{1} << static_cast<uint32_t>(d); }
  uint32_t mask_ = 0;
};

// Convenience sets for claim predicates. `kFloatDTypes` is the fp32/fp16/bf16 target the refactor
// standardizes on — a kernel that supports all three declares exactly this.
inline constexpr DTypeSet kFloatDTypes{DType::F32, DType::F16, DType::BF16};
inline constexpr DTypeSet kFloatAndBF16{DType::F32, DType::F16, DType::BF16};

}  // namespace ort_mps
