// Copyright (c) 2026. Licensed under the MIT License.
//
// op_registry.h — modular op-handler registry for the Metal EP (PROTOTYPE / scaffolding).
//
// WHY THIS EXISTS
// ---------------
// Today `src/ep/ep.cc` claims nodes with hand-written, author-keyed predicates
// (`AddClaimable` / `CocoClaimable` / `MarietteClaimable`) and dispatches with a matching
// `if/else` chain in Compile. Adding an op means editing the 1654-line monolith in two places;
// there is no opset awareness and no dtype abstraction. See docs/OP_ARCHITECTURE.md.
//
// This registry makes each op (or op family) a SELF-CONTAINED unit keyed by
// (domain, op_type, opset-range):
//   * a claim predicate  — does this EP support this node? (op_type/domain/opset/dtypes/attrs)
//   * a kernel factory   — build the KernelBase that runs it.
// GetCapability iterates the registry instead of hardcoded `if (claim_x)`, and Compile looks up
// the factory instead of an `if/else` cascade. Adding an op = drop `src/ops/<family>/<op>.cc`
// with one `Register...()` call; NO ep.cc surgery. A new opset version of an existing op is just
// another registration with a different [min_opset, max_opset] range.
//
// DECOUPLING: this header pulls in NO ORT headers and NO Metal headers. Nodes are read through the
// abstract `NodeView` (implemented in ep.cc over Ort::ConstNode; a fake in tests), and kernels are
// built through the opaque `KernelBuildContext`. So op modules and their unit tests compile in
// isolation, and this header is the stable seam the migration wires ep.cc into.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "dtype/dtype_traits.h"

// Forward declarations only — definitions live in the ORT/Metal translation units.
struct KernelBase;                 // src/ep/ep.h
struct OrtApi;                     // onnxruntime_c_api.h
namespace ort_mps {
class MetalContext;                // src/ep/metal_context.h
}

namespace ort_mps {

// Effective opset import version for a node's domain ("" / "ai.onnx", or "com.microsoft").
using OpsetVersion = int;

// Sentinels for an unbounded opset range end.
inline constexpr OpsetVersion kAnyOpsetMin = 1;
inline constexpr OpsetVersion kAnyOpsetMax = 100000;

// ---------------------------------------------------------------------------
// NodeView — ORT-agnostic read view of a graph node used by claim predicates.
// ep.cc provides a concrete adapter over Ort::ConstNode; tests provide a fake.
// ---------------------------------------------------------------------------
class NodeView {
 public:
  virtual ~NodeView() = default;

  virtual std::string_view OpType() const = 0;
  virtual std::string_view Domain() const = 0;      // "" == ai.onnx
  virtual OpsetVersion Opset() const = 0;           // import version for Domain()

  virtual size_t InputCount() const = 0;
  virtual size_t OutputCount() const = 0;

  // dtype of input/output `i`; DType::Undefined if the (optional) operand is absent or untyped.
  virtual DType InputType(size_t i) const = 0;
  virtual DType OutputType(size_t i) const = 0;

  // Static shape of input `i` into `out` (dynamic dims as -1). False if unavailable.
  virtual bool InputShape(size_t i, std::vector<int64_t>& out) const = 0;

  virtual int64_t IntAttr(std::string_view name, int64_t default_value) const = 0;
  virtual float FloatAttr(std::string_view name, float default_value) const = 0;

  // Convenience: input `i`'s dtype is one of `set` (missing operands never match).
  bool InputTypeIn(size_t i, const DTypeSet& set) const {
    DType t = InputType(i);
    return t != DType::Undefined && set.Contains(t);
  }
};

// ---------------------------------------------------------------------------
// Kernel construction. `KernelBuildContext` type-erases the ORT/Metal handles so this header stays
// dependency-free; the ep.cc adapter casts them back when it builds the concrete kernel.
// ---------------------------------------------------------------------------
struct KernelBuildContext {
  const OrtApi* ort_api = nullptr;
  ort_mps::MetalContext* metal = nullptr;
  const NodeView* node = nullptr;   // attribute/dtype access at build time
  const void* ort_node = nullptr;   // opaque Ort::ConstNode handle (bridged in ep.cc)
};

using ClaimPredicate = std::function<bool(const NodeView&)>;
using KernelFactory = std::function<std::unique_ptr<KernelBase>(const KernelBuildContext&)>;

// A single self-contained op handler. One per (domain, op_type, opset-range[, dtype set]).
struct OpHandler {
  std::string domain;                       // "" == ai.onnx
  std::string op_type;                      // e.g. "MatMulNBits"
  OpsetVersion min_opset = kAnyOpsetMin;    // inclusive
  OpsetVersion max_opset = kAnyOpsetMax;    // inclusive
  const char* family = "";                  // grouping tag for logging (e.g. "matmulnbits")
  ClaimPredicate claim;                     // fine-grained support test (dtypes/attrs/shape)
  KernelFactory make_kernel;                // builds the runnable kernel

  bool MatchesKey(std::string_view d, std::string_view op, OpsetVersion opset) const {
    return op == op_type && d == domain && opset >= min_opset && opset <= max_opset;
  }
};

// ---------------------------------------------------------------------------
// OpRegistry — the lookup table. Not thread-safe for concurrent Register (registration happens once
// at startup); const lookups (Find/Claims) are safe to call concurrently afterward.
// ---------------------------------------------------------------------------
class OpRegistry {
 public:
  // Process-wide registry the EP consults in GetCapability/Compile.
  static OpRegistry& Instance();

  void Register(OpHandler handler) { handlers_.push_back(std::move(handler)); }

  // First handler whose (domain, op_type, opset) key matches AND whose claim predicate accepts
  // `node`; nullptr if none. Registration order is priority order (register specializations first).
  const OpHandler* Find(const NodeView& node) const {
    const std::string_view d = node.Domain();
    const std::string_view op = node.OpType();
    const OpsetVersion opset = node.Opset();
    for (const OpHandler& h : handlers_) {
      if (h.MatchesKey(d, op, opset) && (!h.claim || h.claim(node))) return &h;
    }
    return nullptr;
  }

  bool Claims(const NodeView& node) const { return Find(node) != nullptr; }

  const std::vector<OpHandler>& Handlers() const { return handlers_; }
  size_t Size() const { return handlers_.size(); }

 private:
  std::vector<OpHandler> handlers_;
};

// ---------------------------------------------------------------------------
// Registration entry points. We use EXPLICIT registration (not static-init self-registration)
// because op modules live in a static lib and unreferenced TUs get stripped by the linker, which
// would silently drop self-registering ops. Each family file defines `void Register<Family>Ops
// (OpRegistry&)`; `RegisterAllOps` (src/ops/register_all.cc, added in the migration) calls them in
// priority order. See docs/OP_ARCHITECTURE.md §3.3.
// ---------------------------------------------------------------------------
void RegisterAllOps(OpRegistry& registry);

// Small helper so a family file can register a fully-typed handler tersely.
inline void RegisterOp(OpRegistry& registry, std::string domain, std::string op_type,
                       const char* family, ClaimPredicate claim, KernelFactory make_kernel,
                       OpsetVersion min_opset = kAnyOpsetMin,
                       OpsetVersion max_opset = kAnyOpsetMax) {
  OpHandler h;
  h.domain = std::move(domain);
  h.op_type = std::move(op_type);
  h.family = family;
  h.min_opset = min_opset;
  h.max_opset = max_opset;
  h.claim = std::move(claim);
  h.make_kernel = std::move(make_kernel);
  registry.Register(std::move(h));
}

}  // namespace ort_mps
