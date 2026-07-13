// Copyright (c) 2026. Licensed under the MIT License.
//
// The modular ONNX->MLX op-translation registry — the single source of truth for which ops the
// MLX-native EP can translate. Handlers are keyed by (domain, op_type, [min_opset, max_opset]) and
// dispatched by BOTH the claim-time membership check (mlx_backend::Supported, consulted from
// ep.cc::GetCapability) and the run-time translator (TranslationContext::Translate). Because both
// paths consult the SAME table, "claimed" and "translatable" can never disagree.
//
// Adding a new op = add one handler function (typically in a new src/ep/ops/<family>.cc module) and
// one registration line in that module's RegisterXxxOps(). See docs/OP_ARCHITECTURE.md §6.

#pragma once

#include <limits>
#include <string>
#include <vector>

#include "mlx_backend.h"  // NodeDesc

namespace ort_mps_mlx {

// The engine object a handler uses to resolve inputs, bind outputs, and emit MLX ops. Fully defined
// in mlx_engine.h; handler modules include that header. Forward-declared here so op_registry.h stays
// free of the (heavy) mlx-c include.
class TranslationContext;

// A translation handler: reads NodeDesc, emits MLX ops through the context, binds the node outputs.
using OpHandler = void (*)(TranslationContext& ctx, const NodeDesc& node);

// A claim-time predicate registered NEXT TO a handler: given the concrete ONNX node, decide whether
// the MLX backend can translate it exactly (dtypes, shapes, attributes, input/output form). The
// registry key (domain, op_type, opset) is already matched before this runs, so a predicate only
// needs the node-specific checks. Lives in the same ops/<family>.cc module as its handler, using the
// shared helpers in op_claim.h. Replaces the old per-family *Claimable funcs in ep.cc.
using ClaimPredicate = bool (*)(Ort::ConstNode node);

// Sentinel for an unbounded opset endpoint. A registration with [kAnyOpset, kAnyOpset] matches any
// opset (used for version-insensitive ops and contrib/com.microsoft ops). A version-split op
// registers two handlers with adjacent, non-overlapping ranges (e.g. [1,22] and [23, kAnyOpset]).
inline constexpr int kAnyOpset = -1;

// One entry in the registry table: match (domain, op_type) with since_version in [min_opset,
// max_opset] (endpoints are inclusive; kAnyOpset means unbounded on that side).
struct OpRegistration {
  std::string domain;
  std::string op_type;
  int min_opset = kAnyOpset;
  int max_opset = kAnyOpset;
  OpHandler handler = nullptr;
  // Claim-time predicate for this (domain, op, opset). A node is claimed only if its matching entry
  // has a claimable that accepts it. Adding an op = handler + claimable in one place; ep.cc never
  // changes. May be nullptr for an op that is registered but not (yet) claimable on its own.
  ClaimPredicate claimable = nullptr;
};

// The opset-aware (domain, op) -> handler table. A process-wide singleton, lazily populated with the
// built-in op modules on first use (thread-safe via a function-local static).
class OpRegistry {
 public:
  static OpRegistry& Instance();

  void Register(OpRegistration entry);

  // Returns the handler whose (domain, op_type) match and whose [min,max] opset range contains
  // `since_version`, or nullptr if the op has no MLX translation.
  OpHandler Find(const std::string& domain, const std::string& op_type, int since_version) const;

  // Returns the full matching entry (handler + claim predicate) for (domain, op_type, since_version),
  // or nullptr if the op has no MLX translation. Used by the claim-time check (Claimable).
  const OpRegistration* FindEntry(const std::string& domain, const std::string& op_type,
                                  int since_version) const;

  const std::vector<OpRegistration>& Entries() const { return table_; }

 private:
  OpRegistry() = default;
  std::vector<OpRegistration> table_;
};

// Per-family registration entry points. Each is implemented in its src/ep/ops/<family>.cc module and
// called once from OpRegistry::Instance(). Adding a new module = declare its RegisterXxxOps here and
// call it in op_registry.cc::RegisterBuiltinOps.
void RegisterElementwiseOps(OpRegistry& registry);
void RegisterNormOps(OpRegistry& registry);
void RegisterAttentionOps(OpRegistry& registry);
void RegisterQuantOps(OpRegistry& registry);
void RegisterMathOps(OpRegistry& registry);
void RegisterShapeOps(OpRegistry& registry);
void RegisterReductionOps(OpRegistry& registry);
void RegisterNormExtOps(OpRegistry& registry);
void RegisterAttentionExtOps(OpRegistry& registry);
void RegisterConvOps(OpRegistry& registry);
void RegisterSsmMiscOps(OpRegistry& registry);

}  // namespace ort_mps_mlx
