// Copyright (c) 2026. Licensed under the MIT License.
//
// The opset-aware ONNX->MLX op-translation registry (see op_registry.h). Holds the single (domain,
// op, opset-range) -> handler table consulted by both the claim-time membership check and the
// run-time translator.

#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// Populate the table with every built-in op module. Add a new module's RegisterXxxOps call here.
void RegisterBuiltinOps(OpRegistry& registry) {
  RegisterElementwiseOps(registry);
  RegisterNormOps(registry);
  RegisterAttentionOps(registry);
  RegisterQuantOps(registry);
}

}  // namespace

OpRegistry& OpRegistry::Instance() {
  // Function-local static: thread-safe one-time init (C++11), populated before first use.
  static OpRegistry* instance = [] {
    auto* registry = new OpRegistry();
    RegisterBuiltinOps(*registry);
    return registry;
  }();
  return *instance;
}

void OpRegistry::Register(OpRegistration entry) { table_.push_back(std::move(entry)); }

const OpRegistration* OpRegistry::FindEntry(const std::string& domain, const std::string& op_type,
                                            int since_version) const {
  for (const OpRegistration& entry : table_) {
    if (entry.domain != domain || entry.op_type != op_type) continue;
    if (entry.min_opset != kAnyOpset && since_version < entry.min_opset) continue;
    if (entry.max_opset != kAnyOpset && since_version > entry.max_opset) continue;
    return &entry;
  }
  return nullptr;
}

OpHandler OpRegistry::Find(const std::string& domain, const std::string& op_type,
                           int since_version) const {
  const OpRegistration* entry = FindEntry(domain, op_type, since_version);
  return entry ? entry->handler : nullptr;
}

// Claim-time membership check (declared in mlx_backend.h) — consults the same registry the
// translator dispatches through, so claim and translation can never disagree.
bool Supported(const std::string& domain, const std::string& op_type, int since_version) {
  return OpRegistry::Instance().Find(domain, op_type, since_version) != nullptr;
}

// Claim-time node predicate (declared in mlx_backend.h). A node is claimable iff the registry has a
// matching (domain, op, opset) entry AND that entry's claim predicate accepts this concrete node.
// This folds the old ep.cc AddClaimable/MarietteClaimable/CocoClaimable per-op logic (now living in
// the ops/*.cc modules) AND the Supported() membership AND-gate into a single lookup, so adding an
// op needs zero ep.cc edits.
bool Claimable(Ort::ConstNode node) {
  const OpRegistration* entry = OpRegistry::Instance().FindEntry(
      node.GetDomain(), node.GetOperatorType(), node.GetSinceVersion());
  return entry != nullptr && entry->claimable != nullptr && entry->claimable(node);
}

}  // namespace ort_mps_mlx
