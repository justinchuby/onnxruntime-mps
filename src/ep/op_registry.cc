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

OpHandler OpRegistry::Find(const std::string& domain, const std::string& op_type,
                           int since_version) const {
  for (const OpRegistration& entry : table_) {
    if (entry.domain != domain || entry.op_type != op_type) continue;
    if (entry.min_opset != kAnyOpset && since_version < entry.min_opset) continue;
    if (entry.max_opset != kAnyOpset && since_version > entry.max_opset) continue;
    return entry.handler;
  }
  return nullptr;
}

// Claim-time membership check (declared in mlx_backend.h) — consults the same registry the
// translator dispatches through, so claim and translation can never disagree.
bool Supported(const std::string& domain, const std::string& op_type, int since_version) {
  return OpRegistry::Instance().Find(domain, op_type, since_version) != nullptr;
}

}  // namespace ort_mps_mlx
