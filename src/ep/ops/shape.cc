// Copyright (c) 2026. Licensed under the MIT License.
//
// Shape / data-movement op handlers (Gather, Slice, Concat, Reshape, Transpose, Unsqueeze, Split,
// Expand, Pad, ...). Populated by the op-coverage work; see docs/OP_ARCHITECTURE.md.

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

void RegisterShapeOps(OpRegistry& registry) {
  (void)registry;  // no ops registered yet
}

}  // namespace ort_mps_mlx
