// Copyright (c) 2026. Licensed under the MIT License.
//
// Extended normalization op handlers (LayerNormalization, GroupNormalization, BatchNormalization,
// LpNormalization, SimplifiedLayerNormalization, SkipLayerNormalization). Populated by the
// op-coverage work; see docs/OP_ARCHITECTURE.md for the add-an-op recipe.

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

void RegisterNormExtOps(OpRegistry& registry) {
  (void)registry;  // no ops registered yet
}

}  // namespace ort_mps_mlx
