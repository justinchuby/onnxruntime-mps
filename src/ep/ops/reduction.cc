// Copyright (c) 2026. Licensed under the MIT License.
//
// Reduction op handlers (ReduceSum/Max/Mean/Min/SumSquare, CumSum, TopK). Populated by the
// op-coverage work; see docs/OP_ARCHITECTURE.md for the add-an-op recipe.

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

void RegisterReductionOps(OpRegistry& registry) {
  (void)registry;  // no ops registered yet
}

}  // namespace ort_mps_mlx
