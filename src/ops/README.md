# `src/ops/` — modular op modules for the Metal EP

This directory holds the Metal EP's **op handlers**, one self-contained module per op (or op
family). It replaces the author-keyed `AddClaimable` / `CocoClaimable` / `MarietteClaimable`
predicates and the `if/else` dispatch cascade in `src/ep/ep.cc`. See
[`docs/OP_ARCHITECTURE.md`](../../docs/OP_ARCHITECTURE.md) for the full design and the phased
migration plan.

> **Status:** scaffolding. `op_registry.h` and `../dtype/dtype_traits.h` are in place; the family
> modules and the `ep.cc` wiring land in the migration phase **after** Mariette's prefill-GEMM work
> merges. Do not move kernels out of `ep.cc` yet.

## The pattern

Each op is `(claim predicate) + (kernel factory)` registered against the process-wide
`OpRegistry`, keyed by `(domain, op_type, [min_opset, max_opset])`:

- **Claim predicate** — `bool(const NodeView&)`. Checks op_type/domain (via the key), plus dtypes,
  attributes, input counts, and shape constraints. This is exactly the logic that lives in the
  `*Claimable` functions today, but scoped to one op.
- **Kernel factory** — `unique_ptr<KernelBase>(const KernelBuildContext&)`. Builds the runnable
  kernel (reading attributes off the node). This is what the `if/else` in `Compile` does today.

`GetCapability` calls `OpRegistry::Claims(node)`; `Compile` calls `OpRegistry::Find(node)->make_kernel(ctx)`.
**Adding an op touches only its own file** — no `ep.cc` edit.

## Directory layout

```
src/ops/
├── README.md            # this file
├── op_registry.h        # OpRegistry, OpHandler, NodeView, KernelBuildContext (interfaces)
├── register_all.cc      # RegisterAllOps(): calls each family's Register…Ops() in priority order
├── elementwise/         # Add, Mul, Sub, Div, Sigmoid, SiLU, Gelu, Cast, …
│   ├── elementwise_ops.h
│   ├── elementwise_ops.cc   # claim + factory + RegisterElementwiseOps()
│   └── (kernels live in ../../kernels/elementwise.metal — unchanged)
├── matmulnbits/         # com.microsoft::MatMulNBits (GEMV decode + GEMM prefill)
├── gqa/                 # com.microsoft::GroupQueryAttention
├── norm/                # RMSNormalization, SkipSimplifiedLayerNormalization, LayerNormalization
├── quant/              # GatherBlockQuantized (+ dequant/gather)
├── data_movement/       # Reshape, Transpose, Concat, (Slice, Split, Expand…)
└── attention/           # ai.onnx::Attention (opset 23 & 24), MultiHeadAttention, …
```

`.metal` shader sources stay in `src/kernels/` (already split per family — keep that). An op module
is the **C++ host side** (claim + factory + launch glue); it dispatches into `MetalContext`, which
owns the pipeline-state cache built from `src/kernels/*.metal`.

## Adding a new op (worked example)

Create `src/ops/elementwise/elementwise_ops.cc`:

```cpp
#include "ops/op_registry.h"
#include "ep/ep.h"                 // KernelBase, CocoKernel/AddKernel during migration
#include "dtype/dtype_traits.h"

namespace ort_mps {

void RegisterElementwiseOps(OpRegistry& reg) {
  // Add: fp32/fp16/bf16, 2 inputs, scalar-or-suffix broadcast.
  RegisterOp(reg, /*domain=*/"", "Add", "elementwise",
      /*claim=*/[](const NodeView& n) {
        return n.InputCount() == 2 &&
               n.InputTypeIn(0, kFloatDTypes) &&
               n.InputType(0) == n.InputType(1) &&
               n.OutputType(0) == n.InputType(0);
      },
      /*factory=*/[](const KernelBuildContext& c) -> std::unique_ptr<KernelBase> {
        return std::make_unique<ElementwiseKernel>(*c.ort_api, c.metal, c.ort_node);
      });
  // …Mul/Sub/Div/Sigmoid/SiLU/Gelu/Cast registered the same way…
}

}  // namespace ort_mps
```

Then add one line to `register_all.cc`:

```cpp
void RegisterAllOps(OpRegistry& reg) {
  RegisterElementwiseOps(reg);   // <-- new families slot in here, in priority order
  RegisterMatMulNBitsOps(reg);
  RegisterGqaOps(reg);
  // …
}
```

That is the entire change. No `ep.cc` edit.

## Adding a new opset version of an existing op

Register a second handler with a disjoint opset range. `OpRegistry::Find` picks the handler whose
`[min_opset, max_opset]` contains the node's opset:

```cpp
// ai.onnx::Attention — the input signature differs across opsets (opset 24 adds input #6
// nonpad_kv_seqlen + TensorScatter-backed static cache). Two handlers, two kernels, one op_type.
RegisterOp(reg, "", "Attention", "attention", ClaimAttentionV23, MakeAttentionV23,
           /*min_opset=*/23, /*max_opset=*/23);
RegisterOp(reg, "", "Attention", "attention", ClaimAttentionV24, MakeAttentionV24,
           /*min_opset=*/24, /*max_opset=*/kAnyOpsetMax);
```

## dtype strategy (see `../dtype/dtype_traits.h`)

Kernels are written **once** as MSL templates over the storage type `T` (fp32 accumulators) and
explicitly instantiated for `f32`/`f16`/`bf16` with `[[host_name("<base>_<suffix>")]]`. The host
side selects the pipeline by name via `MslKernelName(base, DType)` and reasons about element size /
Metal type through `DTypeInfoOf(DType)` (runtime) or `DTypeTraits<D>` (compile time). **bfloat16 is
a first-class dtype** — verified to compile at runtime on this M1 Max (Metal 3.1). Adding a dtype is
one enum value + one `DTypeInfo` row + one MSL instantiation line; kernel bodies do not change.

## Testing an op module in isolation

Because `op_registry.h` and `dtype_traits.h` pull in no ORT/Metal headers, a module's claim logic is
unit-testable with a fake `NodeView` (no session, no graph). Kernel *numerics* are tested as today,
against the ORT CPU reference (`tests/kernels/`).
