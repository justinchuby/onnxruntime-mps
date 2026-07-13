# Metal EP — Modular Op + dtype Architecture

**Status:** Design + scaffolding (Phase 2 groundwork)
**Author:** Nabil (ORT Plugin EP Engineer — architecture owner, Metal EP)
**Date:** 2026-07-13
**Repo:** `onnxruntime-mps`
**Requested by:** Justin Chu
**Companion:** [`DESIGN.md`](./DESIGN.md) (Roy, Phase 0). This document is the op-layer refactor
DESIGN.md §3/§5 left open: how the C++ side scales past the Qwen hot path.

> **This is a design + non-conflicting scaffolding drop, not a migration.** Mariette is actively in
> `src/ep/ep.cc`, `src/kernels/matmulnbits.metal`, `src/ep/metal_context.{h,mm}` (prefill GEMM).
> The new files here — `src/ops/op_registry.h`, `src/ops/README.md`, `src/dtype/dtype_traits.h` —
> touch none of hers. The actual extraction of `ep.cc` kernels into modules is the **phased
> migration** in §6, executed after her GEMM lands.

---

## 1. Problem

`src/ep/ep.cc` is **1654 lines**. Op support is organized **by author, not by op**:

- Claiming is three monolithic predicates — `AddClaimable`, `CocoClaimable`, `MarietteClaimable`
  (`ep.cc:178/295/388`) — each a long `if (op == …)` chain. `NodeClaimable` ORs them.
- Dispatch in `Compile` is a parallel `if (IsAddNode) … else if (MarietteClaimable) … else if
  (CocoClaimable)` cascade (`ep.cc:1576`) that re-runs the same predicates to pick a kernel class.
- Every kernel body (`AddKernel`, `CocoKernel`, `MarietteKernel`) is inline in `ep.cc`.

Consequences: (1) adding an op edits the monolith in ≥2 places; (2) **no opset awareness** in
claiming (an `Attention` node is treated identically at opset 23 and 24 even though the input
signature differs); (3) **no dtype abstraction** — fp32 is hard-coded in most kernels, fp16 is
partial, **bf16 is absent**; adding a dtype means copy-pasting kernels.

The `.metal` sources are already split per family (`src/kernels/*.metal`) and use `template
<typename T>` bodies (e.g. `rope.metal:19`, `gather_block_quantized.metal:16`). **That is the right
model — the C++ side must catch up to it.**

### Goal (user directive, verbatim intent)

1. **Cover every op Mobius emits** — not just the Qwen decode path (§2).
2. **Modular** — one op/family per module, not a pile in `ep.cc` (§3).
3. **Extensible for new ops AND new opsets** — adding either is a localized registration (§3.2–3.4).
4. **dtype strategy — templates?** Share kernel logic across fp32/fp16/**bf16** with no copy-paste;
   research the optimal mechanism (§4).

---

## 2. Full Mobius op inventory (coverage target)

Enumerated from `mobius/src/mobius/components/` and `…/tasks/` by scanning every `op.<OpType>(…)`
emission site across **all** model families (not just Qwen). Counts are distinct emission sites
(a proxy for breadth, not runtime node count). Domain: `ai.onnx` unless marked `com.microsoft`
(contrib) — contrib ops carry `_domain="com.microsoft"` / `_MICROSOFT_DOMAIN` in Mobius.

**Opset context (critical for §3.4):** Mobius targets **ai.onnx opset 24** by default and *lowers to
23* for EPs lacking opset-24 kernels (`mobius/src/mobius/_builder.py:_maybe_apply_opset_lowering`).
Opset-24-only semantics it relies on: **`TensorScatter`** (static KV cache) and **`Attention` with
input #6 `nonpad_kv_seqlen`**. So our EP must be **opset-aware**, not just op-type-aware.

Legend: ✅ implemented · ⚠️ partial · ❌ missing.

### 2.1 Decode hot path (P0 — the Qwen-class census)

| Op | Domain | Opset | Status | Notes |
|---|---|---|---|---|
| MatMulNBits | com.microsoft | — | ✅ | GEMV decode + GEMM prefill (Mariette). fp32 only today. |
| GroupQueryAttention | com.microsoft | — | ✅ | RoPE+KV+causal fused (Mariette). fp32, 9-input layout. |
| RMSNormalization | ai.onnx | 23 | ✅ | axis=-1, fp32 (Mariette). |
| SkipSimplifiedLayerNormalization | com.microsoft | — | ✅ | residual+RMS, 3-in (Mariette). |
| Softmax | ai.onnx | 13 | ✅ | last-axis, fp32 (Mariette). |
| RotaryEmbedding | com.microsoft / ai.onnx(23) | — | ✅ | standalone, interleaved+non (Coco). |

### 2.2 Elementwise + activations

| Op | Domain | Status | Notes |
|---|---|---|---|
| Add / Mul / Sub / Div | ai.onnx | ✅ | float; Sub also int64. Suffix-broadcast only. |
| Sigmoid / SiLU / Swish / Gelu | ai.onnx / com.microsoft | ✅ | Coco; SiLU/SwiGLU fusion of Sigmoid∘Mul. |
| Cast | ai.onnx | ⚠️ | only f32↔f16, i64→i32 (Coco). Needs bf16 + general. |
| CastLike | ai.onnx | ❌ | **40 sites** — very common; same as Cast w/ type from operand. |
| Relu / Tanh / Softplus / Clip | ai.onnx | ❌ | activations across MLP/audio/vision. |
| Exp / Log / Sqrt / Reciprocal / Neg / Abs / Floor | ai.onnx | ❌ | unary math (norm/softmax/SSM). |
| Sin / Cos | ai.onnx | ❌ | RoPE-cache construction, diffusion time-embeds. |
| Where / Equal / Less / Greater / GreaterOrEqual / And / Or / Not | ai.onnx | ❌ | masking / control. |
| Min / Max / Mod / Pow | ai.onnx | ❌ | misc. |
| Identity | ai.onnx | ❌ | pass-through (can alias, no kernel). |

### 2.3 Normalization

| Op | Domain | Opset | Status | Notes |
|---|---|---|---|---|
| RMSNormalization | ai.onnx | 23 | ✅ | (see 2.1) |
| SkipSimplifiedLayerNormalization | com.microsoft | — | ✅ | (see 2.1) |
| LayerNormalization | ai.onnx | 17 | ❌ | **7 sites** — vision/audio/encoder-decoder. mean+var. |
| GroupNormalization | ai.onnx | 21 | ❌ | diffusion UNet / VAE. |
| BatchNormalization | ai.onnx | 15 | ❌ | vision backbones. |
| LpNormalization | ai.onnx | 22 | ❌ | embeddings. |
| SimplifiedLayerNormalization / SkipLayerNormalization | com.microsoft | — | ❌ | other RMS/LN fusions. |

### 2.4 Quantization + gather

| Op | Domain | Status | Notes |
|---|---|---|---|
| MatMulNBits | com.microsoft | ✅ | (see 2.1) |
| GatherBlockQuantized | com.microsoft | ✅ | int4 embedding, gather_axis=0 (Coco). |
| Gather | ai.onnx | ❌ | **34 sites** — plain index gather (embeddings, indexing). |
| GatherElements | ai.onnx | ❌ | 3 sites. |

### 2.5 Shape + data movement

| Op | Domain | Opset | Status | Notes |
|---|---|---|---|---|
| Reshape / Transpose / Concat | ai.onnx | — | ✅ | fixed-size, ≤8-D (Coco). |
| Unsqueeze / Squeeze | ai.onnx | — | ❌ | **104 / 44 sites** — everywhere. Often no-op reshapes. |
| Slice / Split / Expand | ai.onnx | — | ❌ | 36 / 22 / 19 sites. |
| Shape / Size | ai.onnx | — | ❌ | **64 sites** — shape bookkeeping (cheap; often best on CPU). |
| Constant / ConstantOfShape | ai.onnx | — | ❌ | 208 / 5 — folded as initializers; rarely a kernel. |
| Range / Tile / Pad | ai.onnx | — | ❌ | 19 / 2 / 11. |
| Compress / ScatterElements | ai.onnx | — | ❌ | 4 / 2. |
| TensorScatter | ai.onnx | **24** | ❌ | opset-24 static KV cache. **Opset-gated** (§3.4). |
| Identity | ai.onnx | — | ❌ | alias. |

### 2.6 Reductions + selection

| Op | Domain | Status | Notes |
|---|---|---|---|
| ReduceSum / ReduceMax / ReduceMean / ReduceMin / ReduceSumSquare | ai.onnx | ❌ | 13/10/3/1/2 — norm & attention stats. |
| CumSum | ai.onnx | ❌ | 6 — position ids / SSM. |
| TopK | ai.onnx | ❌ | 6 — MoE routing, sampling. |

### 2.7 Attention variants

| Op | Domain | Opset | Status | Notes |
|---|---|---|---|---|
| GroupQueryAttention | com.microsoft | — | ✅ | (see 2.1) |
| Attention | ai.onnx | **23 / 24** | ❌ | **16 sites**. opset-24 adds input #6 nonpad_kv_seqlen + TensorScatter static cache. **Opset-gated** (§3.4). |
| MultiHeadAttention | com.microsoft | — | ❌ | vision (Pixtral, encoder-decoder). |
| PackedMultiHeadAttention | com.microsoft | — | ❌ | 2 — Qwen2.5-VL / Qwen3-VL vision. |
| LinearAttention / LightningAttention | com.microsoft(custom) | — | ❌ | 3 — linear-attn SSM hybrids. |

### 2.8 State-space / recurrent (new families)

| Op | Domain | Opset | Status | Notes |
|---|---|---|---|---|
| Scan | ai.onnx | — | ❌ | 6 — Mamba/SSM/RNNT loops (subgraph body op). |
| CausalConvWithState | com.microsoft | — | ❌ | 2 — Mamba causal conv. |
| GatedDeltaNet / Mamba block | (composed) | — | ❌ | built from Scan+Conv+elementwise. |

### 2.9 Conv + vision + diffusion + audio

| Op | Domain | Status | Notes |
|---|---|---|---|
| Conv | ai.onnx | ❌ | **20 sites** — vision backbones, audio frontends, diffusion. |
| ConvTranspose | ai.onnx | ❌ | 3 — VAE decoder / codec. |
| AveragePool | ai.onnx | ❌ | 1 — pooling. |
| (vision attention) | com.microsoft | ❌ | via MultiHeadAttention/PackedMHA (2.7). |

**Coverage summary:** **13 of ~90 distinct op types implemented** (the Qwen decode hot path + basic
elementwise/shape). The long tail (CastLike, Gather, Unsqueeze/Squeeze/Slice/Split/Expand, the
reductions, LayerNormalization, Attention, Conv) is the extensibility target — and precisely why a
registry beats three growing `if`-chains.

---

## 3. Modular op-registry design

Every op becomes a **self-contained handler** keyed by `(domain, op_type, [min_opset, max_opset])`,
registered against a process-wide `OpRegistry`. `GetCapability` and `Compile` consult the registry
instead of hardcoded predicate chains. **The interfaces are prototyped in
[`src/ops/op_registry.h`](../src/ops/op_registry.h)** (compiles standalone; see §5).

### 3.1 Interfaces (from `op_registry.h`)

```cpp
// ORT-agnostic node view — ep.cc adapts Ort::ConstNode to this; tests use a fake.
class NodeView {
  std::string_view OpType() const; std::string_view Domain() const; OpsetVersion Opset() const;
  size_t InputCount() const; size_t OutputCount() const;
  DType InputType(size_t) const; DType OutputType(size_t) const;   // DType from dtype_traits.h
  bool InputShape(size_t, std::vector<int64_t>&) const;
  int64_t IntAttr(std::string_view, int64_t def) const; float FloatAttr(std::string_view, float) const;
};

struct KernelBuildContext { const OrtApi* ort_api; MetalContext* metal;
                            const NodeView* node; const void* ort_node; };

using ClaimPredicate = std::function<bool(const NodeView&)>;
using KernelFactory  = std::function<std::unique_ptr<KernelBase>(const KernelBuildContext&)>;

struct OpHandler {                       // one per (domain, op_type, opset-range)
  std::string domain, op_type;
  OpsetVersion min_opset, max_opset;     // inclusive; [1, ∞) by default
  const char* family;                    // logging/grouping tag
  ClaimPredicate claim;                  // dtype/attr/shape check
  KernelFactory  make_kernel;            // builds the runnable kernel
};

class OpRegistry {
  static OpRegistry& Instance();
  void Register(OpHandler);
  const OpHandler* Find(const NodeView&) const;   // key match + claim(); nullptr if unsupported
  bool Claims(const NodeView&) const;
};
```

**Why `NodeView` (not `Ort::ConstNode`) and `DType` (not the ORT enum):** it keeps `op_registry.h`
and every op module free of ORT/Metal headers, so (a) modules compile in isolation, (b) claim logic
is unit-testable with a fake node (no session/graph), (c) `ep.cc` is the *only* file that knows the
ORT C++ API — the seam the migration targets.

### 3.2 How `GetCapability` / `Compile` use it

Today (`ep.cc:1440`, `:1576`):

```cpp
supported[i] = NodeClaimable(nodes[i], config_);           // OR of 3 author predicates
// …and in Compile…
if (IsAddNode(node)) np.kernel = make AddKernel;
else if (MarietteClaimable(node)) np.kernel = make MarietteKernel;   // re-runs predicate
else if (CocoClaimable(node)) np.kernel = make CocoKernel;
```

After migration:

```cpp
NodeViewAdapter view(nodes[i]);
supported[i] = OpRegistry::Instance().Claims(view);        // one registry lookup

// …in Compile…
if (const OpHandler* h = OpRegistry::Instance().Find(view))
  np.kernel = h->make_kernel({&ort_api, metal_, &view, node});
else return Error("no handler for claimed op " + view.OpType());
```

Claim and dispatch now share **one** decision (`Find`), eliminating the double-predicate drift risk.

### 3.3 Directory layout & registration

```
src/ops/
├── op_registry.h        # interfaces (this drop)
├── register_all.cc      # RegisterAllOps(reg): calls each family in priority order (migration)
├── elementwise/  matmulnbits/  gqa/  norm/  quant/  data_movement/  attention/  …
└── README.md            # the pattern (this drop)
```

Each family file exposes `void Register<Family>Ops(OpRegistry&)` and registers its handlers.
`RegisterAllOps` calls them in priority order; the factory calls `RegisterAllOps(OpRegistry::Instance())`
once at startup.

**Registration is explicit, not static-init self-registration.** Op modules live in a static lib;
the linker strips TUs nothing references, which would silently drop self-registering ops. An explicit
`RegisterAllOps` list is deterministic, greppable, and controls priority (§3.4). This is a
deliberate design choice recorded in the decision log.

**Adding an op** = new `src/ops/<family>/<op>.cc` + one line in `register_all.cc`. **No `ep.cc`
edit.** Worked example in [`src/ops/README.md`](../src/ops/README.md).

### 3.4 New-opset extensibility

`Find` scans handlers in registration order and matches the first whose `[min_opset, max_opset]`
contains the node's opset *and* whose `claim` accepts it. A new opset version = a new registration
with a disjoint range; the old handler is untouched:

```cpp
// ai.onnx::Attention — opset 24 adds input #6 nonpad_kv_seqlen + TensorScatter static cache,
// so the claim + kernel differ. Two handlers, one op_type, zero edits to the v23 path.
RegisterOp(reg, "", "Attention", "attention", ClaimAttnV23, MakeAttnV23, /*min*/23, /*max*/23);
RegisterOp(reg, "", "Attention", "attention", ClaimAttnV24, MakeAttnV24, /*min*/24, kAnyOpsetMax);
```

This directly models Mobius's opset-24-vs-23 reality (`TensorScatter`, `Attention` nonpad_kv_seqlen)
that the current EP cannot express. Register the **more specific / newer** handler first so it wins
the priority scan.

---

## 4. dtype strategy — research + recommendation

### 4.1 Options evaluated

| Option | Mechanism | Pros | Cons |
|---|---|---|---|
| **(a) C++ templates over dtype-traits** | `DTypeTraits<D>` gives `T`/`AccT`/size/MSL-name; host launch glue templated | zero host copy-paste; compile-time correctness | host-only — does not template the *shader* |
| **(b) MSL `template<typename T>` + explicit instantiation** | one kernel body, `[[host_name("<base>_<suffix>")]]` per dtype | **shares the shader logic**; matches existing `rope.metal`/`gather_block_quantized.metal` | must list instantiations; larger metallib |
| **(b′) Metal function constants** | one pipeline, `[[function_constant]]` branch on a dtype id | single pipeline variant; runtime-selectable | branchy; still needs typed loads; awkward for storage-type changes |
| **(c) runtime dtype tag + switch** | host `switch(DType)` picks a per-dtype kernel name/path | trivial | explodes into N hand-written kernels — the thing we're removing |

### 4.2 Recommendation (optimal mix for this codebase)

**Adopt (a) + (b): a C++ dtype-traits header for host glue + MSL templates with explicit f32/f16/bf16
instantiation for shaders. Reserve (b′) function-constants for kernels where a single pipeline should
branch on a non-storage attribute (e.g. interleaved RoPE), not for the storage dtype.**

This matches what the `.metal` files *already* do (`template <typename T>` bodies accumulating in
`float`) and what the host side is missing. Concretely:

1. **One dtype vocabulary** — [`src/dtype/dtype_traits.h`](../src/dtype/dtype_traits.h) (this drop):
   - `enum class DType { F32, F16, BF16, I8, U8, I32, I64, Bool, … }`.
   - `template<DType> struct DTypeTraits` → `T` (storage), **`AccT = float` for all float dtypes**
     (the DESIGN.md §4 numerics contract), byte size, MSL type name, name suffix.
   - Runtime mirror `DTypeInfoOf(DType)` for value-level dispatch; `DTypeFromOnnx(int)` bridges the
     ORT/ONNX element enum **without an ORT include** (uses spec-frozen ONNX integer values).
   - `DTypeSet` + `kFloatDTypes` so a claim reads `n.InputTypeIn(0, kFloatDTypes)`.

2. **Shader side** — write each kernel once as `template <typename T>` (fp32 accumulators), and
   explicitly instantiate:
   ```metal
   template [[host_name("mps_rmsnorm_f32")]]  kernel void rmsnorm<float>(…);
   template [[host_name("mps_rmsnorm_f16")]]  kernel void rmsnorm<half>(…);
   template [[host_name("mps_rmsnorm_bf16")]] kernel void rmsnorm<bfloat>(…);
   ```
3. **Host↔shader naming contract** — `MslKernelName(base, DType)` → `"<base>_<suffix>"`, the single
   string the pipeline cache and the `[[host_name]]` share. Selecting a dtype at runtime is a
   suffix swap; no host branching.

**Adding a dtype = one enum value + one `DTypeInfo` row + one MSL instantiation line per kernel.**
Kernel bodies never change. That is the "not copy-pasted across dtypes" the directive demands.

### 4.3 bfloat16 availability — FINDING

**Verified available on this machine.** `bfloat` scalars, MSL `template<typename T>` kernels, and
`[[host_name(...)]]` explicit instantiation all **compile at runtime** via `id<MTLDevice>
newLibraryWithSource:` — the *exact* path `src/ep/metal_context.mm` uses.

- Machine: **Apple M1 Max**, **macOS 26.5.1 (25F80)**.
- Language version: **`MTLLanguageVersion3_1`** (Metal 3.1; `bfloat` requires Metal 3.1+ / macOS 14+).
- Probe: `.nabil_probe/bf16_probe.mm` compiled a `bfloat` kernel **and** a
  `scale_probe<bfloat|half|float>` template trio → all `[OK]`.
- Note: the standalone `xcrun metal` CLI is **not installed** on this box ("missing Metal
  Toolchain"), but the EP never uses it — it compiles from source at runtime — so this is a non-issue
  for us. If we later switch to a precompiled `default.metallib` (DESIGN.md §6), we must ensure the
  Metal Toolchain component is installed in CI, or keep runtime compilation.

**Conclusion:** design bf16 into the traits from the start (done — `DType::BF16`, `msl="bfloat"`,
`AccT=float`). bf16 eases testing (wider range than fp16, cheaper than fp32) and is a required target.

---

## 5. Scaffolding delivered (this drop, non-conflicting)

| File | What |
|---|---|
| `src/dtype/dtype_traits.h` | The dtype abstraction: `DType`, `DTypeTraits<D>`, `DTypeInfoOf`, `DTypeFromOnnx`, `MslKernelName`, `DTypeSet`/`kFloatDTypes`. ORT-free, header-only. |
| `src/ops/op_registry.h` | `OpRegistry`, `OpHandler`, `NodeView`, `KernelBuildContext`, `ClaimPredicate`/`KernelFactory`, `RegisterOp`. ORT/Metal-free (forward decls only). |
| `src/ops/README.md` | The module pattern, layout, and worked new-op / new-opset examples. |
| `docs/OP_ARCHITECTURE.md` | This document. |

**None of these are wired into `ep.cc` or the build yet** — they are the interfaces the migration
(§6) wires in *after* Mariette's GEMM work lands. Where the migration will need an `ep.cc` hook it is
documented in §6, not edited here.

**Verification:** both headers compile in isolation and their inline logic runs — see
`.nabil_probe/header_compile_check.cc` (`clang++ -std=c++17 -Isrc`, asserts bf16 traits, the ONNX
enum bridge, `MslKernelName` suffixing, and registry claim + opset routing via a fake `NodeView`).
The existing `cmake --build build` + `ctest` remain green (no build files changed).

---

## 6. Phased migration plan (executed AFTER Mariette's prefill GEMM lands)

Ordered, low-risk, coherence-green at every step. Each phase is a separate PR; the coherence gate
(DESIGN.md §7.3: "capital of France" ⇒ "Paris") and `ctest` must pass before the next.

**Precondition:** Mariette's prefill-GEMM PR (touching `ep.cc`, `matmulnbits.metal`,
`metal_context.{h,mm}`) is merged. Rebase onto it so we extract the *final* kernel bodies.

| Phase | Change | Files touched | Risk |
|---|---|---|---|
| **M0 — land scaffolding** | Merge `dtype_traits.h`, `op_registry.h`, `src/ops/README.md`, this doc. Add `src/ops/` + `src/dtype/` to CMake **include dirs only** (no new sources compiled). | `CMakeLists.txt` (include dirs), new files | none (nothing compiled) |
| **M1 — registry seam** | Add `NodeViewAdapter` (Ort::ConstNode → NodeView) + `OpRegistry::Instance()`/`RegisterAllOps` definitions in a new `src/ops/register_all.cc`. Add the ORT/ONNX→`DType` conversion in the adapter. **`ep.cc` unchanged** — registry not yet consulted. | new `.cc`, CMake source add | low (dead code) |
| **M2 — elementwise family** | Move `AddKernel` + Coco elementwise (Add/Mul/Sub/Div/Sigmoid/SiLU/Gelu/Cast) into `src/ops/elementwise/`. Register them. In `ep.cc`, replace those branches of `NodeClaimable`/Compile with a registry consult **guarded so unmigrated ops still use the old path**. | `ep.cc` (delete moved branches, add registry consult), new module | med |
| **M3 — data movement + quant** | Move Reshape/Transpose/Concat + GatherBlockQuantized + RotaryEmbedding (Coco) into `src/ops/data_movement/`, `src/ops/quant/`. Register. | `ep.cc`, new modules | med |
| **M4 — core compute** | Move MatMulNBits, RMSNorm, SkipSimplifiedLayerNorm, Softmax, GQA (Mariette) into `src/ops/matmulnbits/`, `src/ops/norm/`, `src/ops/gqa/`. Register. **This is where GEMV/GEMM variant selection moves behind the factory.** | `ep.cc`, new modules | med-high |
| **M5 — retire monolith** | `NodeClaimable`/`*Claimable` and the Compile `if/else` are now empty → delete. `ep.cc` shrinks to ABI glue + partitioning + subgraph plan. Registry is the sole authority. | `ep.cc` | low (removal) |
| **M6 — dtype rollout** | Convert the moved kernels to templated MSL (f32/f16/bf16) + `MslKernelName` host selection. Add per-dtype correctness tests (bf16 vs CPU ref). Broaden claims to `kFloatDTypes`. | `src/kernels/*.metal`, modules, tests | med |
| **M7 — coverage expansion** | Add the §2 missing ops as demand dictates, newest-first by model family: CastLike, Gather, Unsqueeze/Squeeze/Slice/Split/Expand, LayerNormalization, reductions, then Attention (opset 23 **and** 24), MultiHeadAttention, Conv. Each = one module + one `register_all.cc` line. | new modules only | low per-op |

**Ordering rationale:** elementwise first (simplest, highest node count, lowest blast radius) proves
the seam end-to-end; core-compute last (highest value, most churn) so it rebases cleanly onto
Mariette's final kernels. `ep.cc` never has *both* systems claiming the same op — each phase moves a
family atomically and deletes its old branch in the same PR.

**Coherence discipline at every phase:** run the DESIGN.md §7.3 coherence gate + full `ctest` +
partition-assertion (node placement unchanged) before merge. Memory-safety: any new device-buffer
caching in a factory must release on kernel destruction (the MRR `=nil` leak that crashed the box —
bounded tests only, no unbounded allocation loops).

---

## 7. Open questions

1. **Registry granularity** — one handler per op_type, or per (op_type, dtype)? Recommend **per
   op_type** with a `DTypeSet` inside the claim (fewer registrations; dtype is a claim detail), and
   split only when two dtypes need genuinely different kernels/factories.
2. **Shape/bookkeeping ops (Shape/Size/Constant/Unsqueeze)** — claim them (fuse more, fewer CPU copy
   nodes) or leave on CPU (they're cheap, and claiming forces them into the Metal partition)?
   Recommend measuring partition copy-node count both ways in M7; default to CPU unless claiming
   demonstrably tightens the partition.
3. **`Attention` (ai.onnx 23/24) vs `GroupQueryAttention` (com.microsoft)** — as Mobius moves to the
   opset-24 `Attention`+`TensorScatter` static-cache path, do we port GQA numerics under a new
   `Attention` handler, or keep both? The registry supports both concurrently; decide by which graph
   Mobius emits per EP.
