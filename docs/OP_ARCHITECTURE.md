# ONNX→MLX Op Translation Architecture

**Status:** Final post-pivot architecture  
**Date:** 2026-07-13  
**Repo:** `onnxruntime-mlx`  
**Companion:** [`DESIGN.md`](./DESIGN.md)

---

## 0. Summary

The op architecture is now an ONNX→MLX translator, not a hand-kernel registry.

`src/ep/ep.cc` decides which ONNX nodes are claimable, asks ORT to fuse the supported partition, and compiles the partition into an MLX-oriented node-descriptor plan. `src/ep/mlx_backend.cc` runs that plan through `mlx-c`, dispatching each node through a **modular, opset-aware, dtype-generic op registry** (`src/ep/op_registry.{h,cc}` + the per-family handler modules under `src/ep/ops/`). MLX is the **only** compute path for both prefill and decode.

The registered EP name is **`MLXExecutionProvider`** (the name passed to `RegisterExecutionProviderLibrary` and returned by the factory/EP `GetName`). The repo, vendor string, target, and artifact are MLX-native:

- Repo/vendor: `onnxruntime-mlx`
- Target: `onnxruntime_mlx_ep`
- Dylib: `libonnxruntime_mlx_ep.dylib`

There are no active `.metal` kernels, no Metal shader registry, and no dtype/MSL specialization scaffold. Op translations live in the modular registry described in §2.2 — adding an op is one handler file plus one registration line.

---

## 1. Current pipeline

### 1.1 Claim

The EP claims only nodes whose domain, op type, dtype, attributes, and input form exactly match the translation inventory in §2. The claim set is intentionally a subset of what MLX can execute through this EP.

If a node is unsupported, ambiguous, or only supported by the old hand-kernel path, the EP does not claim it. ORT assigns that node to CPU.

### 1.2 Fuse

Claimed nodes are grouped into an ORT partition for the plugin EP. The design goal is a fused decoder subgraph rather than a sequence of per-op custom kernel launches.

### 1.3 Compile

`src/ep/ep.cc` owns `Compile` and builds the ONNX→MLX plan. The plan records:

- The translated MLX operation sequence.
- Input/output binding information for ORT tensors.
- Constants and live-context data that should be converted and cached once.
- KV-cache output bindings used across prefill and decode.

### 1.4 Run

`src/ep/mlx_backend.cc` materializes and executes the MLX graph via `mlx-c`, dispatching each node through the op registry (§2.2). The runtime evaluates once at the fused subgraph boundary with `mlx_eval`, then writes outputs back to the ORT tensors expected by the session.

### 1.5 Fallback

Unclaimed nodes run on ORT CPU. This includes both genuinely unsupported ops and forms that are deliberately excluded from the translator, such as asymmetric `GatherBlockQuantized` with `zero_points`.

---

## 2. Authoritative op translation inventory

The following table is the current support contract. Do not broaden claims without adding the corresponding MLX translation and tests.

| ONNX op | Domain | Claimed ONNX form | MLX op(s) | Notes |
|---|---|---|---|---|
| **MatMulNBits** | `com.microsoft` | int4 block quantized weights, `bits=4`, `block_size=32` | `mlx_quantized_matmul` | Packed uint8 int4 weights are repacked once to MLX affine-quant format and cached persistently on the compiled plan. |
| **GroupQueryAttention** | `com.microsoft` | 9-input separate-QKV form; `fp32` Q/K/V/past_k/past_v/cos/sin; `int32` `seqlens_k` and `total_seq`; RoPE applied in-op | `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope` | Writes present K/V back to the same ORT ctx outputs in `[B, kv_heads, total_seq, head]` `fp32` layout. This keeps prefill→decode KV handoff layout- and position-continuous. |
| **RMSNormalization** | `ai.onnx` | `axis=-1`; `fp32`/`fp16`/`bf16` | `mlx_fast_rms_norm` | Gamma is cached from live context data on first run. Dtype-generic. |
| **SkipSimplifiedLayerNormalization** | `com.microsoft` | `fp32`/`fp16`/`bf16` input/skip/gamma | skip-add + `mlx_fast_rms_norm` | Implements residual add followed by RMS normalization. Dtype-generic. |
| **GatherBlockQuantized** | `com.microsoft` | SYMMETRIC int4 embedding only; 3-input form; `zp=8` | gather + int4 dequant | The asymmetric 4-input `zero_points` form is intentionally not claimed and falls back to CPU. MLX zero-points support is a documented follow-up. |
| **Softmax** | `ai.onnx` | last-axis; `fp32`/`fp16`/`bf16` | `mlx_softmax` | Claimed only for the last-axis form. Dtype-generic. |
| **Add** | `ai.onnx` | `fp32`/`fp16`/`bf16` | MLX elementwise add | Floating forms only (fp32 via the residual-add predicate, fp16/bf16 via the elementwise predicate). |
| **Mul** | `ai.onnx` | `fp32`/`fp16`/`bf16` | MLX elementwise multiply | Floating forms only. |
| **Sub** | `ai.onnx` | `fp32`/`fp16`/`bf16`, `int64` | MLX elementwise subtract | Covers supported numeric/bookkeeping forms only. |
| **Sigmoid** | `ai.onnx`, `com.microsoft` | `fp32`/`fp16`/`bf16` | MLX elementwise sigmoid | Standalone `SiLU`/`Swish` are not claimed. |
| **Cast** | `ai.onnx` | float↔float among `fp32`/`fp16`/`bf16`, `int64`→`int32` | MLX cast | Other casts remain on CPU. |

### 2.1 Ops no longer claimed

The old hand-kernel architecture claimed or planned additional ops. The MLX translator does **not** currently claim these; they run on ORT CPU:

| Former op | Current behavior | Reason |
|---|---|---|
| `Div` | CPU fallback | No current MLX translator entry. |
| `SiLU` | CPU fallback | The translator claims `Sigmoid`, not standalone SiLU. |
| `Swish` | CPU fallback | No current MLX translator entry. |
| `Gelu` | CPU fallback | No current MLX translator entry. |
| standalone `RotaryEmbedding` | CPU fallback | RoPE is translated only inside `GroupQueryAttention` via `mlx_fast_rope`. |
| `Reshape` | CPU fallback | No current MLX translator entry. |
| `Transpose` | CPU fallback | No current MLX translator entry. |
| `Concat` | CPU fallback | No current MLX translator entry. |

This list is intentionally explicit because older design notes and branch history may still mention these ops as hand-kernel coverage.

---

## 2.2 The modular op registry

The translator is a **registry**, not an if-chain. Both the claim-time membership check and the run-time dispatch consult the SAME table, so a claimed op is always translatable and vice-versa.

### Files

| File | Role |
|---|---|
| `src/ep/op_registry.{h,cc}` | The `OpRegistry` singleton: the `(domain, op_type, [min_opset, max_opset]) → handler` table, `Find()`, and `Supported()`. |
| `src/ep/mlx_engine.h` | `MlxDtypeFromOnnx()` (the dtype mapping), `Plan` (persistent MLX state), and `TranslationContext` (the object handlers use to `Resolve`/`Bind` and emit MLX ops). |
| `src/ep/mlx_backend.cc` | The engine: `TranslationContext` method definitions, boundary eval + copy-out, and the `BuildPlan`/`RunPlan`/`DestroyPlan` API. **No op-specific logic.** |
| `src/ep/ops/elementwise.cc` | `Add`, `Mul`, `Sub`, `Sigmoid`, `Softmax`, `Cast` handlers + `RegisterElementwiseOps`. |
| `src/ep/ops/norm.cc` | `RMSNormalization`, `SkipSimplifiedLayerNormalization` + `RegisterNormOps`. |
| `src/ep/ops/attention.cc` | `GroupQueryAttention` (in-op RoPE) + `RegisterAttentionOps`. |
| `src/ep/ops/quant.cc` | `MatMulNBits`, `GatherBlockQuantized` + `RegisterQuantOps`. |

Each module exposes one `RegisterXxxOps(OpRegistry&)` function; `op_registry.cc::RegisterBuiltinOps` calls them all once when the singleton is first used (explicit registration — no reliance on static-init ordering).

### The registry key: `(domain, op_type, opset range)`

A handler is `void(TranslationContext&, const NodeDesc&)`. It is registered under a domain (`""` = `ai.onnx`, or `com.microsoft`), an op type, and an inclusive opset range `[min_opset, max_opset]`. `kAnyOpset` (`-1`) means "unbounded on that side"; a version-insensitive or contrib op registers with `{kAnyOpset, kAnyOpset}`.

The opset is threaded end-to-end: `ep.cc` reads `Ort::ConstNode::GetSinceVersion()` into `NodeDesc::since_version`, and `OpRegistry::Find` dispatches by matching the range. This is the seam that lets opset-23 and opset-24 variants of an op (e.g. `Attention`, `TensorScatter`) map to different handlers — a version-split op registers two handlers with adjacent, non-overlapping ranges (e.g. `[1, 22]` and `[23, kAnyOpset]`). `RMSNormalization` already uses a bounded range (`[23, kAnyOpset]`) as a live example.

### The dtype mapping

`MlxDtypeFromOnnx()` maps every ONNX tensor element type mlx-c can carry to its `mlx_dtype`: `fp32`, `fp16`, **`bf16`**, `fp64`, the signed/unsigned integer widths (`int8/16/32/64`, `uint8/16/32/64`), and `bool`. It is used in `Resolve`/`Bind`, constant materialization, the pre-eval boundary cast, and `CopyOut`, so **every tensor honors its actual dtype** rather than a hard-coded fp32.

Because MLX carries the resolved dtype through its ops with no per-dtype code, the dtype-generic handlers (elementwise, activation, softmax, normalization, cast) work in fp32, fp16 **and** bf16 with a single implementation. `MatMulNBits`, `GroupQueryAttention`, and `GatherBlockQuantized` remain fp32-only (the quant repack / SDPA paths match the cpu-recipe graph); widening them is follow-up work.

---

## 3. Translation details by family

### 3.1 Quantized matmul

`MatMulNBits` is claimed only for the int4 block-32 form. The ONNX packed uint8 weight tensor is repacked once into the affine-quant layout MLX expects, then stored on the compiled plan for reuse across prefill and decode.

Runtime target: `mlx_quantized_matmul`.

### 3.2 Attention and KV cache

`GroupQueryAttention` is the fused attention op used by the target decoder graphs. The claimed form is the 9-input separate-QKV `com.microsoft` op with `fp32` Q/K/V, past K/V, cos/sin, and `int32` sequence-length inputs.

The translator maps it to:

- `mlx_fast_rope` for the in-op RoPE transform.
- `mlx_fast_scaled_dot_product_attention` for attention.

The backend writes present K/V to the same ORT context outputs in `[B, kv_heads, total_seq, head]` `fp32` layout. This preserves the runtime-owned KV-cache handoff across the prefill→decode boundary.

### 3.3 Normalization

`RMSNormalization` maps to `mlx_fast_rms_norm` for `axis=-1`, in `fp32`/`fp16`/`bf16`.

`SkipSimplifiedLayerNormalization` is translated as skip-add followed by `mlx_fast_rms_norm` for `fp32`/`fp16`/`bf16` input/skip/gamma.

### 3.4 Quantized embedding gather

`GatherBlockQuantized` is claimed only for the symmetric int4 embedding form with three inputs and `zp=8`. The backend performs gather plus int4 dequant.

The asymmetric four-input form with explicit `zero_points` is not claimed. It falls back to CPU until the MLX zero-points path is implemented and tested.

### 3.5 Softmax and elementwise

The translator supports the exact elementwise and cast forms listed in §2:

- `Softmax`: last-axis, `fp32`/`fp16`/`bf16` → `mlx_softmax`.
- `Add`: `fp32`/`fp16`/`bf16`.
- `Mul`: `fp32`/`fp16`/`bf16`.
- `Sub`: `fp32`/`fp16`/`bf16` and `int64`.
- `Sigmoid`: `fp32`/`fp16`/`bf16`.
- `Cast`: float↔float among `fp32`/`fp16`/`bf16`, `int64`→`int32`.

Unsupported dtype combinations fall back to CPU.

---

## 4. Caching and lifetime model

The compiled plan owns persistent conversions that are expensive or unnecessary to repeat:

- Repacked `MatMulNBits` weights.
- Cos/sin caches.
- Gammas.
- Embedding table data.
- Biases.

Some values are available only through live context data, so they are cached on the first `Run`. Subsequent prefill/decode invocations reuse the plan cache.

This replaces the old per-kernel pipeline-state model. There are no Metal compute pipeline states, shader names, or MSL dtype suffixes in the current architecture.

---

## 5. Claiming rules

A claim predicate should answer one question: **can the current ONNX node be represented by the MLX translator exactly enough to run inside the fused subgraph?**

That means every claim must validate:

1. Domain and op type.
2. Input count and output count where the op has multiple forms.
3. Dtypes.
4. Required attributes such as `bits`, `block_size`, `axis`, or last-axis softmax.
5. Layout assumptions, especially KV-cache shape for GQA.
6. Whether constants/initializers can be cached in the current plan.

The claim predicate is additionally AND-gated by registry membership: `NodeClaimable` calls `ort_mps_mlx::Supported(domain, op_type, since_version)`, which consults the same `(domain, op, opset)` registry the run-time translator dispatches through. A node the registry has no handler for is never claimed, so "claimed" can never outrun "translatable".

When in doubt, do not claim. CPU fallback is preferred to an approximate translation.

---

## 6. Adding or changing a translated op

The extension path is registry-centric. To add a long-tail op:

1. **Handler.** Add a `void MyOp(TranslationContext& ctx, const NodeDesc& n)` function in the appropriate `src/ep/ops/<family>.cc` module (or a new module — add its `.cc` to `CMakeLists.txt` and declare `RegisterMyFamilyOps` in `op_registry.h`, called from `RegisterBuiltinOps`). Resolve inputs with `ctx.Resolve(n.inputs[i])`, emit MLX ops via the `ctx` helpers (`Reshape`, `Transpose`, `Astype`, `Mul`, `AddA`, …) or `mlx_*` calls wrapped in `ctx.Keep(...)`, and bind results with `ctx.Bind(n.outputs[i], ...)`. Read the tensor's actual dtype through `MlxDtypeFromOnnx` — never hard-code fp32.
2. **Register.** Add one line to that module's `RegisterXxxOps`: `registry.Register({domain, "MyOp", min_opset, max_opset, &MyOp});`. Use `kAnyOpset` for a version-insensitive op, or a bounded range for an opset-split op.
3. **Attributes.** If the handler reads attributes not already threaded, add them to the `mnd.ints`/`mnd.floats` population in `ep.cc`'s node-build loop.
4. **Claim predicate.** Add/extend the dtype/shape/attribute claim logic in `ep.cc` (`MarietteClaimable`/`CocoClaimable`/`AddClaimable`). The registry AND-gate means you must both add the claim predicate AND register the handler.
5. **Tests.** Add op-correctness coverage in `tests/ops/mlx_op_test.py` (ONNX IR API). For a dtype-generic op, add fp16 (vs ORT CPU) and bf16 (bf16-interior subgraph vs a numpy reference) cases.
6. **E2E.** Add or update `tests/e2e/` coverage if the op affects decoder coherence or KV-cache behavior.
7. **Docs.** Update the §2 table and, if relevant, §2.2.

**Add a new opset variant:** register a second handler for the new range and narrow the existing registration (e.g. change `[23, kAnyOpset]` to `[23, 23]` and add `[24, kAnyOpset]`).

**Add a new dtype:** if mlx-c exposes it, add the `ONNX → mlx_dtype` case to `MlxDtypeFromOnnx`, a `CopyOut` memcpy case, and widen the relevant claim predicate. Dtype-generic handlers need no change.

Do not add a new `.metal` kernel or a dtype-traits/MSL specialization layer for new coverage.

---

## 7. Testing expectations

| Layer | Test | Purpose |
|---|---|---|
| Op correctness | `tests/ops/mlx_op_test.py` / `mlx_op_tests` | Confirms each claimed translation matches a reference within accepted tolerances. fp32/fp16 compare against ORT CPU; bf16 keeps the compute inside an MLX-claimed subgraph (fp32 boundaries) and compares against a numpy fp32 reference (~1e-2), since ORT CPU has no bf16 kernels. |
| E2E coherence | `tests/e2e/` / `mlx_e2e` | Confirms the plugin produces coherent model output. |
| Memory stability | `tests/e2e/` / `mlx_leak_test` | Confirms allocator memory stays flat across bounded runs. |

Current post-pivot baseline:

- Build green.
- `ctest`: 3/3 green (`mlx_op_tests`, `mlx_e2e`, `mlx_leak_test`).
- Qwen2.5-0.5B emits `The capital of France is Paris`.
- CPU token stream match for the first 14 tokens; known fp32 decode drift after that is accepted.
- Prefill/TTFT improves from ~33 ms CPU to ~15 ms MLXExecutionProvider for a 26-token prompt.
- Prefill/TTFT improves from ~575 ms CPU to ~165 ms MLXExecutionProvider for a 512-token prompt.
- Warm decode is ~122–148 tok/s at short context.
- Leak test shows flat allocator memory across bounded cycles.

---

## 8. Build and dependency implications

`mlx-c` is a hard dependency:

```sh
brew install mlx-c
```

CMake configure fails if it cannot find `mlx-c`. There is no build flag to disable MLX and no hand-kernel fallback.

Use only the current target/artifact names:

- `onnxruntime_mlx_ep`
- `libonnxruntime_mlx_ep.dylib`

The registered EP name is `MLXExecutionProvider`; do not use the target/dylib name as evidence that the runtime-facing EP name differs.

---

## 9. Removed / historical architecture

This document replaces the older modular-op and dtype plan. The following are historical and must not be described as active:

- `src/kernels/*.metal` hand-written shaders for matmulnbits, GQA, norm, softmax, RoPE, elementwise, data movement, and quantized gather.
- Metal shader compile/registry/encode machinery in `src/ep/metal_context.{h,mm}`.
- `cmake/metal_kernels.inc.in`.
- `src/ops/` and its old **hand-kernel** op-registry scaffold (distinct from the current `src/ep/ops/` ONNX→MLX handler modules, which are active — see §2.2).
- `src/dtype/dtype_traits.h` and the dtype/MSL specialization plan.
- The old `onnxruntime_mps_ep` target and `libonnxruntime_mps_ep.dylib` artifact.
- Transitional MLX/Metal feature flags and any hand-kernel fallback path.

For the data that justified the pivot, see [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md).

---

## 10. Open follow-ups

| Follow-up | Current behavior |
|---|---|
| `GatherBlockQuantized` with explicit `zero_points` | Not claimed; CPU fallback. |
| Additional decoder ops | Not claimed unless/until an MLX translation and tests are added. |
| Further fp32 drift analysis | Current drift is accepted after the first 14 CPU-matching tokens; keep monitoring with E2E tests. |
| Broader model-family coverage | Out of scope for the current claim table; CPU fallback remains the default. |
