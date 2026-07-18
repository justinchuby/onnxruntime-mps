# Compiled Capture (`mlx_compile`) — what can be dynamic, and the model invariants

**Status:** Final Rust post-pivot architecture
**Date:** 2026-07-18
**Repo:** `onnxruntime-mlx`
**Companion:** [`OP_ARCHITECTURE.md`](./OP_ARCHITECTURE.md), [`DESIGN.md`](./DESIGN.md)
**Code:** `rust/src/compiled.rs`, `rust/src/engine.rs`, `rust/src/ops/attention.rs`

---

## 0. Summary

The eager translator (`engine::TranslationContext::execute`) rebuilds and dispatches **every** node of a
claimed subgraph as a separate, unfused MLX primitive launch on **every** `Compute` call — ~393 kernel
launches per token for a decoder (`compiled.rs:3-6`). The compiled fast path traces the whole subgraph
**once** into an `mlx_closure` over its dynamic inputs, compiles it with `mlx_compile` (kernel fusion),
caches the closure on the `Plan`, and on each later call just applies the compiled closure to the
freshly-wrapped live buffers.

This doc explains **how that capture works**, **what is allowed to be dynamic** while still being
captured, and **the invariants a model must satisfy** for the (shapeless) decode capture to be both
correct and retrace-free.

---

## 1. The mechanism

1. **Trace once.** Build an `mlx_closure` over just the **dynamic (non-constant) ctx inputs**. Constants
   (weights, cos/sin caches) are baked in; dynamic inputs become **placeholders** (`build_closure`,
   `trace_body` in `compiled.rs`).
2. **Compile.** `mlx_compile` fuses the traced graph into one optimized kernel program
   (`Closure::compile`, `compiled.rs`).
3. **Cache.** The compiled closure is stored on the `Plan` (per `Slot`).
4. **Apply per call.** Zero-copy wrap the live ORT input buffers (`Array::from_data_managed`) and run the
   fused closure; copy the boundary outputs back (`try_compiled`, `compiled.rs`).

**The central constraint:** *inside a trace, placeholders carry no data.* Anything that must **read a
dynamic value mid-trace** — to decide a shape, an axis, or a slice bound — is illegal, because
`mlx_compile` cannot shape-infer or eval it. Every allowance and every invariant below follows from this
one fact.

**Safety.** Every path falls back to the eager translator on any doubt (ineligible plan, missing cache,
trace/apply/eval error). The compiled path never crashes and never diverges (`compiled.rs:27-28`).

---

## 2. Two capture modes (`ShapeMode`) — the crux of "what can be dynamic"

`ShapeMode` (`engine.rs:241-248`) is the whole answer:

| Mode | Behavior on a changed input shape | Used by |
|---|---|---|
| **`Shapeless`** | The compiled closure is shape-*agnostic*: a dimension may **grow/change every call with zero retrace**. One closure serves every step. | **decode** (growing KV length costs one compile ever, not one per token) |
| **`ShapeKeyed`** | `mlx_compile` keys on input shapes/dtypes and **transparently retraces** (re-invokes the thunk) when they change, so a new shape recompiles rather than miscomputes. | **general** static-shape subgraphs; **prefill** (varying query length `S`) |

Shapeless `mlx_compile` **cannot shape-infer a `Slice`** and cannot eval mid-trace, so a shapeless
subgraph must contain neither. ShapeKeyed can, but pays one recompile per distinct `(shape, dtype)` tuple
— cheap when the shape set is small and repeating (prefill prompt lengths, CNN batch sizes),
pathological if unbounded.

### Two distinct senses of "dynamic"

- **Symbolic-dynamic at claim time** (a `-1`/symbolic dim in the graph): always fine **as long as the
  hidden/feature dim is static**. E.g. the `Attention` claim requires `qshape[2]` (hidden) `> 0` but lets
  `qshape[0]` (batch) and seq be dynamic — literally *"batch/seq may be dynamic"*
  (`ops/attention.rs:1207-1211`).
- **Runtime-varying across calls:** free in `Shapeless`; a transparent recompile in `ShapeKeyed`.

### The three configurations

`CompiledConfig` (`engine.rs:251-315`) expresses all three fast paths as one parameterised core:

- **decode** = `{ Shapeless, kv_alias, rope_as_data, delta_copyout }` — shapeless so a growing KV length
  never retraces; RoPE position + valid-past fed as **data**; delta KV copy-out.
- **general** = `{ ShapeKeyed, contiguous_outputs }` — retraces on a shape change; no attention/KV
  specialisation.
- **prefill** = `{ ShapeKeyed, kv_alias, rope_as_data, delta_copyout }` — the **same** decoder subgraph
  as decode but at query length `S > 1`, so it is shape-keyed (see §4).

---

## 3. Invariants for the shapeless **decode** capture

These are the gates in `general_enabled` / `build_closure` and the reasons behind each. A subgraph that
violates one falls back to eager (or, for attention, to the growing-concat route).

1. **No control flow** (`If` / `Loop` / `Scan`). The graph *structure* would depend on runtime data
   (`compile_enabled`, `compiled.rs:46-51`).

2. **No data-dependent shapes.** `Reshape` / `Expand` target (input 1), `Slice` starts/ends/axes/steps
   (1–4), and `Range` bounds (0–2) must be **constant or shape-const** (derived from `Shape`/`Size`),
   never a plain runtime intermediate (`reads_data_dependent_shape`, `compiled.rs:84-96`). Shape-const is
   OK because it resolves to a compile-time integer; a data-dependent value forces a mid-trace eval that
   crashes shapeless compile. (This is exactly why runtime-bounded `Range` is deferred — see
   `OP_ARCHITECTURE`/README Range note.)

3. **No host-computed / data-dependent-output-shape ops** — `Det`, `NonZero`, `Unique` GPU-eval their
   input data mid-translate and/or emit a shape known only at runtime (`is_general_compile_unsafe`,
   `compiled.rs:61-73`).

4. **Attention uses the shared-buffer (fixed-capacity) KV contract, not growing concat.** With a
   runtime-owned max KV length, `past_k`'s seq axis is a **fixed capacity** every step, so **all shapes
   are static across tokens** (`detect_shared_kv`, `compiled.rs:167-213`). The only thing that varies —
   the valid-past offset — is fed as **data**: the new K/V are written with a `slice_update_dynamic` at
   that offset and attention runs over the whole capacity under a **static-shape additive mask** (buffer
   tail beyond `valid_past + S` masked to `-inf`; causal within the valid prefix)
   (`ops/attention.rs:420-425, 582-590`). Every op stays statically shaped, so the shapeless closure can
   carry it. A *growing* concat KV would change shape each token and force retrace/eager.

5. **`valid_past` must be recoverable as data** — from `total_sequence_length` (GQA input 6) or, when
   that scalar is computed in-subgraph, from the `attention_mask` width (`detect_shared_kv:193-209`,
   `try_compiled:355-360`). That is what lets the per-step position be **data** rather than a baked
   constant.

6. **RoPE must be full-rotary (`rotary_dim == head_dim`).** Then rotate-half is expressed as a `[hd,hd]`
   **matmul** — carrying no `Slice`, which shapeless compile cannot shape-infer — and the cos/sin rows
   are **pre-sliced outside the graph and fed as synthetic closure inputs**, so the per-step position
   never bakes in (`compiled.rs:21-25`, the `rope_as_data` feature). Partial rotary → the build declines
   → eager.

### Mental model

> **Shapeless capture = "the only things allowed to vary at runtime are values fed as *data*; everything
> the graph's *structure / shapes* depend on must be a compile-time constant or a fixed-capacity
> buffer."** Position, valid-past, and the KV delta are data. Capacity, head counts, hidden dims, and
> rotary structure are static. That contract is what lets one compiled closure serve an unbounded decode.

---

## 4. Why prefill is `ShapeKeyed`, not `Shapeless`

Prefill is the same decoder subgraph but at query length `S > 1`. The causal mask's query-position
`arange(0, S)` extent and the KV write width are read as **Rust ints during the trace and baked as
constants** (`CompiledConfig::prefill`, `engine.rs:300-314`). One shapeless closure traced at one `S`
would miscompute at another — so prefill retraces per distinct `S` (a handful of prompt lengths) while
replaying the fused closure for repeated lengths. The shape-keyed prefill trace also statically narrows
attention to the valid prefix `[0, valid_past + S)` (both static per shape key), so it does not run SDPA
over the full KV capacity — which is what turned prefill from a TTFT regression into a win
(`compiled.rs:113-123`).

---

## 5. Empty-tensor guard

For `ShapeKeyed` paths, `try_compiled` skips to eager on **every** call if any dynamic input has a
zero-size dim (an empty tensor): `mlx_compile` over a zero-size shape can abort inside MLX, and a
shape-keyed closure would otherwise retrace straight back into the same abort. Shapeless decode never
sees empty inputs (`compiled.rs:309-321`).

---

## 6. Current limitation (deferred)

The shapeless growing-KV decode capture is currently hardwired to `com.microsoft::GroupQueryAttention`
(`detect_shared_kv`, and the `kv_alias` / `rope_as_data` / `delta_copyout` machinery). Generalizing it to
ai.onnx `Attention` is a larger change because that op has a different KV/mask contract (additive mask +
`is_causal`, no `seqlens_k`), so it does not yet get the shapeless decode fast path. Static-shape
`Attention` subgraphs are still eligible for the **general** `ShapeKeyed` capture.
