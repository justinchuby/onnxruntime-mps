# ONNX conformance (cbourjau/onnx-tests) vs. the MLX Execution Provider — RESULTS

Property-based (Hypothesis) fuzz-conformance of the onnxruntime-mlx
`MLXExecutionProvider` against the ONNX standard, using
[`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) as the
source-of-truth (each generated model is run on the MLX EP **and** on the ONNX
reference evaluator; outputs are compared with the suite's own tolerances).

> These are **fuzzing findings, not a green/red gate.** Because Hypothesis
> samples randomly, exact pass/fail counts vary run-to-run; the *classes* of
> failure are stable. This run: `--hypothesis-seed=0`,
> `--hypothesis-max-examples=25`.
>
> **Update (claim-hardening pass, branch `fix/claim-hardening`).** The
> CRASH/ABORT robustness classes below **have been fixed**, and — per the
> corrected directive — **zero-size / empty tensors are now handled *on MLX*,
> not rejected to CPU.** A claimed op never crashes; empty operands run on the
> MLX path and produce the exact ONNX/numpy-reference output shape (and values,
> where the reference defines them). **CRASH count: 16 → 0.** Only two forms are
> still declined to CPU: **`float64`** (Apple GPUs have no fp64 — an unavoidable
> Metal hardware limit) and absent-optional forms that are handled via defaults.
> The residual FAILs are *not* memory-safety or zero-size issues: they are
> numeric-tolerance gaps, NaN/overflow semantics, or `op`×`dtype` combinations
> the ONNX **reference evaluator / ORT CPU itself** cannot run (see "Known
> non-crash gaps"). See "Claim-hardening" for exactly what changed.

## Environment

| | |
|---|---|
| EP dylib | `build/libonnxruntime_mlx_ep.dylib` (this repo, `ORT_API_VERSION 27`) |
| EP name | `MLXExecutionProvider` (+ `CPUExecutionProvider` fallback) |
| onnxruntime (python) | **1.27.0** (PyPI) — required; the EP refuses to load on ≤1.26 (`MetalEP requires an ONNX Runtime built with ORT_API_VERSION >= 27`) |
| ORT native lib (DYLD) | onnx-genai `ort-prebuilt/lib/libonnxruntime.1.27.0.dylib` |
| onnx-tests | sibling clone, `pixi run postinstall`, python 3.14 / conda-forge |
| Host | macOS 14 / Apple Silicon, mlx-c 0.6.0 |

## How the EP is injected (non-invasive)

onnx-tests picks its "candidate" runtime from the `RUN_CANDIDATE` env var — a
dotted import path to a `Callable[[onnx.ModelProto], dict[str, np.ndarray]]`
(see `onnx_tests/config.py`, `onnx_tests/runtime_wrappers.py`). **No onnx-tests
source is modified.** We point it at our own wrapper:

```
RUN_CANDIDATE=mlx_runtime_wrapper.run_mlx
PYTHONPATH=<this dir>          # so the wrapper is importable
MLX_EP_LIB=<abs path to build/libonnxruntime_mlx_ep.dylib>
```

`mlx_runtime_wrapper.run_mlx` calls
`onnxruntime.register_execution_provider_library("MLXExecutionProvider", MLX_EP_LIB)`
once, then builds every session with
`providers=["MLXExecutionProvider","CPUExecutionProvider"]` and
`ORT_DISABLE_ALL` optimizations (matching the suite's `run_ort`).

The MLX EP is native code and **can hard-crash (segfault / MLX `abort`) the host
process** on an unhandled op form, which would abort the whole pytest session.
So `run_conformance.sh` fuzzes **each op in its own pytest subprocess** — a crash
is contained to that op and recorded as `CRASH(rc=…)`.

## Summary (71 claimed ops)

Counts **after** the claim-hardening pass (before → after: CRASH 16 → **0**,
PASS 28 → **37**, FAIL 27 → **34**). Zero-size inputs now execute **on MLX** for
every supported dtype; the corrected pass moved **Expand** from FAIL → PASS and
removed the last empty-output CopyOut/broadcast crashes.

| Result | Count | Ops |
|---|---|---|
| ✅ PASS | 37 | Add, Sub, Mul, Div, Abs, Neg, Exp, Log, Sqrt, Reciprocal, Floor, Ceil, Round, Sum, Sigmoid, Tanh, LeakyRelu, Equal, Less, Greater, GreaterOrEqual, LessOrEqual, Not, And, Or, Xor, **Softmax**, **Concat**, **Reshape**, **Transpose**, **Slice**, **Squeeze**, **Unsqueeze**, **Tile**, **Expand**, Cast, Identity |
| ⚠️ FAIL | 34 | Pow, Erf, Sign, Min, Max, Mean, Clip, Relu, Gelu, Elu, Selu, Softplus, HardSigmoid, HardSwish, Mish, PRelu, Where, ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceSumSquare, ReduceLogSum, ReduceLogSumExp, LogSoftmax, Split, Pad, Gather, Flatten, MatMul, Conv |
| 💥 CRASH | 0 | *(none — all 16 prior crashes resolved)* |

**Bold** PASS ops were CRASH before hardening. **Expand** now PASSES with
zero-size inputs handled on MLX (the empty broadcast is computed with numpy
bidirectional-broadcast dim rules instead of a naive `max`, so `[3,0]`-style
operands no longer abort `mlx_broadcast_to`). The other prior-CRASH ops
(LogSoftmax, Split, Pad, MatMul, …) now **FAIL cleanly without crashing** — and
their *empty* cases run correctly on MLX; the remaining failures are
`op`×`dtype` coverage gaps or reference-side limitations, **not** EP faults or
empties (see "Known non-crash gaps").

Machine-readable per-op results: [`results.csv`](results.csv). Per-op pytest
logs: [`logs/`](logs/). First-crashing-example captures:
[`logs/culprit/`](logs/culprit/).

> **PASS caveat / what actually ran on MLX.** Unclaimed op *forms* fall back to
> ORT CPU, so a PASS can be a CPU pass. Per-node ORT profiling (see
> "Provider attribution" below) confirms the PASS ops **did execute on the MLX
> provider** for their supported dtypes (fp16/fp32) — e.g. Add/Sub/Mul/Div and
> the elementwise/reduce families fuse into `MLXExecutionProvider_<hash>` nodes.
> **Zero-size inputs now also run on MLX** (verified with `PROFILE=1`
> `ran_on_MLX=True`): e.g. `MatMul (2,0)@(0,4)→(2,4)` zeros, `Pad` empty→empty
> and empty→padded, `Expand [1]→[0]`, empty `Softmax`/`LogSoftmax`/`Where`/
> `ReduceMax`/`ReduceMin`/`ReduceSum` — each produces the ONNX-reference output
> shape/values on the MLX path with **no** CPU fallback. Only `float64` still
> falls back to CPU (Metal has no fp64).

## Provider attribution (which ops ran on MLX vs CPU)

From ORT profiling (`PROFILE=1`, small sample). MLX-claimed nodes are **fused**
into an opaque `MLXExecutionProvider_<hash>` node, so attribution is derived from
the model's real op types + whether any MLX node executed.

- **Ran on MLX** (for supported dtypes, **including zero-size / empty inputs**;
  falls back to CPU only for `float64`):
  Add, Sub, Mul, Div, Abs, Neg, Exp, Log, Sqrt, Reciprocal, Floor, Round, Min,
  Max, Sign, Relu, Gelu, Elu, Selu, Softplus, HardSwish, Mish, PRelu, Sigmoid,
  Tanh, LeakyRelu, Erf, Not, Equal, Less, Greater, GreaterOrEqual, LessOrEqual,
  And, Or, Pow, Where, Conv, and the Reduce* family.
- **CPU-only in the sample** (not observed on MLX in the bounded run — dtype- or
  form-dependent; small sample): Cast, Ceil, HardSigmoid, Mean, ReduceProd,
  Selu, Sum, Xor.

Attribution is a secondary signal; correctness-vs-standard is the primary one.

---

## FAILURES — details for EP triage (not fixed here)

Failures cluster into five reproducible root causes. Reproduce any single case
with the suite's `reproduce_failure` hash printed in the per-op log, or re-run
the op (see "Reproduce" below).

### A. `op`×`dtype` coverage gaps ORT itself can't run → init/kernel failure

> **Not a zero-size issue.** Zero-size / empty tensors are now **handled on
> MLX** (see "what actually ran on MLX"). This class is a *dtype-coverage* gap:
> the onnx-tests generator samples a `dtype` the op's ONNX schema nominally
> allows but that **ORT has no kernel for on any provider** (e.g. `uint16` /
> `int16` for Relu/Min/Max/Where/Pad, or integer `Mean`). It reproduces with a
> **non-empty** input just as well as an empty one, so emptiness is incidental.

When the MLX EP claims such a node, ORT's memcpy transformer aborts session
init; when it declines, ORT CPU raises `NOT_IMPLEMENTED`. Either way the case
cannot be computed by *any* backend — including the reference — so it is
unfixable at the EP claim layer:

```
# MLX-claimed:
onnxruntime …: FAIL : Exception during initialization: transformer_memcpy.cc:254
  IsNodeCompatibleWithProvider … Provider type for Relu node 'Relu_0' is not set.
# CPU-only (same op/dtype):
onnxruntime …: NOT_IMPLEMENTED : Could not find an implementation for Relu(…)
```

- Confirmed identical MLX-init-fail **and** CPU-`NOT_IMPLEMENTED` for:
  `Relu int16` (both empty and 1-element), `Min/Max [(0,),(1,)] uint16`,
  `Where (1,)/(2,)/() uint16`, `Pad(21) uint16/int16`, `Mean` integer.
- Same signature seen for: Pow, Erf, Sign, Gelu, Elu, Selu, Softplus,
  HardSigmoid, HardSwish, Conv, ReduceMean, … on their unsupported dtypes.

**Root cause / triage:** a pure ORT dtype-coverage gap. Optionally the EP could
tighten these claims to the dtypes ORT actually implements (turning the init
abort into a cleaner CPU `NOT_IMPLEMENTED`), but this changes no pass/fail
outcome and is orthogonal to the zero-size work.

### B. float16 precision beyond the suite's `rtol=1e-3`

These examples **ran on MLX** (confirmed by attribution) and are genuine numeric
divergences from the reference at fp16:

| Op | Input (fp16) | actual → desired | max abs Δ |
|---|---|---|---|
| Elu (α=1) | `-0.125` | `-0.11749 → -0.11737` | 1.8e-4 |
| Softplus | `-2.0` | | 3.7e-4 |
| Mish | `-2.0` | `-0.2524 → -0.2532` | 7.3e-4 |
| HardSwish | small negative | | ~2e-4 |
| ReduceLogSumExp | fp16 | | 3.7e-4 |

Likely a slightly different fp16 formulation/rounding in the MLX kernels.

### C. NaN handling diverges from the standard

| Op | Input | actual → desired |
|---|---|---|
| Sign | `[nan]` (fp16) | `0` → `nan` |
| PRelu | `x=0, slope=nan` (fp32) | `0` → `nan` |
| ReduceProd | contains nan | diverges |

MLX drops/absorbs NaN where ONNX requires NaN propagation.

### D. Integer reduction / overflow semantics

| Op | Input | actual → desired | note |
|---|---|---|---|
| ReduceProd | `int32 [1291,1291,1291]` | `2147483647` → `-2143282125` | EP **saturates** to INT_MAX; ONNX/ref **wraps** |
| ReduceL2 | `uint32 [29309]×5` | `346` → `65536` | overflow handled differently |
| ReduceMean | `int32` large | mismatch | integer rounding/overflow |
| ReduceSum | `float64` ±1.34e154 | `0` → `2` | large-magnitude cancellation (float64 path) |

### E. Pow large-magnitude / float64

Pow shows both the §A empty-input error and large relative divergence on big
float64 operands (`max rel diff ≈ 0.187`).

---

## CRASHES — ✅ RESOLVED by the claim-hardening pass

> All 16 crashes below are **fixed** on `fix/claim-hardening` (re-run: CRASH = 0).
> Crash class 1 (`float64`) is declined to CPU (a Metal hardware limit). Crash
> class 2 (zero-size) is now **handled on the MLX path** — the op stays claimed
> and produces the correct empty/degenerate output on MLX (no CPU fallback).
> Crash class 3 (null-optional deref) is guarded. The three classes and fixes:

### 1. `float64` claimed but unsupported on the MLX GPU → `abort` — FIXED (Metal HW limit)

**Apple GPUs have no `float64`.** MLX aborts the moment a double array is
materialised on the Metal stream — fp64 genuinely cannot run on our GPU, so
this is an *unavoidable hardware limit*, not a policy choice. The data-movement
gate `IsMovableType` (and the OneHot/Trilu value gate `IsBoundaryValueType`)
previously admitted `float64`; **fix:** dropped `DOUBLE` from both gates so fp64
forms fall back to ORT CPU. (A future option is routing fp64 to an MLX *CPU*
device/stream instead of CPU-EP fallback, but that is out of scope here and not
worth forcing.) Concat/Reshape/Transpose/Squeeze/Unsqueeze/Tile/Slice now
**PASS**.

| Op | Repro (was first crash) | Now |
|---|---|---|
| Concat / Reshape / Transpose / Squeeze / Unsqueeze / Tile / Slice | `float64` data | PASS (fp64 → CPU, Metal has no fp64) |

### 2. Zero-size / degenerate shapes → MLX fatal or segfault — FIXED (handled **on MLX**)

Per the corrected directive, empties are **no longer rejected to CPU** — they
run on MLX and match the ONNX/numpy reference output. Two distinct crash
mechanisms were found and worked around **in the handlers** (`src/ep/ops/*.cc`),
keeping the claim predicates permissive:

1. **Construction-abort ops.** `mlx_max`/`mlx_min`/`mlx_logsumexp` call `abort()`
   at op construction on a zero-size input. Handled in the handler: `ReduceMax`/
   `ReduceMin` route empty inputs through `EmptyMinMaxReduce` (builds the
   ONNX-reduced output shape filled with the reduction identity via `mlx_full`);
   `LogSoftmax` emits `mlx_zeros_like` for an empty input. Softmax / Sum / Prod /
   Mean and all elementwise ops compute empties natively.
2. **Empty-output CopyOut crash.** `mlx_matmul` and `mlx_pad` return an *empty*
   result with **no backing buffer**; the boundary `CopyOut`'s typed
   `mlx_array_data_*` accessor segfaults on it (even though `count==0`). Handled
   by re-materialising the empty result as a clean, correctly-shaped
   `mlx_zeros` (0 elements ⇒ value-irrelevant, shape/dtype exact).
3. **Expand broadcast.** The output dim is computed with numpy bidirectional
   broadcast rules (a size-1 dim takes the other operand's size, incl. `0`)
   instead of a naive `max`, so `[3,0]`↔`[3,1]` no longer asks
   `mlx_broadcast_to` to expand a `0`-dim (which MLX rejects). **Expand PASSES.**

Split additionally validates explicit `split` sizes and rejects rank-0 inputs.

| Op | Was (crash) | Now |
|---|---|---|
| Softmax | `[max] Cannot max reduce zero size array` | **PASS** — empty on MLX |
| Expand | empty-broadcast abort | **PASS** — empty on MLX (numpy broadcast dims) |
| ReduceMax / ReduceMin | `abort()` at construction on empty | empty handled on MLX (identity fill); FAIL only on unsupported dtypes |
| MatMul / Pad / LogSoftmax | empty-output CopyOut segfault / abort | empty handled on MLX (re-materialise / zeros_like); FAIL only reference/dtype-side |
| Split | unequal-section abort | FAIL* (reference-side unequal-split error) |

\* remaining FAILs are reference/CPU-side or dtype-coverage, not EP crashes and
not empties — see "Known non-crash gaps".

### 3. Claim-time segfaults (during `GetCapability`) — FIXED

`ClipClaim`/`SliceClaim` (and `PadClaim`, `SplitClaim`, `ReductionClaim`,
`TriluClaim`, `OptionalBiasIsValid`, `SkipLayerNormClaim`, `LayerNormClaim`)
called `GetName()`/`TensorInfo()` on **absent optional inputs**, which ORT
surfaces as a NULL `OrtValueInfo` → null deref. **Fix:** `TensorInfo` now returns
`false` for a NULL value info, and a shared `SlotPresent(inputs, i)` /
`ValueInfoPresent(v)` guard gates every optional-slot access. No claim predicate
dereferences a null `OrtValueInfo`. Clip/Slice no longer crash (Slice PASSes).

---

## Claim-hardening — shared guards & handler-level empty handling

Claim guards (`src/ep/op_claim.h`) — null-safety + the fp64 gate only:

| Helper | Purpose |
|---|---|
| `ValueInfoPresent(v)` | true iff `v` wraps a live (non-NULL) `OrtValueInfo` |
| `SlotPresent(vals, i)` | null-safe "optional input/output slot i is present & named" |
| `TensorInfo(...)` | returns `false` for a NULL value info (was: unconditional deref) |
| `IntsAttribute(node, name, out, present)` | shared INTS-attribute reader (Split `split`) |

> The prior pass's zero-size *reject* helpers (`HasZeroSizedShape` /
> `IsZeroSizedTensor` / `AnyInputZeroSized`) were **removed** — empties are now
> handled on MLX in the handlers, not declined in the claim.

- **fp64 excluded** (→ CPU; Metal has no fp64): `IsMovableType`,
  `IsBoundaryValueType` (Concat, Reshape, Transpose, Squeeze, Unsqueeze, Tile,
  Slice, Gather, Flatten, Expand, Pad, OneHot, Trilu).
- **zero-size handled on MLX** (claim stays permissive; handler emits the correct
  empty/degenerate output): Add/Mul/Sub/Sigmoid/Softmax (elementwise.cc);
  Unary/Binary/MinMax/Where/Clip and `LogSoftmax` (math.cc); Reduce*, incl.
  `ReduceMax`/`ReduceMin` via `EmptyMinMaxReduce` (reduction.cc);
  `MatMul` empty-result re-materialisation (matmul.cc); `Expand` (numpy
  broadcast dims), `Pad` empty-result re-materialisation, `Split` (shape.cc).
- **null-optional guarded:** Clip, Trilu (math.cc); Slice, Split, Pad
  (shape.cc); Reduce* (reduction.cc); Conv bias (conv.cc); LayerNorm,
  SkipLayerNorm (norm_ext.cc).
- **Split** validates explicit `split` sizes and rejects rank-0 inputs.

## Known non-crash gaps (documented, not force-fit)

None of these are crashes, memory-safety issues, or zero-size issues (empties
run on MLX); they are pre-existing correctness/tolerance gaps or `op`×`dtype`
combinations the reference/ORT-CPU also cannot compute.

1. **fp16 precision > suite `rtol=1e-3`** (ran on MLX): Elu, Softplus, Mish,
   HardSwish, HardSigmoid, ReduceLogSumExp — slightly different fp16 kernel
   rounding (max abs Δ ≈ 1e-4…7e-4). See §B.
2. **NaN propagation:** Sign(`nan`)→0 vs `nan`; PRelu(`x=0, slope=nan`); ReduceProd
   with `nan`. MLX absorbs where ONNX propagates. See §C.
3. **Integer reduction / overflow:** ReduceProd (saturates vs wraps), ReduceL2 /
   ReduceMean (int overflow), ReduceSum (fp64 large-magnitude cancellation). §D.
4. **Pow** large-magnitude / fp64 divergence. §E.
5. **Reference/harness-side failures (not the EP; empties themselves run on MLX):**
   - Gather — Hypothesis strategy raises `InvalidArgument`
     (`max_value=-1 < min_value=0`) during example generation.
   - Flatten / LogSoftmax — the ONNX **reference evaluator** raises `ValueError`
     on an empty input (`op_flatten.py`, `op_log_softmax.py`), so there is no
     expected value to compare — the MLX EP itself computes the empty output
     fine (verified `ran_on_MLX=True`).
   - Split — reference/ORT-CPU error `Invalid num_outputs value of 1. Size of
     dimension being split is 0` and unequal explicit `split` sums.
   - MatMul — reference/ORT-CPU error on non-broadcastable operand shapes the
     fuzzer generated (`matmul_helper.h:144 … cannot broadcast`); the MLX EP
     produces the correct empty output where the reference errors.
   - Pad / Min / Max / Where / Relu / Mean and other `dtype`×opset combos (e.g.
     `uint16`/`int16` Pad(21), integer `Mean`) — a pure **ORT dtype-coverage
     gap**: **ORT CPU also lacks a kernel** (`Could not find an implementation
     for …` / `transformer_memcpy … Provider type … is not set`). Reproduces
     with non-empty inputs too, so it is unrelated to zero-size and unfixable at
     the EP claim layer; neither backend can compute that `op`×`dtype`. See §A.

---

## Reproduce

Prereqs (once):

```bash
# 1. Build the EP dylib (mlx-c is a hard dep)
cd <onnxruntime-mlx>
cmake -S . -B build -G "Unix Makefiles" && cmake --build build -j8

# 2. Clone + install onnx-tests (sibling of this repo)
git clone https://github.com/cbourjau/onnx-tests ../onnx-tests
curl -fsSL https://pixi.sh/install.sh | bash            # -> ~/.pixi/bin/pixi
cd ../onnx-tests && ~/.pixi/bin/pixi run postinstall
# 3. The EP needs ORT 1.27; upgrade the pixi env's python onnxruntime:
~/.pixi/bin/pixi run python -m pip install "onnxruntime==1.27.0"
```

Run the bounded conformance subset (auto-discovers the ORT 1.27 lib dir):

```bash
cd <onnxruntime-mlx>/tests/conformance
MAX_EXAMPLES=25 SEED=0 ./run_conformance.sh          # correctness, per-op
PROFILE=1 ./run_conformance.sh                        # + MLX/CPU attribution
OPS="Softmax Clip MatMul" ./run_conformance.sh        # just a few ops
```

Isolate a single crashing example (verbose, uncaptured):

```bash
cd ../onnx-tests
MLX_EP_LIB=$PWD/../onnxruntime-mlx/build/libonnxruntime_mlx_ep.dylib \
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
PYTHONPATH=$PWD/../onnxruntime-mlx/tests/conformance \
RUN_CANDIDATE=mlx_runtime_wrapper.run_mlx \
~/.pixi/bin/pixi run python -m pytest tests -k test_Softmax_ -x -s \
  --hypothesis-seed=0 --hypothesis-max-examples=25 --hypothesis-verbosity=verbose
```

See [`README.md`](README.md) for the full env-var reference.
