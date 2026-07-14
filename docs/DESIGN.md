# ONNX Runtime MLX Execution Provider — Design

**Status:** Final post-pivot architecture  
**Date:** 2026-07-13  
**Repo:** `onnxruntime-mlx`  
**Runtime sibling:** `onnx-genai`

---

## 0. TL;DR

`onnxruntime-mlx` is an out-of-tree ONNX Runtime plugin Execution Provider for Apple Silicon. It is loaded by a stock ORT build through the plugin-EP C ABI and shipped as:

- CMake target: `onnxruntime_mlx_ep`
- Dylib: `libonnxruntime_mlx_ep.dylib`
- Vendor string: `onnxruntime-mlx`
- Registered EP/device name: **`MetalEP`**

The registered EP name intentionally remains **`MetalEP`** for wire compatibility with `onnx-genai`, which binds by that device/EP name. The repo, vendor string, target, and dylib were renamed from the former `onnxruntime-mps` naming.

The EP no longer runs hand-written Metal compute kernels. It claims only ONNX nodes it can translate to MLX, fuses the claimed decoder subgraph, compiles that subgraph into an MLX plan, and runs the plan through `mlx-c`. MLX is the **sole compute path** for both prefill and decode. Unsupported or deliberately unclaimed ONNX nodes remain on ORT's CPU EP.

The pivot promotes the Phase-0 MLX path to the default architecture: MLX was Pareto-dominant versus the hand kernels, with decode never slower, prefill substantially faster, coherent output, and stable memory. See [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md) for the Phase-0 data.

---

## 1. Motivation and goals

### 1.1 Why MLX is the architecture

The original project tried to close the Apple Silicon performance gap with custom `.metal` kernels for int4 matmul, attention, normalization, softmax, RoPE, elementwise, data movement, and quantized gather. Phase-0 then evaluated an MLX-native path and found it better on the axes that matter for the target decoder workload:

- Decode: **1.02–1.09×** versus the hand-kernel path, never slower.
- Prefill: **2.5–3×** faster than the hand-kernel path.
- Correctness/coherence: stable, coherent generations.
- Memory: allocator memory stayed flat across bounded cycles.

Post-pivot verification on Qwen2.5-0.5B (`qwen2.5-0.5b-cpu-recipe`, M1 Max, warm) showed:

- Build green.
- `ctest` green for `mlx_op_tests`, `mlx_e2e`, and `mlx_leak_test`.
- E2E prompt emits: `The capital of France is Paris`.
- Token stream matches ORT CPU for the first 14 tokens, then exhibits the known fp32 decode drift accepted by the team.
- Prefill/TTFT: 26-token prompt about **15 ms** on MetalEP versus **33 ms** on CPU (~2.2×).
- Prefill/TTFT: 512-token prompt about **165 ms** on MetalEP versus **575 ms** on CPU (~3.3–3.5×).
- Warm decode: about **122–148 tok/s** at short context.
- Leak test: allocator memory flat across bounded cycles.

### 1.2 Goals

1. **MLX-native execution for the fused decoder subgraph.** Translate the supported ONNX decoder hot path into MLX and evaluate at the subgraph boundary.
2. **Zero ORT fork.** Remain an out-of-tree plugin EP loaded by ORT's public plugin-EP C ABI.
3. **Compatibility with `onnx-genai`.** Keep the registered EP name `MetalEP` while renaming the repo/vendor/artifact to MLX.
4. **Deliberate CPU fallback.** Claim only ops whose exact dtype/attribute/layout contract is implemented by the MLX translator.
5. **Stable KV-cache handoff.** Preserve the ORT IoBinding / KV-cache contract across prefill and decode.
6. **Coherent output before performance claims.** Keep the accepted fp32 drift bounded and documented.

### 1.3 Non-goals

- General ONNX opset completeness.
- A hand-written Metal kernel fallback.
- A build mode without MLX.
- Reintroducing the removed `src/ops/` registry scaffold or dtype/MSL specialization layer.
- Training or non-Apple GPU support.

---

## 2. Plugin-EP integration architecture

### 2.1 Public ORT plugin-EP ABI

ORT exposes a public C ABI for registering an out-of-tree EP as a shared library. The library exports the two plugin entry points ORT loads with `dlsym`:

```c
OrtStatus* CreateEpFactories(const char* registered_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

The EP continues to use this ABI; no ORT fork or ORT rebuild is required.

### 2.2 EP identity and compatibility

| Field | Current value | Notes |
|---|---|---|
| Repository | `onnxruntime-mlx` | Formerly `onnxruntime-mps`. |
| Vendor string | `onnxruntime-mlx` | Replaces the old vendor string. |
| CMake target | `onnxruntime_mlx_ep` | Current CMake target name. |
| Dylib | `libonnxruntime_mlx_ep.dylib` | Current plugin artifact loaded by ORT at runtime. |
| Registered EP name | **`MetalEP`** | Intentionally unchanged for `onnx-genai` compatibility. |

Do not rename the registered EP/device name unless `onnx-genai` changes its binding contract.

### 2.3 Runtime objects and responsibilities

| Object / file | Responsibility |
|---|---|
| EP factory | Exports the plugin factory, reports the Apple GPU device metadata, creates the per-session EP, and sets the ORT API compatibility fields. |
| `src/ep/ep.cc` | Owns subgraph support checks, claim/fuse decisions, and `Compile`. During `Compile`, it builds the ONNX→MLX node-descriptor plan for the claimed subgraph. |
| `src/ep/mlx_backend.{h,cc}` | Owns MLX graph construction/execution details for the compiled plan and calls into `mlx-c`. |
| `src/ep/metal_context.{h,mm}` | Minimal Metal allocator / `MTLBuffer` bridge used for ORT device I/O and KV-cache binding only. It no longer owns shader compilation, a Metal library, pipeline states, or kernel encoding. |
| ORT allocator / data transfer | Provides the device-buffer surface ORT and `onnx-genai` bind to. Apple unified memory keeps CPU/GPU movement cheap, but compute is performed by MLX. |

### 2.4 Claim → fuse → compile → run

The EP is compile-based at the fused subgraph boundary:

1. **Claim.** `GetCapability` inspects ONNX nodes and claims only the exact ops, domains, dtypes, attributes, and input forms listed in §3.
2. **Fuse.** ORT gives the EP a supported partition for the claimed nodes. Unclaimed nodes remain assigned to CPU.
3. **Compile.** `src/ep/ep.cc` converts the ONNX nodes in that partition into a compact MLX-oriented plan. Constants and live initializers are described but not repeatedly repacked.
4. **Run.** `src/ep/mlx_backend.{h,cc}` materializes/runs the MLX graph through `mlx-c`, evaluates once at the subgraph boundary, and writes outputs back to the ORT tensors expected by the session.

The boundary is intentionally coarse: instead of ORT driving many tiny custom kernels, the EP gives MLX the fused decoder region and performs **one `mlx_eval` at the subgraph boundary**.

### 2.5 CPU fallback and partitioning contract

The claim set is a subset of the MLX translator's implemented set. If an op is not in the table below, or if its dtype/attributes/input form do not match the listed contract, the EP does **not** claim it. ORT then runs it on the CPU EP and inserts any required partition copies.

This is a correctness feature, not a gap: unsupported graph pieces degrade to CPU instead of failing or taking an unimplemented path.

### 2.6 KV cache and IoBinding

The runtime-owned KV cache and ORT IoBinding design remain central:

- The GQA translator writes present K/V back to the same ORT context outputs used by the runtime.
- The layout is `[B, kv_heads, total_seq, head]` in `fp32`.
- Prefill and decode share the same layout and position convention, so the prefill→decode handoff is continuous.
- `metal_context` exists to allocate/bind the `MTLBuffer` surfaces ORT and the runtime expect; it is not a compute subsystem.

---

## 3. ONNX ops translated to MLX

The EP claims only the following ONNX op forms. All other ops remain on CPU.

| ONNX op | Domain | Claimed form | MLX target(s) | Notes |
|---|---|---|---|---|
| **MatMulNBits** | `com.microsoft` | int4 block quantized weights, `bits=4`, `block_size=32` | `mlx_quantized_matmul` | Packed uint8 int4 weights are repacked once to MLX affine-quant format and cached persistently on the plan. |
| **GroupQueryAttention** | `com.microsoft` | 9-input separate-QKV form; matching `fp32`/`fp16`/`bf16` Q/K/V/past_k/past_v/cos/sin; `int32` `seqlens_k`/`total_seq`; RoPE applied in-op | `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope` | Writes present K/V to the same ORT ctx outputs in `[B, kv_heads, total_seq, head]` native-float layout. |
| **RMSNormalization** | `ai.onnx` | `axis=-1`, `fp32` | `mlx_fast_rms_norm` | Gamma is cached from live context data on first run. |
| **SkipSimplifiedLayerNormalization** | `com.microsoft` | `fp32` input/skip/gamma | skip-add + `mlx_fast_rms_norm` | Preserves the residual+RMS behavior expected by the decoder graph. |
| **GatherBlockQuantized** | `com.microsoft` | SYMMETRIC int4 embedding only, 3-input form, `zp=8` | gather + int4 dequant | The asymmetric 4-input `zero_points` form is intentionally not claimed and falls back to CPU. MLX `zero_points` support is a follow-up. |
| **Softmax** | `ai.onnx` | last-axis, `fp32` | `mlx_softmax` | Standalone softmax; attention-internal softmax is handled by MLX fast attention. |
| **Add** | `ai.onnx` | `fp32` and `fp16` | MLX elementwise add | Claimed only for supported floating dtypes. |
| **Mul** | `ai.onnx` | floating point | MLX elementwise multiply | Claimed only for supported floating dtypes. |
| **Sub** | `ai.onnx` | floating point and `int64` | MLX elementwise subtract | `int64` is kept for position/bookkeeping forms the translator handles. |
| **Sigmoid** | `ai.onnx` | floating point | MLX elementwise sigmoid | No standalone SiLU/Swish claim. |
| **Cast** | `ai.onnx` | `fp32`↔`fp16`, `int64`→`int32` | MLX cast | Other casts remain on CPU. |

### 3.1 Formerly claimed ops now left on CPU

The hand-kernel era claimed or planned several ops that the current MLX translator does **not** translate. These are no longer claimed and run on ORT CPU unless/until an MLX translation is added:

- `Div`
- `SiLU`
- `Swish`
- `Gelu`
- standalone `RotaryEmbedding`
- `Reshape`
- `Transpose`
- `Concat`

The important distinction is that RoPE inside `GroupQueryAttention` is still translated through `mlx_fast_rope`; the standalone ONNX `RotaryEmbedding` op is not.

### 3.2 Constant and initializer caching

The compiled plan caches immutable or effectively constant data instead of rebuilding it every token:

- `MatMulNBits` uint8 int4 weights are repacked once to MLX affine-quant format.
- Cos/sin tables, gammas, embedding tables, and biases are cached once from live context data on first `Run`.
- The cache belongs to the compiled plan so prefill and decode reuse the same converted data.

---

## 4. Repository structure

Current high-level structure relevant to the EP:

```text
onnxruntime-mlx/
├── docs/
│   ├── DESIGN.md
│   ├── OP_ARCHITECTURE.md
│   └── MLX_EVALUATION.md
├── src/
│   └── ep/
│       ├── ep.cc                  # claim/fuse/compile and ONNX→MLX NodeDesc build
│       ├── mlx_backend.h          # MLX backend interface
│       ├── mlx_backend.cc         # mlx-c graph execution backend
│       ├── metal_context.h        # minimal MTLBuffer allocator bridge
│       └── metal_context.mm
├── tests/
│   ├── ops/
│   │   └── mlx_op_test.py         # op-correctness tests
│   └── e2e/                       # coherence and leak tests
└── CMakeLists.txt
```

Removed historical paths are listed in §8.

---

## 5. Build system

`mlx-c` is a **hard build dependency**:

```sh
brew install mlx-c
```

CMake configure fails if `mlx-c` is absent. There is no build flag to disable MLX and no hand-kernel fallback.

Build outputs and references must use the MLX names:

- Target: `onnxruntime_mlx_ep`
- Dylib: `libonnxruntime_mlx_ep.dylib`

The build links the ORT plugin-EP ABI surface plus `mlx-c`. It must not reference removed Metal shader compilation machinery, precompiled `.metallib` resources, or MLX opt-in/feature flags from the old transition period.

---

## 6. Testing and validation

Docs-only edits do not require test execution, but the architecture is validated by the following existing tests:

| Test area | Location / name | Purpose |
|---|---|---|
| Op correctness | `tests/ops/mlx_op_test.py` / `mlx_op_tests` | Verifies each claimed ONNX→MLX translation against ORT CPU expectations. |
| E2E coherence | `tests/e2e/` / `mlx_e2e` | Runs the model through the plugin EP and verifies coherent generation. |
| Leak stability | `tests/e2e/` / `mlx_leak_test` | Ensures bounded repeated runs do not grow allocator memory. |

Current post-pivot validation on Qwen2.5-0.5B:

- `ctest`: 3/3 green (`mlx_op_tests`, `mlx_e2e`, `mlx_leak_test`).
- E2E text: `The capital of France is Paris`.
- CPU token stream match: first 14 tokens; known fp32 drift afterward is accepted.
- Prefill/TTFT: ~15 ms vs ~33 ms CPU for a 26-token prompt.
- Prefill/TTFT: ~165 ms vs ~575 ms CPU for a 512-token prompt.
- Warm decode: ~122–148 tok/s at short context.
- Leak test: flat allocator memory across bounded cycles.

---

## 7. Risks and follow-ups

| Risk / follow-up | Impact | Mitigation |
|---|---|---|
| `mlx-c` availability/version | Configure-time blocker | Treat `mlx-c` as a hard dependency and fail early with a clear CMake error. |
| Plugin-EP ABI compatibility | Runtime load failure if ORT ABI changes | Continue targeting the ORT plugin ABI used by `onnx-genai`; keep exported symbols stable. |
| Deliberate CPU fallback | More CPU partitions if a model emits unsupported forms | Keep the claim predicates exact; add MLX translations only with tests. |
| `GatherBlockQuantized` asymmetric zero-points | 4-input form falls back to CPU | Track MLX zero-points support as a follow-up. |
| Accepted fp32 decode drift | Token stream diverges after the known prefix | Preserve the coherence gate and compare drift against the accepted baseline. |

---

## 8. Removed / historical hand-kernel era

The previous architecture used custom Metal compute kernels and host-side kernel dispatch scaffolding. That era has ended. The following were removed and must not be referenced as active architecture:

- `src/kernels/*.metal`, including matmulnbits, GQA, norm, softmax, RoPE, elementwise, data-movement, and gather-block-quantized shaders.
- Metal shader compile, registry, and encode machinery formerly in `src/ep/metal_context.{h,mm}`.
- `cmake/metal_kernels.inc.in`.
- `src/ops/` op-registry scaffold.
- `src/dtype/dtype_traits.h` dtype/MSL-specialization scaffold.
- The old `onnxruntime_mps_ep` target and `libonnxruntime_mps_ep.dylib` artifact.
- Transitional MLX/Metal feature flags and hand-kernel fallback paths.

Historical docs and comments should point readers to this section and to [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md) rather than describing the removed hand-kernel plan as current.

---

## 9. References

- [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md) — Phase-0 MLX-vs-hand-kernel evaluation and pivot justification.
- `src/ep/ep.cc` — subgraph claim logic and ONNX→MLX compile plan construction.
- `src/ep/mlx_backend.{h,cc}` — MLX backend and `mlx-c` execution path.
- `src/ep/metal_context.{h,mm}` — minimal `MTLBuffer` allocator / ORT IoBinding bridge.
- ORT plugin-EP C ABI: `onnxruntime_ep_c_api.h`, `RegisterExecutionProviderLibrary`, `SessionOptionsAppendExecutionProvider_V2`, `CreateEpFactories`, `ReleaseEpFactory`.
- MLX / `mlx-c`: <https://github.com/ml-explore/mlx>.
