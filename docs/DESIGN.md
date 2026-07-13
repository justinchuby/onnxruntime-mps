# ONNX Runtime Metal/MPS Execution Provider — Design

**Status:** Draft / Plan (Phase 0)
**Author:** Roy (Lead/Architect)
**Date:** 2026-07-12
**Requested by:** Justin Chu
**Repo:** `onnxruntime-mps`
**Siblings:** `mobius` (ONNX model builder), `onnx-genai` (Rust ORT runtime — our test harness)

---

## 0. TL;DR

Build an **out-of-tree ("plugin") ONNX Runtime Execution Provider** for **Apple Metal/MPS**, distributed as a
standalone `libonnxruntime_mps_ep.dylib` that a stock prebuilt `libonnxruntime.dylib` (ORT 1.27) loads at runtime
via `RegisterExecutionProviderLibrary` — **no ORT fork, no ORT rebuild**. It supplies hand-tuned Metal kernels for
the exact operator set our Qwen-class decoder models emit (`MatMulNBits` int4, `GroupQueryAttention`,
`GatherBlockQuantized`, `SkipSimplifiedLayerNormalization`, `RMSNormalization`, and the standard elementwise/shape
ops), with CPU fallback for everything else. Goal: **be the fastest ORT EP on Apple Silicon and close/beat
llama.cpp-Metal (LM Studio / Foundry Local) on Mac**, which our own benchmarks show ORT's generic CPU/WebGPU int4
kernels currently trail — especially long-context.

We validate end-to-end through **onnx-genai** by adding an `ONNX_GENAI_EP=metal` path that registers this plugin
library and runs the existing Qwen2.5-0.5B packages already in `onnx-genai/models/`.

> **This document is a plan.** No kernel or EP implementation code ships in this phase. Phases 1–3 below assign the
> implementation to Nabil (EP integration), Mariette (compute kernels), Coco (data/quant kernels), and Freysa
> (perf/test).

---

## 1. Motivation & Goals

### 1.1 Why

onnx-genai has, through a long correctness+perf campaign, proven it is **ORT-kernel-bound on Apple Silicon**:

- The correctness-verified Mobius CPU recipe (169 int4 `MatMulNBits` at `accuracy_level=4`, quantized
  embedding/head) reaches ~158/115 tok/s (short/long) — it **beats Ollama CPU and edges LM Studio CPU short**, but
  **trails LM Studio by ~28% long** and trails **Foundry Local** (an ORT stack with a better-packed 121-node int4
  graph) in both regimes. (See `onnx-genai/.squad/decisions.md`, 2026-07-12 "Treat the full CPU recipe as
  competitive, not a universal llama.cpp win".)
- The WebGPU path is coherent but **KV round-trips host↔device every step** (CPU-allocated shared KV), capping
  decode at ~19–21 tok/s — below the CPU EP. Making KV device-resident on ORT 1.27 WebGPU **SIGSEGVs** during
  multi-step decode (see decisions.md "Device-resident GQA KV … blocked by ORT 1.27 WebGPU SIGSEGV").
- llama.cpp's Metal backend uses **hand-tuned int4 matvec with int8 accumulation** and fused attention; that is the
  gap we cannot close with ORT's generic CPU/WebGPU kernels.

The plugin-EP ABI (below) lets us drop in **our own Metal kernels** under the same ORT graph/session our runtime
already drives — without maintaining an ORT fork.

### 1.2 Goals

1. **G1 — Fastest ORT EP on Mac.** Beat ORT CPU and ORT WebGPU decode+prefill on Apple Silicon for our models.
2. **G2 — Beat llama.cpp-Metal class runtimes** (LM Studio, Foundry Local) on decode throughput and TTFT for
   Qwen2.5-0.5B, short and long context, at parity quantization (Q4 int4 block-32).
3. **G3 — Zero ORT fork.** Ship a plugin `.dylib` loadable against the **stock prebuilt ORT 1.27** that onnx-genai
   already links (`ort-sys` downloads `onnxruntime-osx-arm64-1.27.0`).
4. **G4 — Cover every op our models emit**, with correct CPU fallback so any graph still runs.
5. **G5 — Numerically faithful.** fp16 compute with int8-accumulated int4 matmul must stay coherent and within a
   bounded tolerance of the ORT CPU reference.

### 1.3 Non-goals (this initiative)

- Training, or non-Apple GPUs.
- General ONNX opset completeness beyond what onnx-genai / Mobius emit (we fall back to CPU for the rest).
- Replacing CoreML EP (CoreML is a black box; we want hand-tuned control — this is complementary).

---

## 2. ORT Plugin-EP Integration Architecture

### 2.1 The plugin-EP C ABI (confirmed surface & version)

ORT exposes a **public, additive C ABI** for registering an out-of-tree EP as a shared library. It was **introduced
in ORT 1.22** (`ORT_API_VERSION 22`) and extended through 1.23–1.27. Our target, **ORT 1.27 (`ORT_API_VERSION 27`)**
— the exact version `onnx-genai/crates/onnx-genai-ort/ort-sys/build.rs` pins (`ORT_VERSION = "1.27.0"`,
`ORT_API_VERSION = "27"`, downloading `onnxruntime-osx-arm64-1.27.0.tgz`) — **contains the full plugin-EP ABI in the
stock prebuilt binary. No ORT rebuild is required.**

**Primary header:** `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` (auto-included at the end of
`onnxruntime_c_api.h`).

**Sources (microsoft/onnxruntime, tag/branch as noted):**
- `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` — the ABI (structs & `OrtEpApi`).
- `include/onnxruntime/core/session/onnxruntime_c_api.h` — `#define ORT_API_VERSION 27` at `v1.27.x`.
- `onnxruntime/core/session/plugin_ep/ep_library_plugin.{h,cc}` — ORT-internal `dlopen`/`dlsym` loader.
- `onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc` — bridges plugin EP into ORT's provider system.
- `onnxruntime/core/session/abi_ep_types.h` — `OrtEpGraphSupportInfo` internal struct.
- `onnxruntime/core/session/ort_apis.h` — `RegisterExecutionProviderLibrary`, `SessionOptionsAppendExecutionProvider_V2`, `GetEpApi`.
- **Reference implementations:**
  - `onnxruntime/test/autoep/library/example_plugin_ep/` — canonical compile-based example (entry point, factory, EP).
  - `onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/` — kernel-registry-based example.
  - `onnxruntime/test/autoep/library/example_plugin_ep_virt_gpu/` — stream-aware virtual-GPU example.
  - `onnxruntime/core/providers/cuda/plugin/cuda_plugin_ep.cc` + `docs/cuda_plugin_ep/cuda_plugin_ep_design.md` — production plugin EP + design doc; **our closest architectural analog**.

### 2.2 Required exported symbols (the `.dylib` contract)

The plugin library **must export exactly two C symbols** (with `__attribute__((visibility("default")))` on macOS, or
`dlsym` will not find them):

```c
// typedef CreateEpApiFactoriesFn  (\since 1.22)
OrtStatus* CreateEpFactories(const char* registered_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories,
                             size_t* num_factories);

// typedef ReleaseEpApiFactoryFn   (\since 1.22)
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

ORT loads them via `Env::GetSymbolFromLibrary(handle, "CreateEpFactories", …)` in `ep_library_plugin.cc`.

### 2.3 Objects & responsibilities

| Object | Since | Our implementation |
|---|---|---|
| `OrtEpFactory` | 1.22 | `MetalEpFactory` — reports the Apple GPU as an `OrtEpDevice`, creates `OrtEp` per session, owns the `MTLDevice`, `MTLCommandQueue`, compiled `MTLLibrary` (kernel pipeline states), the device allocator, and the data-transfer impl. Must set `ort_version_supported = ORT_API_VERSION`. |
| `OrtEpDevice` | 1.22 | Created via `OrtEpApi::CreateEpDevice`; carries EP metadata + allocator info (`EpDevice_AddAllocatorInfo`). Because Apple Metal may not surface as an enumerable `OrtHardwareDevice`, we synthesize one with `OrtEpApi::CreateHardwareDevice(GPU, vendor=Apple, …)` **(open question O3)**. |
| `OrtEp` | 1.22 | `MetalEp` — per-session. Implements `GetName`, `GetCapability` (partitioning), and **either** `GetKernelRegistry` (kernel-based, preferred) **or** `Compile`+`OrtNodeComputeInfo` (compile-based). |
| `OrtEpGraphSupportInfo` | 1.23 | Filled in `GetCapability` via `EpGraphSupportInfo_AddSingleNode` (kernel path) or `EpGraphSupportInfo_AddNodesToFuse` (fusion path). |
| `OrtNodeComputeInfo` | 1.23 | Only if compile-based: `CreateState`/`Compute`/`ReleaseState` for each fused subgraph. |
| `OrtAllocator` (via factory `CreateAllocator`) | 1.23 | Metal buffer allocator (`MTLBuffer` pool, `MTLResourceStorageModeShared` on unified memory). |
| `OrtDataTransferImpl` (via factory `CreateDataTransfer`) | 1.23 | Host↔device copy. On Apple unified memory this is largely a `memcpy`/zero-copy against shared-storage `MTLBuffer`s; blit-encoder for private storage. |
| `OrtSyncStreamImpl` | 1.23 | Optional. A stream = `MTLCommandQueue`; `GetHandle` returns the queue, `Flush` commits. We start **not** stream-aware (`IsStreamAware=false`) and add it in Phase 3 if it wins. |

### 2.4 Kernel-based vs compile-based: **decision**

ORT plugin EPs can be **kernel-based** (register per-op kernels in an `OrtKernelRegistry`, `AddSingleNode` in
`GetCapability`; ORT drives node-by-node) or **compile-based** (fuse a subgraph, hand back one `Compute`). 

**Decision: start kernel-based (Phase 1–2), evaluate a fused-attention compile path in Phase 3.**

- Kernel-based matches ORT's own CUDA/WebGPU model, gives clean per-op correctness testing against the ORT CPU
  reference, and lets partitioning + CPU fallback fall out of `AddSingleNode` for exactly the ops we support.
- The one place fusion clearly pays off is the **decode attention block** (RoPE + GQA + KV update); however
  `GroupQueryAttention` is **already a fused meta-op** in our graphs, so a single hand-tuned GQA kernel captures most
  of that benefit without graph fusion. We keep compile-based fusion (e.g. `MatMulNBits`→`SwiGLU`) as a Phase-3
  perf lever, gated behind measured wins.

Kernel-based requires ORT `\since 1.24` (`OrtKernelImpl`, `OrtKernelRegistry`, `GetKernelRegistry`,
`KernelRegistry_AddKernel`) — satisfied by 1.27.

### 2.5 Graph partitioning & CPU fallback

In `MetalEp::GetCapability(graph, graph_support_info)` we iterate nodes and `AddSingleNode` **only** for
(op_type, domain, dtype) tuples we have a registered kernel for (§4 coverage table). Every other node is left
unclaimed → ORT assigns it to the CPU EP automatically and inserts the necessary `MemcpyToHost`/`MemcpyFromHost`
copy nodes across the partition boundary. This guarantees **any** graph runs; unsupported ops degrade to CPU, not
to failure. We log the resulting partition (nodes-on-Metal vs nodes-on-CPU vs copy-node count) exactly as onnx-genai
already logs for WebGPU, so we can watch the partition tighten as coverage grows.

### 2.6 Allocators & data transfer on unified memory

Apple Silicon has **unified memory**: CPU and GPU share physical RAM. We exploit this:

- Allocator: pool of `MTLBuffer`s created with `MTLResourceStorageModeShared`, so the same pages are addressable by
  CPU and GPU with no copy. `OrtMemoryInfo` registered as a `GPU`/`DEFAULT` device named e.g. `"MetalEP_Buffer"`
  (mirrors how onnx-genai already handles the WebGPU `"WebGPU_Buffer"` allocator in
  `crates/onnx-genai-ort/src/allocator.rs`).
- Data transfer: `CanCopy` returns true for host↔MetalEP and MetalEP↔MetalEP; `CopyTensors` is a `memcpy` for
  shared-storage buffers (or a blit-encoder for any private-storage buffers we later introduce for perf). Unified
  memory is the key reason a Metal EP can beat WebGPU here — **the KV-cache host↔device round-trips that cap our
  WebGPU decode simply do not exist** if KV lives in a shared `MTLBuffer`.

### 2.7 Objective-C++ ↔ Metal bridging

- The ABI callbacks are plain C. Files that touch Metal are compiled as **Objective-C++ (`.mm`)**; pure ABI glue is
  C++ (`.cc`). Kernel sources are `.metal`, compiled to a `.metallib` (embedded in the dylib or loaded from a known
  path) and turned into `MTLComputePipelineState`s at factory init.
- A thin `metal_context` (Obj-C++) owns `id<MTLDevice>`, `id<MTLCommandQueue>`, the `id<MTLLibrary>`, and a
  name→`MTLComputePipelineState` map. C++ kernel-dispatch code calls into it through a C++-facing header that hides
  all Obj-C types (PIMPL), so the rest of the EP stays in portable C++.

---

## 3. Operator Coverage (derived from our actual models)

Enumerated from the graphs onnx-genai actually runs (`onnx-genai/models/qwen2.5-0.5b-cpu-recipe/model.onnx`,
`…/qwen2.5-0.5b-q4-gqa-webgpu`, `…/qwen2.5-0.5b-cuda`) and Mobius emission
(`mobius/src/mobius/components/`). The Qwen2.5-0.5B decode graph is **394 nodes**:

```
169  com.microsoft   MatMulNBits                       (int4, block_size=32, accuracy_level=4)
 72  ai.onnx         Add
 48  com.microsoft   SkipSimplifiedLayerNormalization  (RMS + residual, fused)
 48  ai.onnx         Mul
 24  com.microsoft   GroupQueryAttention               (num_heads=14, kv_num_heads=2, do_rotary=1)
 24  ai.onnx         Sigmoid                            (SwiGLU: Sigmoid∘Mul)
  2  ai.onnx         Cast
  1  com.microsoft   GatherBlockQuantized               (int4, block_size=32 — quantized embedding)
  1  ai.onnx         RMSNormalization
  1  ai.onnx         ReduceSum / Sub / Shape / Gather / Constant  (seqlen/position bookkeeping)
```

Broader Mobius families additionally emit: `RotaryEmbedding` (standalone, when RoPE is not folded into GQA),
`Attention` / `PackedMultiHeadAttention`, `LayerNormalization`, `Gelu`, `Softplus`, `Where`, `Split`, `Concat`,
`Transpose`, `Reshape`, `Expand`, `Conv`, `Slice`, `Unsqueeze`, `CumSum`. These are the second-tier target set.

### 3.1 Coverage table (priority = decode hot path first)

Confirmed op attributes (from `qwen2.5-0.5b-cpu-recipe/model.onnx`):
- `MatMulNBits`: `K=896, N=896, bits=4, block_size=32, accuracy_level=4`, inputs `(A, B_packed_uint8, scales)` (symmetric, no zero-point input in this graph).
- `GroupQueryAttention`: `num_heads=14, kv_num_heads=2, scale=0.125, do_rotary=1, rotary_interleaved=0`, 9 inputs `(Q,K,V, past_key, past_value, seqlens_k, total_seq_len, cos_cache, sin_cache)`, 3 outputs `(out, present_key, present_value)`.
- `GatherBlockQuantized`: `bits=4, block_size=32, gather_axis=0, quantize_axis=1`, inputs `(data_packed, indices, scales)`.
- `SkipSimplifiedLayerNormalization`: `epsilon`, 3 inputs `(input, skip, gamma)`, 4 outputs.
- `RMSNormalization`: `epsilon, axis=-1`, 2 inputs.

| Op | Domain | Priority | Kernel strategy | Reference impl |
|---|---|---|---|---|
| **MatMulNBits** (int4, blk32, acc4) | com.microsoft | **P0** | Custom `.metal`: int4 dequant → **int8×int8 matmul with int32 accumulate**, fp16 I/O; per-32 block scales. Decode = tall-skinny **GEMV** (M=1) simdgroup-reduction kernel; prefill = tiled **GEMM** (`simdgroup_matrix`). Two specializations. | llama.cpp `ggml-metal` `mul_mv_q4/q8` + `mul_mm`; ExecuTorch `backends/apple/mps`; PyTorch `aten/src/ATen/native/mps` (int4mm/`_weight_int4pack_mm`). |
| **GroupQueryAttention** (+RoPE, KV) | com.microsoft | **P0** | Custom fused attention: apply RoPE to Q/K (interleaved=0), append K/V to KV cache, causal (+ optional sliding-window) flash-style online-softmax attention, GQA head broadcast (14 q-heads → 2 kv-heads). fp16 compute, fp32 softmax accumulators. | ExecuTorch MPS SDPA; PyTorch MPS SDPA / `MPSGraph` scaled-dot-product; llama.cpp `flash_attn` Metal. |
| **RoPE** (standalone) | com.microsoft/ai.onnx | **P0** | Custom `.metal`: rotate pairs using cos/sin cache; interleaved & non-interleaved variants. Also inlined into GQA. | PyTorch MPS; ExecuTorch RoPE. |
| **RMSNormalization** | ai.onnx | **P0** | Custom `.metal`: simdgroup reduction for sum-of-squares, fp32 accumulate, fp16 out; `axis=-1`. | ExecuTorch/PyTorch MPS layer/rms norm; llama.cpp `rms_norm`. |
| **SkipSimplifiedLayerNormalization** | com.microsoft | **P0** | Custom `.metal`: fused `(input+skip)` residual → RMS-norm × gamma; emits normed output **and** the residual sum (4 outputs). | Same as RMSNorm; fuse residual add. |
| **GatherBlockQuantized** (int4 embed) | com.microsoft | **P1** | Custom `.metal`: gather rows from packed int4 table, dequant per-block (block_size=32, quantize_axis=1) to fp16. | Reference impl in ORT contrib; llama.cpp `get_rows_q4`. |
| **MatMul** (fp16) | ai.onnx | **P1** | `simdgroup_matrix` GEMM, or `MPSMatrixMultiplication`/MPSGraph for correctness-first. | PyTorch MPS `mm`; MPS `MPSMatrixMultiplication`. |
| **Add / Mul / Sub** | ai.onnx | **P1** | Elementwise `.metal` with broadcasting. | trivial; PyTorch MPS binary ops. |
| **Sigmoid / SiLU(=x·σ(x))** | ai.onnx | **P1** | Elementwise `.metal`; fuse `Sigmoid`+`Mul` → SiLU when adjacent. | PyTorch MPS activations. |
| **Cast** | ai.onnx | **P1** | Elementwise dtype convert. | trivial. |
| **Softmax** | ai.onnx | **P1** | simdgroup online softmax (also embedded in GQA). | PyTorch MPS softmax; llama.cpp. |
| **Gather / Concat / Reshape / Transpose / Shape / Unsqueeze / Constant / ReduceSum** | ai.onnx | **P2** | Mostly shape/index bookkeeping; cheap `.metal` or leave on CPU if not hot. | PyTorch MPS. |
| **RotaryEmbedding / LayerNormalization / Gelu / Softplus / Where / Split / Expand / Slice / Conv / CumSum / Attention / PackedMHA** | mixed | **P2/P3** | Broader Mobius coverage; add as models demand. Conv/Attention only when a model that needs them enters scope. | PyTorch/ExecuTorch MPS. |

**Fallback:** anything not in this table stays on the ORT CPU EP (§2.5).

---

## 4. Hot-op Kernel Design

Common conventions: **fp16 storage/compute, fp32 reduction accumulators**; threadgroup = one output tile;
simdgroup (32 lanes) reductions; exploit unified memory (no explicit staging copies). Weight layout is chosen to
match Mobius's packing so no re-pack is needed at load.

### 4.1 MatMulNBits (int4 → int8, block-32, accuracy_level=4)

- **Layout:** `B` is packed uint8 (two int4 per byte), `N×(K/block)` blocks, one fp16 scale per block (symmetric;
  zero-point absent in our graph — support the optional zero-point input for generality). `accuracy_level=4` ⇒ ORT
  contract is **int8 matmul accumulation**: quantize the fp16 activation row to int8 per group, do int8×int4→int32
  MAC, then scale.
- **Decode (M=1) — GEMV:** one threadgroup per output tile of N; each simdgroup computes a strip of N. Loop over K
  in blocks of 32: load 32 packed int4 weights, unpack to int8, load the int8-quantized 32-activation block,
  `dot`-accumulate into int32, then `+= int32 * (act_scale * w_scale)` in fp32. This is the llama.cpp `mul_mv_q4`
  pattern. **This is the single most important kernel** (169 nodes; dominates decode).
- **Prefill (M>1) — GEMM:** tiled `simdgroup_matrix<half>` (8×8) GEMM; dequant int4→fp16 tiles into threadgroup
  memory, MMA-accumulate. Optionally an int8 MMA path if measured faster. Matches llama.cpp `mul_mm`.
- **Refs:** llama.cpp `ggml/src/ggml-metal/ggml-metal.metal` (`kernel_mul_mv_q4_0`, `kernel_mul_mm`); PyTorch
  `aten/src/ATen/native/mps/kernels/…` `int4pack_mm`; ExecuTorch `backends/apple/mps`.
- **Numerics:** int8-accumulated int4 with per-32 scales matches `accuracy_level=4`; validate ≤ small rel-error vs
  ORT CPU MatMulNBits. Guard against fp16 overflow in long-K reductions by fp32 block accumulation.

### 4.2 GroupQueryAttention (fused RoPE + KV cache + causal/sliding-window)

- **Inputs:** Q/K/V (possibly packed), `past_key`/`past_value`, `seqlens_k`, `total_seq_len`, `cos_cache`,
  `sin_cache`. **Outputs:** attention out, `present_key`, `present_value`.
- **Steps in one (or two) kernels:**
  1. **RoPE** on Q and K (`rotary_interleaved=0`) using cos/sin cache.
  2. **KV append:** write new K/V at the correct cache offset from `seqlens_k`; `present_*` shares the past buffer
     (in-place, past==present share-buffer) — the runtime owns a max-length KV buffer (matches onnx-genai's
     runtime-owned GQA share-buffer contract, decisions.md "batty-inference-metadata-gqa").
  3. **Attention:** flash-style **online softmax** (running max/sum, fp32) over the causal window; **GQA head
     broadcast** 14 q-heads → 2 kv-heads (each kv-head serves 7 q-heads); optional **sliding-window** mask.
  - `scale=0.125` (=1/sqrt(head_dim=64)).
- **Decode (q_len=1):** one threadgroup per (q-head); stream KV tiles from the cache, online-softmax accumulate;
  ideal for unified memory since KV never leaves RAM. **Prefill:** tiled flash-attention over q_len×kv_len.
- **Refs:** ExecuTorch MPS SDPA (`backends/apple/mps`), PyTorch MPS SDPA / `MPSGraph` scaledDotProductAttention,
  llama.cpp Metal `flash_attn_ext`.

### 4.3 RoPE (standalone)

- Pairwise rotation from cos/sin cache; interleaved and non-interleaved variants; one thread per (token, head,
  dim-pair). Trivial, memory-bound. Shared with the GQA-inlined version.

### 4.4 RMSNormalization / SkipSimplifiedLayerNormalization

- One threadgroup per row (`axis=-1`, D=896). simdgroup reduction of Σx² in fp32 → `rms = rsqrt(mean + eps)` →
  `y = x * rms * gamma` in fp16. **Skip variant** first computes `residual = input + skip`, emits both the
  normalized output and `residual` (used by the next block), matching the 4-output contract. `epsilon` from attr.
- **Refs:** llama.cpp `kernel_rms_norm`; PyTorch/ExecuTorch MPS norm kernels.

### 4.5 Softmax

- Online/streaming simdgroup softmax, fp32 max+sum accumulators, fp16 out. Standalone kernel plus the inlined
  attention version.

### 4.6 GatherBlockQuantized (int4 embedding)

- Gather rows selected by `indices` from the packed int4 table (`gather_axis=0`), dequantize per block
  (`block_size=32`, `quantize_axis=1`) with per-block fp16 scale → fp16 rows. One threadgroup per gathered row.
- **Refs:** ORT contrib reference for correctness; llama.cpp `kernel_get_rows_q4_0`.

### 4.7 Elementwise (Add/Mul/Sub/Sigmoid/SiLU/Cast)

- Grid-stride elementwise with NumPy-style broadcasting; **fuse `Sigmoid`+`Mul` → SiLU/SwiGLU** when adjacent (the
  `Sigmoid`(24)/`Mul`(48) pattern is the MLP gate). fp16 compute.

---

## 5. Repository Structure

```
onnxruntime-mps/
├── docs/
│   ├── DESIGN.md                 # this document
│   └── (kernel notes, benchmarks per phase)
├── include/
│   └── onnxruntime_mps/
│       ├── metal_ep.h            # public C entry decls (CreateEpFactories/ReleaseEpFactory)
│       └── version.h
├── src/
│   ├── ep/                       # ABI glue (C++/.cc + Obj-C++/.mm)
│   │   ├── ep_factory.mm         # MetalEpFactory : OrtEpFactory
│   │   ├── ep.mm                 # MetalEp : OrtEp (GetCapability, kernel registry)
│   │   ├── allocator.mm          # MTLBuffer pool allocator (shared storage)
│   │   ├── data_transfer.mm      # OrtDataTransferImpl
│   │   ├── kernel_registry.cc    # op → pipeline-state dispatch table
│   │   ├── metal_context.mm      # id<MTLDevice/CommandQueue/Library> owner (PIMPL)
│   │   └── entry.cc              # CreateEpFactories / ReleaseEpFactory exports
│   └── kernels/                  # Metal shader sources
│       ├── matmul_nbits.metal
│       ├── group_query_attention.metal
│       ├── rope.metal
│       ├── rmsnorm.metal         # + skip-simplified variant
│       ├── softmax.metal
│       ├── gather_block_quantized.metal
│       └── elementwise.metal     # add/mul/sub/sigmoid/silu/cast
├── cmake/
│   └── FindONNXRuntime.cmake
├── tests/
│   ├── kernels/                  # per-op correctness vs ORT CPU reference
│   └── e2e/                      # onnx-genai-driven coherence + benchmarks
├── CMakeLists.txt
├── LICENSE
└── README.md
```

---

## 6. Build System

- **CMake** (Xcode + Ninja generators). Language: `C`, `CXX`, `OBJCXX`.
- **Output:** `libonnxruntime_mps_ep.dylib` with `CreateEpFactories`/`ReleaseEpFactory` exported
  (`-fvisibility=hidden` + explicit `__attribute__((visibility("default")))`).
- **Link:** `libonnxruntime.dylib` (the prebuilt ORT 1.27 that `onnx-genai/ort-sys` already downloads/caches — point
  `FindONNXRuntime.cmake` at `ORT_LIB_DIR`/`ORT_INCLUDE_DIR`); Apple frameworks `Metal`, `Foundation`,
  `MetalPerformanceShaders`, `MetalPerformanceShadersGraph`.
- **Shaders:** compile `.metal` → `default.metallib` via `xcrun -sdk macosx metal`/`metallib`, embedded as a resource
  or loaded next to the dylib.
- **Headers:** we depend only on the ORT C ABI headers (`onnxruntime_c_api.h` + `onnxruntime_ep_c_api.h`) — vendored
  from the pinned 1.27 release to guarantee ABI-version match.
- **Min deployment target:** macOS 14+ (Sonoma) for the MPSGraph SDPA and simdgroup features we rely on; confirm
  during Phase 1.

---

## 7. Testing Strategy (via onnx-genai)

The whole point of co-developing with onnx-genai is a **real** end-to-end harness, not synthetic graphs.

### 7.1 Wire a `metal` EP into onnx-genai

Add an `ExecutionProvider::Metal` variant and an `ONNX_GENAI_EP=metal` path
(`crates/onnx-genai-ort/src/session.rs` `execution_providers_from_env`, currently `cpu|webgpu|cuda|coreml`). The
registration differs from the built-in EPs — it uses the **plugin path**:

1. On environment creation (`env.rs`), call `RegisterExecutionProviderLibrary(env, "metal",
   $ONNX_GENAI_METAL_EP_LIB)` where the env var points at `libonnxruntime_mps_ep.dylib`.
2. `GetEpDevices(env, …)`, select the MetalEP device.
3. `SessionOptionsAppendExecutionProvider_V2(opts, env, ep_devices, …)` instead of the string
   `SessionOptionsAppendExecutionProvider` used for WebGPU/CoreML.
4. `UnregisterExecutionProviderLibrary` on teardown.

This is additive and feature-gated (`--features metal`, default-off, like the existing `cuda` feature) so default
Mac builds are unaffected. It reuses the runtime-owned shared-KV / IoBinding plumbing that already exists for
WebGPU/CUDA — KV simply becomes a shared `MTLBuffer` with **no host round-trip**.

### 7.2 Correctness

- **Per-kernel (tests/kernels):** for each op, run the same node on MetalEP and on the ORT CPU EP over random +
  edge-case inputs; assert fp16 tolerance (e.g. rel err ≤ 2e-2 for int4 matmul, tighter for norms/elementwise).
  Golden inputs drawn from real layer-0 tensors of `qwen2.5-0.5b-cpu-recipe`.
- **Partition assertion:** load the Qwen graph with MetalEP and assert the expected node placement (e.g. all 169
  `MatMulNBits`, 24 GQA on Metal; only shape/seqlen bookkeeping + copies on CPU), logged like the WebGPU partition
  logging.

### 7.3 E2E coherence + benchmark

- **Coherence gate (non-negotiable, per decisions.md WebGPU correctness gate):** `ONNX_GENAI_EP=metal
  onnx-genai-server --model models/qwen2.5-0.5b-cpu-recipe` → "capital of France" ⇒ "Paris"; 128-token essay fluent.
  A perf number is only reported if coherence passes.
- **Benchmark:** reuse `scripts/compare_runtimes.sh` (decisions.md) to compare **MetalEP vs LM Studio Metal vs
  Foundry Local vs Ollama Metal** at parity Q4_0/Q4 int4, short (≈59-tok) and long (≈858-tok) prompts, measuring
  TTFT, decode tok/s (isolated by differencing max_tokens), and prefill throughput. Target: beat Foundry Local
  (ORT-vs-ORT) first, then LM Studio/llama.cpp-Metal.

---

## 8. Phased Implementation Plan

| Phase | Scope | Owner(s) | Exit criteria |
|---|---|---|---|
| **P1 — EP skeleton + fallback + 1–2 ops** | `MetalEpFactory`/`MetalEp`, device+allocator+data-transfer, kernel registry, `GetCapability` with CPU fallback; register + load via onnx-genai `ONNX_GENAI_EP=metal`; implement **elementwise (Add/Mul/Sigmoid)** + **RMSNorm** as first kernels. | Nabil (EP/ABI), Mariette (first kernels), + onnx-genai wiring | Qwen graph loads with MetalEP active, partitions correctly, runs coherently (rest on CPU fallback), per-op tests green for the 2 ops. |
| **P2 — Decode hot path** | **MatMulNBits GEMV (decode)**, **GroupQueryAttention** (RoPE+KV+causal), **SkipSimplifiedLayerNorm**, **Softmax**, **GatherBlockQuantized**, SiLU fusion; shared-KV `MTLBuffer` (no host round-trip). | Mariette (matmul/attn/norm/softmax), Coco (MatMulNBits int4/int8 + GatherBlockQuantized quant), Nabil (KV binding) | Full Qwen decode runs on Metal (near-zero CPU nodes except bookkeeping); coherent; **decode tok/s beats ORT CPU & WebGPU**. |
| **P3 — Full coverage + perf** | Prefill GEMM (`simdgroup_matrix`), fused `MatMulNBits→SiLU`, sliding-window attention, remaining P2/P3 ops, stream-aware option, graph-capture-style replay if it wins; tune tiling/occupancy. | Mariette + Coco (kernels), Freysa (perf/bench), Nabil (compile-path fusion) | **Beat Foundry Local, then LM Studio/llama.cpp-Metal** on short & long; prefill competitive; benchmark report vs all runtimes. |

**Team ownership:** **Nabil** — EP integration / ABI / partitioning / onnx-genai registration & KV binding.
**Mariette** — compute kernels (matmul, attention, norm, softmax, elementwise). **Coco** — data/quant kernels
(int4/int8 MatMulNBits numerics, GatherBlockQuantized, dequant/pack layout). **Freysa** — perf harness, correctness
gates, cross-runtime benchmarking. **Roy** — architecture, review, decisions.

---

## 9. Risks & Open Questions

| # | Risk / Question | Impact | Mitigation / needed input |
|---|---|---|---|
| **R1** | **ABI stability.** Plugin-EP ABI has no formal "stable" label, though it's additive-only and used by production EPs (CUDA/WebGPU/QNN). | Med | Pin to ORT 1.27 headers; `ort_version_supported` guards new fields; CI against the exact 1.27.0 prebuilt onnx-genai uses. |
| **R2** | **Prebuilt-ORT compatibility.** Must confirm the *stock* `onnxruntime-osx-arm64-1.27.0` binary exports `RegisterExecutionProviderLibrary`/`GetEpApi`/`SessionOptionsAppendExecutionProvider_V2` (guarded by `!ORT_MINIMAL_BUILD`). | High | **Phase 1 first task:** `nm -gU libonnxruntime.1.27.0.dylib | grep -i ExecutionProviderLibrary`. If missing (minimal build), fall back to a self-built ORT or a matching non-minimal release. **(needs a 5-min check)** |
| **O3** | **Apple hardware-device enumeration.** ORT may not surface a Metal `OrtHardwareDevice`; we may need `OrtEpApi::CreateHardwareDevice(GPU, Apple, …)`. Exact `OrtHardwareDeviceType` for Apple GPU unverified. | Med | Verify against the example plugin EP + `onnxruntime_ep_device_ep_metadata_keys.h` in Phase 1; synthesize a virtual device if needed. |
| **R4** | **fp16 accuracy.** int8-accumulated int4 matmul + fp16 attention may drift on long context. | Med | fp32 reduction accumulators; per-kernel tolerance tests vs ORT CPU; coherence gate before any perf claim. |
| **R5** | **MatMulNBits int4 Metal perf.** Beating llama.cpp's years-tuned `mul_mv_q4` is the core bet. | High | Follow llama.cpp/ExecuTorch layouts closely; profile with Metal System Trace/`MTLCounters`; GEMV vs GEMM specialization. |
| **R6** | **KV share-buffer semantics on Metal.** WebGPU's in-place shared-KV SIGSEGV'd on ORT 1.27; must confirm the plugin-EP allocator + IoBinding path is stable. | Med | Unified memory means we control the `MTLBuffer` directly; validate the past==present in-place contract early (P2). |
| **O7** | **MPSGraph vs custom shaders** per op — where is MPSGraph "good enough" vs where we need custom `.metal`? | Low | Correctness-first with MPS primitives for P1 non-hot ops; custom shaders for all P0 hot ops. |

### Blocking unknowns for Justin to resolve
1. **R2 check** — is our prebuilt ORT 1.27.0 a non-minimal build that exports the plugin-EP registration symbols? (One `nm` command; if no, we need a non-minimal / self-built ORT — a scope decision.)
2. **Version pin** — stay on **1.27.0** (what onnx-genai links today) or move to **1.27.1** (public release, same API 27)? Recommend staying on 1.27.0 for exact parity with the runtime.
3. **Scope of op coverage** — confirm we target **only** the Qwen2.5-decoder op set for P1–P2, deferring Conv/Attention/VLM ops (Mobius emits them for other model families) to a later initiative.

---

## 10. References (sources cited)

- **ORT plugin-EP ABI:** `microsoft/onnxruntime` — `include/onnxruntime/core/session/onnxruntime_ep_c_api.h`,
  `onnxruntime_c_api.h` (`ORT_API_VERSION 27` @ `v1.27.x`); `onnxruntime/core/session/plugin_ep/ep_library_plugin.{h,cc}`,
  `ep_plugin_provider_interfaces.cc`; `abi_ep_types.h`; `ort_apis.h`
  (`RegisterExecutionProviderLibrary`, `SessionOptionsAppendExecutionProvider_V2`, `GetEpApi`).
- **Example plugin EPs:** `onnxruntime/test/autoep/library/example_plugin_ep/`,
  `…/example_plugin_ep_kernel_registry/`, `…/example_plugin_ep_virt_gpu/`.
- **Production plugin EP analog:** `onnxruntime/core/providers/cuda/plugin/cuda_plugin_ep.cc` +
  `docs/cuda_plugin_ep/cuda_plugin_ep_design.md`.
- **MPS kernel references:** ExecuTorch `backends/apple/mps`; PyTorch `aten/src/ATen/native/mps` (incl.
  `_weight_int4pack_mm`, MPS SDPA); llama.cpp `ggml/src/ggml-metal/ggml-metal.metal`
  (`mul_mv_q4_0`, `mul_mm`, `rms_norm`, `flash_attn_ext`, `get_rows_q4_0`).
- **Our models / ops:** `onnx-genai/models/qwen2.5-0.5b-cpu-recipe/model.onnx` (op census + attributes),
  `…/qwen2.5-0.5b-q4-gqa-webgpu`, `…/qwen2.5-0.5b-cuda`; `mobius/src/mobius/components/`.
- **Our runtime & prior findings:** `onnx-genai/crates/onnx-genai-ort/{ort-sys/build.rs, src/session.rs,
  src/allocator.rs, src/env.rs}`; `onnx-genai/.squad/decisions.md` (CPU recipe competitiveness, WebGPU KV
  round-trip, ORT 1.27 WebGPU device-KV SIGSEGV, correctness gate, inference-metadata GQA share-buffer).
