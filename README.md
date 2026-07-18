# onnxruntime-mlx

> **PyPI package: [`onnxruntime-ep-mlx`](https://pypi.org/project/onnxruntime-ep-mlx/)** — `pip install onnxruntime-ep-mlx`, `import onnxruntime_ep_mlx`. (Formerly published as `onnxruntime-mlx`, now renamed.)

An **MLX-native execution provider** for ONNX Runtime on Apple Silicon, built as an out-of-tree
**plugin EP** (ORT plugin-EP C ABI, ORT 1.27 / `ORT_API_VERSION 27`). It ships as a standalone
`libonnxruntime_mlx_ep.dylib` loaded by a stock prebuilt `libonnxruntime.dylib` via
`RegisterExecutionProviderLibrary` — **no ONNX Runtime fork required**.

Instead of hand-tuned Metal shaders, the EP **translates a fused ONNX decoder subgraph into an
[MLX](https://github.com/ml-explore/mlx) graph** and lets MLX compile/schedule the Metal work. One
efficient implementation (MLX) covers the whole decoder for **both prefill and decode** — there are
no `.metal` kernels to maintain.

> **Why MLX-only?** A Phase-0 head-to-head (see [`docs/MLX_EVALUATION.md`](docs/MLX_EVALUATION.md))
> found the MLX path Pareto-dominant vs. the previous hand-written kernels: **decode 1.02–1.09×
> (never slower)**, **prefill ~2.5–3.5× faster**, coherent output, and memory-stable. The hand
> kernels were deleted and MLX promoted to the sole compute path.

## Compute path

`ONNX fused subgraph → MLX graph → single mlx_eval at the subgraph boundary → ORT outputs`

- **MatMulNBits** → `mlx_quantized_matmul` (int4 weights repacked once, cached on the plan)
- **QMoE** (quantized Mixture-of-Experts, `quant_type='int'`) → dense per-expert matmuls + top-k
  softmax routing + SwiGLU/silu/gelu/relu activation (int4/int8 experts dequantized in-graph)
- **GroupQueryAttention** (RoPE in-op) → `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope`
- **RMSNormalization / SkipSimplifiedLayerNormalization** → `mlx_fast_rms_norm`
- **GatherBlockQuantized** (symmetric int4 embedding) → gather + dequant
- **Softmax / Add / Mul / Sub / Sigmoid / Cast** → the matching MLX elementwise ops

Ops the EP does not translate are left unclaimed and run on ORT's CPU EP.

The translator covers the **full set of ops Mobius emits** (~85 op types) via a modular, opset-aware
registry (`rust/src/ops/*.rs`) — math/logical, reductions, shape/data-movement, normalizations,
attention (GQA, Attention 23/24, MHA, RoPE), dense MatMul/Gemm, Conv/pooling, quantized matmul &
embedding, quantized Mixture-of-Experts (`QMoE`), and more, in fp32/fp16/bf16. A handful of ops that
need engine-level control-flow or recurrence (`Scan`, `LSTM`, `LinearAttention`, float `MoE`,
`PackedMultiHeadAttention`) run on ORT CPU by design. See
[`docs/OP_ARCHITECTURE.md`](docs/OP_ARCHITECTURE.md) for the full coverage table.

## When is a graph fast? (claim + compile rules of thumb)

Peak performance comes from the EP fusing a large region into **one MLX closure** that is traced +
`mlx_compile`d **once** and replayed — one dispatch instead of hundreds. Whether that happens depends
on how the graph is shaped. Rules of thumb, fastest → slowest:

1. **Static-shape, fully-claimable feed-forward** (audio / CNN / vision encoders) — *ideal*. The whole
   graph is one convex cluster, compiled once, replayed. (Perch: 725/725 nodes, 1 subgraph.)
2. **Dynamic dims that resolve at trace time** are fine. A symbolic batch/sequence/spatial dim, or a
   `shape`/`starts` derived from `Shape(x)` (a *shape-const* value), is resolved to a concrete extent
   per **shape key** — so dynamic-spatial Conv/Resize, `[B,S,-1]` reshapes, etc. still compile. But a
   **shape *change*** at run time **retraces** the closure (the `general` = shape-keyed path), so
   many distinct shapes = many compiles. Bucket/pad your shapes for best reuse.
3. **Attention decoders** (`GroupQueryAttention`) get a dedicated **shapeless decode/prefill** path:
   the growing KV length is a shapeless dim, so per-token decode **never retraces** (KV aliased
   in-place, delta copy-out). This is the one case where a growing dimension stays fast.
4. **What forces a slow fallback (per-node eager, or CPU):**
   - **Control flow** — `If` / `Loop` / `Scan` bodies are never compiled (the whole plan runs eager).
   - **Data-dependent output shapes** — `NonZero` / `Unique` / a `Reshape` whose target is computed
     from tensor *values* (not shapes) — these need a mid-graph host read that a single trace can't
     express, so their subgraph runs eager.
   - **fp64** anywhere (MLX has no float64), or any op the registry doesn't claim.
5. **Fragmentation is the real cost.** One unclaimed op *in the middle* of a graph splits it into two
   islands with a CPU round-trip between them. Sub-5 ms graphs are dispatch/eval-overhead-bound, so a
   few islands can make MLX *slower* than CPU — the win scales with fused-region compute size, not
   claim rate alone. Aim to keep declined ops at the graph's edges.

**Diagnosing it yourself:** run with `MLX_EP_CLAIM_DEBUG=1` (or the tracer) to print exactly which
ops were declined, how many, and an actionable reason for each — the fastest way to see why a graph
fragmented and what to change (e.g. re-export at a higher opset, give a static shape, drop an fp64
cast).

## Requirements

- macOS on Apple Silicon, ORT 1.27 prebuilt (`ORT_API_VERSION >= 27`)
- **`mlx-c` (and `mlx`) — a HARD build dependency**: `brew install mlx-c`
- A **Rust toolchain** (`rustup`) to build the EP from source

## Build

The EP is a Rust `cdylib` crate under [`rust/`](rust/). Point it at an ONNX Runtime C-API
include directory and `cargo build`:

```sh
brew install mlx-c                                  # HARD dependency (mlx-c + mlx)
cd rust
# Either point ORT_INCLUDE_DIR at the ORT headers directly, or set ORT_HOME to an
# ONNX Runtime release root (build.rs will look in $ORT_HOME/include):
export ORT_INCLUDE_DIR=/path/to/onnxruntime/include   # or: export ORT_HOME=/path/to/onnxruntime-osx-arm64-1.27.0
cargo build --release
# => rust/target/release/libonnxruntime_mlx_ep.dylib  (registers the EP as "MLXExecutionProvider")
```

The crate binds the ORT plugin-EP C ABI and `mlx-c` directly via `bindgen`; it does **not** link
`libonnxruntime` (ORT is reached through the `OrtApi` function-pointer table passed to
`CreateEpFactories`).

## Install & use

### Python (recommended)

```sh
pip install -U onnxruntime-ep-mlx        # macOS/Apple-Silicon wheel; bundles the mlx runtime
```

```python
import onnxruntime as ort
import onnxruntime_ep_mlx

# Register the plugin EP once, then select it (with CPU fallback) like any provider.
onnxruntime_ep_mlx.register_execution_provider_library()          # name: "MLXExecutionProvider"
sess = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
out = sess.run(None, feeds)
```

`onnxruntime_ep_mlx` also exposes `library_path()`, `ep_name()`, `version()`, and
`append_to_session_options(so)`.

### C / C++ (or any onnxruntime binding)

Point onnxruntime at the built dylib and select the provider by name:

```c
// 1. Register the plugin library with the environment (once).
RegisterExecutionProviderLibrary(env, "MLXExecutionProvider",
                                 "/abs/path/libonnxruntime_mlx_ep.dylib");
// 2. Append it to a session's options (falls back to CPU for unclaimed ops).
const char* ep = "MLXExecutionProvider";
SessionOptionsAppendExecutionProvider_V2(options, env, &ep, /*count*/ 1, ...);
```

From Rust via **onnx-genai**: `ONNX_GENAI_EP=metal` +
`ONNX_GENAI_METAL_EP_LIB=/abs/path/libonnxruntime_mlx_ep.dylib`.

## Performance (M1 Max, warm)

Real end-to-end models, median of 10 runs, MLX EP vs the ORT **CPU** EP on the same machine — top-1
identical and max abs diff ≤ 6e-5 in every case:

| Model | Workload | CPU EP | MLX EP | Speedup |
|---|---|---:|---:|---:|
| Perch v2 | audio encoder (with DFT front-end) | 64.0 ms | 12.0 ms | **5.3×** |
| Perch v2 (no DFT) | audio encoder | 56.5 ms | 12.0 ms | **4.7×** |
| BirdNET | audio classifier (CNN) | 14.9 ms | 7.3 ms | **2.0×** |
| gemma-4-E2B | vision encoder (fp16 ViT) | 267 ms | 47 ms | **5.7×** |

Feed-forward encoders (audio / CNN / vision) are the EP's sweet spot: the whole graph fuses into a
single MLX closure that is traced + `mlx_compile`d once and replayed, so a static-shape model runs
end-to-end on the GPU with one dispatch (e.g. Perch: 725/725 nodes claimed, 1 fused subgraph).

For **LLMs**, the EP accelerates both prefill / TTFT and — on larger quantized decoders — decode.
The **Foundry Local** q4f16 decoders below run on the same M1 Max, warm, MLX EP vs the ORT CPU EP
(decode = 1 token with 128 past; prefill = 128-token step):

| Model | Arch | Prefill | Decode |
|---|---|---:|---:|
| Qwen2.5-0.5B | GQA, external rotary | **5.2×** | dispatch-bound (CPU-favored) |
| Phi-3.5-mini | Phi3, GQA | **5.29×** | 1.19× |
| Phi-4-mini | Phi4, long-context RoPE | **5.78×** | 1.10× |
| Mistral-7B-Instruct | GQA, growing KV | **11.89×** | **3.30×** |
| gemma-4-E2B | Gemma3n, 15-layer | 3.3× | **3.3×** |

The prefill lead grows with prompt length and with model size (Mistral-7B: **11.9×**). Decode is
weight-bandwidth-bound: on a small 0.5B model the CPU `accuracy_level=4` int8 MatMulNBits path wins
per-token, but on larger **q4f16** decoders the MLX path pulls ahead — the **gemma-4-E2B** decoder
(Gemma3n, int4 weights + fp16 activations) runs a decode step in **33 ms vs 111 ms on CPU (3.3×)**,
and **Mistral-7B** reaches **3.30×** decode — once their fp16 MatMulNBits, `num_heads`-inferred
RotaryEmbedding, and GroupQueryAttention (9-/11-input, external rotary + attention_bias) all run on
MLX.

Phi-4-mini additionally exercises a data-dependent `If` (long-context RoPE-cache selection): the EP
leaves that control-flow node on the CPU (its condition is a runtime value) while still offloading the
rest of the decoder, so it lands at **5.78×** prefill instead of falling entirely back to CPU.


Any op the EP doesn't claim falls back to the ORT CPU EP, so **every** graph still runs correctly —
the EP is a safe drop-in. The audio numbers above are the public Hugging Face Perch v2 / BirdNET ONNX
exports, timed as the median of 10 warm runs against the CPU EP on the same machine.

## Concurrency

MLX evaluation is **thread-affine** — a given `InferenceSession`'s MLX work must run on the thread
that first drove it. The rule is simple:

> **Use one `InferenceSession` per thread.** Do not call `Run()` on a single shared session from
> multiple threads.

Session-per-thread scales cleanly (each thread creates and runs its own session). If you *do* call a
shared session from another thread, the EP detects it and returns a clean `EP_FAIL` — ORT then
transparently falls back to the CPU EP for that call, so you get a correct result instead of a crash.
Internally, each session's compiled-graph cache is mutex-guarded, so there is no data race even under
misuse.

## Numerical accuracy

Op outputs match the ORT CPU EP to ~1e-5 (float32), and are validated MLX-vs-CPU across the 900+
`tests/ops` cases plus ONNX's own backend node tests. MLX and CPU use different math libraries, so
results are *close* but not bit-identical: they can differ in the last ULP or two of float32.

For autoregressive decoding this is worth understanding. A per-step argmax is stable for many tokens
(early tokens are typically bit-identical to a CPU run), but any float32 reduction-order difference is
amplified across a long greedy loop — once two candidate logits are within rounding of each other, MLX
and CPU can pick different tokens and the sequences then diverge. This is expected floating-point
behavior, not a bug; it does not indicate lower quality, only a different-but-equally-valid rounding.
If you require bit-exact parity with a CPU reference over a long generation, run decode on the CPU EP.

## Layout

```
docs/     design docs (DESIGN, OP_ARCHITECTURE, MLX_EVALUATION)
rust/     the Rust EP: plugin-EP C-ABI vtables (factory/ep) + the modular ONNX->MLX
          translator (engine, registry, ops/*.rs) over a mlx-c RAII layer (mlx.rs)
python/   pure-Python pip package (onnxruntime-ep-mlx): a locator that bundles + registers
          the cargo-built dylib (hatchling build hook, hatch_build.py)
tests/    MLX op-correctness (tests/ops, pytest) + ONNX-standard conformance (tests/conformance)
.github/  CI (cargo build + op tests) and PyPI trusted-publishing workflows
```

## Testing

Build the EP (above), then run the pytest op-correctness suite (MLX vs ORT CPU reference):

```sh
export ONNXRUNTIME_MLX_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib
export DYLD_LIBRARY_PATH=<ort-prebuilt/lib>
python -m pytest tests/ops -q
```

- `tests/ops` — each translated decoder op via MLX vs. ORT CPU reference (tolerance-gated, pytest)
- `tests/conformance` — opt-in fuzz-conformance of the MLX EP against the ONNX standard
  (`cbourjau/onnx-tests`); see [`tests/conformance/README.md`](tests/conformance/README.md)

