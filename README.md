# onnxruntime-mlx

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
- **GroupQueryAttention** (RoPE in-op) → `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope`
- **RMSNormalization / SkipSimplifiedLayerNormalization** → `mlx_fast_rms_norm`
- **GatherBlockQuantized** (symmetric int4 embedding) → gather + dequant
- **Softmax / Add / Mul / Sub / Sigmoid / Cast** → the matching MLX elementwise ops

Ops the EP does not translate are left unclaimed and run on ORT's CPU EP.

The translator covers the **full set of ops Mobius emits** (~85 op types) via a modular, opset-aware
registry (`src/ep/ops/*.cc`) — math/logical, reductions, shape/data-movement, normalizations,
attention (GQA, Attention 23/24, MHA, RoPE), dense MatMul/Gemm, Conv/pooling, quantized matmul &
embedding, and more, in fp32/fp16/bf16. A handful of ops that need engine-level control-flow or
recurrence (`Scan`, `LSTM`, `LinearAttention`, `MoE`, `PackedMultiHeadAttention`) run on ORT CPU by
design. See [`docs/OP_ARCHITECTURE.md`](docs/OP_ARCHITECTURE.md) for the full coverage table.

## Requirements

- macOS on Apple Silicon, ORT 1.27 prebuilt (`ORT_API_VERSION >= 27`)
- **`mlx-c` (and `mlx`) — a HARD build dependency**: `brew install mlx-c`

## Build

```sh
cmake -S . -B build -G "Unix Makefiles"   # FAILS if mlx-c is not installed
cmake --build build -j8
# => build/libonnxruntime_mlx_ep.dylib   (registers the EP under the name "MLXExecutionProvider")
```

## Install & use

### Python (recommended)

```sh
pip install onnxruntime-mlx        # macOS/Apple-Silicon wheel; bundles the mlx runtime
```

```python
import onnxruntime as ort
import onnxruntime_mlx

# Register the plugin EP once, then select it (with CPU fallback) like any provider.
onnxruntime_mlx.register_execution_provider_library()          # name: "MLXExecutionProvider"
sess = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
out = sess.run(None, feeds)
```

`onnxruntime_mlx` also exposes `library_path()`, `ep_name()`, `version()`, and
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

The EP is a **prefill / TTFT accelerator**: MLX prefill runs **1.85–2.77× faster than the ORT CPU EP**
(and the lead grows with prompt length). Decode is weight-bandwidth-bound — on small models the CPU
`accuracy_level=4` int8 path is very fast, so decode stays competitive-to-CPU-favored there; the MLX
decode edge widens on larger models. Unclaimed ops fall back to ORT CPU, so any graph still runs.

## Layout

```
docs/     design docs (DESIGN, OP_ARCHITECTURE, MLX_EVALUATION)
include/  public C entry-point headers (CreateEpFactories / ReleaseEpFactory)
src/ep/   plugin-EP ABI glue + the modular ONNX->MLX translator (ops/*.cc, mlx_backend, op_registry)
python/   nanobind + scikit-build-core pip package (onnxruntime-mlx)
cmake/    build helpers
tests/    MLX op-correctness (tests/ops, pytest) + e2e coherence & leak (tests/e2e)
.github/  CI (build + op tests) and PyPI trusted-publishing workflows
```

## Testing

```sh
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> ctest --test-dir build
```

- `mlx_op_tests` — each translated decoder op via MLX vs. ORT CPU reference (tolerance-gated)
- `mlx_e2e` — full-MLX prefill+decode coherence gate ("The capital of France is Paris")
- `mlx_leak_test` — allocator memory flat across bounded session/inference cycles
