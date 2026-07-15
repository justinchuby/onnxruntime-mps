# Example MLX EP traces

Perfetto / Chrome-trace captures from the Rust MLX execution provider's built-in
tracer. **Open any `.json` here at <https://ui.perfetto.dev>** (drag-and-drop or
"Open trace file") or at `chrome://tracing`.

Each trace was produced by setting `ONNX_GENAI_MLX_TRACE=<path>` while running a
model through the EP (the tracer is env-gated and off by default, near-zero cost
when off). Events are stamped with the real process id, so an EP trace merges
into onnx-genai's own timeline under the same process.

## The traces

| File | What it is | What to look for |
|------|-----------|------------------|
| `qwen2.5-0.5b-decode.json` | 8 decode steps of qwen2.5-0.5B (q4) through the EP | The whole decoder fuses into **one `mlx.eval`** per token (GPU-inclusive time); `GroupQueryAttention` runs `op.fast` (fused SDPA + RoPE); **0 composed events**; `mlx.gpu_mem_bytes` counter track |
| `gqa-attention-fused.json` | GroupQueryAttention op tests | `op.fast` spans (`mlx_fast_scaled_dot_product_attention`, `mlx_fast_rope`) — no composed markers |
| `matmulnbits-fast-vs-composed.json` | MatMulNBits block-16 vs block-32 | Both a `op.fast` (`mlx_quantized_matmul`, block 32) **and** `op.composed` path (block 16 → dequant+dense), with a `⚠ composed-path: MatMulNBits (...)` instant marker + `mlx.composed_path_count` counter |
| `ep-events-conv.json` | Conv op suite (the general **compiled** path) | `mlx.compute[general]` instant events (`ep.path`); the `cache` arg is `MISS` on each fresh session; `mlx.compile` + `mlx.eval` + `mlx.copy` **timing-attribution** spans (`ep.phase`); `mlx.getcapability` claim events (`ep.claim`) with `claimed`/`fused_subgraphs`; `mlx.mem_wrap_bytes` / `mlx.copyout_bytes` **memory** counters; a `mlx.session_summary` digest instant |
| `ep-events-attention.json` | GQA / attention op suite (the **eager** path) | `mlx.compute[eager]` events (`ep.path`); `mlx.translate` + `mlx.eval` timing spans; the claim, memory and summary events as above |
| `perch-audio-encoder.json` | A real model — Perch v2 bird-vocalization audio encoder (725 nodes, 5s@32kHz), the general **compiled** path (MLX ~5× faster than CPU) | `ep.claim` shows **724/725 nodes claimed** into 2 fused subgraphs (1 `Max` → CPU with a `fallback_Max` reason); `ep.path` `mlx.compute[general]` with `cache` HIT/MISS/RETRACE; the `mlx.session_summary` digest (claim rate, per-path breakdown, zero-copy/delta memory, timing attribution) — the at-a-glance view on a substantial graph |

## The observability views (new)

The EP's key runtime events are surfaced as structured, env-gated tracer events
(no `eprintln` on the traced-off fast path). Four views + a summary:

1. **Claiming view** (`ep.claim`) — `mlx.getcapability` instant per GetCapability
   with `claimed` / `total` / `unclaimed` / `fused_subgraphs` (fragmentation
   signal) plus per-op `fallback_<Op>` reasons; counters `mlx.claimed_nodes`,
   `mlx.unclaimed_nodes`, `mlx.fused_subgraphs`.
2. **Execution-path view** (`ep.path`) — `mlx.compute[<path>]` instant per Compute
   naming the path (`decode` / `prefill` / `general` / `eager`), the compile-cache
   state (`HIT` / `MISS` / `RETRACE`), the `shape_key`, and node count; counter
   `mlx.compute_path` (one series per path).
3. **Memory view** — counters `mlx.mem_wrap_bytes` (zero-copy boundary wrap) and
   `mlx.copyout_bytes` (`delta` vs `full` series). The summary reports the
   zero-copy-aligned vs internal-copy split and delta-vs-full copy-out bytes.
4. **Timing attribution** (`ep.phase`) — `mlx.translate` / `mlx.compile` /
   `mlx.eval` / `mlx.copy` spans per subgraph.
5. **Session summary** — a human-readable digest printed to stderr on EP teardown
   (claim rate, per-path Compute breakdown, memory movement, time attribution) and
   embedded as the `mlx.session_summary` instant. Force it without full JSON tracing
   with `ONNX_GENAI_MLX_VERBOSE=1`.

## Reading the tracks

- **Categories** (Perfetto colors by category):
  - `ep` — `mlx.subgraph`, one per fused-node Compute.
  - `ep.claim` — `mlx.getcapability` claim events.
  - `ep.path` — `mlx.compute[...]` execution-path events.
  - `ep.phase` — `mlx.translate` / `mlx.compile` / `mlx.eval` / `mlx.copy` timing.
  - `gpu` — `mlx.eval`, the synchronous `mlx_eval`; its wall time is the fused
    subgraph's **GPU-inclusive** time.
  - `op` — one span per node during graph build (op structure).
  - **`op.fast`** — the node used a fused MLX kernel (`optimized=true`, `kernel=...`).
  - **`op.composed`** — the node HAS a fused kernel but took a slower composed
    fallback (`optimized=false`, `reason=...`) — these are what you want to hunt
    down for perf. Each also emits a `⚠ composed-path: <Op>` instant marker.
  - `gpu_counter` — counter tracks (`mlx.gpu_mem_bytes`, `mlx.composed_path_count`,
    `mlx.claimed_nodes`, `mlx.compute_path`, `mlx.mem_wrap_bytes`, …).

- **Args** on a span carry `op_type`, shapes, dtype, and (for fast/composed) the
  kernel name or fallback reason.

## Regenerating

```sh
# op-level (any pytest op suite)
ORT_LIB=<...ort-prebuilt/lib>
DYLD_LIBRARY_PATH=$ORT_LIB \
  ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_TRACE=examples/traces/gqa-attention-fused.json \
  python -m pytest tests/ops -q -k gqa

# the new claim/path/memory/timing event traces
DYLD_LIBRARY_PATH=$ORT_LIB \
  ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_TRACE=examples/traces/ep-events-conv.json \
  python -m pytest tests/ops/test_conv.py -q            # general compiled path
DYLD_LIBRARY_PATH=$ORT_LIB \
  ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_TRACE=examples/traces/ep-events-attention.json \
  python -m pytest tests/ops/test_attention_ext.py -q -k "gqa or group"

# just the human-readable session summary (no JSON trace file, near-zero overhead)
DYLD_LIBRARY_PATH=$ORT_LIB \
  ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_VERBOSE=1 python -m pytest tests/ops/test_conv.py -q -s

# real decode (via onnx-genai profile_decode)
DYLD_LIBRARY_PATH=$ORT_LIB ONNX_GENAI_EP=metal \
  ONNX_GENAI_METAL_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_TRACE=examples/traces/qwen2.5-0.5b-decode.json \
  ../onnx-genai/target/release/profile_decode \
  --model ../onnx-genai/models/qwen2.5-0.5b-cpu-recipe --tokens 8
```

## Example session summary

```
[rust-mlx-ep] ===== MLX EP session summary =====
  claim:   22/22 nodes claimed (100.0%) across 22 fused subgraph(s), 22 GetCapability call(s)
  compute: decode=0 prefill=0 general=22 eager=0  (cache: 0 HIT / 22 MISS / 0 RETRACE)
  memory:  managed-wrap 22 (1 zero-copy aligned), 0.01 MiB borrowed; copy-wrap 18, 0.00 MiB
           copy-out: delta 0 (0.00 MiB) vs full 22 (0.00 MiB)
  timing:  compile=370us (x22), copy=18us (x22), eval=37472us (x22)
[rust-mlx-ep] ===================================
```

