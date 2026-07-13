# onnxruntime-mps

A custom **Apple Metal/MPS execution provider** for ONNX Runtime, built as an out-of-tree
**plugin EP** (ORT plugin-EP C ABI, ORT 1.27 / `ORT_API_VERSION 27`). It ships as a standalone
`libonnxruntime_mps_ep.dylib` loaded by a stock prebuilt `libonnxruntime.dylib` via
`RegisterExecutionProviderLibrary` — **no ONNX Runtime fork required**.

Goal: hand-tuned Metal kernels (int4 `MatMulNBits`, `GroupQueryAttention`, RoPE, RMSNorm,
`GatherBlockQuantized`, …) that make ORT the **fastest runtime on Apple Silicon**, beating
llama.cpp-Metal-class stacks (LM Studio, Foundry Local) for the models our
[`onnx-genai`](../onnx-genai) runtime and [`mobius`](../mobius) builder use.

> **Status: design/plan.** See [`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture,
> operator coverage, kernel design, build/test strategy, and phased plan. No kernel/EP
> implementation has landed yet.

## Layout

```
docs/     design docs
include/  public C entry-point headers
src/ep/   plugin-EP ABI glue (C++/Obj-C++)
src/kernels/  Metal shader sources (.metal)
cmake/    build helpers
tests/    per-kernel correctness + onnx-genai-driven e2e
```

## Testing

End-to-end validation runs through onnx-genai: `ONNX_GENAI_EP=metal` registers this plugin
library and drives the existing Qwen2.5-0.5B packages in `onnx-genai/models/`. See
[`docs/DESIGN.md` §7](docs/DESIGN.md).
