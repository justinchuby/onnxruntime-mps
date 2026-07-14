// Copyright (c) 2026. Licensed under the MIT License.
//
// Dense linear-algebra op handlers (MatMul, Gemm) — the non-quantized matmul family that maps onto
// MLX's dense GEMM primitives. See docs/OP_ARCHITECTURE.md §2/§6 for the add-an-op recipe.
//
//   * MatMul (ai.onnx) — dense A @ B via mlx_matmul. Handles rank-2 and rank>2 batched forms;
//     mlx_matmul carries numpy/ONNX-matching batch-dim broadcasting (e.g. [b,m,k] @ [k,n]) with no
//     manual reshape, so this is a single MLX op per node. Dtype-generic (fp32/fp16/bf16): mlx_matmul
//     carries the resolved input dtype through with no per-dtype code. The 1-D (vector) MatMul forms
//     (which ONNX pads to a matrix and then squeezes) are left to ORT CPU — they are rare in decoder
//     graphs and not worth the reshape/squeeze special-casing.
//   * Gemm (ai.onnx) — Y = alpha * A' @ B' (+ beta * C) with optional transA/transB and an optional
//     broadcast bias C. A'/B' are the (optionally) transposed rank-2 operands; alpha/beta scale
//     through dtype-matched scalars so the output keeps the input dtype (a raw fp32 scalar multiply
//     would promote an fp16/bf16 GEMM to fp32). C is unidirectionally broadcast to (M,N) by MLX add.
//
// Both claims are conservative: float dtypes only (fp32/fp16/bf16), all operands the same dtype as
// the output, and the exact rank each op requires (MatMul: rank>=2 on both sides; Gemm: rank-2 A/B).
// Every other form falls back to ORT CPU, which is always correct.

#include <cstdint>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// A node input slot is present when it exists and is not an omitted optional (Gemm's bias C).
bool Present(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent;
}

// Claim-time: an ONNX node value slot is present when it exists, is a non-null value info, and
// carries a non-empty name (ORT hands back a NULL OrtValueInfo for an omitted optional input).
bool ClaimPresent(const std::vector<Ort::ConstValueInfo>& vals, size_t i) {
  if (i >= vals.size()) return false;
  if (static_cast<const OrtValueInfo*>(vals[i]) == nullptr) return false;
  return !vals[i].GetName().empty();
}

// A dtype-matched scalar (float value cast to `dt`), so alpha/beta scaling keeps the GEMM dtype
// instead of promoting an fp16/bf16 product to fp32.
mlx_array ScalarLike(TranslationContext& ctx, float value, mlx_dtype dt) {
  return ctx.Astype(ctx.Keep(mlx_array_new_float32(value)), dt);
}

// ---- MatMul (ai.onnx) -----------------------------------------------------------------------

// Y = A @ B. mlx_matmul broadcasts the leading (batch) dims exactly like numpy/ONNX MatMul, so a
// single call covers the rank-2 and rank>2 batched/broadcast forms. When the result is empty (a
// zero-sized contraction/batch/broadcast dim), mlx_matmul returns an array with no backing buffer,
// and the boundary CopyOut's typed data access would segfault on it — so re-materialise the empty
// result as a clean, correctly-shaped zeros array (matches numpy.matmul's empty output exactly).
void MatMulOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array a = ctx.Resolve(n.inputs[0]);
  mlx_array b = ctx.Resolve(n.inputs[1]);
  mlx_array y = mlx_array_new();
  MLX_CHECK(mlx_matmul(&y, a, b, ctx.stream()));
  ctx.Keep(y);
  if (mlx_array_size(y) == 0) {
    const std::vector<int> shp = TranslationContext::ShapeOf(y);
    mlx_array z = mlx_array_new();
    MLX_CHECK(mlx_zeros(&z, shp.data(), shp.size(), mlx_array_dtype(y), ctx.stream()));
    ctx.Bind(n.outputs[0], ctx.Keep(z));
    return;
  }
  ctx.Bind(n.outputs[0], y);
}

// ---- Gemm (ai.onnx) -------------------------------------------------------------------------

// Y = alpha * A' @ B' (+ beta * C), where A'/B' are the (optionally) transposed rank-2 operands.
void GemmOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array a = ctx.Resolve(n.inputs[0]);
  mlx_array b = ctx.Resolve(n.inputs[1]);
  const mlx_dtype dt = mlx_array_dtype(a);

  const bool trans_a = n.ints.count("transA") && n.ints.at("transA") != 0;
  const bool trans_b = n.ints.count("transB") && n.ints.at("transB") != 0;
  if (trans_a) a = ctx.Transpose(a, {1, 0});
  if (trans_b) b = ctx.Transpose(b, {1, 0});

  const float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 1.0f;
  const float beta = n.floats.count("beta") ? n.floats.at("beta") : 1.0f;

  mlx_array mm = mlx_array_new();
  MLX_CHECK(mlx_matmul(&mm, a, b, ctx.stream()));
  ctx.Keep(mm);

  mlx_array y = alpha != 1.0f ? ctx.Mul(mm, ScalarLike(ctx, alpha, dt)) : mm;
  if (Present(n, 2)) {
    mlx_array c = ctx.Resolve(n.inputs[2]);
    if (beta != 1.0f) c = ctx.Mul(c, ScalarLike(ctx, beta, dt));
    y = ctx.AddA(y, c);
  }
  ctx.Bind(n.outputs[0], y);
}

// ---- claim predicates (dtype/shape checks; registry already matched domain/op/opset) ------------

// MatMul: two float operands of the same dtype as the output, both rank>=2 (the batched/broadcast
// matrix forms mlx_matmul expresses). 1-D vector operands are left to ORT CPU.
bool MatMulClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.empty()) return false;
  ONNXTensorElementDataType ad, bd, od;
  std::vector<int64_t> ashape, bshape;
  if (!TensorInfo(inputs[0], ad, &ashape) || !TensorInfo(inputs[1], bd, &bshape) ||
      !TensorInfo(outputs[0], od)) {
    return false;
  }
  if (!IsMlxFloatType(ad) || bd != ad || od != ad) return false;
  return ashape.size() >= 2 && bshape.size() >= 2;
}

// Gemm: rank-2 float A/B of the same dtype as the output, and (if present) a float bias C of the
// same dtype broadcastable to (M,N). alpha/beta/transA/transB are read generically in the handler.
bool GemmClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 && inputs.size() != 3) return false;
  if (outputs.empty()) return false;
  ONNXTensorElementDataType ad, bd, od;
  std::vector<int64_t> ashape, bshape;
  if (!TensorInfo(inputs[0], ad, &ashape) || !TensorInfo(inputs[1], bd, &bshape) ||
      !TensorInfo(outputs[0], od)) {
    return false;
  }
  if (!IsMlxFloatType(ad) || bd != ad || od != ad) return false;
  if (ashape.size() != 2 || bshape.size() != 2) return false;
  if (ClaimPresent(inputs, 2)) {
    ONNXTensorElementDataType cd;
    std::vector<int64_t> cshape;
    if (!TensorInfo(inputs[2], cd, &cshape) || cd != ad) return false;
    if (cshape.size() > 2) return false;  // C must broadcast to the rank-2 output
  }
  return true;
}

}  // namespace

void RegisterMatMulOps(OpRegistry& registry) {
  registry.Register({"", "MatMul", kAnyOpset, kAnyOpset, &MatMulOp, &MatMulClaim});
  registry.Register({"", "Gemm", kAnyOpset, kAnyOpset, &GemmOp, &GemmClaim});
}

}  // namespace ort_mps_mlx
