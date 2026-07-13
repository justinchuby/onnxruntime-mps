// Copyright (c) 2026. Licensed under the MIT License.
//
// Quantized op handlers: MatMulNBits (int4 block-quantized weight matmul via mlx_quantized_matmul)
// and GatherBlockQuantized (int4 embedding gather + symmetric dequant). These are the fp32 quant
// path used by the cpu-recipe decoder; the weight repack runs once and is cached on the Plan.

#include <cstdint>

#include "mlx_engine.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// MatMulNBits: Y[M,N] = A[M,K] @ dequant(B)^T (+bias). Repack our uint8 [N,nblocks,16] to MLX affine
// uint32 words + biases=-8*scale ONCE (cached by weight name), then mlx_quantized_matmul.
void MatMulNBitsOp(TranslationContext& ctx, const NodeDesc& n) {
  const int64_t K = n.ints.at("K");
  const int64_t N = n.ints.at("N");
  const int64_t block = 32, bits = 4;
  const int64_t nblocks = K / block;

  mlx_array a = ctx.Resolve(n.inputs[0]);
  const TensorRef& wref = n.inputs[1];
  const TensorRef& sref = n.inputs[2];

  // Repacked uint32 weight [N, K/8] (8 nibbles/word, low->high).
  mlx_array w = ctx.Cached(wref.name + "#qw", [&]() {
    const uint8_t* src = static_cast<const uint8_t*>(ctx.RawHost(wref).data);
    const int words = static_cast<int>(K / 8);
    std::vector<uint32_t> packed(static_cast<size_t>(N) * words, 0);
    for (int64_t row = 0; row < N; ++row) {
      for (int64_t k = 0; k < K; ++k) {
        const int64_t blk = k / block;
        const int64_t within = k % block;
        const int64_t byte = within / 2;
        const int nib = static_cast<int>(within % 2);
        const uint8_t b = src[(row * nblocks + blk) * 16 + byte];
        const uint32_t q = nib == 0 ? (b & 0x0F) : (b >> 4);
        const int64_t word = row * words + k / 8;
        packed[word] |= q << ((k % 8) * bits);
      }
    }
    int sh[2] = {static_cast<int>(N), words};
    return mlx_array_new_data(packed.data(), sh, 2, MLX_UINT32);
  });
  // Scales [N, nblocks] (wrapped raw) and biases = -8*scale.
  mlx_array scales = ctx.Cached(sref.name + "#sc", [&]() {
    int sh[2] = {static_cast<int>(N), static_cast<int>(nblocks)};
    return mlx_array_new_data(ctx.RawHost(sref).data, sh, 2, MLX_FLOAT32);
  });
  mlx_array biases = ctx.Cached(wref.name + "#bi", [&]() {
    const float* sc = static_cast<const float*>(ctx.RawHost(sref).data);
    std::vector<float> bi(static_cast<size_t>(N) * nblocks);
    for (size_t i = 0; i < bi.size(); ++i) bi[i] = -8.0f * sc[i];
    int sh[2] = {static_cast<int>(N), static_cast<int>(nblocks)};
    return mlx_array_new_data(bi.data(), sh, 2, MLX_FLOAT32);
  });

  // Flatten leading dims of A to [M, K].
  std::vector<int> ashape = TranslationContext::ShapeOf(a);
  int M = 1;
  for (size_t i = 0; i + 1 < ashape.size(); ++i) M *= ashape[i];
  mlx_array a2 = ctx.Reshape(a, {M, static_cast<int>(K)});

  mlx_array y = mlx_array_new();
  mlx_optional_int gs = {static_cast<int>(block), true};
  mlx_optional_int bb = {static_cast<int>(bits), true};
  MLX_CHECK(mlx_quantized_matmul(&y, a2, w, scales, biases, /*transpose=*/true, gs, bb, "affine",
                                 ctx.stream()));
  ctx.Keep(y);

  mlx_array out = y;
  if (n.inputs.size() == 4) out = ctx.AddA(out, ctx.Resolve(n.inputs[3]));

  // Restore leading dims with N as the last dim.
  std::vector<int> oshape(ashape);
  oshape.back() = static_cast<int>(N);
  ctx.Bind(n.outputs[0], ctx.Reshape(out, oshape));
}

// GatherBlockQuantized: gather int4 rows for input_ids and dequantize (symmetric zp=8).
void GatherBlockQuantizedOp(TranslationContext& ctx, const NodeDesc& n) {
  const int64_t block = n.ints.count("block_size") ? n.ints.at("block_size") : 32;
  const TensorRef& dref = n.inputs[0];  // uint8 [V, packed]
  mlx_array idx_in = ctx.Resolve(n.inputs[1]);
  const TensorRef& sref = n.inputs[2];  // f32 [V, nblocks]

  mlx_array data = ctx.Resolve(dref);    // uint8 [V, D/2]
  mlx_array scales = ctx.Resolve(sref);  // f32 [V, nblocks]

  // Flatten indices to 1D int32.
  std::vector<int> ish = TranslationContext::ShapeOf(idx_in);
  int BS = 1;
  for (int d : ish) BS *= d;
  mlx_array idx = ctx.Astype(ctx.Reshape(idx_in, {BS}), MLX_INT32);

  mlx_array g = mlx_array_new();  // [BS, D/2] uint8
  MLX_CHECK(mlx_take_axis(&g, data, idx, 0, ctx.stream()));
  ctx.Keep(g);
  mlx_array sg = mlx_array_new();  // [BS, nblocks]
  MLX_CHECK(mlx_take_axis(&sg, scales, idx, 0, ctx.stream()));
  ctx.Keep(sg);

  const int packed = TranslationContext::ShapeOf(g)[1];
  const int D = packed * 2;
  const int nblocks = static_cast<int>(D / block);

  mlx_array g32 = ctx.Astype(g, MLX_UINT32);
  mlx_array low = mlx_array_new();
  MLX_CHECK(mlx_bitwise_and(&low, g32, ctx.ScalarU32(0x0F), ctx.stream()));
  ctx.Keep(low);
  mlx_array hi_sh = mlx_array_new();
  MLX_CHECK(mlx_right_shift(&hi_sh, g32, ctx.ScalarU32(4), ctx.stream()));
  ctx.Keep(hi_sh);
  mlx_array high = mlx_array_new();
  MLX_CHECK(mlx_bitwise_and(&high, hi_sh, ctx.ScalarU32(0x0F), ctx.stream()));
  ctx.Keep(high);

  // Interleave low/high -> [BS, packed, 2] -> [BS, D].
  mlx_vector_array pair = mlx_vector_array_new();
  mlx_vector_array_append_value(pair, low);
  mlx_vector_array_append_value(pair, high);
  mlx_array stacked = mlx_array_new();
  MLX_CHECK(mlx_stack_axis(&stacked, pair, 2, ctx.stream()));
  ctx.Keep(stacked);
  mlx_vector_array_free(pair);
  mlx_array q = ctx.Reshape(stacked, {BS, D});
  mlx_array qf = ctx.Astype(q, MLX_FLOAT32);

  // Dequant: (q - 8) * scale, scale broadcast per 32-wide block.
  mlx_array eight = ctx.Keep(mlx_array_new_float32(8.0f));
  mlx_array centered = ctx.SubA(qf, eight);
  mlx_array sc_blocks = ctx.Reshape(sg, {BS, nblocks, 1});
  int bshape[3] = {BS, nblocks, static_cast<int>(block)};
  mlx_array sc_b = mlx_array_new();
  MLX_CHECK(mlx_broadcast_to(&sc_b, sc_blocks, bshape, 3, ctx.stream()));
  ctx.Keep(sc_b);
  mlx_array sc_full = ctx.Reshape(sc_b, {BS, D});
  mlx_array w = ctx.Mul(centered, sc_full);

  // Restore [.., D] output shape from the index tensor's shape.
  std::vector<int> oshape = ish;
  oshape.push_back(D);
  ctx.Bind(n.outputs[0], ctx.Reshape(w, oshape));
}

}  // namespace

void RegisterQuantOps(OpRegistry& registry) {
  registry.Register({"com.microsoft", "MatMulNBits", kAnyOpset, kAnyOpset, &MatMulNBitsOp});
  registry.Register(
      {"com.microsoft", "GatherBlockQuantized", kAnyOpset, kAnyOpset, &GatherBlockQuantizedOp});
}

}  // namespace ort_mps_mlx
