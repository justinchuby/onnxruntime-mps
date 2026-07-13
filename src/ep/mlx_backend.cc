// Copyright (c) 2026. Licensed under the MIT License.
//
// MLX (mlx-c) translation of a fused decoder subgraph — Phase-0 GO/NO-GO prototype (Nabil).
//
// Compiled ALWAYS, but only does real work when the plugin was configured with
// -DORT_MPS_ENABLE_MLX=ON (which defines ORT_MPS_HAS_MLX and links libmlxc/libmlx). Otherwise every
// entry point is an inert stub so the default hand-kernel build/behaviour is completely unchanged.
//
// See mlx_backend.h and docs/MLX_EVALUATION.md §6 (Phase 0).

#include "mlx_backend.h"

#include <cstdlib>
#include <cstring>

namespace ort_mps_mlx {

bool Enabled() { return Available() && std::getenv("ONNX_GENAI_METAL_EP_MLX") != nullptr; }

}  // namespace ort_mps_mlx

#ifndef ORT_MPS_HAS_MLX
// ---------------------------------------------------------------------------------------------
// Stub build (MLX not compiled in). Keeps the symbols so ep.cc links either way.
// ---------------------------------------------------------------------------------------------
namespace ort_mps_mlx {

bool Available() { return false; }
Plan* BuildPlan(std::vector<NodeDesc>, std::string& error) {
  error = "MLX support not compiled in (configure with -DORT_MPS_ENABLE_MLX=ON)";
  return nullptr;
}
void DestroyPlan(Plan*) {}
bool RunPlan(Plan&, Ort::KernelContext&, std::string& error) {
  error = "MLX support not compiled in";
  return false;
}

}  // namespace ort_mps_mlx

#else
// ---------------------------------------------------------------------------------------------
// Real MLX build.
// ---------------------------------------------------------------------------------------------
#include <cmath>
#include <stdexcept>

#include "mlx/c/mlx.h"

namespace ort_mps_mlx {

bool Available() { return true; }

namespace {

mlx_dtype ToMlx(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return MLX_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return MLX_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return MLX_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return MLX_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return MLX_UINT8;
    default: return MLX_FLOAT32;
  }
}

struct MlxError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define MLX_CHECK(expr)                                                    \
  do {                                                                     \
    if ((expr) != 0) throw MlxError(std::string("mlx call failed: ") + #expr); \
  } while (0)

}  // namespace

// Persistent, per-subgraph MLX state: the stream, tuned memory bounds, and the cache of
// repacked-weight / wrapped-initializer MLX arrays keyed by initializer name (reused every step so
// weights are repacked exactly once, not per token).
struct Plan {
  std::vector<NodeDesc> nodes;
  mlx_stream stream;
  std::unordered_map<std::string, mlx_array> cache;  // persistent (freed in DestroyPlan)

  Plan() { stream = mlx_default_gpu_stream_new(); }
  ~Plan() {
    for (auto& kv : cache) mlx_array_free(kv.second);
    mlx_stream_free(stream);
  }
};

bool Available();  // fwd (defined above)

namespace {

// Per-Compute execution context: builds the MLX graph for one forward pass, evals once, copies out.
class Run {
 public:
  Run(Plan& plan, Ort::KernelContext& ctx) : plan_(plan), ctx_(ctx), s_(plan.stream) {}

  ~Run() {
    for (mlx_array a : transient_) mlx_array_free(a);
  }

  void Execute() {
    for (const NodeDesc& node : plan_.nodes) Translate(node);

    // Collect boundary outputs and evaluate the whole graph in one shot.
    mlx_vector_array outs = mlx_vector_array_new();
    std::vector<const OutRef*> ext;
    for (const NodeDesc& node : plan_.nodes) {
      for (const OutRef& o : node.outputs) {
        if (o.external && env_.count(o.name)) {
          ext.push_back(&o);
          mlx_vector_array_append_value(outs, env_.at(o.name));
        }
      }
    }
    MLX_CHECK(mlx_eval(outs));
    mlx_vector_array_free(outs);

    // Copy each boundary output back across the ORT boundary (accepted boundary copy).
    for (const OutRef* o : ext) CopyOut(*o);
  }

 private:
  // ---- array bookkeeping ----
  mlx_array Keep(mlx_array a) {
    transient_.push_back(a);
    return a;
  }

  static std::vector<int> ToInt(const std::vector<int64_t>& v) {
    std::vector<int> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = static_cast<int>(v[i]);
    return r;
  }
  static std::vector<int> ShapeOf(mlx_array a) {
    size_t nd = mlx_array_ndim(a);
    const int* sh = mlx_array_shape(a);
    return std::vector<int>(sh, sh + nd);
  }

  // Raw host bytes for a weight/scale tensor. Constant initializers are surfaced by ORT either as
  // compile-time initializers (init_data) or, with drop_constant_initializers=false, as runtime
  // context inputs. Handle both so weight repack works regardless of how ORT hoisted them. The
  // returned pointer is valid for the current Run; MatMulNBits repacks once and caches, so reading
  // at the first Run is sufficient.
  struct HostBytes {
    const void* data = nullptr;
    std::vector<int64_t> shape;
    size_t count = 0;
  };
  HostBytes RawHost(const TensorRef& ref) {
    HostBytes h;
    if (ref.source == Src::Initializer) {
      h.data = ref.init_data;
      h.shape = ref.init_shape;
      h.count = ref.init_count;
    } else if (ref.source == Src::CtxInput) {
      Ort::ConstValue v = ctx_.GetInput(ref.ctx_index);
      auto info = v.GetTensorTypeAndShapeInfo();
      h.data = v.GetTensorRawData();
      h.shape = info.GetShape();
      h.count = info.GetElementCount();
    } else {
      throw MlxError("MLX: RawHost on non-constant input " + ref.name);
    }
    return h;
  }

  // ---- input resolution ----
  // Intermediate -> produced env; CtxInput -> wrap ORT input (per-run); Initializer -> wrap raw
  // once and cache persistently on the plan (gammas, biases, cos/sin, embedding table, ...).
  mlx_array Resolve(const TensorRef& ref) {
    switch (ref.source) {
      case Src::Intermediate: {
        auto it = env_.find(ref.name);
        if (it == env_.end()) throw MlxError("MLX: missing intermediate " + ref.name);
        return it->second;
      }
      case Src::CtxInput: {
        // Constant ctx inputs (hoisted initializers) are wrapped once and cached persistently on the
        // plan; genuinely dynamic inputs (ids, position, KV cache) are wrapped per-run in env_.
        if (ref.constant) {
          auto ci = plan_.cache.find(ref.name);
          if (ci != plan_.cache.end()) return ci->second;
        } else {
          auto it = env_.find(ref.name);
          if (it != env_.end()) return it->second;
        }
        Ort::ConstValue v = ctx_.GetInput(ref.ctx_index);
        auto info = v.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shp = info.GetShape();
        std::vector<int> ishp = ToInt(shp);
        mlx_array raw = mlx_array_new_data(v.GetTensorRawData(), ishp.data(),
                                           static_cast<int>(ishp.size()),
                                           ToMlx(info.GetElementType()));
        if (ref.constant) {
          plan_.cache[ref.name] = raw;  // persistent copy; ctx data is read only on the first Run
          return raw;
        }
        mlx_array a = Keep(raw);
        env_[ref.name] = a;
        return a;
      }
      case Src::Initializer: {
        auto it = plan_.cache.find(ref.name);
        if (it != plan_.cache.end()) return it->second;
        std::vector<int> ishp = ToInt(ref.init_shape);
        mlx_array a = mlx_array_new_data(ref.init_data, ishp.data(),
                                         static_cast<int>(ishp.size()), ToMlx(ref.init_type));
        plan_.cache[ref.name] = a;  // persistent
        return a;
      }
      default:
        throw MlxError("MLX: absent input");
    }
  }

  // Fetch-or-build a persistent cached array under `key` using `build` (for repacked weights).
  template <typename F>
  mlx_array Cached(const std::string& key, F&& build) {
    auto it = plan_.cache.find(key);
    if (it != plan_.cache.end()) return it->second;
    mlx_array a = build();
    plan_.cache[key] = a;
    return a;
  }

  void Bind(const OutRef& o, mlx_array a) { env_[o.name] = a; }

  // ---- op translations ----
  void Translate(const NodeDesc& n) {
    const std::string& op = n.op_type;
    if (op == "MatMulNBits") return MatMulNBits(n);
    if (op == "GroupQueryAttention") return GroupQueryAttention(n);
    if (op == "RMSNormalization") return RmsNorm(n);
    if (op == "SkipSimplifiedLayerNormalization") return SkipRmsNorm(n);
    if (op == "GatherBlockQuantized") return GatherBlockQuantized(n);
    if (op == "Add") return Binary(n, /*mul=*/false);
    if (op == "Mul") return Binary(n, /*mul=*/true);
    if (op == "Sub") return SubNode(n);
    if (op == "Sigmoid") return Sigmoid(n);
    if (op == "Softmax") return Softmax(n);
    if (op == "Cast") return Cast(n);
    throw MlxError("MLX: no translation for op " + n.op_type);
  }

  mlx_array Reshape(mlx_array a, const std::vector<int>& shape) {
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_reshape(&r, a, shape.data(), shape.size(), s_));
    return Keep(r);
  }
  mlx_array Transpose(mlx_array a, const std::vector<int>& axes) {
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_transpose_axes(&r, a, axes.data(), axes.size(), s_));
    return Keep(r);
  }
  mlx_array Astype(mlx_array a, mlx_dtype t) {
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_astype(&r, a, t, s_));
    return Keep(r);
  }
  mlx_array Mul(mlx_array a, mlx_array b) {
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_multiply(&r, a, b, s_));
    return Keep(r);
  }
  mlx_array AddA(mlx_array a, mlx_array b) {
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_add(&r, a, b, s_));
    return Keep(r);
  }
  mlx_array SubA(mlx_array a, mlx_array b) {
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_subtract(&r, a, b, s_));
    return Keep(r);
  }
  mlx_array Slice(mlx_array a, const std::vector<int>& start, const std::vector<int>& stop) {
    std::vector<int> strides(start.size(), 1);
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_slice(&r, a, start.data(), start.size(), stop.data(), stop.size(),
                        strides.data(), strides.size(), s_));
    return Keep(r);
  }
  mlx_array Concat2(mlx_array a, mlx_array b, int axis) {
    mlx_vector_array v = mlx_vector_array_new();
    mlx_vector_array_append_value(v, a);
    mlx_vector_array_append_value(v, b);
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_concatenate_axis(&r, v, axis, s_));
    mlx_vector_array_free(v);
    return Keep(r);
  }
  mlx_array ScalarU32(uint32_t val) {
    mlx_array r = mlx_array_new_data(&val, nullptr, 0, MLX_UINT32);
    return Keep(r);
  }

  // MatMulNBits: Y[M,N] = A[M,K] @ dequant(B)^T (+bias). Repack our uint8 [N,nblocks,16] to MLX
  // affine uint32 words + biases=-8*scale ONCE (cached by weight name), then mlx_quantized_matmul.
  void MatMulNBits(const NodeDesc& n) {
    const int64_t K = n.ints.at("K");
    const int64_t N = n.ints.at("N");
    const int64_t block = 32, bits = 4;
    const int64_t nblocks = K / block;

    mlx_array a = Resolve(n.inputs[0]);
    const TensorRef& wref = n.inputs[1];
    const TensorRef& sref = n.inputs[2];

    // Repacked uint32 weight [N, K/8] (8 nibbles/word, low->high).
    mlx_array w = Cached(wref.name + "#qw", [&]() {
      const uint8_t* src = static_cast<const uint8_t*>(RawHost(wref).data);
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
    mlx_array scales = Cached(sref.name + "#sc", [&]() {
      int sh[2] = {static_cast<int>(N), static_cast<int>(nblocks)};
      return mlx_array_new_data(RawHost(sref).data, sh, 2, MLX_FLOAT32);
    });
    mlx_array biases = Cached(wref.name + "#bi", [&]() {
      const float* sc = static_cast<const float*>(RawHost(sref).data);
      std::vector<float> bi(static_cast<size_t>(N) * nblocks);
      for (size_t i = 0; i < bi.size(); ++i) bi[i] = -8.0f * sc[i];
      int sh[2] = {static_cast<int>(N), static_cast<int>(nblocks)};
      return mlx_array_new_data(bi.data(), sh, 2, MLX_FLOAT32);
    });

    // Flatten leading dims of A to [M, K].
    std::vector<int> ashape = ShapeOf(a);
    int M = 1;
    for (size_t i = 0; i + 1 < ashape.size(); ++i) M *= ashape[i];
    mlx_array a2 = Reshape(a, {M, static_cast<int>(K)});

    mlx_array y = mlx_array_new();
    mlx_optional_int gs = {static_cast<int>(block), true};
    mlx_optional_int bb = {static_cast<int>(bits), true};
    MLX_CHECK(mlx_quantized_matmul(&y, a2, w, scales, biases, /*transpose=*/true, gs, bb, "affine", s_));
    Keep(y);

    mlx_array out = y;
    if (n.inputs.size() == 4) out = AddA(out, Resolve(n.inputs[3]));

    // Restore leading dims with N as the last dim.
    std::vector<int> oshape(ashape);
    oshape.back() = static_cast<int>(N);
    Bind(n.outputs[0], Reshape(out, oshape));
  }

  void RmsNorm(const NodeDesc& n) {
    mlx_array x = Resolve(n.inputs[0]);
    mlx_array g = Resolve(n.inputs[1]);
    float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-6f;
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_fast_rms_norm(&r, x, g, eps, s_));
    Bind(n.outputs[0], Keep(r));
  }

  // residual = input + skip; out = rms_norm(residual) * gamma. out[0]=normalized, out[last]=residual.
  void SkipRmsNorm(const NodeDesc& n) {
    mlx_array input = Resolve(n.inputs[0]);
    mlx_array skip = Resolve(n.inputs[1]);
    mlx_array gamma = Resolve(n.inputs[2]);
    float eps = n.floats.count("epsilon") ? n.floats.at("epsilon") : 1e-6f;
    mlx_array residual = AddA(input, skip);
    mlx_array norm = mlx_array_new();
    MLX_CHECK(mlx_fast_rms_norm(&norm, residual, gamma, eps, s_));
    Keep(norm);
    Bind(n.outputs[0], norm);
    if (n.outputs.size() >= 2) Bind(n.outputs.back(), residual);
  }

  void Binary(const NodeDesc& n, bool mul) {
    mlx_array a = Resolve(n.inputs[0]);
    mlx_array b = Resolve(n.inputs[1]);
    Bind(n.outputs[0], mul ? Mul(a, b) : AddA(a, b));
  }

  // Integer Sub used only by the seqlens-prep chain (seqlens_k = ReduceSum(mask) - 1). MLX-GQA does
  // not consume seqlens (it uses causal masking on the full KV length), so this node is dead in the
  // MLX graph, but it must still translate to keep the whole subgraph mappable.
  void SubNode(const NodeDesc& n) {
    mlx_array a = Resolve(n.inputs[0]);
    mlx_array b = Resolve(n.inputs[1]);
    Bind(n.outputs[0], SubA(a, b));
  }

  void Sigmoid(const NodeDesc& n) {
    mlx_array a = Resolve(n.inputs[0]);
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_sigmoid(&r, a, s_));
    Bind(n.outputs[0], Keep(r));
  }

  void Softmax(const NodeDesc& n) {
    mlx_array a = Resolve(n.inputs[0]);
    int axis = -1;
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_softmax_axis(&r, a, axis, /*precise=*/true, s_));
    Bind(n.outputs[0], Keep(r));
  }

  void Cast(const NodeDesc& n) {
    mlx_array a = Resolve(n.inputs[0]);
    Bind(n.outputs[0], Astype(a, ToMlx(n.outputs[0].type)));
  }

  // Embedding: gather int4 rows for input_ids and dequantize (symmetric zp=8, block=32).
  void GatherBlockQuantized(const NodeDesc& n) {
    const int64_t block = n.ints.count("block_size") ? n.ints.at("block_size") : 32;
    const TensorRef& dref = n.inputs[0];  // uint8 [V, packed]
    mlx_array idx_in = Resolve(n.inputs[1]);
    const TensorRef& sref = n.inputs[2];  // f32 [V, nblocks]

    mlx_array data = Resolve(dref);    // uint8 [V, D/2]
    mlx_array scales = Resolve(sref);  // f32 [V, nblocks]

    // Flatten indices to 1D int32.
    std::vector<int> ish = ShapeOf(idx_in);
    int BS = 1;
    for (int d : ish) BS *= d;
    mlx_array idx = Astype(Reshape(idx_in, {BS}), MLX_INT32);

    mlx_array g = mlx_array_new();  // [BS, D/2] uint8
    MLX_CHECK(mlx_take_axis(&g, data, idx, 0, s_));
    Keep(g);
    mlx_array sg = mlx_array_new();  // [BS, nblocks]
    MLX_CHECK(mlx_take_axis(&sg, scales, idx, 0, s_));
    Keep(sg);

    const int packed = ShapeOf(g)[1];
    const int D = packed * 2;
    const int nblocks = static_cast<int>(D / block);

    mlx_array g32 = Astype(g, MLX_UINT32);
    mlx_array low = mlx_array_new();
    MLX_CHECK(mlx_bitwise_and(&low, g32, ScalarU32(0x0F), s_));
    Keep(low);
    mlx_array hi_sh = mlx_array_new();
    MLX_CHECK(mlx_right_shift(&hi_sh, g32, ScalarU32(4), s_));
    Keep(hi_sh);
    mlx_array high = mlx_array_new();
    MLX_CHECK(mlx_bitwise_and(&high, hi_sh, ScalarU32(0x0F), s_));
    Keep(high);

    // Interleave low/high -> [BS, packed, 2] -> [BS, D].
    mlx_vector_array pair = mlx_vector_array_new();
    mlx_vector_array_append_value(pair, low);
    mlx_vector_array_append_value(pair, high);
    mlx_array stacked = mlx_array_new();
    MLX_CHECK(mlx_stack_axis(&stacked, pair, 2, s_));
    Keep(stacked);
    mlx_vector_array_free(pair);
    mlx_array q = Reshape(stacked, {BS, D});
    mlx_array qf = Astype(q, MLX_FLOAT32);

    // Dequant: (q - 8) * scale, scale broadcast per 32-wide block.
    mlx_array eight = Keep(mlx_array_new_float32(8.0f));
    mlx_array centered = SubA(qf, eight);
    mlx_array sc_blocks = Reshape(sg, {BS, nblocks, 1});
    int bshape[3] = {BS, nblocks, static_cast<int>(block)};
    mlx_array sc_b = mlx_array_new();
    MLX_CHECK(mlx_broadcast_to(&sc_b, sc_blocks, bshape, 3, s_));
    Keep(sc_b);
    mlx_array sc_full = Reshape(sc_b, {BS, D});
    mlx_array w = Mul(centered, sc_full);

    // Restore [.., D] output shape from the index tensor's shape.
    std::vector<int> oshape = ish;
    oshape.push_back(D);
    Bind(n.outputs[0], Reshape(w, oshape));
  }

  // GroupQueryAttention: rope(q, new-k) with the provided cos/sin cache, append to KV cache, GQA
  // causal SDPA. Layout matches com.microsoft.GroupQueryAttention (fp32, batch-first).
  void GroupQueryAttention(const NodeDesc& n) {
    const int num_heads = static_cast<int>(n.ints.at("num_heads"));
    const int kv_heads = static_cast<int>(n.ints.at("kv_num_heads"));
    const bool interleaved = n.ints.count("rotary_interleaved") && n.ints.at("rotary_interleaved");
    const bool do_rotary = !n.ints.count("do_rotary") || n.ints.at("do_rotary");

    mlx_array q = Resolve(n.inputs[0]);       // [B,S,num*hd]
    mlx_array k = Resolve(n.inputs[1]);       // [B,S,kv*hd]
    mlx_array v = Resolve(n.inputs[2]);       // [B,S,kv*hd]
    mlx_array past_k = Resolve(n.inputs[3]);  // [B,kv,past,hd]
    mlx_array past_v = Resolve(n.inputs[4]);

    std::vector<int> qs = ShapeOf(q);
    const int B = qs[0], S = qs[1];
    const int head = qs[2] / num_heads;
    const int past = ShapeOf(past_k)[2];

    float scale = n.floats.count("scale") && n.floats.at("scale") != 0.0f
                      ? n.floats.at("scale")
                      : 1.0f / std::sqrt(static_cast<float>(head));

    // [B,S,H*hd] -> [B,H,S,hd].
    auto to_heads = [&](mlx_array x, int h) {
      return Transpose(Reshape(x, {B, S, h, head}), {0, 2, 1, 3});
    };
    mlx_array qh = to_heads(q, num_heads);
    mlx_array kh = to_heads(k, kv_heads);
    mlx_array vh = to_heads(v, kv_heads);

    if (do_rotary) {
      mlx_array cos = Resolve(n.inputs[7]);  // [max_seq, rot/2]
      mlx_array sin = Resolve(n.inputs[8]);
      const int half = ShapeOf(cos)[1];  // rot/2
      // cos/sin rows for positions [past, past+S).
      mlx_array cr = Reshape(Slice(cos, {past, 0}, {past + S, half}), {1, 1, S, half});
      mlx_array sr = Reshape(Slice(sin, {past, 0}, {past + S, half}), {1, 1, S, half});
      qh = Rope(qh, cr, sr, half, interleaved);
      kh = Rope(kh, cr, sr, half, interleaved);
    }

    // Append to KV cache along the sequence axis.
    mlx_array present_k = Concat2(past_k, kh, 2);
    mlx_array present_v = Concat2(past_v, vh, 2);

    mlx_array attn = mlx_array_new();
    MLX_CHECK(mlx_fast_scaled_dot_product_attention(&attn, qh, present_k, present_v, scale,
                                                    "causal", /*mask=*/mlx_array_empty,
                                                    /*sinks=*/mlx_array_empty, s_));
    Keep(attn);
    // [B,H,S,hd] -> [B,S,H*hd].
    mlx_array out = Reshape(Transpose(attn, {0, 2, 1, 3}), {B, S, num_heads * head});

    Bind(n.outputs[0], out);
    if (n.outputs.size() >= 2) Bind(n.outputs[1], present_k);
    if (n.outputs.size() >= 3) Bind(n.outputs[2], present_v);
  }

  // Rotate-half (non-interleaved) or interleaved rotary over the first 2*half head dims.
  mlx_array Rope(mlx_array x, mlx_array cos, mlx_array sin, int half, bool interleaved) {
    std::vector<int> xs = ShapeOf(x);  // [B,H,S,hd]
    const int hd = xs[3];
    const int rot = 2 * half;
    mlx_array rotated;
    if (!interleaved) {
      mlx_array x1 = Slice(x, {0, 0, 0, 0}, {xs[0], xs[1], xs[2], half});
      mlx_array x2 = Slice(x, {0, 0, 0, half}, {xs[0], xs[1], xs[2], rot});
      mlx_array o1 = SubA(Mul(x1, cos), Mul(x2, sin));
      mlx_array o2 = AddA(Mul(x2, cos), Mul(x1, sin));
      rotated = Concat2(o1, o2, 3);
    } else {
      // interleaved: pairs (2i, 2i+1). Reshape to [...,half,2], split, recombine.
      mlx_array xr = Reshape(Slice(x, {0, 0, 0, 0}, {xs[0], xs[1], xs[2], rot}),
                             {xs[0], xs[1], xs[2], half, 2});
      mlx_array xe = Slice(xr, {0, 0, 0, 0, 0}, {xs[0], xs[1], xs[2], half, 1});
      mlx_array xo = Slice(xr, {0, 0, 0, 0, 1}, {xs[0], xs[1], xs[2], half, 2});
      mlx_array c = Reshape(cos, {xs[0], 1, xs[2], half, 1});  // broadcast on heads
      mlx_array sn = Reshape(sin, {xs[0], 1, xs[2], half, 1});
      xe = Reshape(xe, {xs[0], xs[1], xs[2], half, 1});
      xo = Reshape(xo, {xs[0], xs[1], xs[2], half, 1});
      mlx_array oe = SubA(Mul(xe, c), Mul(xo, sn));
      mlx_array oo = AddA(Mul(xo, c), Mul(xe, sn));
      rotated = Reshape(Concat2(oe, oo, 4), {xs[0], xs[1], xs[2], rot});
    }
    if (rot == hd) return rotated;
    mlx_array tail = Slice(x, {0, 0, 0, rot}, {xs[0], xs[1], xs[2], hd});
    return Concat2(rotated, tail, 3);
  }

  void CopyOut(const OutRef& o) {
    mlx_array a = env_.at(o.name);
    std::vector<int> sh = ShapeOf(a);
    std::vector<int64_t> shp(sh.begin(), sh.end());
    size_t count = 1;
    for (int d : sh) count *= static_cast<size_t>(d);
    Ort::UnownedValue out = ctx_.GetOutput(o.ctx_index, shp);
    const float* src = mlx_array_data_float32(a);
    std::memcpy(out.GetTensorMutableRawData(), src, count * sizeof(float));
  }

  Plan& plan_;
  Ort::KernelContext& ctx_;
  mlx_stream s_;
  std::unordered_map<std::string, mlx_array> env_;
  std::vector<mlx_array> transient_;
};

// Ops we can translate; used to reject a subgraph (keeping the hand path) rather than throw later.
bool Supported(const std::string& op) {
  static const char* kOps[] = {"MatMulNBits", "GroupQueryAttention", "RMSNormalization",
                               "SkipSimplifiedLayerNormalization", "GatherBlockQuantized",
                               "Add", "Mul", "Sub", "Sigmoid", "Softmax", "Cast"};
  for (const char* o : kOps)
    if (op == o) return true;
  return false;
}

}  // namespace

Plan* BuildPlan(std::vector<NodeDesc> nodes, std::string& error) {
  for (const NodeDesc& n : nodes) {
    if (!Supported(n.op_type)) {
      error = "MLX backend cannot translate op '" + n.op_type + "'";
      return nullptr;
    }
  }
  // Bound MLX's caching allocator so it coexists with our MTLBuffer pool (memory-safety note).
  size_t prev = 0;
  mlx_set_cache_limit(&prev, static_cast<size_t>(512) << 20);   // 512 MB cache cap
  mlx_set_wired_limit(&prev, static_cast<size_t>(1) << 30);     // 1 GB wired cap
  auto* plan = new Plan();
  plan->nodes = std::move(nodes);
  return plan;
}

void DestroyPlan(Plan* plan) { delete plan; }

bool RunPlan(Plan& plan, Ort::KernelContext& ctx, std::string& error) {
  try {
    Run run(plan, ctx);
    run.Execute();
    return true;
  } catch (const std::exception& ex) {
    error = ex.what();
    return false;
  }
}

}  // namespace ort_mps_mlx

#endif  // ORT_MPS_HAS_MLX
