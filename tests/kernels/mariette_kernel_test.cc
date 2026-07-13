// Copyright (c) 2026. Licensed under the MIT License.
//
// Standalone CPU-reference correctness tests for Mariette's hot-path Metal
// kernels: MatMulNBits (int4 block-quantized), RMSNormalization,
// SkipSimplifiedLayerNormalization, and Softmax. Each test builds a small input,
// runs the Metal kernel, and compares against a scalar CPU reference with an
// fp32-aware tolerance.

#include "metal_context.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckNear(float actual, float expected, float tolerance, const std::string& label) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(label + ": got " + std::to_string(actual) + ", expected " +
                             std::to_string(expected));
  }
}

void Require(bool ok, const std::string& error) {
  if (!ok) throw std::runtime_error(error);
}

// Pack a [N, nblocks, 16] uint8 int4 weight tensor and matching scales, and
// return the fully dequantized weight w[n][k] = (q - 8) * scale[n][block].
// Nibble convention (ORT MatMulNBits): within a 32-element block, byte = k/2,
// low nibble holds the even element, high nibble the odd element.
void BuildQuantWeights(size_t N, size_t K, size_t block, std::mt19937& rng,
                       std::vector<uint8_t>& packed, std::vector<float>& scales,
                       std::vector<float>& dequant) {
  const size_t nblocks = K / block;
  packed.assign(N * nblocks * (block / 2), 0);
  scales.assign(N * nblocks, 0.0f);
  dequant.assign(N * K, 0.0f);
  std::uniform_int_distribution<int> qdist(0, 15);
  std::uniform_real_distribution<float> sdist(0.005f, 0.05f);
  for (size_t n = 0; n < N; ++n) {
    for (size_t b = 0; b < nblocks; ++b) {
      const float scale = sdist(rng);
      scales[n * nblocks + b] = scale;
      uint8_t* blk = packed.data() + (n * nblocks + b) * (block / 2);
      for (size_t e = 0; e < block; ++e) {
        const int q = qdist(rng);
        const size_t byte = e / 2;
        if ((e & 1) == 0) {
          blk[byte] = static_cast<uint8_t>((blk[byte] & 0xF0) | (q & 0x0F));
        } else {
          blk[byte] = static_cast<uint8_t>((blk[byte] & 0x0F) | ((q & 0x0F) << 4));
        }
        dequant[n * K + b * block + e] = (static_cast<float>(q) - 8.0f) * scale;
      }
    }
  }
}

void TestMatMulNBits(ort_mps::MetalContext& metal) {
  std::mt19937 rng(1234);
  const size_t block = 32;
  // Two shapes: a decode GEMV (M=1) and a small prefill GEMM (M>1).
  for (const size_t M : {size_t(1), size_t(3)}) {
    const size_t N = 8;
    const size_t K = 64;  // 2 blocks
    std::vector<uint8_t> packed;
    std::vector<float> scales, dequant;
    BuildQuantWeights(N, K, block, rng, packed, scales, dequant);

    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::vector<float> a(M * K);
    for (auto& v : a) v = adist(rng);
    std::vector<float> bias(N);
    for (auto& v : bias) v = adist(rng);

    std::vector<float> y(M * N, 0.0f);
    std::string error;
    Require(metal.MatMulNBitsF32(a.data(), packed.data(), scales.data(), bias.data(), y.data(), M,
                                 N, K, K / block, error),
            error);

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float ref = bias[n];
        for (size_t k = 0; k < K; ++k) ref += a[m * K + k] * dequant[n * K + k];
        CheckNear(y[m * N + n], ref, 1e-3f, "MatMulNBits M=" + std::to_string(M));
      }
    }
  }
}

void TestRmsNorm(ort_mps::MetalContext& metal) {
  const size_t rows = 4, d = 96;
  const float eps = 1e-6f;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> x(rows * d), gamma(d), y(rows * d, 0.0f);
  for (auto& v : x) v = dist(rng);
  for (auto& v : gamma) v = dist(rng);

  std::string error;
  Require(metal.RmsNormF32(x.data(), gamma.data(), y.data(), rows, d, eps, error), error);

  for (size_t r = 0; r < rows; ++r) {
    float ss = 0.0f;
    for (size_t i = 0; i < d; ++i) ss += x[r * d + i] * x[r * d + i];
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
    for (size_t i = 0; i < d; ++i) {
      CheckNear(y[r * d + i], x[r * d + i] * inv * gamma[i], 1e-3f, "RMSNorm");
    }
  }
}

void TestSkipSimplifiedLayerNorm(ort_mps::MetalContext& metal) {
  const size_t rows = 3, d = 64;
  const float eps = 1e-6f;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
  std::vector<float> in(rows * d), skip(rows * d), gamma(d);
  for (auto& v : in) v = dist(rng);
  for (auto& v : skip) v = dist(rng);
  for (auto& v : gamma) v = dist(rng);
  std::vector<float> out(rows * d, 0.0f), residual(rows * d, 0.0f);

  std::string error;
  Require(metal.SkipSimplifiedLayerNormF32(in.data(), skip.data(), gamma.data(), out.data(),
                                           residual.data(), rows, d, eps, error),
          error);

  for (size_t r = 0; r < rows; ++r) {
    std::vector<float> res(d);
    float ss = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      res[i] = in[r * d + i] + skip[r * d + i];
      ss += res[i] * res[i];
    }
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
    for (size_t i = 0; i < d; ++i) {
      CheckNear(residual[r * d + i], res[i], 1e-4f, "SkipLN residual");
      CheckNear(out[r * d + i], res[i] * inv * gamma[i], 1e-3f, "SkipLN normalized");
    }
  }
}

void TestSoftmax(ort_mps::MetalContext& metal) {
  const size_t rows = 4, d = 50;
  std::mt19937 rng(23);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  std::vector<float> x(rows * d), y(rows * d, 0.0f);
  for (auto& v : x) v = dist(rng);

  std::string error;
  Require(metal.SoftmaxF32(x.data(), y.data(), rows, d, error), error);

  for (size_t r = 0; r < rows; ++r) {
    float mx = -1e30f;
    for (size_t i = 0; i < d; ++i) mx = std::max(mx, x[r * d + i]);
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) sum += std::exp(x[r * d + i] - mx);
    for (size_t i = 0; i < d; ++i) {
      CheckNear(y[r * d + i], std::exp(x[r * d + i] - mx) / sum, 1e-5f, "Softmax");
    }
  }
}

// Applies half-split / interleaved rotary (ORT convention) to a single head vector at `pos`.
void RopeRef(std::vector<float>& x, size_t pos, size_t head, size_t rotary_dim, bool interleaved,
             const std::vector<float>& cos, const std::vector<float>& sin) {
  const size_t half = rotary_dim / 2;
  std::vector<float> out(x);
  for (size_t d = 0; d < rotary_dim; ++d) {
    size_t pair, partner;
    float sign;
    if (interleaved) {
      pair = d >> 1;
      partner = d ^ 1u;
      sign = (d & 1u) == 0 ? -1.0f : 1.0f;
    } else {
      pair = d < half ? d : d - half;
      partner = d < half ? d + half : d - half;
      sign = d < half ? -1.0f : 1.0f;
    }
    const float c = cos[pos * half + pair];
    const float s = sin[pos * half + pair];
    out[d] = x[d] * c + sign * x[partner] * s;
  }
  x = out;
  (void)head;
}

// CPU reference for com.microsoft.GroupQueryAttention (semantics verified against ORT CPU EP).
void GqaReference(size_t B, size_t S, size_t H, size_t KVH, size_t HS, size_t past, size_t present_seq,
                  bool do_rotary, bool interleaved, int local_window, float scale,
                  const std::vector<float>& q, const std::vector<float>& k,
                  const std::vector<float>& v, const std::vector<float>& past_k,
                  const std::vector<float>& past_v, const std::vector<float>& cos,
                  const std::vector<float>& sin, size_t rotary_dim, std::vector<float>& out,
                  std::vector<float>& present_k, std::vector<float>& present_v) {
  const size_t group = H / KVH;
  present_k.assign(B * KVH * present_seq * HS, 0.0f);
  present_v.assign(B * KVH * present_seq * HS, 0.0f);
  out.assign(B * S * H * HS, 0.0f);
  const size_t past_seq = past;  // reference past buffer sized exactly to `past`
  for (size_t b = 0; b < B; ++b) {
    const size_t total = past + S;
    for (size_t pos = 0; pos < total; ++pos) {
      for (size_t kh = 0; kh < KVH; ++kh) {
        float* pk = &present_k[((b * KVH + kh) * present_seq + pos) * HS];
        float* pv = &present_v[((b * KVH + kh) * present_seq + pos) * HS];
        if (pos < past) {
          const float* sk = &past_k[((b * KVH + kh) * past_seq + pos) * HS];
          const float* sv = &past_v[((b * KVH + kh) * past_seq + pos) * HS];
          for (size_t d = 0; d < HS; ++d) { pk[d] = sk[d]; pv[d] = sv[d]; }
        } else {
          const size_t i = pos - past;
          std::vector<float> kk(k.begin() + ((b * S + i) * KVH + kh) * HS,
                                k.begin() + ((b * S + i) * KVH + kh) * HS + HS);
          if (do_rotary) RopeRef(kk, pos, HS, rotary_dim, interleaved, cos, sin);
          for (size_t d = 0; d < HS; ++d) {
            pk[d] = kk[d];
            pv[d] = v[((b * S + i) * KVH + kh) * HS + d];
          }
        }
      }
    }
    for (size_t i = 0; i < S; ++i) {
      const size_t posn = past + i;
      for (size_t h = 0; h < H; ++h) {
        const size_t kh = h / group;
        std::vector<float> qq(q.begin() + ((b * S + i) * H + h) * HS,
                              q.begin() + ((b * S + i) * H + h) * HS + HS);
        if (do_rotary) RopeRef(qq, posn, HS, rotary_dim, interleaved, cos, sin);
        size_t lo = 0;
        if (local_window >= 0) {
          const long l = (long)posn - local_window + 1;
          lo = l > 0 ? (size_t)l : 0;
        }
        std::vector<float> scores;
        for (size_t j = lo; j <= posn; ++j) {
          const float* pk = &present_k[((b * KVH + kh) * present_seq + j) * HS];
          float s = 0.0f;
          for (size_t d = 0; d < HS; ++d) s += qq[d] * pk[d];
          scores.push_back(s * scale);
        }
        float m = -1e30f;
        for (float s : scores) m = std::max(m, s);
        float sum = 0.0f;
        for (float& s : scores) { s = std::exp(s - m); sum += s; }
        std::vector<float> acc(HS, 0.0f);
        for (size_t idx = 0; idx < scores.size(); ++idx) {
          const size_t j = lo + idx;
          const float* pv = &present_v[((b * KVH + kh) * present_seq + j) * HS];
          for (size_t d = 0; d < HS; ++d) acc[d] += scores[idx] * pv[d];
        }
        for (size_t d = 0; d < HS; ++d) out[((b * S + i) * H + h) * HS + d] = acc[d] / sum;
      }
    }
  }
}

void TestGroupQueryAttention(ort_mps::MetalContext& metal) {
  std::mt19937 rng(4242);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t H = 4, KVH = 2, HS = 64;
  const size_t rotary_dim = HS;

  struct Case {
    const char* tag;
    size_t B, S, past;
    bool do_rotary, interleaved;
    int local_window;
  };
  const Case cases[] = {
      {"decode", 1, 1, 5, true, false, -1},
      {"decode-norope", 1, 1, 5, false, false, -1},
      {"decode-batch", 2, 1, 7, true, false, -1},
      {"prefill", 1, 6, 0, true, false, -1},
      {"prefill-interleaved", 1, 4, 0, true, true, -1},
      {"chunked", 1, 3, 4, true, false, -1},
      {"decode-swa", 1, 1, 10, true, false, 3},
      {"prefill-swa", 1, 6, 0, true, false, 2},
  };

  for (const Case& c : cases) {
    const size_t total = c.past + c.S;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HS));
    std::vector<float> q(c.B * c.S * H * HS), k(c.B * c.S * KVH * HS), v(c.B * c.S * KVH * HS);
    std::vector<float> past_k(c.B * KVH * c.past * HS), past_v(c.B * KVH * c.past * HS);
    std::vector<float> cos(total * (rotary_dim / 2)), sin(total * (rotary_dim / 2));
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);
    for (auto& x : past_k) x = dist(rng);
    for (auto& x : past_v) x = dist(rng);
    for (size_t p = 0; p < total; ++p) {
      for (size_t j = 0; j < rotary_dim / 2; ++j) {
        const float ang = dist(rng);
        cos[p * (rotary_dim / 2) + j] = std::cos(ang);
        sin[p * (rotary_dim / 2) + j] = std::sin(ang);
      }
    }

    std::vector<float> ref_out, ref_pk, ref_pv;
    GqaReference(c.B, c.S, H, KVH, HS, c.past, total, c.do_rotary, c.interleaved, c.local_window,
                 scale, q, k, v, past_k, past_v, cos, sin, rotary_dim, ref_out, ref_pk, ref_pv);

    std::vector<float> out(c.B * c.S * H * HS, 0.0f);
    std::vector<float> pk(c.B * KVH * total * HS, 0.0f), pv(c.B * KVH * total * HS, 0.0f);
    std::vector<int32_t> seqlens(c.B, static_cast<int32_t>(total) - 1);

    ort_mps::GroupQueryAttentionParams params;
    params.batch_size = static_cast<uint32_t>(c.B);
    params.sequence_length = static_cast<uint32_t>(c.S);
    params.num_heads = static_cast<uint32_t>(H);
    params.kv_num_heads = static_cast<uint32_t>(KVH);
    params.head_size = static_cast<uint32_t>(HS);
    params.rotary_dim = c.do_rotary ? static_cast<uint32_t>(rotary_dim) : 0u;
    params.past_seq = static_cast<uint32_t>(c.past);
    params.present_seq = static_cast<uint32_t>(total);
    params.do_rotary = c.do_rotary;
    params.interleaved = c.interleaved;
    params.local_window_size = c.local_window;
    params.scale = scale;

    std::string error;
    Require(metal.GroupQueryAttention(q.data(), k.data(), v.data(), past_k.data(), past_v.data(),
                                      seqlens.data(), c.do_rotary ? cos.data() : nullptr,
                                      c.do_rotary ? sin.data() : nullptr, out.data(), pk.data(),
                                      pv.data(), params, error),
            error);

    const std::string tag = c.tag;
    for (size_t idx = 0; idx < out.size(); ++idx) {
      CheckNear(out[idx], ref_out[idx], 2e-3f, "GQA out " + tag);
    }
    for (size_t b = 0; b < c.B; ++b)
      for (size_t kh = 0; kh < KVH; ++kh)
        for (size_t pos = 0; pos < total; ++pos)
          for (size_t d = 0; d < HS; ++d) {
            const size_t o = ((b * KVH + kh) * total + pos) * HS + d;
            CheckNear(pk[o], ref_pk[o], 2e-3f, "GQA present_key " + tag);
            CheckNear(pv[o], ref_pv[o], 2e-3f, "GQA present_value " + tag);
          }
  }
}

}  // namespace

int main() {
  try {
    std::string error;
    std::unique_ptr<ort_mps::MetalContext> metal = ort_mps::MetalContext::Create(error);
    Require(metal != nullptr, error);
    TestMatMulNBits(*metal);
    TestRmsNorm(*metal);
    TestSkipSimplifiedLayerNorm(*metal);
    TestSoftmax(*metal);
    TestGroupQueryAttention(*metal);
    std::cout << "All Mariette hot-path kernel CPU-reference checks passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Mariette kernel test failed: " << ex.what() << "\n";
    return 1;
  }
}
