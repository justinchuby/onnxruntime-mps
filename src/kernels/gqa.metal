// Copyright (c) 2026. Licensed under the MIT License.
//
// Metal GroupQueryAttention (com.microsoft.GroupQueryAttention) for the Metal/MPS EP.
//
// Two passes, encoded back-to-back into one serial command encoder by
// MetalContext::GroupQueryAttention (so the second observes the first's writes):
//
//   1. mps_gqa_write_kv_f32  — fills the present K/V cache. For cache positions < past it copies
//      past K/V into present (a no-op when present aliases past, i.e. the share-buffer path); for
//      the S new tokens it applies rotary embedding to K and appends new K/V at present[past+i].
//   2. mps_gqa_attention_f32 — one threadgroup per (batch, query-head, query-token). Applies
//      rotary to Q, then does online-softmax (flash-decoding) over the valid causal / sliding
//      window of the present cache, reading the grouped KV head shared by num_heads/kv_num_heads
//      query heads. All dot products / softmax accumulate in fp32.
//
// Semantics match the ORT CPU op exactly (verified differentially): seqlens_k[b] = total valid
// keys - 1, past = total - S, token i is at absolute position past+i, causal keys are
// [lo, posn] with lo = max(0, posn - local_window + 1) when local_window >= 0 else 0, and query
// head h reads kv head h / (num_heads / kv_num_heads).

#include <metal_stdlib>
using namespace metal;

struct GqaParams {
  uint batch;
  uint seq_len;        // S (number of new query tokens)
  uint num_heads;      // query heads
  uint kv_num_heads;   // key/value heads (grouped)
  uint head_size;      // per-head dim
  uint rotary_dim;     // rotary width (<= head_size); cos/sin cache stride is rotary_dim/2
  uint past_seq;       // seq dimension of the past K/V buffers
  uint present_seq;    // seq dimension of the present K/V buffers
  uint do_rotary;      // 1 => apply RoPE to Q and new K
  uint interleaved;    // 1 => interleaved RoPE pairing, else half-split
  int  local_window;   // sliding-window left size; -1 => full causal
  float scale;         // softmax pre-scale
};

// Rotary value for element `d` of the head vector at `x + base`, at absolute `position`.
// Matches src/kernels/rope.metal (ORT half-split / interleaved conventions).
static inline float gqa_rope_value(device const float* x, ulong base, uint d, uint position,
                                   constant GqaParams& p, device const float* cosc,
                                   device const float* sinc) {
  const uint half_dim = p.rotary_dim >> 1;
  uint pair;
  uint partner;
  float sign;
  if (p.interleaved != 0) {
    pair = d >> 1;
    partner = d ^ 1u;
    sign = (d & 1u) == 0u ? -1.0f : 1.0f;
  } else {
    pair = d < half_dim ? d : d - half_dim;
    partner = d < half_dim ? d + half_dim : d - half_dim;
    sign = d < half_dim ? -1.0f : 1.0f;
  }
  const ulong ci = (ulong)position * half_dim + pair;
  const float c = cosc[ci];
  const float s = sinc[ci];
  const float xd = x[base + d];
  const float xo = x[base + partner];
  return xd * c + sign * xo * s;
}

// Pass 1: build the present K/V cache (copy past region + rope/append new tokens).
// Grid threadgroups: (present positions, kv_num_heads, batch); threads: head_size.
kernel void mps_gqa_write_kv_f32(device const float* key          [[buffer(0)]],
                                 device const float* value        [[buffer(1)]],
                                 device const float* past_key     [[buffer(2)]],
                                 device const float* past_value   [[buffer(3)]],
                                 device const int*   seqlens_k    [[buffer(4)]],
                                 device const float* cos_cache    [[buffer(5)]],
                                 device const float* sin_cache    [[buffer(6)]],
                                 device float*       present_key  [[buffer(7)]],
                                 device float*       present_value[[buffer(8)]],
                                 constant GqaParams& p            [[buffer(9)]],
                                 uint3 tg  [[threadgroup_position_in_grid]],
                                 uint3 tid [[thread_position_in_threadgroup]]) {
  const uint d = tid.x;
  if (d >= p.head_size) return;
  const uint pos = tg.x;
  const uint kh = tg.y;
  const uint b = tg.z;
  const uint total = uint(seqlens_k[b]) + 1u;
  if (pos >= total) return;
  const uint past = total - p.seq_len;
  const uint head = p.head_size;
  const ulong pres_base = (((ulong)b * p.kv_num_heads + kh) * p.present_seq + pos) * head;

  if (pos < past) {
    const ulong past_base = (((ulong)b * p.kv_num_heads + kh) * p.past_seq + pos) * head;
    present_key[pres_base + d] = past_key[past_base + d];
    present_value[pres_base + d] = past_value[past_base + d];
  } else {
    const uint i = pos - past;
    // Input K/V layout is [B, S, kv_num_heads, head_size].
    const ulong kv_base = (((ulong)b * p.seq_len + i) * p.kv_num_heads + kh) * head;
    present_value[pres_base + d] = value[kv_base + d];
    if (p.do_rotary != 0 && d < p.rotary_dim) {
      present_key[pres_base + d] = gqa_rope_value(key, kv_base, d, pos, p, cos_cache, sin_cache);
    } else {
      present_key[pres_base + d] = key[kv_base + d];
    }
  }
}

// Pass 2: flash attention over the present cache. Grid threadgroups: (S, num_heads, batch).
// Threadgroup memory `shared` is partitioned as: qrot[head], m[T], l[T], acc[T*head].
kernel void mps_gqa_attention_f32(device const float* query        [[buffer(0)]],
                                  device const float* present_key  [[buffer(1)]],
                                  device const float* present_value[[buffer(2)]],
                                  device const int*   seqlens_k    [[buffer(3)]],
                                  device const float* cos_cache    [[buffer(4)]],
                                  device const float* sin_cache    [[buffer(5)]],
                                  device float*       output       [[buffer(6)]],
                                  constant GqaParams& p            [[buffer(7)]],
                                  threadgroup float*  shared       [[threadgroup(0)]],
                                  uint3 tg   [[threadgroup_position_in_grid]],
                                  uint3 tid  [[thread_position_in_threadgroup]],
                                  uint3 tpg  [[threads_per_threadgroup]]) {
  const uint i = tg.x;
  const uint h = tg.y;
  const uint b = tg.z;
  const uint head = p.head_size;
  const uint t = tid.x;
  const uint T = tpg.x;

  threadgroup float* qrot = shared;
  threadgroup float* m_sh = qrot + head;
  threadgroup float* l_sh = m_sh + T;
  threadgroup float* acc_sh = l_sh + T;

  const uint total = uint(seqlens_k[b]) + 1u;
  const uint past = total - p.seq_len;
  const uint posn = past + i;
  const uint kh = h / (p.num_heads / p.kv_num_heads);

  const ulong qbase = ((ulong)b * p.seq_len + i) * p.num_heads * head + (ulong)h * head;
  for (uint d = t; d < head; d += T) {
    if (p.do_rotary != 0 && d < p.rotary_dim) {
      qrot[d] = gqa_rope_value(query, qbase, d, posn, p, cos_cache, sin_cache);
    } else {
      qrot[d] = query[qbase + d];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint lo = 0u;
  if (p.local_window >= 0) {
    const int l = int(posn) - p.local_window + 1;
    lo = l > 0 ? uint(l) : 0u;
  }
  const uint hi = posn;  // inclusive (causal)

  // Online-softmax partials for this thread's stride of keys.
  threadgroup float* acc_t = acc_sh + (ulong)t * head;
  for (uint d = 0; d < head; ++d) acc_t[d] = 0.0f;
  float m_t = -INFINITY;
  float l_t = 0.0f;
  const ulong kv_head_base = ((ulong)b * p.kv_num_heads + kh) * p.present_seq;
  for (uint j = lo + t; j <= hi; j += T) {
    const ulong kbase = (kv_head_base + j) * head;
    float s = 0.0f;
    for (uint d = 0; d < head; ++d) s += qrot[d] * present_key[kbase + d];
    s *= p.scale;
    const float m_new = max(m_t, s);
    const float corr = exp(m_t - m_new);
    const float pw = exp(s - m_new);
    l_t = l_t * corr + pw;
    for (uint d = 0; d < head; ++d) acc_t[d] = acc_t[d] * corr + pw * present_value[kbase + d];
    m_t = m_new;
  }
  m_sh[t] = m_t;
  l_sh[t] = l_t;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Combine the T partials (online-softmax merge) and write the output.
  float m_g = -INFINITY;
  for (uint tt = 0; tt < T; ++tt) m_g = max(m_g, m_sh[tt]);
  float l_g = 0.0f;
  for (uint tt = 0; tt < T; ++tt) l_g += l_sh[tt] * exp(m_sh[tt] - m_g);
  const float inv = l_g > 0.0f ? 1.0f / l_g : 0.0f;
  const ulong obase = ((ulong)b * p.seq_len + i) * p.num_heads * head + (ulong)h * head;
  for (uint d = t; d < head; d += T) {
    float o = 0.0f;
    for (uint tt = 0; tt < T; ++tt) o += exp(m_sh[tt] - m_g) * acc_sh[(ulong)tt * head + d];
    output[obase + d] = o * inv;
  }
}
