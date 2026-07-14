//! Attention op handlers: the scaled-dot-product-attention family mapping onto the MLX fast SDPA
//! primitive. Faithful port of the C++ `ops/attention.cc` + `ops/attention_ext.cc`:
//!
//!   * GroupQueryAttention (com.microsoft) — separate Q/K/V, in-op RoPE + KV-cache append + causal
//!     SDPA. Multi-output (attn, present_key, present_value). Decode-critical.
//!   * Attention (ai.onnx opset 23 & 24)   — MHA / GQA / MQA, 3D (B,S,H*hd) or 4D (B,H,S,hd),
//!     optional attn_mask, is_causal, custom scale, in-op past/present KV concat.
//!   * MultiHeadAttention (com.microsoft)  — separate Q/K/V with optional projection bias,
//!     unidirectional (causal), custom scale.
//!   * RotaryEmbedding (ai.onnx opset 23 & com.microsoft) — standalone RoPE with an explicit
//!     cos/sin cache indexed by position_ids (rotate-half or interleaved, partial rotation).
//!
//! Every op honors the resolved input dtype (fp32/fp16/bf16). GQA head broadcast (q_num_heads a
//! multiple of kv_num_heads) is handled inside MLX SDPA, so K/V are passed with their own head count.
//! This is the eager (single-`mlx_eval`) path only: the compiled-decode fast-path (dynamic
//! cos/sin slice, rotate-half matmul) is next-wave and not implemented here.

use std::os::raw::c_char;

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::{is_mlx_float, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET};
use crate::sys::mlx;
use crate::sys::ort;

// ---- small local MLX helpers -------------------------------------------------------------------

#[inline]
fn empty_array() -> mlx::mlx_array {
    mlx::mlx_array_ {
        ctx: std::ptr::null_mut(),
    }
}

fn mul(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_multiply, a, b)
}
fn add(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_add, a, b)
}
fn sub(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_subtract, a, b)
}

/// A row-major slice [start, stop) with unit stride over all axes.
fn slice(ctx: &mut TranslationContext, a: mlx::mlx_array, start: &[i32], stop: &[i32]) -> Result<mlx::mlx_array, MlxError> {
    let stride = vec![1i32; start.len()];
    ctx.emit(|res, s| unsafe {
        mlx::mlx_slice(
            res,
            a,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            stride.as_ptr(),
            stride.len(),
            s,
        )
    })
}

/// Concatenate two arrays along `axis`.
fn concat2(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array, axis: i32) -> Result<mlx::mlx_array, MlxError> {
    let mut vec = VectorArray::new();
    vec.append(a);
    vec.append(b);
    ctx.emit(|res, s| unsafe { mlx::mlx_concatenate_axis(res, vec.as_raw(), axis, s) })
}

/// Run MLX fast SDPA. `mask_mode` is a NUL-terminated byte string ("", "causal", or "array");
/// `mask` is used only for "array".
fn sdpa(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    scale: f32,
    mask_mode: &[u8],
    mask: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let mode = mask_mode.as_ptr() as *const c_char;
    ctx.emit(|res, s| unsafe {
        mlx::mlx_fast_scaled_dot_product_attention(res, q, k, v, scale, mode, mask, empty_array(), s)
    })
}

/// Dispatch SDPA over the (mutually exclusive) causal / array-mask / no-mask cases.
fn sdpa_dispatch(
    ctx: &mut TranslationContext,
    q: mlx::mlx_array,
    k: mlx::mlx_array,
    v: mlx::mlx_array,
    scale: f32,
    causal: bool,
    mask: Option<mlx::mlx_array>,
    compute_dtype: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    if causal {
        return sdpa(ctx, q, k, v, scale, b"causal\0", empty_array());
    }
    if let Some(mut m) = mask {
        // Bool masks stay bool (True = attend); float additive masks cast to the compute dtype.
        if ctx.dtype_of(m) != mlx::mlx_dtype__MLX_BOOL {
            m = ctx.astype(m, compute_dtype)?;
        }
        return sdpa(ctx, q, k, v, scale, b"array\0", m);
    }
    sdpa(ctx, q, k, v, scale, b"\0", empty_array())
}

/// [B,S,H*hd] -> [B,H,S,hd] (head-major split then transpose).
fn split_heads(ctx: &mut TranslationContext, x: mlx::mlx_array, b: i32, s: i32, h: i32, hd: i32) -> Result<mlx::mlx_array, MlxError> {
    let r = ctx.reshape(x, &[b, s, h, hd])?;
    ctx.transpose(r, &[0, 2, 1, 3])
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

fn attr_int(n: &NodeDesc, name: &str, default: i64) -> i64 {
    n.ints.get(name).copied().unwrap_or(default)
}

fn attr_scale(n: &NodeDesc, hd: i32) -> f32 {
    match n.floats.get("scale") {
        Some(&s) if s != 0.0 => s,
        _ => 1.0 / (hd as f32).sqrt(),
    }
}

// ---- GroupQueryAttention in-op RoPE ------------------------------------------------------------

/// Rotate-half (non-interleaved) or interleaved rotary over the first 2*half head dims of x [B,H,S,hd].
/// cos/sin are the per-position rows [S, half], broadcast over B and the head axis.
fn gqa_rope(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    cos: mlx::mlx_array,
    sin: mlx::mlx_array,
    half: i32,
    interleaved: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let xs = ctx.shape_of(x); // [B,H,S,hd]
    let (b, h, s, hd) = (xs[0], xs[1], xs[2], xs[3]);
    let rot = 2 * half;
    let rotated = if !interleaved {
        let x1 = slice(ctx, x, &[0, 0, 0, 0], &[b, h, s, half])?;
        let x2 = slice(ctx, x, &[0, 0, 0, half], &[b, h, s, rot])?;
        let x1c = mul(ctx, x1, cos)?;
        let x2s = mul(ctx, x2, sin)?;
        let o1 = sub(ctx, x1c, x2s)?;
        let x2c = mul(ctx, x2, cos)?;
        let x1s = mul(ctx, x1, sin)?;
        let o2 = add(ctx, x2c, x1s)?;
        concat2(ctx, o1, o2, 3)?
    } else {
        let sl = slice(ctx, x, &[0, 0, 0, 0], &[b, h, s, rot])?;
        let xr = ctx.reshape(sl, &[b, h, s, half, 2])?;
        let xe = slice(ctx, xr, &[0, 0, 0, 0, 0], &[b, h, s, half, 1])?;
        let xo = slice(ctx, xr, &[0, 0, 0, 0, 1], &[b, h, s, half, 2])?;
        let c = ctx.reshape(cos, &[b, 1, s, half, 1])?;
        let sn = ctx.reshape(sin, &[b, 1, s, half, 1])?;
        let xe = ctx.reshape(xe, &[b, h, s, half, 1])?;
        let xo = ctx.reshape(xo, &[b, h, s, half, 1])?;
        let xec = mul(ctx, xe, c)?;
        let xosn = mul(ctx, xo, sn)?;
        let oe = sub(ctx, xec, xosn)?;
        let xoc = mul(ctx, xo, c)?;
        let xesn = mul(ctx, xe, sn)?;
        let oo = add(ctx, xoc, xesn)?;
        let cat = concat2(ctx, oe, oo, 4)?;
        ctx.reshape(cat, &[b, h, s, rot])?
    };
    if rot == hd {
        return Ok(rotated);
    }
    let tail = slice(ctx, x, &[0, 0, 0, rot], &[b, h, s, hd])?;
    concat2(ctx, rotated, tail, 3)
}

/// Slice the cos/sin cache rows for positions [past, past+S) -> [S, half] (eager static slice).
fn cos_sin_row(ctx: &mut TranslationContext, cache: mlx::mlx_array, past: i32, seq: i32, half: i32) -> Result<mlx::mlx_array, MlxError> {
    slice(ctx, cache, &[past, 0], &[past + seq, half])
}

fn group_query_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_heads = attr_int(n, "num_heads", 0) as i32;
    let kv_heads = attr_int(n, "kv_num_heads", 0) as i32;
    let interleaved = attr_int(n, "rotary_interleaved", 0) != 0;
    let do_rotary = !n.ints.contains_key("do_rotary") || attr_int(n, "do_rotary", 1) != 0;

    let q = ctx.resolve(&n.inputs[0])?; // [B,S,num*hd]
    let k = ctx.resolve(&n.inputs[1])?; // [B,S,kv*hd]
    let v = ctx.resolve(&n.inputs[2])?; // [B,S,kv*hd]
    let past_k = ctx.resolve(&n.inputs[3])?; // [B,kv,past,hd]
    let past_v = ctx.resolve(&n.inputs[4])?;

    let qs = ctx.shape_of(q);
    let (b, s) = (qs[0], qs[1]);
    let head = qs[2] / num_heads;
    let past = ctx.shape_of(past_k)[2];

    let scale = attr_scale(n, head);

    let mut qh = split_heads(ctx, q, b, s, num_heads, head)?;
    let mut kh = split_heads(ctx, k, b, s, kv_heads, head)?;
    let vh = split_heads(ctx, v, b, s, kv_heads, head)?;

    if do_rotary {
        let cos = ctx.resolve(&n.inputs[7])?; // [max_seq, rot/2]
        let sin = ctx.resolve(&n.inputs[8])?;
        let half = ctx.shape_of(cos)[1]; // rot/2
        let cr = cos_sin_row(ctx, cos, past, s, half)?;
        let sr = cos_sin_row(ctx, sin, past, s, half)?;
        qh = gqa_rope(ctx, qh, cr, sr, half, interleaved)?;
        kh = gqa_rope(ctx, kh, cr, sr, half, interleaved)?;
    }

    // Append to KV cache along the sequence axis.
    let present_k = concat2(ctx, past_k, kh, 2)?;
    let present_v = concat2(ctx, past_v, vh, 2)?;

    let attn = sdpa(ctx, qh, present_k, present_v, scale, b"causal\0", empty_array())?;
    // [B,H,S,hd] -> [B,S,H*hd].
    let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
    let out = ctx.reshape(t, &[b, s, num_heads * head])?;

    ctx.bind(&n.outputs[0], out);
    if n.outputs.len() >= 2 {
        ctx.bind(&n.outputs[1], present_k);
    }
    if n.outputs.len() >= 3 {
        ctx.bind(&n.outputs[2], present_v);
    }
    Ok(())
}

// ---- Attention (ai.onnx) -----------------------------------------------------------------------

fn attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let q_in = ctx.resolve(&n.inputs[0])?;
    let k_in = ctx.resolve(&n.inputs[1])?;
    let v_in = ctx.resolve(&n.inputs[2])?;
    let dt = ctx.dtype_of(q_in);

    let qs = ctx.shape_of(q_in);
    let is3d = qs.len() == 3;
    let b = qs[0];

    let (qh, s, hd_v, qh4, kh4, vh4) = if is3d {
        let qh = attr_int(n, "q_num_heads", 0) as i32;
        let kvh = attr_int(n, "kv_num_heads", 0) as i32;
        let s = qs[1];
        let ks = ctx.shape_of(k_in);
        let vs = ctx.shape_of(v_in);
        let hd_q = qs[2] / qh;
        let hd_k = ks[2] / kvh;
        let hd_v = vs[2] / kvh;
        let qh4 = split_heads(ctx, q_in, b, s, qh, hd_q)?;
        let kh4 = split_heads(ctx, k_in, b, ks[1], kvh, hd_k)?;
        let vh4 = split_heads(ctx, v_in, b, vs[1], kvh, hd_v)?;
        (qh, s, hd_v, qh4, kh4, vh4)
    } else {
        let qh = qs[1];
        let s = qs[2];
        let hd_v = ctx.shape_of(v_in)[3];
        (qh, s, hd_v, q_in, k_in, v_in)
    };

    let has_past = present(n, 4) && present(n, 5);
    let (present_k, present_v) = if has_past {
        let pk = ctx.resolve(&n.inputs[4])?;
        let pv = ctx.resolve(&n.inputs[5])?;
        (concat2(ctx, pk, kh4, 2)?, concat2(ctx, pv, vh4, 2)?)
    } else {
        (kh4, vh4)
    };

    let hd_q = ctx.shape_of(qh4)[3];
    let scale = attr_scale(n, hd_q);
    let causal = attr_int(n, "is_causal", 0) != 0;
    let mask = if present(n, 3) {
        Some(ctx.resolve(&n.inputs[3])?)
    } else {
        None
    };

    let attn = sdpa_dispatch(ctx, qh4, present_k, present_v, scale, causal, mask, dt)?;

    if is3d {
        // [B,qh,S,hd_v] -> [B,S,qh*hd_v].
        let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
        let out = ctx.reshape(t, &[b, s, qh * hd_v])?;
        ctx.bind(&n.outputs[0], out);
    } else {
        ctx.bind(&n.outputs[0], attn); // already [B,qh,S,hd_v]
    }
    if has_past {
        if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
            ctx.bind(&n.outputs[1], present_k);
        }
        if n.outputs.len() >= 3 && !n.outputs[2].name.is_empty() {
            ctx.bind(&n.outputs[2], present_v);
        }
    }
    Ok(())
}

// ---- MultiHeadAttention (com.microsoft) --------------------------------------------------------

fn multihead_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let h = attr_int(n, "num_heads", 0) as i32;
    let mut q_in = ctx.resolve(&n.inputs[0])?;
    let mut k_in = ctx.resolve(&n.inputs[1])?;
    let mut v_in = ctx.resolve(&n.inputs[2])?;

    let qs = ctx.shape_of(q_in);
    let ks = ctx.shape_of(k_in);
    let vs = ctx.shape_of(v_in);
    let (b, s, dq) = (qs[0], qs[1], qs[2]);
    let (lk, dk) = (ks[1], ks[2]);
    let (lv, dv) = (vs[1], vs[2]);
    let (hd_q, hd_k, hd_v) = (dq / h, dk / h, dv / h);

    if present(n, 3) {
        let bias = ctx.resolve(&n.inputs[3])?; // 1D [Dq+Dk+Dv]
        let qb = slice(ctx, bias, &[0], &[dq])?;
        let qb = ctx.reshape(qb, &[1, 1, dq])?;
        q_in = add(ctx, q_in, qb)?;
        let kb = slice(ctx, bias, &[dq], &[dq + dk])?;
        let kb = ctx.reshape(kb, &[1, 1, dk])?;
        k_in = add(ctx, k_in, kb)?;
        let vb = slice(ctx, bias, &[dq + dk], &[dq + dk + dv])?;
        let vb = ctx.reshape(vb, &[1, 1, dv])?;
        v_in = add(ctx, v_in, vb)?;
    }

    let qh4 = split_heads(ctx, q_in, b, s, h, hd_q)?;
    let kh4 = split_heads(ctx, k_in, b, lk, h, hd_k)?;
    let vh4 = split_heads(ctx, v_in, b, lv, h, hd_v)?;

    let scale = attr_scale(n, hd_q);
    let causal = attr_int(n, "unidirectional", 0) != 0;
    let dt = ctx.dtype_of(qh4);

    let attn = sdpa_dispatch(ctx, qh4, kh4, vh4, scale, causal, None, dt)?;

    // [B,H,S,hd_v] -> [B,S,H*hd_v].
    let t = ctx.transpose(attn, &[0, 2, 1, 3])?;
    let out = ctx.reshape(t, &[b, s, h * hd_v])?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- RotaryEmbedding (ai.onnx opset 23 / com.microsoft) ----------------------------------------

/// Apply RoPE over the first rot (= 2*half) head dims of x4 [B,N,S,hd]; cos4/sin4 are [Bc,1,S,half]
/// (Bc == B for per-position gather, 1 for the offset/absent forms) and broadcast over the head axis.
fn rope_apply(
    ctx: &mut TranslationContext,
    x4: mlx::mlx_array,
    cos4: mlx::mlx_array,
    sin4: mlx::mlx_array,
    half: i32,
    interleaved: bool,
) -> Result<mlx::mlx_array, MlxError> {
    let xs = ctx.shape_of(x4); // [B,N,S,hd]
    let (b, nh, s, hd) = (xs[0], xs[1], xs[2], xs[3]);
    let rot = 2 * half;
    let rotated = if !interleaved {
        let x1 = slice(ctx, x4, &[0, 0, 0, 0], &[b, nh, s, half])?;
        let x2 = slice(ctx, x4, &[0, 0, 0, half], &[b, nh, s, rot])?;
        let x1c = mul(ctx, x1, cos4)?;
        let x2s = mul(ctx, x2, sin4)?;
        let o1 = sub(ctx, x1c, x2s)?;
        let x1s = mul(ctx, x1, sin4)?;
        let x2c = mul(ctx, x2, cos4)?;
        let o2 = add(ctx, x1s, x2c)?;
        concat2(ctx, o1, o2, 3)?
    } else {
        let bc = ctx.shape_of(cos4)[0];
        let sl = slice(ctx, x4, &[0, 0, 0, 0], &[b, nh, s, rot])?;
        let xr = ctx.reshape(sl, &[b, nh, s, half, 2])?;
        let xe = slice(ctx, xr, &[0, 0, 0, 0, 0], &[b, nh, s, half, 1])?; // even lanes (x1)
        let xo = slice(ctx, xr, &[0, 0, 0, 0, 1], &[b, nh, s, half, 2])?; // odd lanes (x2)
        let c = ctx.reshape(cos4, &[bc, 1, s, half, 1])?;
        let sn = ctx.reshape(sin4, &[bc, 1, s, half, 1])?;
        let xec = mul(ctx, xe, c)?;
        let xosn = mul(ctx, xo, sn)?;
        let oe = sub(ctx, xec, xosn)?; // real -> even lanes
        let xesn = mul(ctx, xe, sn)?;
        let xoc = mul(ctx, xo, c)?;
        let oo = add(ctx, xesn, xoc)?; // imag -> odd lanes
        let cat = concat2(ctx, oe, oo, 4)?;
        ctx.reshape(cat, &[b, nh, s, rot])?
    };
    if rot == hd {
        return Ok(rotated);
    }
    let tail = slice(ctx, x4, &[0, 0, 0, rot], &[b, nh, s, hd])?;
    concat2(ctx, rotated, tail, 3)
}

/// Build a [Bc,1,S,half] cos/sin tensor (broadcastable over heads) from a cos/sin cache. pos_rank < 0
/// = absent position_ids (cache is already [B,S,half]); pos_rank == 2 = per-position gather
/// (position_ids [B,S], Bc == B); otherwise the offset form (position_ids [1], positions offset+[0,S),
/// Bc == 1).
fn gather_cache(
    ctx: &mut TranslationContext,
    cache: mlx::mlx_array,
    pos: Option<mlx::mlx_array>,
    pos_rank: i32,
    s: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if pos_rank < 0 {
        let cs = ctx.shape_of(cache); // [B,S,half]
        return ctx.reshape(cache, &[cs[0], 1, cs[1], cs[2]]);
    }
    let half = ctx.shape_of(cache)[1]; // cache: [max_seq, half]
    let pos = pos.expect("position_ids present when pos_rank >= 0");
    let (idx, bc) = if pos_rank == 2 {
        let idx = ctx.astype(pos, mlx::mlx_dtype__MLX_INT32)?; // [B,S]
        let bc = ctx.shape_of(pos)[0];
        (idx, bc)
    } else {
        let off = ctx.astype(pos, mlx::mlx_dtype__MLX_INT32)?; // [1]
        let ar = ctx.emit(|res, st| unsafe {
            mlx::mlx_arange(res, 0.0, s as f64, 1.0, mlx::mlx_dtype__MLX_INT32, st)
        })?; // [S]
        let idx = add(ctx, off, ar)?; // [S] = offset + [0,S)
        (idx, 1)
    };
    let g = ctx.emit(|res, st| unsafe { mlx::mlx_take_axis(res, cache, idx, 0, st) })?; // [B,S,half] or [S,half]
    ctx.reshape(g, &[bc, 1, s, half])
}

fn rotary_embedding_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let ms = n.domain == "com.microsoft";
    let ci = if ms { 2 } else { 1 };
    let si = if ms { 3 } else { 2 };
    let pi = if ms { 1 } else { 3 };
    let interleaved = attr_int(n, "interleaved", 0) != 0;

    let x = ctx.resolve(&n.inputs[0])?;
    let xs = ctx.shape_of(x);
    let rank = xs.len();

    let (b, nh, s, hd, x4) = if rank == 4 {
        (xs[0], xs[1], xs[2], xs[3], x)
    } else {
        let b = xs[0];
        let s = xs[1];
        let nh = attr_int(n, "num_heads", 0) as i32;
        let hd = xs[2] / nh;
        let r = ctx.reshape(x, &[b, s, nh, hd])?;
        let x4 = ctx.transpose(r, &[0, 2, 1, 3])?; // [B,N,S,hd]
        (b, nh, s, hd, x4)
    };

    let (pos, pos_rank) = if present(n, pi) {
        let p = ctx.resolve(&n.inputs[pi])?;
        (Some(p), ctx.ndim(p) as i32)
    } else {
        (None, -1)
    };

    let cos_cache = ctx.resolve(&n.inputs[ci])?;
    let sin_cache = ctx.resolve(&n.inputs[si])?;
    let cos4 = gather_cache(ctx, cos_cache, pos, pos_rank, s)?;
    let sin4 = gather_cache(ctx, sin_cache, pos, pos_rank, s)?;
    let half = ctx.shape_of(cos4)[3];

    let out4 = rope_apply(ctx, x4, cos4, sin4, half, interleaved)?;

    if rank == 4 {
        ctx.bind(&n.outputs[0], out4); // already [B,N,S,hd]
    } else {
        let t = ctx.transpose(out4, &[0, 2, 1, 3])?;
        let out = ctx.reshape(t, &[b, s, nh * hd])?;
        ctx.bind(&n.outputs[0], out);
    }
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn dtype_of(node: &NodeView, i: usize) -> Option<ort::ONNXTensorElementDataType> {
    node.input_info(i).map(|s| s.dtype)
}

fn is_int32(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
}

/// GroupQueryAttention (com.microsoft): 9-input separate-QKV decode/prefill layout. All floating
/// inputs/outputs one dtype; seqlens_k / total_sequence_length are int32.
fn group_query_attention_claim(node: &NodeView) -> bool {
    if node.num_outputs() == 0 {
        return false;
    }
    let out_type = match node.output_info(0) {
        Some(o) if is_mlx_float(o.dtype) => o.dtype,
        _ => return false,
    };
    if node.num_inputs() != 9 {
        return false; // q,k,v,past_k,past_v,seqlens_k,total_seq,cos,sin
    }
    for &idx in &[0usize, 1, 2, 3, 4, 7, 8] {
        match dtype_of(node, idx) {
            Some(t) if t == out_type => {}
            _ => return false,
        }
    }
    let nouts = node.num_outputs();
    for idx in 1..nouts.min(3) {
        match node.output_info(idx) {
            Some(o) if o.dtype == out_type => {}
            _ => return false,
        }
    }
    if !matches!(dtype_of(node, 5), Some(t) if is_int32(t)) {
        return false;
    }
    if !matches!(dtype_of(node, 6), Some(t) if is_int32(t)) {
        return false;
    }
    let nh = node.int_attr("num_heads", 0);
    let kvh = node.int_attr("kv_num_heads", 0);
    if nh <= 0 || kvh <= 0 || nh % kvh != 0 {
        return false;
    }
    // Unsupported (rare) variants fall back to CPU. Only a genuine enable (== 1) is rejected.
    if node.int_attr("smooth_softmax", 0) == 1 {
        return false;
    }
    if node.int_attr("qk_output", 0) != 0 {
        return false;
    }
    if node.float_attr("softcap", 0.0) != 0.0 {
        return false;
    }
    true
}

/// Q/K/V present and same MLX float dtype as the output.
fn check_qkv_float(node: &NodeView) -> Option<ort::ONNXTensorElementDataType> {
    if node.num_inputs() < 3 || node.num_outputs() == 0 {
        return None;
    }
    let qd = dtype_of(node, 0)?;
    if !node.input_present(1) || !node.input_present(2) {
        return None;
    }
    let kd = dtype_of(node, 1)?;
    let vd = dtype_of(node, 2)?;
    let od = node.output_info(0)?.dtype;
    if is_mlx_float(qd) && kd == qd && vd == qd && od == qd {
        Some(qd)
    } else {
        None
    }
}

/// A past/present pair must be used together, share the query dtype; present outputs require past.
fn check_kv_cache(node: &NodeView, past_k: usize, past_v: usize, qd: ort::ONNXTensorElementDataType) -> bool {
    let pk = node.input_present(past_k);
    let pv = node.input_present(past_v);
    if pk != pv {
        return false;
    }
    if pk {
        match (dtype_of(node, past_k), dtype_of(node, past_v)) {
            (Some(a), Some(b)) if a == qd && b == qd => {}
            _ => return false,
        }
    }
    if !pk && (node.output_present(1) || node.output_present(2)) {
        return false;
    }
    true
}

/// An attn/attention_bias mask must be bool or the query float dtype, and cannot co-exist with causal.
fn check_mask(node: &NodeView, mask_idx: usize, causal: bool, _qd: ort::ONNXTensorElementDataType) -> bool {
    if !node.input_present(mask_idx) {
        return true;
    }
    if causal {
        return false;
    }
    match dtype_of(node, mask_idx) {
        Some(md) => md == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL || is_mlx_float(md),
        None => false,
    }
}

/// Attention (ai.onnx, opset 23 and 24). 3D/4D SDPA, optional attn_mask + past/present KV.
fn attention_claim(node: &NodeView) -> bool {
    let qd = match check_qkv_float(node) {
        Some(qd) => qd,
        None => return false,
    };
    let qshape = match node.input_info(0) {
        Some(i) => i.shape,
        None => return false,
    };
    let kshape = match node.input_info(1) {
        Some(i) => i.shape,
        None => return false,
    };
    let vshape = match node.input_info(2) {
        Some(i) => i.shape,
        None => return false,
    };
    let rank = qshape.len();
    if rank != 3 && rank != 4 {
        return false;
    }
    if kshape.len() != qshape.len() || vshape.len() != qshape.len() {
        return false;
    }

    let (qh, kvh) = if rank == 3 {
        let qh = node.int_attr("q_num_heads", 0);
        let kvh = node.int_attr("kv_num_heads", 0);
        if qh <= 0 || kvh <= 0 {
            return false;
        }
        for &d in &qshape {
            if d <= 0 {
                return false;
            }
        }
        if kshape[1] <= 0 || vshape[1] <= 0 || kshape[2] <= 0 || vshape[2] <= 0 {
            return false;
        }
        if qshape[2] % qh != 0 || kshape[2] % kvh != 0 || vshape[2] % kvh != 0 {
            return false;
        }
        (qh, kvh)
    } else {
        let qh = qshape[1];
        let kvh = kshape[1];
        if qh <= 0 || kvh <= 0 {
            return false;
        }
        (qh, kvh)
    };
    if qh % kvh != 0 {
        return false;
    }

    if node.float_attr("softcap", 0.0) != 0.0 {
        return false; // logit soft-cap unsupported
    }
    if node.output_present(3) {
        return false; // qk_matmul_output unsupported
    }
    if node.input_present(6) {
        return false; // nonpad_kv_seqlen (opset 24)
    }

    let causal = node.int_attr("is_causal", 0) != 0;
    if !check_mask(node, 3, causal, qd) {
        return false;
    }
    check_kv_cache(node, 4, 5, qd)
}

/// MultiHeadAttention (com.microsoft). Separate 3D Q/K/V + optional projection bias.
fn multihead_attention_claim(node: &NodeView) -> bool {
    if node.int_attr("num_heads", 0) <= 0 {
        return false;
    }
    let qd = match check_qkv_float(node) {
        Some(qd) => qd,
        None => return false,
    };
    let h = node.int_attr("num_heads", 0);
    let qshape = match node.input_info(0) {
        Some(i) => i.shape,
        None => return false,
    };
    let kshape = match node.input_info(1) {
        Some(i) => i.shape,
        None => return false,
    };
    let vshape = match node.input_info(2) {
        Some(i) => i.shape,
        None => return false,
    };
    if qshape.len() != 3 || kshape.len() != 3 || vshape.len() != 3 {
        return false;
    }
    for sh in [&qshape, &kshape, &vshape] {
        for &d in sh.iter() {
            if d <= 0 {
                return false;
            }
        }
    }
    if qshape[2] % h != 0 || kshape[2] % h != 0 || vshape[2] % h != 0 {
        return false;
    }

    if node.input_present(3) {
        // bias: 1D [Dq+Dk+Dv]
        let (bd, bshape) = match node.input_info(3) {
            Some(i) => (i.dtype, i.shape),
            None => return false,
        };
        if bd != qd || bshape.len() != 1 || bshape[0] != qshape[2] + kshape[2] + vshape[2] {
            return false;
        }
    }
    // key_padding_mask (#4), attention_bias (#5), past/present KV (#6/#7), past_seq_len (#8),
    // cache_indirection (#9) -> CPU.
    for i in 4..=9 {
        if node.input_present(i) {
            return false;
        }
    }
    if node.output_present(1) || node.output_present(2) || node.output_present(3) {
        return false;
    }
    true
}

/// RotaryEmbedding (ai.onnx opset 23 / com.microsoft). Float 3D (B,S,H*hd)+num_heads or 4D input;
/// [B,S] gather, [1] offset (com.microsoft), or (ai.onnx only) absent pos with [B,S,half] cache.
fn rotary_embedding_claim(node: &NodeView) -> bool {
    if node.num_outputs() == 0 {
        return false;
    }
    let ms = node.domain() == "com.microsoft";
    let ci = if ms { 2 } else { 1 };
    let si = if ms { 3 } else { 2 };
    let pi = if ms { 1 } else { 3 };
    let min_inputs = if ms { 4 } else { 3 };
    if node.num_inputs() < min_inputs {
        return false;
    }

    let (xd, xshape) = match node.input_info(0) {
        Some(i) => (i.dtype, i.shape),
        None => return false,
    };
    let (cd, cshape) = match node.input_info(ci) {
        Some(i) => (i.dtype, i.shape),
        None => return false,
    };
    if !node.input_present(si) {
        return false;
    }
    let sd = match dtype_of(node, si) {
        Some(t) => t,
        None => return false,
    };
    let od = match node.output_info(0) {
        Some(o) => o.dtype,
        None => return false,
    };
    if !is_mlx_float(xd) || cd != xd || sd != xd || od != xd {
        return false;
    }

    let rank = xshape.len();
    if rank == 3 {
        let nh = node.int_attr("num_heads", 0);
        if nh <= 0 || xshape[2] <= 0 || xshape[2] % nh != 0 {
            return false;
        }
    } else if rank != 4 {
        return false;
    }

    let has_pos = node.input_present(pi);
    if ms && !has_pos {
        return false; // com.microsoft position_ids is mandatory
    }
    if has_pos {
        let (pd, pshape) = match node.input_info(pi) {
            Some(i) => (i.dtype, i.shape),
            None => return false,
        };
        if pd != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && pd != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        {
            return false;
        }
        let gather = pshape.len() == 2; // position_ids [B,S] (either domain)
        let offset = ms && pshape.len() == 1 && pshape[0] == 1; // position_ids [1] (com.microsoft)
        if !gather && !offset {
            return false;
        }
        if cshape.len() != 2 {
            return false; // gather/offset caches are [max_seq, half]
        }
    } else if cshape.len() != 3 {
        return false; // absent form: cache must be per-position [B,S,half]
    }
    true
}

// ---- registration ------------------------------------------------------------------------------

fn reg(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    min_opset: i32,
    max_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain,
        op_type,
        min_opset,
        max_opset,
        handler,
        claim,
    });
}

pub fn register_attention(registry: &mut OpRegistry) {
    reg(registry, "com.microsoft", "GroupQueryAttention", K_ANY_OPSET, K_ANY_OPSET, group_query_attention_op, group_query_attention_claim);
    // Attention entered ai.onnx at opset 23; opset 24 adds the trailing nonpad_kv_seqlen input.
    reg(registry, "", "Attention", 23, 23, attention_op, attention_claim);
    reg(registry, "", "Attention", 24, K_ANY_OPSET, attention_op, attention_claim);
    reg(registry, "com.microsoft", "MultiHeadAttention", K_ANY_OPSET, K_ANY_OPSET, multihead_attention_op, multihead_attention_claim);
    // RotaryEmbedding: ai.onnx entered at opset 23; com.microsoft is version-insensitive.
    reg(registry, "", "RotaryEmbedding", 23, K_ANY_OPSET, rotary_embedding_op, rotary_embedding_claim);
    reg(registry, "com.microsoft", "RotaryEmbedding", K_ANY_OPSET, K_ANY_OPSET, rotary_embedding_op, rotary_embedding_claim);
}
