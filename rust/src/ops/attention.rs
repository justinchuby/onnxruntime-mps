//! Attention op handlers: the scaled-dot-product-attention family mapping onto the MLX fast SDPA
//! primitive. Faithful port of the C++ `ops/attention.cc` + `ops/attention_ext.cc`:
//!
//!   * GroupQueryAttention (com.microsoft) — separate Q/K/V, in-op RoPE + KV-cache append + causal
//!     SDPA. Multi-output (attn, present_key, present_value). Decode-critical.
//!   * Attention (ai.onnx opset 23 & 24)   — MHA / GQA / MQA, 3D (B,S,H*hd) or 4D (B,H,S,hd),
//!     optional attn_mask, is_causal, custom scale, in-op past/present KV concat.
//!   * MultiHeadAttention (com.microsoft)  — separate Q/K/V with optional projection bias,
//!     unidirectional (causal), custom scale.
//!   * RotaryEmbedding (ai.onnx opset 23 & com.microsoft) — standalone RoPE (rotate-half or
//!     interleaved, partial rotation). The fp32 gather / offset forms recover the RoPE period from
//!     the cos/sin cache and run the fused `mlx_fast_rope` kernel; the absent-position-ids form
//!     (explicit per-position cache) and reduced-precision caches keep the composed path.
//!
//! Every op honors the resolved input dtype (fp32/fp16/bf16). GQA head broadcast (q_num_heads a
//! multiple of kv_num_heads) is handled inside MLX SDPA, so K/V are passed with their own head count.
//! GQA's in-op RoPE also fuses onto `mlx_fast_rope` for fp32 caches (composed fallback otherwise).
//! This is the eager (single-`mlx_eval`) path only.

use std::os::raw::c_char;

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::{
    is_mlx_float, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- small local MLX helpers -------------------------------------------------------------------

#[inline]
fn empty_array() -> mlx::mlx_array {
    mlx::mlx_array_ {
        ctx: std::ptr::null_mut(),
    }
}

fn mul(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_multiply, a, b)
}
fn add(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_add, a, b)
}
fn sub(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_subtract, a, b)
}

/// A row-major slice [start, stop) with unit stride over all axes.
fn slice(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    start: &[i32],
    stop: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
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
fn concat2(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
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
    ctx.mark_fast("mlx_fast_scaled_dot_product_attention");
    ctx.emit(|res, s| unsafe {
        mlx::mlx_fast_scaled_dot_product_attention(
            res,
            q,
            k,
            v,
            scale,
            mode,
            mask,
            empty_array(),
            s,
        )
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

/// Build an ONNX-semantics causal additive mask `[q_len, k_len]` for SDPA.
///
/// ONNX `Attention` `is_causal` masks query `i` against key `j` iff `j > past_seq + i`
/// (`onnx/reference/ops/op_attention.py::_apply_causal`), i.e. an upper-left alignment where the
/// first `past_seq` keys are always visible and the causal triangle starts at the new-key region.
/// MLX's built-in `"causal"` mode instead uses a lower-right alignment (`j <= i + (k_len - q_len)`),
/// which only matches ONNX when `past_seq == k_len - q_len`. For the non-square no-past case
/// (`k_len > q_len`, `past_seq == 0`) the two disagree, so we build the additive mask explicitly.
fn causal_mask_topleft(
    ctx: &mut TranslationContext,
    q_len: i32,
    k_len: i32,
    past_seq: i32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let i32t = mlx::mlx_dtype__MLX_INT32;
    let key_pos = ctx.arange(0.0, k_len as f64, 1.0, i32t)?; // [k_len]
    let key_pos = ctx.reshape(key_pos, &[1, k_len])?; // [1,k_len]
    let q_pos = ctx.arange(past_seq as f64, (past_seq + q_len) as f64, 1.0, i32t)?; // [q_len]
    let q_pos = ctx.reshape(q_pos, &[q_len, 1])?; // [q_len,1]
    let allow = ctx.less_equal(key_pos, q_pos)?; // [q_len,k_len] bool
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    ctx.where_(allow, zero, neg)
}

/// GQA eager SDPA over the attended keys `[0, k_len)`, with the queries at positions
/// `[valid_past, valid_past+S)`. Without an `attention_bias` this is plain causal SDPA (bit-for-bit
/// the legacy behavior). With the 11-input Gemma3n `attention_bias` (input 10, `[B,1,S,total]`
/// additive mask encoding the causal + sliding-window mask), we fold it into an array mask:
/// `mask = causal_topleft[q_len,k_len] + attention_bias[..,:k_len]`. Adding the causal triangle is
/// idempotent where the bias already masks (−inf + finite = −inf) and supplies causality where it
/// doesn't; the sliding window is carried entirely by the bias.
#[allow(clippy::too_many_arguments)]
fn gqa_eager_sdpa(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    qh: mlx::mlx_array,
    ak: mlx::mlx_array,
    av: mlx::mlx_array,
    scale: f32,
    q_len: i32,
    valid_past: i32,
    k_len: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if present(n, 10) {
        let dt = ctx.dtype_of(qh);
        let bias = ctx.resolve(&n.inputs[10])?; // [B,1,S,total]
        let bs = ctx.shape_of(bias);
        let bias = slice(ctx, bias, &[0, 0, 0, 0], &[bs[0], bs[1], q_len, k_len])?;
        let causal = causal_mask_topleft(ctx, q_len, k_len, valid_past, dt)?; // [q_len,k_len]
        let mask = add(ctx, causal, bias)?; // -> [B,1,S,k_len]
        return sdpa(ctx, qh, ak, av, scale, b"array\0", mask);
    }
    sdpa(ctx, qh, ak, av, scale, b"causal\0", empty_array())
}

/// [B,S,H*hd] -> [B,H,S,hd] (head-major split then transpose).
fn split_heads(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    b: i32,
    s: i32,
    h: i32,
    hd: i32,
) -> Result<mlx::mlx_array, MlxError> {
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

// ---- Fused RoPE (mlx_fast_rope) ----------------------------------------------------------------

/// A `mlx_optional_float` with no value — signals `mlx_fast_rope` to take frequencies from the
/// explicit `freqs` array (its reciprocal) rather than deriving them from a `base`.
#[inline]
fn opt_float_none() -> mlx::mlx_optional_float {
    mlx::mlx_optional_float {
        value: 0.0,
        has_value: false,
    }
}

/// Recover the RoPE **period** array (length `half`) that `mlx_fast_rope` expects for its `freqs`
/// argument, from a standard cos/sin cache whose row `p` holds `cos(p·invfreq)` / `sin(p·invfreq)`.
///
/// MLX internally computes `invfreq = 1/freqs` and `theta = (offset + s)·invfreq`, so passing the
/// period (= `1/invfreq`) makes MLX reproduce the exact angles ORT baked into the cache. `invfreq`
/// is read off absolute-position row 1 (`theta = 1·invfreq = atan2(sin[1], cos[1])`; every standard
/// RoPE `invfreq ∈ (0,1]`, so this angle is unambiguous), then reciprocated.
fn rope_freqs_from_cache(
    ctx: &mut TranslationContext,
    cos_cache: mlx::mlx_array,
    sin_cache: mlx::mlx_array,
    half: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let cos1 = slice(ctx, cos_cache, &[1, 0], &[2, half])?; // [1,half] @ position 1
    let sin1 = slice(ctx, sin_cache, &[1, 0], &[2, half])?;
    // Derive in fp32: for fp16/bf16 caches an atan2 in the low-precision dtype would corrupt the
    // recovered period (mlx casts `freqs` to fp32 internally regardless).
    let cos1 = ctx.astype(cos1, mlx::mlx_dtype__MLX_FLOAT32)?;
    let sin1 = ctx.astype(sin1, mlx::mlx_dtype__MLX_FLOAT32)?;
    let invfreq = ctx.emit(|res, s| unsafe { mlx::mlx_arctan2(res, sin1, cos1, s) })?;
    let period = ctx.emit(|res, s| unsafe { mlx::mlx_reciprocal(res, invfreq, s) })?;
    ctx.reshape(period, &[half])
}

/// Fused RoPE over the first `rot` head dims of x [B,N,S,hd] with a compile-time integer position
/// `offset` (positions `offset + s`). `traditional` = interleaved (consecutive-pair) rotation.
fn fast_rope_static(
    ctx: &mut TranslationContext,
    x4: mlx::mlx_array,
    rot: i32,
    traditional: bool,
    offset: i32,
    freqs: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.mark_fast("mlx_fast_rope");
    ctx.emit(|res, s| unsafe {
        mlx::mlx_fast_rope(
            res,
            x4,
            rot,
            traditional,
            opt_float_none(),
            1.0,
            offset,
            freqs,
            s,
        )
    })
}

/// Fused RoPE with a runtime position `offset` array (scalar/[1], or per-row [B] for [B,S]
/// position_ids). Positions are `offset + s`; `traditional` = interleaved rotation.
fn fast_rope_dynamic(
    ctx: &mut TranslationContext,
    x4: mlx::mlx_array,
    rot: i32,
    traditional: bool,
    offset: mlx::mlx_array,
    freqs: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.mark_fast("mlx_fast_rope");
    ctx.emit(|res, s| unsafe {
        mlx::mlx_fast_rope_dynamic(
            res,
            x4,
            rot,
            traditional,
            opt_float_none(),
            1.0,
            offset,
            freqs,
            s,
        )
    })
}

/// True when `a`'s dtype is fp32. The fused kernel recomputes cos/sin from frequencies recovered
/// out of the cos/sin cache; recovering those frequencies from a reduced-precision (fp16/bf16)
/// cache drifts from the exact values ORT stored, so such caches keep the composed path (which
/// consumes the provided cache directly).
#[inline]
fn is_fp32(ctx: &TranslationContext, a: mlx::mlx_array) -> bool {
    ctx.dtype_of(a) == mlx::mlx_dtype__MLX_FLOAT32
}

// ---- GroupQueryAttention composed RoPE (reduced-precision fallback) -----------------------------

/// Rotate-half (non-interleaved) or interleaved rotary over the first 2*half head dims of x [B,H,S,hd].
/// cos/sin are the per-position rows [S, half], broadcast over B and the head axis. Used only when the
/// cache dtype rules out the fused `mlx_fast_rope` path (see `is_fp32`).
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
fn cos_sin_row(
    ctx: &mut TranslationContext,
    cache: mlx::mlx_array,
    past: i32,
    seq: i32,
    half: i32,
) -> Result<mlx::mlx_array, MlxError> {
    slice(ctx, cache, &[past, 0], &[past + seq, half])
}

/// Compiled-decode RoPE: rotate-half via a `[hd,hd]` matmul (no Slice) with FULL-width cos/sin rows
/// (`[1,1,S,hd]`, each half duplicated) fed as closure inputs. `out = x*cos + (x @ M)*sin`, matching
/// standard non-interleaved RoPE. Eligibility guarantees `rot == hd == 2*half`, so there is no
/// pass-through tail. Everything is cast to `x`'s dtype so the result dtype matches the eager path.
fn gqa_rope_matmul(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    cos_full: mlx::mlx_array,
    sin_full: mlx::mlx_array,
    hd: i32,
    half: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let dt = ctx.dtype_of(x);
    let m = ctx.rotate_half_matrix(hd, half);
    let m = ctx.astype(m, dt)?;
    let cos_c = ctx.astype(cos_full, dt)?;
    let sin_c = ctx.astype(sin_full, dt)?;
    let xrot = ctx.matmul(x, m)?; // rotate_half(x)
    let xc = mul(ctx, x, cos_c)?;
    let xs = mul(ctx, xrot, sin_c)?;
    add(ctx, xc, xs)
}

/// Compiled-decode shared-buffer KV update + masked SDPA. `valid_past` is DATA (recovered from the
/// in-graph `total_sequence_length` = input[6], minus S), so the new K/V are written in place with a
/// dynamic slice-update at that offset and attention runs over the whole `cap` buffer under a
/// static-shape additive mask (buffer tail beyond `valid_past+S` masked to -inf; causal within the
/// valid prefix). Every op is statically shaped, so the shapeless compiled closure can carry it.
/// Returns `(present_k, present_v, attn)` with present at the full `[B,kv,cap,hd]` capacity.
#[allow(clippy::too_many_arguments)]
fn gqa_shared_compiled(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    past_k: mlx::mlx_array,
    past_v: mlx::mlx_array,
    kh: mlx::mlx_array,
    vh: mlx::mlx_array,
    qh: mlx::mlx_array,
    s: i32,
    cap: i32,
    scale: f32,
) -> Result<(mlx::mlx_array, mlx::mlx_array, mlx::mlx_array), MlxError> {
    let i32t = mlx::mlx_dtype__MLX_INT32;
    // valid_past is fed as a live [1] int32 closure input (appended after the synth RoPE rows) so it
    // advances every decode step. Resolving the in-graph total_sequence_length here instead would let
    // shapeless compile freeze it at trace time (baking Shape(attention_mask) as a constant), which
    // pins the write offset + mask and corrupts generation. Fall back defensively if unavailable.
    let vp = match ctx.shared_valid_past() {
        Some(a) => a,
        None => {
            let ts = ctx.resolve(&n.inputs[6])?;
            let ts = ctx.astype(ts, i32t)?;
            let ts = ctx.reshape(ts, &[1])?;
            let s_scalar = ctx.scalar_i32(s);
            let s_scalar = ctx.reshape(s_scalar, &[1])?;
            ctx.sub(ts, s_scalar)?
        }
    }; // [1] valid_past

    // In-place write of the S new rows at axis-2 offset valid_past; present keeps the full capacity.
    let present_k = ctx.slice_update_dynamic(past_k, kh, vp, &[2])?;
    let present_v = ctx.slice_update_dynamic(past_v, vh, vp, &[2])?;

    // Additive causal mask [1,1,S,cap]: key j attends query i iff j <= valid_past + i.
    let key_pos = ctx.arange(0.0, cap as f64, 1.0, i32t)?; // [cap]
    let key_pos = ctx.reshape(key_pos, &[1, cap])?; // [1,cap]
    let q_off = ctx.arange(0.0, s as f64, 1.0, i32t)?; // [S]
    let q_pos = ctx.add(vp, q_off)?; // [S] (valid_past + i, broadcast [1]+[S])
    let q_pos = ctx.reshape(q_pos, &[s, 1])?; // [S,1]
    let allow = ctx.less_equal(key_pos, q_pos)?; // [S,cap] bool
    let dt = ctx.dtype_of(qh);
    let zero = ctx.scalar_f32(0.0);
    let zero = ctx.astype(zero, dt)?;
    let neg = ctx.scalar_f32(f32::NEG_INFINITY);
    let neg = ctx.astype(neg, dt)?;
    let mask = ctx.where_(allow, zero, neg)?; // [S,cap]
    let mask = ctx.reshape(mask, &[1, 1, s, cap])?;

    let attn = sdpa(ctx, qh, present_k, present_v, scale, b"array\0", mask)?;
    Ok((present_k, present_v, attn))
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
    // Sequence capacity of the past-KV cache (its axis-2 length). In the growing (ZeroCopyRebind)
    // contract this equals the number of valid past keys; in the fixed-capacity shared-buffer
    // contract it is the runtime-owned max length and exceeds the valid past.
    let cap = ctx.shape_of(past_k)[2];

    // Recover how many past keys are actually valid and whether the runtime handed us a max-length
    // shared buffer to write in place. `total_sequence_length` (input[6], int32 scalar) counts the
    // valid keys after this step (= valid_past + S), so `valid_past = total_sequence_length[0] - S`.
    // Shared-buffer mode is exactly `cap > valid_past` (past_k is a max-length buffer). The compiled
    // decode trace cannot eval mid-graph, so it takes the mode from the once-detected plan flag
    // (`ctx.shared_kv()`) and drives the in-place write with a DATA offset instead (see below).
    let (shared_buffer, valid_past) = if ctx.rope_dynamic() {
        // valid_past (int) is unused on the shapeless compiled DECODE path: RoPE uses the pre-sliced
        // synth rows and the KV write/mask use a data offset derived from total_sequence_length. On
        // the shape-keyed compiled PREFILL path valid_past IS static (known per shape key), so use it
        // to narrow the attention to the valid prefix below.
        if ctx.compiled_shape_keyed() {
            (ctx.shared_kv(), ctx.compiled_valid_past())
        } else {
            (ctx.shared_kv(), cap)
        }
    } else {
        let ts = ctx.resolve(&n.inputs[6])?;
        let total = ctx.read_scalar_i64(ts)? as i32;
        let vp = total - s;
        if vp < 0 || vp > cap {
            // Defensive: an unexpected mask/total width — honor the growing contract.
            (false, cap)
        } else {
            (cap > vp, vp)
        }
    };
    // RoPE positions for the new Q/K rows run [valid_past, valid_past+S). In growing mode this is the
    // past length exactly as before (bit-for-bit); in shared-buffer mode it must be valid_past, NOT
    // the buffer capacity.
    let past = valid_past;

    let scale = attr_scale(n, head);

    let mut qh = split_heads(ctx, q, b, s, num_heads, head)?;
    let mut kh = split_heads(ctx, k, b, s, kv_heads, head)?;
    let vh = split_heads(ctx, v, b, s, kv_heads, head)?;

    if do_rotary {
        let cos = ctx.resolve(&n.inputs[7])?; // [max_seq, rot/2]
        let sin = ctx.resolve(&n.inputs[8])?;
        let half = ctx.shape_of(cos)[1]; // rot/2
        if ctx.rope_dynamic() {
            // Compiled-decode trace: the position offset is DATA (the cos/sin rows arrive as
            // pre-sliced closure inputs), and rotate-half is a [hd,hd] matmul so the shapeless graph
            // carries no Slice. Eligibility (checked before compiling) guarantees rot == head_dim.
            // The matmul rotate-half matrix is non-interleaved only; an interleaved model errors here
            // so the trace fails and the plan falls back to the eager path (never miscomputed).
            if interleaved {
                return Err(
                    "MLX: compiled-decode RoPE does not support rotary_interleaved (falls back to eager)"
                        .to_string(),
                );
            }
            let cos_full = ctx.rope_row_full(&n.inputs[7].name, s, half)?; // [1,1,S,2*half]
            let sin_full = ctx.rope_row_full(&n.inputs[8].name, s, half)?;
            qh = gqa_rope_matmul(ctx, qh, cos_full, sin_full, head, half)?;
            kh = gqa_rope_matmul(ctx, kh, cos_full, sin_full, head, half)?;
        } else if is_fp32(ctx, cos) {
            // Fused RoPE: positions run [past, past+S); the standard [max_seq, half] caches encode a
            // base/scale formula, so recover the period and let mlx_fast_rope apply it.
            let rot = 2 * half;
            let freqs = rope_freqs_from_cache(ctx, cos, sin, half)?;
            qh = fast_rope_static(ctx, qh, rot, interleaved, past, freqs)?;
            kh = fast_rope_static(ctx, kh, rot, interleaved, past, freqs)?;
        } else {
            // Reduced-precision cache: consume it directly via the composed path.
            let cr = cos_sin_row(ctx, cos, past, s, half)?;
            let sr = cos_sin_row(ctx, sin, past, s, half)?;
            qh = gqa_rope(ctx, qh, cr, sr, half, interleaved)?;
            kh = gqa_rope(ctx, kh, cr, sr, half, interleaved)?;
            ctx.mark_composed(
                "GroupQueryAttention RoPE composed: reduced-precision (fp16/bf16) cos/sin cache — recovered frequencies would drift from the stored values",
            );
        }
    }

    // Append the new K/V to the cache. Three contracts:
    //   * Shared-buffer, EAGER: write the S new rows in place at [valid_past, valid_past+S) of the
    //     [B,kv,cap,hd] buffer via slice_update, emit present at the FULL cap shape (matches ORT's
    //     pre-bound shared output), and attend over the valid prefix [0, valid_past+S) so causal
    //     alignment places the S queries at their true positions [valid_past, valid_past+S).
    //   * Shared-buffer, COMPILED PREFILL (shape-keyed): valid_past + S are static for the shape key,
    //     so narrow exactly like the eager path — write the S new rows in place at
    //     [valid_past, valid_past+S), then attend over only the valid prefix [0, valid_past+S) under a
    //     causal SDPA (STATIC slices, no full-cap mask). This drops the compute-bound full-capacity
    //     attention that made prefill-compile a TTFT regression while staying byte-identical to eager.
    //   * Shared-buffer, COMPILED DECODE (shapeless): valid_past is DATA (total_sequence_length - S),
    //     so the write is a slice_update_dynamic at that offset and attention uses a static-shape
    //     additive mask over the whole cap buffer (the tail beyond valid_past+S masked to -inf).
    //     Keeping every op statically shaped lets the shapeless compiled closure express it.
    //   * Growing: concat past+new along the sequence axis (unchanged legacy behavior).
    let (present_k, present_v, attn) =
        if shared_buffer && ctx.rope_dynamic() && ctx.compiled_shape_keyed() {
            let start = [0, 0, valid_past, 0];
            let stop = [b, kv_heads, valid_past + s, head];
            let pk = ctx.slice_update(past_k, kh, &start, &stop)?;
            let pv = ctx.slice_update(past_v, vh, &start, &stop)?;
            let vp1 = valid_past + s;
            let ak = slice(ctx, pk, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
            let av = slice(ctx, pv, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
            let attn = sdpa(ctx, qh, ak, av, scale, b"causal\0", empty_array())?;
            (pk, pv, attn)
        } else if shared_buffer && ctx.rope_dynamic() {
            gqa_shared_compiled(ctx, n, past_k, past_v, kh, vh, qh, s, cap, scale)?
        } else if shared_buffer {
            let start = [0, 0, valid_past, 0];
            let stop = [b, kv_heads, valid_past + s, head];
            let pk = ctx.slice_update(past_k, kh, &start, &stop)?;
            let pv = ctx.slice_update(past_v, vh, &start, &stop)?;
            let vp1 = valid_past + s;
            let ak = slice(ctx, pk, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
            let av = slice(ctx, pv, &[0, 0, 0, 0], &[b, kv_heads, vp1, head])?;
            let attn = gqa_eager_sdpa(ctx, n, qh, ak, av, scale, s, valid_past, vp1)?;
            (pk, pv, attn)
        } else {
            let pk = concat2(ctx, past_k, kh, 2)?;
            let pv = concat2(ctx, past_v, vh, 2)?;
            let attn = gqa_eager_sdpa(ctx, n, qh, pk, pv, scale, s, valid_past, valid_past + s)?;
            (pk, pv, attn)
        };

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
    // Shared-buffer `present` aliases `past` in ORT memory, so copy-out only needs the S new rows
    // written this step at axis-2 offset `valid_past`. Register the delta so the boundary copy-out
    // is O(new-tokens) instead of O(capacity). (No-op for the growing path, which never sets
    // `shared_buffer`, keeping its full copy-out bit-for-bit unchanged.)
    if shared_buffer {
        if n.outputs.len() >= 2 {
            ctx.record_kv_present(
                &n.outputs[1].name,
                valid_past as i64,
                s as i64,
                n.inputs[3].ctx_index,
            );
        }
        if n.outputs.len() >= 3 {
            ctx.record_kv_present(
                &n.outputs[2].name,
                valid_past as i64,
                s as i64,
                n.inputs[4].ctx_index,
            );
        }
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

    let attn = if causal {
        // ONNX `is_causal` uses upper-left alignment: the past keys (present K length minus the new
        // K length) are always visible and the causal triangle covers the new-key region. MLX's
        // built-in "causal" mode is lower-right aligned, so build the ONNX mask explicitly instead.
        let k_len = ctx.shape_of(present_k)[2];
        let cur_kv = ctx.shape_of(kh4)[2];
        let past_seq = k_len - cur_kv;
        let cmask = causal_mask_topleft(ctx, s, k_len, past_seq, dt)?;
        sdpa(ctx, qh4, present_k, present_v, scale, b"array\0", cmask)?
    } else {
        sdpa_dispatch(ctx, qh4, present_k, present_v, scale, false, mask, dt)?
    };

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

    let cos_cache = ctx.resolve(&n.inputs[ci])?;
    let sin_cache = ctx.resolve(&n.inputs[si])?;

    let (b, nh, s, hd, x4) = if rank == 4 {
        (xs[0], xs[1], xs[2], xs[3], x)
    } else {
        let b = xs[0];
        let s = xs[1];
        let nh_attr = attr_int(n, "num_heads", 0) as i32;
        // num_heads=0 (Gemma3n form): infer the head size from the cos cache. With
        // rotary_embedding_dim=0 (full-head rotary) head_size = 2 * cos_cache_last_dim, so
        // num_heads = hidden / head_size.
        let hd = if nh_attr > 0 {
            xs[2] / nh_attr
        } else {
            let cos_shape = ctx.shape_of(cos_cache);
            2 * cos_shape[cos_shape.len() - 1]
        };
        let nh = xs[2] / hd;
        let r = ctx.reshape(x, &[b, s, nh, hd])?;
        let x4 = ctx.transpose(r, &[0, 2, 1, 3])?; // [B,N,S,hd]
        (b, nh, s, hd, x4)
    };

    let has_pos = present(n, pi);
    // The fused mlx_fast_rope path derives each position as `offset + s`, i.e. it assumes the
    // per-row positions are contiguous. That only holds for the [1] offset form (positions =
    // offset + [0,S)). A 2D [B,S] position_ids tensor may carry arbitrary (non-contiguous)
    // positions, so it must be served by the composed gather path (which indexes the cos/sin
    // cache by the exact position_ids). ndim is statically known from the graph shapes.
    let pos_rank = if has_pos {
        let p = ctx.resolve(&n.inputs[pi])?;
        ctx.ndim(p) as i32
    } else {
        -1
    };
    let fast_ok = has_pos && pos_rank == 1 && is_fp32(ctx, cos_cache);

    let out4 = if fast_ok {
        // Standard fp32 [max_seq, half] cache indexed by a [1] position offset: recover the RoPE
        // period and apply the fused mlx_fast_rope kernel (positions = offset + [0,S)).
        let pos = ctx.resolve(&n.inputs[pi])?;
        let half = ctx.shape_of(cos_cache)[1];
        let rot = 2 * half;
        let freqs = rope_freqs_from_cache(ctx, cos_cache, sin_cache, half)?;
        // position_ids [1] absolute offset; MLX applies positions = offset + [0,S).
        let offset = ctx.astype(pos, mlx::mlx_dtype__MLX_INT32)?;
        fast_rope_dynamic(ctx, x4, rot, interleaved, offset, freqs)?
    } else {
        // Composed fallback for the two forms mlx_fast_rope cannot faithfully reproduce:
        //   * absent position_ids — cos/sin is an explicit per-position [B,S,half] cache that need
        //     not follow any base/scale formula;
        //   * reduced-precision (fp16/bf16) cache — frequencies recovered from it would drift from
        //     the exact values the cache stored.
        let (pos, pos_rank) = if has_pos {
            let p = ctx.resolve(&n.inputs[pi])?;
            let r = ctx.ndim(p) as i32;
            (Some(p), r)
        } else {
            (None, -1)
        };
        let cos4 = gather_cache(ctx, cos_cache, pos, pos_rank, s)?;
        let sin4 = gather_cache(ctx, sin_cache, pos, pos_rank, s)?;
        let half = ctx.shape_of(cos4)[3];
        let out = rope_apply(ctx, x4, cos4, sin4, half, interleaved)?;
        ctx.mark_composed(match (has_pos, pos_rank) {
            // [B,S] position_ids may carry arbitrary (non-contiguous) positions, so the cache is
            // gathered per exact position rather than via the fused offset+s kernel.
            (true, 2) => "RotaryEmbedding composed: [B,S] position_ids gather (positions may be non-contiguous — mlx_fast_rope's offset+s form does not apply)",
            (true, _) => "RotaryEmbedding composed: reduced-precision (fp16/bf16) cos/sin cache — recovered frequencies would drift from the stored values",
            (false, _) => "RotaryEmbedding composed: absent position_ids supplies an explicit per-position cos/sin cache (no base/scale formula) — mlx_fast_rope not applicable",
        });
        out
    };

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

/// GroupQueryAttention (com.microsoft): separate-QKV decode/prefill layout. Two accepted layouts:
///   * 9-input (q, k, v, past_k, past_v, seqlens_k, total_seq, cos, sin) — in-op RoPE decoder.
///   * 11-input (…, seqlens_k, total_seq, cos, sin, position_ids, attention_bias) — the Gemma3n
///     variant with `do_rotary=0` (cos/sin absent, rotary applied by external RotaryEmbedding nodes)
///     and an additive `attention_bias` mask (input 10). `position_ids` (input 9) is ignored.
/// All floating inputs/outputs share one dtype; seqlens_k / total_sequence_length are int32.
fn group_query_attention_claim(node: &NodeView) -> ClaimResult {
    require!(node.num_outputs() > 0, "requires at least 1 output");
    let out_type = match node.output_info(0) {
        Some(o) if is_mlx_float(o.dtype) => o.dtype,
        Some(o) => deny!(
            "output must have an MLX float dtype, got {}",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("output lacks tensor type/shape info"),
    };
    let ninputs = node.num_inputs();
    require!(
        ninputs == 9 || ninputs == 11,
        "expects 9 inputs (q, k, v, past_k, past_v, seqlens_k, total_seq, cos, sin) or 11 inputs \
         (…, position_ids, attention_bias), got {}",
        ninputs
    );
    // The 11-input Gemma3n variant only maps to MLX when rotary is external (do_rotary=0): cos/sin at
    // 7,8 are absent, so we must not require their dtype and must not resolve them in the handler.
    let has_bias = ninputs == 11;
    let do_rotary = node.int_attr("do_rotary", 1) != 0;
    if has_bias {
        require!(
            !do_rotary,
            "11-input GroupQueryAttention is only supported with do_rotary=0 (external rotary); \
             got do_rotary=1"
        );
    }
    // Float inputs that must match the output dtype. cos/sin (7,8) only exist in the 9-input form.
    let float_idx: &[usize] = if has_bias {
        &[0usize, 1, 2, 3, 4]
    } else {
        &[0usize, 1, 2, 3, 4, 7, 8]
    };
    for &idx in float_idx {
        let dtype = match dtype_of(node, idx) {
            Some(dtype) => dtype,
            None => deny!("input {} lacks tensor type/shape info", idx),
        };
        require!(
            dtype == out_type,
            "input {} must have output dtype {}, got {}",
            idx,
            crate::registry::ort_dtype_name(out_type),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    for idx in 1..node.num_outputs().min(3) {
        let dtype = match node.output_info(idx) {
            Some(o) => o.dtype,
            None => deny!("output {} lacks tensor type/shape info", idx),
        };
        require!(
            dtype == out_type,
            "output {} must have dtype {}, got {}",
            idx,
            crate::registry::ort_dtype_name(out_type),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    for idx in [5, 6] {
        let dtype = match dtype_of(node, idx) {
            Some(dtype) => dtype,
            None => deny!("input {} lacks tensor type/shape info", idx),
        };
        require!(
            is_int32(dtype),
            "input {} must have int32 dtype, got {}",
            idx,
            crate::registry::ort_dtype_name(dtype)
        );
    }
    // 11-input variant: attention_bias (input 10) is an additive [B,1,S,total] mask folded into the
    // SDPA scores. Require it present, sharing the output float dtype, and rank 4. position_ids
    // (input 9) is ignored when do_rotary=0, so it may be present or absent.
    if has_bias {
        require!(
            node.input_present(10),
            "11-input GroupQueryAttention requires attention_bias (input 10)"
        );
        let bias = match node.input_info(10) {
            Some(info) => info,
            None => deny!("attention_bias (input 10) lacks tensor type/shape info"),
        };
        require!(
            bias.dtype == out_type,
            "attention_bias (input 10) must have output dtype {}, got {}",
            crate::registry::ort_dtype_name(out_type),
            crate::registry::ort_dtype_name(bias.dtype)
        );
        require!(
            bias.shape.len() == 4,
            "attention_bias (input 10) must have rank 4, got rank {}",
            bias.shape.len()
        );
    }
    let nh = node.int_attr("num_heads", 0);
    let kvh = node.int_attr("kv_num_heads", 0);
    require!(
        nh > 0 && kvh > 0 && nh % kvh == 0,
        "num_heads ({}) must be a positive multiple of kv_num_heads ({})",
        nh,
        kvh
    );
    require!(
        node.int_attr("smooth_softmax", 0) != 1,
        "smooth_softmax=1 is unsupported"
    );
    require!(
        node.int_attr("qk_output", 0) == 0,
        "qk_output is unsupported"
    );
    require!(
        node.float_attr("softcap", 0.0) == 0.0,
        "softcap is unsupported"
    );
    Ok(())
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
fn check_kv_cache(
    node: &NodeView,
    past_k: usize,
    past_v: usize,
    qd: ort::ONNXTensorElementDataType,
) -> bool {
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
fn check_mask(
    node: &NodeView,
    mask_idx: usize,
    causal: bool,
    _qd: ort::ONNXTensorElementDataType,
) -> bool {
    if !node.input_present(mask_idx) {
        return true;
    }
    if causal {
        return false;
    }
    match dtype_of(node, mask_idx) {
        Some(md) => {
            md == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                || is_mlx_float(md)
        }
        None => false,
    }
}

/// Attention (ai.onnx, opset 23 and 24). 3D/4D SDPA, optional attn_mask + past/present KV.
fn attention_claim(node: &NodeView) -> ClaimResult {
    let qd = match check_qkv_float(node) {
        Some(qd) => qd,
        None => deny!("Q, K, V, and output 0 must be present and share one MLX float dtype"),
    };
    let qshape = match node.input_info(0) {
        Some(i) => i.shape,
        None => deny!("Q lacks tensor type/shape info"),
    };
    let kshape = match node.input_info(1) {
        Some(i) => i.shape,
        None => deny!("K lacks tensor type/shape info"),
    };
    let vshape = match node.input_info(2) {
        Some(i) => i.shape,
        None => deny!("V lacks tensor type/shape info"),
    };
    let rank = qshape.len();
    require!(
        rank == 3 || rank == 4,
        "Q must have rank 3 or 4, got rank {}",
        rank
    );
    require!(
        kshape.len() == rank && vshape.len() == rank,
        "Q, K, and V must have equal rank, got {}, {}, {}",
        rank,
        kshape.len(),
        vshape.len()
    );

    let (qh, kvh) = if rank == 3 {
        let qh = node.int_attr("q_num_heads", 0);
        let kvh = node.int_attr("kv_num_heads", 0);
        require!(
            qh > 0 && kvh > 0,
            "q_num_heads and kv_num_heads must be positive, got {} and {}",
            qh,
            kvh
        );
        require!(
            qshape.iter().all(|&d| d > 0),
            "Q must have static positive dimensions, got shape {:?}",
            qshape
        );
        require!(
            kshape[1] > 0 && vshape[1] > 0 && kshape[2] > 0 && vshape[2] > 0,
            "K and V must have static positive sequence and hidden dimensions, got {:?} and {:?}",
            kshape,
            vshape
        );
        require!(
            qshape[2] % qh == 0 && kshape[2] % kvh == 0 && vshape[2] % kvh == 0,
            "hidden dimensions Q/K/V ({}/{}/{}) must divide evenly by head counts {}/{}",
            qshape[2],
            kshape[2],
            vshape[2],
            qh,
            kvh
        );
        (qh, kvh)
    } else {
        let qh = qshape[1];
        let kvh = kshape[1];
        require!(
            qh > 0 && kvh > 0,
            "Q and K head dimensions must be positive, got {} and {}",
            qh,
            kvh
        );
        (qh, kvh)
    };
    require!(
        qh % kvh == 0,
        "Q head count {} must be a multiple of KV head count {}",
        qh,
        kvh
    );
    require!(
        node.float_attr("softcap", 0.0) == 0.0,
        "logit soft-cap is unsupported"
    );
    require!(!node.output_present(3), "qk_matmul_output is unsupported");
    require!(!node.input_present(6), "nonpad_kv_seqlen is unsupported");
    let causal = node.int_attr("is_causal", 0) != 0;
    require!(check_mask(node, 3, causal, qd), "attention mask must be bool or float and cannot be used with is_causal; this guards SDPA mask dispatch");
    require!(
        check_kv_cache(node, 4, 5, qd),
        "past K/V must be paired and match query dtype {}; present K/V require a cache",
        crate::registry::ort_dtype_name(qd)
    );
    Ok(())
}

/// MultiHeadAttention (com.microsoft). Separate 3D Q/K/V + optional projection bias.
fn multihead_attention_claim(node: &NodeView) -> ClaimResult {
    let h = node.int_attr("num_heads", 0);
    require!(h > 0, "num_heads must be positive, got {}", h);
    let qd = match check_qkv_float(node) {
        Some(qd) => qd,
        None => deny!("Q, K, V, and output 0 must be present and share one MLX float dtype"),
    };
    let qshape = match node.input_info(0) {
        Some(i) => i.shape,
        None => deny!("Q lacks tensor type/shape info"),
    };
    let kshape = match node.input_info(1) {
        Some(i) => i.shape,
        None => deny!("K lacks tensor type/shape info"),
    };
    let vshape = match node.input_info(2) {
        Some(i) => i.shape,
        None => deny!("V lacks tensor type/shape info"),
    };
    require!(
        qshape.len() == 3 && kshape.len() == 3 && vshape.len() == 3,
        "Q, K, and V must be rank 3, got ranks {}, {}, {}",
        qshape.len(),
        kshape.len(),
        vshape.len()
    );
    for (name, shape) in [("Q", &qshape), ("K", &kshape), ("V", &vshape)] {
        require!(
            shape.iter().all(|&d| d > 0),
            "{} must have static positive dimensions, got shape {:?}",
            name,
            shape
        );
    }
    require!(
        qshape[2] % h == 0 && kshape[2] % h == 0 && vshape[2] % h == 0,
        "Q/K/V hidden dimensions ({}/{}/{}) must divide evenly by num_heads {}",
        qshape[2],
        kshape[2],
        vshape[2],
        h
    );
    if node.input_present(3) {
        let (bd, bshape) = match node.input_info(3) {
            Some(i) => (i.dtype, i.shape),
            None => deny!("projection bias lacks tensor type/shape info"),
        };
        require!(
            bd == qd && bshape.len() == 1 && bshape[0] == qshape[2] + kshape[2] + vshape[2],
            "projection bias must be a 1D {} tensor of length {}, got {} shape {:?}",
            crate::registry::ort_dtype_name(qd),
            qshape[2] + kshape[2] + vshape[2],
            crate::registry::ort_dtype_name(bd),
            bshape
        );
    }
    for i in 4..=9 {
        require!(
            !node.input_present(i),
            "optional input {} is unsupported",
            i
        );
    }
    require!(
        !node.output_present(1) && !node.output_present(2) && !node.output_present(3),
        "cache and qk_matmul outputs are unsupported"
    );
    Ok(())
}

/// RotaryEmbedding (ai.onnx opset 23 / com.microsoft). Float 3D (B,S,H*hd)+num_heads or 4D input;
/// [B,S] gather, [1] offset (com.microsoft), or (ai.onnx only) absent pos with [B,S,half] cache.
fn rotary_embedding_claim(node: &NodeView) -> ClaimResult {
    require!(node.num_outputs() > 0, "requires at least 1 output");
    let ms = node.domain() == "com.microsoft";
    let ci = if ms { 2 } else { 1 };
    let si = if ms { 3 } else { 2 };
    let pi = if ms { 1 } else { 3 };
    let min_inputs = if ms { 4 } else { 3 };
    require!(
        node.num_inputs() >= min_inputs,
        "expects at least {} inputs, got {}",
        min_inputs,
        node.num_inputs()
    );
    let (xd, xshape) = match node.input_info(0) {
        Some(i) => (i.dtype, i.shape),
        None => deny!("input lacks tensor type/shape info"),
    };
    let (cd, cshape) = match node.input_info(ci) {
        Some(i) => (i.dtype, i.shape),
        None => deny!("cos cache lacks tensor type/shape info"),
    };
    require!(node.input_present(si), "sin cache input is required");
    let sd = match dtype_of(node, si) {
        Some(t) => t,
        None => deny!("sin cache lacks tensor type/shape info"),
    };
    let od = match node.output_info(0) {
        Some(o) => o.dtype,
        None => deny!("output lacks tensor type/shape info"),
    };
    require!(is_mlx_float(xd) && cd == xd && sd == xd && od == xd, "input, cos cache, sin cache, and output must share one MLX float dtype, got {}, {}, {}, {}", crate::registry::ort_dtype_name(xd), crate::registry::ort_dtype_name(cd), crate::registry::ort_dtype_name(sd), crate::registry::ort_dtype_name(od));
    let rank = xshape.len();
    if rank == 3 {
        let nh = node.int_attr("num_heads", 0);
        if nh > 0 {
            require!(
                xshape[2] > 0 && xshape[2] % nh == 0,
                "rank-3 input hidden dimension {} must divide evenly by positive num_heads {}",
                xshape[2],
                nh
            );
        } else {
            // num_heads=0 (Gemma3n form): infer head size from the cos cache. Only the full-head
            // rotary case (rotary_embedding_dim=0) lets us derive head_size = 2 * cos_last_dim.
            let red = node.int_attr("rotary_embedding_dim", 0);
            require!(
                red == 0,
                "rank-3 RotaryEmbedding with num_heads=0 needs rotary_embedding_dim=0 (full-head \
                 rotary) to infer the head size; got rotary_embedding_dim={red}"
            );
            let half = cshape.last().copied().unwrap_or(0);
            let head = 2 * half;
            require!(
                head > 0 && xshape[2] > 0 && xshape[2] % head == 0,
                "rank-3 input hidden dimension {} must divide evenly by the inferred head size {} \
                 (2 × cos-cache last dim); num_heads=0",
                xshape[2],
                head
            );
        }
    } else {
        require!(rank == 4, "input must have rank 3 or 4, got rank {}", rank);
    }
    let has_pos = node.input_present(pi);
    require!(
        !ms || has_pos,
        "com.microsoft RotaryEmbedding requires position_ids"
    );
    if has_pos {
        let (pd, pshape) = match node.input_info(pi) {
            Some(i) => (i.dtype, i.shape),
            None => deny!("position_ids lacks tensor type/shape info"),
        };
        require!(
            pd == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                || pd == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            "position_ids must have int32 or int64 dtype, got {}",
            crate::registry::ort_dtype_name(pd)
        );
        let gather = pshape.len() == 2;
        let offset = ms && pshape.len() == 1 && pshape[0] == 1;
        require!(
            gather || offset,
            "position_ids must be [B,S] or, for com.microsoft, [1], got shape {:?}",
            pshape
        );
        require!(
            cshape.len() == 2,
            "gather/offset cos cache must have rank 2, got shape {:?}",
            cshape
        );
    } else {
        require!(
            cshape.len() == 3,
            "absent-position cos cache must have rank 3, got shape {:?}",
            cshape
        );
    }
    Ok(())
}

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
    reg(
        registry,
        "com.microsoft",
        "GroupQueryAttention",
        K_ANY_OPSET,
        K_ANY_OPSET,
        group_query_attention_op,
        group_query_attention_claim,
    );
    // Attention entered ai.onnx at opset 23; opset 24 adds the trailing nonpad_kv_seqlen input.
    reg(
        registry,
        "",
        "Attention",
        23,
        23,
        attention_op,
        attention_claim,
    );
    reg(
        registry,
        "",
        "Attention",
        24,
        K_ANY_OPSET,
        attention_op,
        attention_claim,
    );
    reg(
        registry,
        "com.microsoft",
        "MultiHeadAttention",
        K_ANY_OPSET,
        K_ANY_OPSET,
        multihead_attention_op,
        multihead_attention_claim,
    );
    // RotaryEmbedding: ai.onnx entered at opset 23; com.microsoft is version-insensitive.
    reg(
        registry,
        "",
        "RotaryEmbedding",
        23,
        K_ANY_OPSET,
        rotary_embedding_op,
        rotary_embedding_claim,
    );
    reg(
        registry,
        "com.microsoft",
        "RotaryEmbedding",
        K_ANY_OPSET,
        K_ANY_OPSET,
        rotary_embedding_op,
        rotary_embedding_claim,
    );
}
