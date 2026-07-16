//! Quantized / integer op handlers — the fp32 quant path used by the cpu-recipe decoder and the
//! ai.onnx integer/QLinear coverage. Faithful port of the C++ `ops/quant.cc`, `ops/quantize.cc` and
//! `ops/quant2.cc`:
//!
//!   * MatMulNBits          — int4 block-quantized weight matmul. BOTH the 3-input SYMMETRIC form
//!                            (implicit zero point 8) AND the 4-input ASYMMETRIC form with an explicit
//!                            packed-int4 `zero_points` input (uint8). Dequant is w = (q - zp) * scale.
//!                            For MLX-supported group sizes (32/64/128) the weight is repacked ONCE to
//!                            MLX affine uint32 words + per-block scales/biases and run through
//!                            `mlx_quantized_matmul` (weights stay compressed — the decode memory win).
//!                            For block_size 16 (which `mlx_quantized_matmul` does not support) the
//!                            weight is dequantized in-graph to fp32 [N,K] and a dense `mlx_matmul` runs.
//!   * GatherBlockQuantized — int4 block-quantized embedding gather + dequant, SYMMETRIC (zp=8) and
//!                            ASYMMETRIC (explicit packed-int4 `zero_points`) forms.
//!   * QuantizeLinear        — y = saturate(round(x / scale) + zero_point).
//!   * DequantizeLinear      — y = (x - zero_point) * scale.
//!   * DynamicQuantizeLinear — compute affine uint8 scale + zero point from x's [min,0]..[max,0] range.
//!   * MatMulInteger         — (A - a_zp) @ (B - b_zp), int32 (exact via a small-K fp32 GEMM gate).
//!   * ConvInteger           — conv(x - x_zp, w - w_zp), int32 (exact via a small-accumulation gate).
//!   * QLinearMatMul         — dequant a,b -> matmul -> requantize to int8/uint8.
//!   * QLinearConv           — dequant x,w -> conv (+int32 bias) -> requantize to int8/uint8.
//!
//! Quantized weights arrive as constant initializers (surfaced as constant ctx inputs); the MatMulNBits
//! weight repack runs once and is cached on the Plan (keyed by weight name).

use std::os::raw::c_void;

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    is_int_index, is_mlx_float, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- small MLX helpers (each keeps + returns the raw result) -------------------------------------

fn mul(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_multiply, a, b)
}
fn sub(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_subtract, a, b)
}
fn add(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_add, a, b)
}
fn div(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_divide, a, b)
}

/// round-half-to-even (matches ONNX / ORT CPU rounding).
fn round_e(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_round(res, a, 0, s) })
}

fn clip(ctx: &mut TranslationContext, a: mlx::mlx_array, lo: mlx::mlx_array, hi: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, a, lo, hi, s) })
}

/// Full reduction to a scalar.
fn reduce_max(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_max(res, a, false, s) })
}
fn reduce_min(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_min(res, a, false, s) })
}

fn f32(ctx: &mut TranslationContext, v: f32) -> mlx::mlx_array {
    ctx.scalar_f32(v)
}

/// A kept 0-D uint32 scalar (bit masks / shift amounts for nibble unpacking).
fn u32_scalar(ctx: &mut TranslationContext, v: u32) -> mlx::mlx_array {
    let sh: [i32; 0] = [];
    ctx.keep(Array::from_data(
        &v as *const u32 as *const c_void,
        &sh,
        mlx::mlx_dtype__MLX_UINT32,
    ))
}

/// A node input slot is present (not an omitted optional).
fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

/// Reshape a 1-D per-axis parameter (scale / zero point) of length L to a rank-`rank` shape that is 1
/// on every axis except `axis` (= L), so it broadcasts against the data tensor. A scalar (size <= 1)
/// parameter already broadcasts and is returned unchanged.
fn align_per_axis(
    ctx: &mut TranslationContext,
    param: mlx::mlx_array,
    rank: usize,
    axis: usize,
) -> Result<mlx::mlx_array, MlxError> {
    let len = ctx.size_of(param);
    if len <= 1 {
        return Ok(param);
    }
    let mut shape = vec![1i32; rank.max(1)];
    let ax = axis.min(shape.len() - 1);
    shape[ax] = len as i32;
    ctx.reshape(param, &shape)
}

fn norm_axis(n: &NodeDesc, rank: usize) -> usize {
    let mut axis = n.ints.get("axis").copied().unwrap_or(1);
    let r = rank as i64;
    if axis < 0 {
        axis += r;
    }
    if axis < 0 {
        axis = 0;
    }
    if axis >= r {
        axis = if r > 0 { r - 1 } else { 0 };
    }
    axis as usize
}

/// The integer range and MLX dtype for a quantized ONNX element type.
fn range_for(t: ort::ONNXTensorElementDataType) -> Option<(f32, f32, mlx::mlx_dtype)> {
    use ort::*;
    #[allow(non_upper_case_globals)]
    match t {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
            Some((-128.0, 127.0, mlx::mlx_dtype__MLX_INT8))
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
            Some((0.0, 255.0, mlx::mlx_dtype__MLX_UINT8))
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            Some((-32768.0, 32767.0, mlx::mlx_dtype__MLX_INT16))
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
            Some((0.0, 65535.0, mlx::mlx_dtype__MLX_UINT16))
        }
        _ => None,
    }
}

// ---- shared int4 unpack / block-broadcast helpers ------------------------------------------------

/// Gather rows `idx` (0-axis) of `src` → [BS, ...].
fn gather_rows(ctx: &mut TranslationContext, src: mlx::mlx_array, idx: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, src, idx, 0, s) })
}

/// Unpack the interleaved low/high int4 nibbles of a packed uint8 tensor [BS, P] into the flattened
/// int4 values [BS, 2P] (uint32): column order low(byte0), high(byte0), low(byte1), high(byte1), …
/// This is the nibble layout both the packed int4 weight `data` and the packed int4 `zero_points`
/// use along the quantize axis.
fn unpack_nibbles(ctx: &mut TranslationContext, packed_u8: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    let sh = ctx.shape_of(packed_u8);
    let bs = sh[0];
    let p = sh[1];
    let g32 = ctx.astype(packed_u8, mlx::mlx_dtype__MLX_UINT32)?;
    let mask = u32_scalar(ctx, 0x0F);
    let low = ctx.binary(mlx::mlx_bitwise_and, g32, mask)?;
    let four = u32_scalar(ctx, 4);
    let hi_sh = ctx.binary(mlx::mlx_right_shift, g32, four)?;
    let mask2 = u32_scalar(ctx, 0x0F);
    let high = ctx.binary(mlx::mlx_bitwise_and, hi_sh, mask2)?;

    let mut pair = VectorArray::new();
    pair.append(low);
    pair.append(high);
    let stacked = ctx.emit(|res, s| unsafe { mlx::mlx_stack_axis(res, pair.as_raw(), 2, s) })?; // [BS,P,2]
    drop(pair);
    ctx.reshape(stacked, &[bs, p * 2])
}

/// Broadcast a per-block float tensor [BS, nblocks] up to per-element [BS, nblocks*block]: element j
/// of a row picks its block value from block j/block.
fn broadcast_blocks(
    ctx: &mut TranslationContext,
    blocks: mlx::mlx_array,
    bs: i32,
    nblocks: i32,
    block: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let r = ctx.reshape(blocks, &[bs, nblocks, 1])?;
    let bshape = [bs, nblocks, block];
    let b = ctx.emit(|res, s| unsafe {
        mlx::mlx_broadcast_to(res, r, bshape.as_ptr(), bshape.len(), s)
    })?;
    ctx.reshape(b, &[bs, nblocks * block])
}

// ---- MatMulNBits --------------------------------------------------------------------------------

/// The float dtype to run an in-graph dequant + dense matmul in: the activation's own float dtype
/// when it is fp16/bf16 (halving weight bytes + speeding the matmul), otherwise fp32.
fn compute_float_dtype(act_dt: mlx::mlx_dtype) -> mlx::mlx_dtype {
    if act_dt == mlx::mlx_dtype__MLX_FLOAT16 || act_dt == mlx::mlx_dtype__MLX_BFLOAT16 {
        act_dt
    } else {
        mlx::mlx_dtype__MLX_FLOAT32
    }
}


/// Repack our uint8 [N, nblocks, block/2] int4 weight to MLX affine uint32 words [N, K/8] (8 nibbles
/// per word, low→high along K). Cached (constant) or kept (dynamic).
fn matmulnbits_repack(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    big_n: i64,
    k: i64,
    block: i64,
) -> Result<mlx::mlx_array, MlxError> {
    let wref = &n.inputs[1];
    let key = format!("{}#qw", wref.name);
    if wref.constant || wref.source == Src::Initializer {
        if let Some(w) = ctx.cache_get(&key) {
            return Ok(w);
        }
    }
    let host = ctx.raw_host(wref)?;
    let src = host.data as *const u8;
    let nblocks = k / block;
    let blob = block / 2;
    let words = (k / 8) as usize;
    let mut packed = vec![0u32; (big_n as usize) * words];
    if !src.is_null() {
        let bytes = unsafe { std::slice::from_raw_parts(src, host.count) };
        for row in 0..big_n {
            for kk in 0..k {
                let blk = kk / block;
                let within = kk % block;
                let byte = within / 2;
                let nib = within % 2;
                let idx = ((row * nblocks + blk) * blob + byte) as usize;
                let b = bytes[idx];
                let q = if nib == 0 { (b & 0x0F) as u32 } else { (b >> 4) as u32 };
                let word = (row as usize) * words + (kk / 8) as usize;
                packed[word] |= q << (((kk % 8) as u32) * 4);
            }
        }
    }
    let sh = [big_n as i32, words as i32];
    let arr = Array::from_data(
        packed.as_ptr() as *const c_void,
        &sh,
        mlx::mlx_dtype__MLX_UINT32,
    );
    if wref.constant || wref.source == Src::Initializer {
        Ok(ctx.cache_put(key, arr))
    } else {
        Ok(ctx.keep(arr))
    }
}

/// Per-block int4 zero points [N, nblocks] as fp32 (asymmetric form): unpack the packed uint8
/// `zero_points` input and trim any trailing padding nibble.
fn matmulnbits_zp_f32(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    big_n: i32,
    nblocks: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let zp_in = ctx.resolve(&n.inputs[3])?; // uint8, packed int4 (N * ceil(nblocks/2) elements)
    let cols_per_row = (nblocks + 1) / 2; // ceil(nblocks/2)
    let zp_packed = ctx.reshape(zp_in, &[big_n, cols_per_row])?; // [N, ceil(nblocks/2)]
    let zp_un = unpack_nibbles(ctx, zp_packed)?; // [N, 2*ceil(nblocks/2)]
    let cols = ctx.shape_of(zp_un)[1];
    let zp_un = if cols != nblocks {
        let start = [0i32, 0];
        let stop = [big_n, nblocks];
        let strides = [1i32, 1];
        let sl = ctx.emit(|res, s| unsafe {
            mlx::mlx_slice(
                res,
                zp_un,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                s,
            )
        })?;
        ctx.contiguous(sl)?
    } else {
        zp_un
    };
    ctx.astype(zp_un, mlx::mlx_dtype__MLX_FLOAT32)
}

fn matmulnbits_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let k = *n.ints.get("K").ok_or("MatMulNBits: missing K")?;
    let big_n = *n.ints.get("N").ok_or("MatMulNBits: missing N")?;
    let block = n.ints.get("block_size").copied().unwrap_or(32);
    if block <= 0 || k % block != 0 {
        return Err("MatMulNBits: bad block_size".to_string());
    }
    let nblocks = (k / block) as i32;
    let has_zp = present(n, 3);

    let a = ctx.resolve(&n.inputs[0])?;
    let ashape = ctx.shape_of(a);
    let mut m: i32 = 1;
    for i in 0..ashape.len().saturating_sub(1) {
        m *= ashape[i];
    }
    let a2 = ctx.reshape(a, &[m, k as i32])?;
    let act_dt = ctx.dtype_of(a2);

    let supported = block == 32 || block == 64 || block == 128;
    let y = if supported {
        ctx.mark_fast("mlx_quantized_matmul");
        // Fast path: repacked uint32 weight + per-block scales/biases through mlx_quantized_matmul.
        let w = matmulnbits_repack(ctx, n, big_n, k, block)?;
        let scales = ctx.resolve(&n.inputs[2])?;
        let scales2d = ctx.reshape(scales, &[big_n as i32, nblocks])?;
        let biases = if has_zp {
            // bias = -(zp * scale)
            let zpf = matmulnbits_zp_f32(ctx, n, big_n as i32, nblocks)?;
            let neg_zp = ctx.unary(mlx::mlx_negative, zpf)?;
            mul(ctx, neg_zp, scales2d)?
        } else {
            let neg8 = f32(ctx, -8.0);
            mul(ctx, scales2d, neg8)?
        };
        let gs = mlx::mlx_optional_int_ { value: block as i32, has_value: true };
        let bb = mlx::mlx_optional_int_ { value: 4, has_value: true };
        let mode = b"affine\0".as_ptr() as *const std::os::raw::c_char;
        ctx.emit(|res, s| unsafe {
            mlx::mlx_quantized_matmul(res, a2, w, scales2d, biases, true, gs, bb, mode, s)
        })?
    } else {
        // Fallback (block_size mlx_quantized_matmul cannot handle, e.g. 16): dequantize in-graph.
        // Dequant + dense matmul run in the activation's float dtype (fp16/bf16 halves the weight
        // bytes + speeds the matmul; fp32 activations keep fp32 — no change).
        let comp_dt = compute_float_dtype(act_dt);
        ctx.mark_composed(format!(
            "block_size {block} unsupported by mlx_quantized_matmul → dequant + dense matmul ({})",
            crate::engine::dtype_name(comp_dt)
        ));
        let wpacked = ctx.resolve(&n.inputs[1])?; // uint8 [N, nblocks, block/2]
        let wflat = ctx.reshape(wpacked, &[big_n as i32, (k / 2) as i32])?;
        let q = unpack_nibbles(ctx, wflat)?; // uint32 [N, K]
        let qf = ctx.astype(q, comp_dt)?;
        let centered = if has_zp {
            let zpf = matmulnbits_zp_f32(ctx, n, big_n as i32, nblocks)?;
            let zpf = ctx.astype(zpf, comp_dt)?;
            let zp_full = broadcast_blocks(ctx, zpf, big_n as i32, nblocks, block as i32)?;
            sub(ctx, qf, zp_full)?
        } else {
            let eight_f = f32(ctx, 8.0);
            let eight = ctx.astype(eight_f, comp_dt)?;
            sub(ctx, qf, eight)?
        };
        let scales = ctx.resolve(&n.inputs[2])?;
        let scales2d = ctx.reshape(scales, &[big_n as i32, nblocks])?;
        let scales2d = ctx.astype(scales2d, comp_dt)?;
        let sc_full = broadcast_blocks(ctx, scales2d, big_n as i32, nblocks, block as i32)?;
        let wdeq = mul(ctx, centered, sc_full)?; // [N, K] comp_dt
        let wt = ctx.transpose(wdeq, &[1, 0])?; // [K, N]
        let a2c = ctx.astype(a2, comp_dt)?;
        ctx.binary(mlx::mlx_matmul, a2c, wt)?
    };

    // Restore leading dims with N as the last dim.
    let mut oshape: Vec<i32> = ashape.clone();
    if let Some(last) = oshape.last_mut() {
        *last = big_n as i32;
    } else {
        oshape.push(big_n as i32);
    }
    let out = ctx.reshape(y, &oshape)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- GatherBlockQuantized -----------------------------------------------------------------------

fn gather_block_quantized_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let block = n.ints.get("block_size").copied().unwrap_or(32) as i32;
    let idx_in = ctx.resolve(&n.inputs[1])?;
    let data = ctx.resolve(&n.inputs[0])?; // uint8 [V, D/2]
    let scales = ctx.resolve(&n.inputs[2])?; // f32 [V, nblocks]

    let ish = ctx.shape_of(idx_in);
    let mut bs: i32 = 1;
    for &d in &ish {
        bs *= d;
    }
    let idx_r = ctx.reshape(idx_in, &[bs])?;
    let idx = ctx.astype(idx_r, mlx::mlx_dtype__MLX_INT32)?;

    let g = gather_rows(ctx, data, idx)?; // [BS, D/2] uint8
    let sg = gather_rows(ctx, scales, idx)?; // [BS, nblocks]

    let packed = ctx.shape_of(g)[1];
    let d = packed * 2;
    let nblocks = d / block;

    let q = unpack_nibbles(ctx, g)?; // [BS, D] uint32
    let qf = ctx.astype(q, mlx::mlx_dtype__MLX_FLOAT32)?;

    let centered = if present(n, 3) {
        let zp_data = ctx.resolve(&n.inputs[3])?; // uint8 [V, nblocks/2] packed int4
        let zpg = gather_rows(ctx, zp_data, idx)?; // [BS, nblocks/2]
        let zp_un = unpack_nibbles(ctx, zpg)?; // [BS, 2*(nblocks/2)]
        let zp_un = if ctx.shape_of(zp_un)[1] != nblocks {
            let start = [0i32, 0];
            let stop = [bs, nblocks];
            let strides = [1i32, 1];
            let sl = ctx.emit(|res, s| unsafe {
                mlx::mlx_slice(
                    res,
                    zp_un,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    s,
                )
            })?;
            ctx.contiguous(sl)?
        } else {
            zp_un
        };
        let zpf = ctx.astype(zp_un, mlx::mlx_dtype__MLX_FLOAT32)?;
        let zp_full = broadcast_blocks(ctx, zpf, bs, nblocks, block)?;
        sub(ctx, qf, zp_full)?
    } else {
        let eight = f32(ctx, 8.0);
        sub(ctx, qf, eight)?
    };

    let sc_full = broadcast_blocks(ctx, sg, bs, nblocks, block)?;
    let w = mul(ctx, centered, sc_full)?;

    let mut oshape: Vec<i32> = ish.clone();
    oshape.push(d);
    let out = ctx.reshape(w, &oshape)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- QuantizeLinear -----------------------------------------------------------------------------

fn quantize_linear_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let x = ctx.astype(x0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let s0 = ctx.resolve(&n.inputs[1])?;
    let mut scale = ctx.astype(s0, mlx::mlx_dtype__MLX_FLOAT32)?;

    let rank = ctx.shape_of(x).len();
    let axis = norm_axis(n, rank);
    scale = align_per_axis(ctx, scale, rank, axis)?;

    let d = div(ctx, x, scale)?;
    let mut q = round_e(ctx, d)?;
    if present(n, 2) {
        let z0 = ctx.resolve(&n.inputs[2])?;
        let zp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let zp = align_per_axis(ctx, zp, rank, axis)?;
        q = add(ctx, q, zp)?;
    }
    let (lo, hi, dt) = range_for(n.outputs[0].otype).ok_or("QuantizeLinear: bad output dtype")?;
    let flo = f32(ctx, lo);
    let fhi = f32(ctx, hi);
    let q = clip(ctx, q, flo, fhi)?;
    let out = ctx.astype(q, dt)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- DequantizeLinear ---------------------------------------------------------------------------

fn dequantize_linear_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let mut x = ctx.astype(x0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let s0 = ctx.resolve(&n.inputs[1])?;
    let mut scale = ctx.astype(s0, mlx::mlx_dtype__MLX_FLOAT32)?;

    let rank = ctx.shape_of(x).len();
    let axis = norm_axis(n, rank);

    if present(n, 2) {
        let z0 = ctx.resolve(&n.inputs[2])?;
        let zp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let zp = align_per_axis(ctx, zp, rank, axis)?;
        x = sub(ctx, x, zp)?;
    }
    scale = align_per_axis(ctx, scale, rank, axis)?;
    let out = mul(ctx, x, scale)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- DynamicQuantizeLinear ----------------------------------------------------------------------

fn dynamic_quantize_linear_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let x = ctx.astype(x0, mlx::mlx_dtype__MLX_FLOAT32)?;

    let zero = f32(ctx, 0.0);
    let rmax = reduce_max(ctx, x)?;
    let xmax = ctx.binary(mlx::mlx_maximum, rmax, zero)?;
    let rmin = reduce_min(ctx, x)?;
    let xmin = ctx.binary(mlx::mlx_minimum, rmin, zero)?;
    let span = sub(ctx, xmax, xmin)?;
    let f255 = f32(ctx, 255.0);
    let scale = div(ctx, span, f255)?;

    let lo = f32(ctx, 0.0);
    let hi = f32(ctx, 255.0);
    let zero2 = f32(ctx, 0.0);
    let neg_min = sub(ctx, zero2, xmin)?;
    let zpq = div(ctx, neg_min, scale)?;
    let zpr = round_e(ctx, zpq)?;
    let zpf = clip(ctx, zpr, lo, hi)?;

    let xq = div(ctx, x, scale)?;
    let xr = round_e(ctx, xq)?;
    let yq = add(ctx, xr, zpf)?;
    let yc = clip(ctx, yq, lo, hi)?;

    let y = ctx.astype(yc, mlx::mlx_dtype__MLX_UINT8)?;
    ctx.bind(&n.outputs[0], y);
    ctx.bind(&n.outputs[1], scale);
    let zpu = ctx.astype(zpf, mlx::mlx_dtype__MLX_UINT8)?;
    ctx.bind(&n.outputs[2], zpu);
    Ok(())
}

// ---- MatMulInteger ------------------------------------------------------------------------------

fn matmul_integer_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a0 = ctx.resolve(&n.inputs[0])?;
    let mut a = ctx.astype(a0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let b0 = ctx.resolve(&n.inputs[1])?;
    let mut b = ctx.astype(b0, mlx::mlx_dtype__MLX_FLOAT32)?;

    if present(n, 2) {
        let z0 = ctx.resolve(&n.inputs[2])?;
        let mut azp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let sz = ctx.size_of(azp);
        if sz > 1 {
            azp = ctx.reshape(azp, &[sz as i32, 1])?;
        }
        a = sub(ctx, a, azp)?;
    }
    if present(n, 3) {
        let z0 = ctx.resolve(&n.inputs[3])?;
        let mut bzp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let sz = ctx.size_of(bzp);
        if sz > 1 {
            bzp = ctx.reshape(bzp, &[1, sz as i32])?;
        }
        b = sub(ctx, b, bzp)?;
    }

    let y = ctx.binary(mlx::mlx_matmul, a, b)?;
    let yr = round_e(ctx, y)?;
    let out = ctx.astype(yr, mlx::mlx_dtype__MLX_INT32)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- conv NCHW<->NHWC transforms (mlx convs are channels-last) ----------------------------------

fn to_channels_last(ctx: &mut TranslationContext, x: mlx::mlx_array, spatial_rank: usize) -> Result<mlx::mlx_array, MlxError> {
    let t = if spatial_rank == 1 {
        ctx.transpose(x, &[0, 2, 1])?
    } else {
        ctx.transpose(x, &[0, 2, 3, 1])?
    };
    ctx.contiguous(t)
}

fn from_channels_last(ctx: &mut TranslationContext, x: mlx::mlx_array, spatial_rank: usize) -> Result<mlx::mlx_array, MlxError> {
    let t = if spatial_rank == 1 {
        ctx.transpose(x, &[0, 2, 1])?
    } else {
        ctx.transpose(x, &[0, 3, 1, 2])?
    };
    ctx.contiguous(t)
}

fn weight_to_channels_last(ctx: &mut TranslationContext, w: mlx::mlx_array, spatial_rank: usize) -> Result<mlx::mlx_array, MlxError> {
    let t = if spatial_rank == 1 {
        ctx.transpose(w, &[0, 2, 1])?
    } else {
        ctx.transpose(w, &[0, 2, 3, 1])?
    };
    ctx.contiguous(t)
}

fn attr_or(n: &NodeDesc, name: &str, size: usize, value: i64) -> Vec<i64> {
    match n.int_arrays.get(name) {
        Some(v) if !v.is_empty() => v.clone(),
        _ => vec![value; size],
    }
}

/// Centered fp32 conv of an integer x/w pair: subtract zero points, widen, transform to channels-last,
/// conv, transform back to NCHW. `x_zp` is per-tensor scalar; `w_zp` is scalar or per-output-channel.
#[allow(clippy::too_many_arguments)]
fn centered_conv(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    x: mlx::mlx_array,
    w: mlx::mlx_array,
    has_x_zp: bool,
    x_zp: mlx::mlx_array,
    has_w_zp: bool,
    w_zp: mlx::mlx_array,
    spatial_rank: usize,
) -> Result<mlx::mlx_array, MlxError> {
    let mut x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    let mut w = ctx.astype(w, mlx::mlx_dtype__MLX_FLOAT32)?;
    if has_x_zp {
        let xz = ctx.astype(x_zp, mlx::mlx_dtype__MLX_FLOAT32)?;
        x = sub(ctx, x, xz)?;
    }
    if has_w_zp {
        let wrank = ctx.shape_of(w).len();
        let wz = ctx.astype(w_zp, mlx::mlx_dtype__MLX_FLOAT32)?;
        let wz = align_per_axis(ctx, wz, wrank, 0)?; // per-output-channel = weight axis 0
        w = sub(ctx, w, wz)?;
    }

    let strides = attr_or(n, "strides", spatial_rank, 1);
    let pads = attr_or(n, "pads", 2 * spatial_rank, 0);
    let dilations = attr_or(n, "dilations", spatial_rank, 1);
    let group = n.ints.get("group").copied().unwrap_or(1) as i32;

    let xl = to_channels_last(ctx, x, spatial_rank)?;
    let wl = weight_to_channels_last(ctx, w, spatial_rank)?;
    let out = if spatial_rank == 1 {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_conv1d(
                res,
                xl,
                wl,
                strides[0] as i32,
                pads[0] as i32,
                dilations[0] as i32,
                group,
                s,
            )
        })?
    } else {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_conv2d(
                res,
                xl,
                wl,
                strides[0] as i32,
                strides[1] as i32,
                pads[0] as i32,
                pads[1] as i32,
                dilations[0] as i32,
                dilations[1] as i32,
                group,
                s,
            )
        })?
    };
    from_channels_last(ctx, out, spatial_rank)
}

// ---- ConvInteger --------------------------------------------------------------------------------

fn conv_integer_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let w = ctx.resolve(&n.inputs[1])?;
    let spatial_rank = ctx.shape_of(x).len() - 2;

    let has_x_zp = present(n, 2);
    let has_w_zp = present(n, 3);
    let x_zp = if has_x_zp { ctx.resolve(&n.inputs[2])? } else { x };
    let w_zp = if has_w_zp { ctx.resolve(&n.inputs[3])? } else { w };

    let out = centered_conv(ctx, n, x, w, has_x_zp, x_zp, has_w_zp, w_zp, spatial_rank)?;
    let r = round_e(ctx, out)?;
    let out = ctx.astype(r, mlx::mlx_dtype__MLX_INT32)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- QLinearMatMul ------------------------------------------------------------------------------
//
// The dequant->matmul->requantize intermediate stays fp32 on purpose. Although the final output is
// requantized to int8/uint8 (so an fp16 intermediate looked near-free), the matmul here accumulates
// the UN-scaled centered integers: each product is O(2^8 * 2^8) and summed over K, the accumulator
// routinely exceeds the fp16 max (65504) — even at K=16 full-range int8 inputs overflow to inf,
// which then corrupts the requantized output. Measured: fp16 here is unsafe, so we keep fp32.

fn qlinear_matmul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a0 = ctx.resolve(&n.inputs[0])?;
    let mut a = ctx.astype(a0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let asc0 = ctx.resolve(&n.inputs[1])?;
    let mut a_scale = ctx.astype(asc0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let b0 = ctx.resolve(&n.inputs[3])?;
    let mut b = ctx.astype(b0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let bsc0 = ctx.resolve(&n.inputs[4])?;
    let mut b_scale = ctx.astype(bsc0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let ysc0 = ctx.resolve(&n.inputs[6])?;
    let mut y_scale = ctx.astype(ysc0, mlx::mlx_dtype__MLX_FLOAT32)?;

    if present(n, 2) {
        let z0 = ctx.resolve(&n.inputs[2])?;
        let mut azp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let sz = ctx.size_of(azp);
        if sz > 1 {
            azp = ctx.reshape(azp, &[sz as i32, 1])?;
        }
        a = sub(ctx, a, azp)?;
    }
    if present(n, 5) {
        let z0 = ctx.resolve(&n.inputs[5])?;
        let mut bzp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let sz = ctx.size_of(bzp);
        if sz > 1 {
            bzp = ctx.reshape(bzp, &[1, sz as i32])?;
        }
        b = sub(ctx, b, bzp)?;
    }

    let acc = ctx.binary(mlx::mlx_matmul, a, b)?;

    let sz = ctx.size_of(a_scale);
    if sz > 1 {
        a_scale = ctx.reshape(a_scale, &[sz as i32, 1])?;
    }
    let sz = ctx.size_of(b_scale);
    if sz > 1 {
        b_scale = ctx.reshape(b_scale, &[1, sz as i32])?;
    }
    let sz = ctx.size_of(y_scale);
    if sz > 1 {
        y_scale = ctx.reshape(y_scale, &[1, sz as i32])?;
    }

    let m1 = mul(ctx, acc, a_scale)?;
    let m2 = mul(ctx, m1, b_scale)?;
    let scaled = div(ctx, m2, y_scale)?;
    let mut q = round_e(ctx, scaled)?;
    if present(n, 7) {
        let z0 = ctx.resolve(&n.inputs[7])?;
        let mut yzp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let sz = ctx.size_of(yzp);
        if sz > 1 {
            yzp = ctx.reshape(yzp, &[1, sz as i32])?;
        }
        q = add(ctx, q, yzp)?;
    }

    let (lo, hi, dt) = range_for(n.outputs[0].otype).ok_or("QLinearMatMul: bad output dtype")?;
    let flo = f32(ctx, lo);
    let fhi = f32(ctx, hi);
    let q = clip(ctx, q, flo, fhi)?;
    let out = ctx.astype(q, dt)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- QLinearConv --------------------------------------------------------------------------------
//
// Like QLinearMatMul, the conv intermediate stays fp32: the un-scaled centered-integer accumulation
// over the receptive field overflows the fp16 range (see the QLinearMatMul note), so fp16 is unsafe
// despite the int8/uint8 requantized output.

fn qlinear_conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let xsc0 = ctx.resolve(&n.inputs[1])?;
    let x_scale = ctx.astype(xsc0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let w = ctx.resolve(&n.inputs[3])?;
    let wsc0 = ctx.resolve(&n.inputs[4])?;
    let w_scale = ctx.astype(wsc0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let ysc0 = ctx.resolve(&n.inputs[6])?;
    let y_scale = ctx.astype(ysc0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let spatial_rank = ctx.shape_of(x).len() - 2;

    let has_x_zp = present(n, 2);
    let has_w_zp = present(n, 5);
    let x_zp = if has_x_zp { ctx.resolve(&n.inputs[2])? } else { x };
    let w_zp = if has_w_zp { ctx.resolve(&n.inputs[5])? } else { w };

    let mut acc = centered_conv(ctx, n, x, w, has_x_zp, x_zp, has_w_zp, w_zp, spatial_rank)?;
    let out_rank = ctx.shape_of(acc).len();

    if present(n, 8) {
        let b0 = ctx.resolve(&n.inputs[8])?;
        let bias = ctx.astype(b0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let bias = align_per_axis(ctx, bias, out_rank, 1)?;
        acc = add(ctx, acc, bias)?;
    }

    let xw = mul(ctx, x_scale, w_scale)?;
    let mult = div(ctx, xw, y_scale)?;
    let mult = align_per_axis(ctx, mult, out_rank, 1)?;
    let scaled = mul(ctx, acc, mult)?;
    let mut q = round_e(ctx, scaled)?;
    if present(n, 7) {
        let z0 = ctx.resolve(&n.inputs[7])?;
        let yzp = ctx.astype(z0, mlx::mlx_dtype__MLX_FLOAT32)?;
        let yzp = align_per_axis(ctx, yzp, out_rank, 1)?;
        q = add(ctx, q, yzp)?;
    }

    let (lo, hi, dt) = range_for(n.outputs[0].otype).ok_or("QLinearConv: bad output dtype")?;
    let flo = f32(ctx, lo);
    let fhi = f32(ctx, hi);
    let q = clip(ctx, q, flo, fhi)?;
    let out = ctx.astype(q, dt)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim predicates ---------------------------------------------------------------------------

const MAX_EXACT_ACCUM: i64 = 256;

fn is_uint8(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
}
fn is_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
}
fn is_int8or(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
}
fn is_quant_output(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
}
fn is_dequant_input(t: ort::ONNXTensorElementDataType) -> bool {
    is_quant_output(t) || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
}

/// A zero-point / scale parameter is claimable when absent, or present with dtype `want` and shape
/// scalar or 1-D of length `axis_len` (per-axis). `axis_len < 0` disables per-axis (scalar only).
fn param_ok(node: &NodeView, i: usize, want: ort::ONNXTensorElementDataType, axis_len: i64) -> bool {
    if !node.input_present(i) {
        return true;
    }
    let info = match node.input_info(i) {
        Some(x) => x,
        None => return false,
    };
    if info.dtype != want {
        return false;
    }
    if info.shape.is_empty() || (info.shape.len() == 1 && info.shape[0] == 1) {
        return true;
    }
    axis_len >= 0 && info.shape.len() == 1 && info.shape[0] == axis_len
}

fn static_positive(shape: &[i64]) -> bool {
    !shape.is_empty() && shape.iter().all(|&d| d > 0)
}

fn matmulnbits_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(node.num_outputs() >= 1, "requires at least one output");
    let out = match node.output_info(0) {
        Some(o) => o,
        None => deny!("missing output type/shape info"),
    };
    require!(is_float(out.dtype), "output must be float32");
    let (a, b, s) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(a), Some(b), Some(s)) => (a, b, s),
        _ => deny!("missing tensor type/shape info on an input"),
    };
    require!(is_float(a.dtype) && is_uint8(b.dtype) && is_float(s.dtype),
        "activation/scales must be float32 and packed weights must be uint8");
    // Only the 3-input symmetric or 4-input asymmetric (uint8 packed int4 zero_points) forms.
    if node.input_present(3) {
        match node.input_info(3) {
            Some(zp) if is_uint8(zp.dtype) => {}
            _ => deny!("zero_points must be uint8 when present"),
        }
    }
    // Reject g_idx / bias (any present input beyond slot 3) — left to ORT CPU.
    for i in 4..nin {
        if node.input_present(i) {
            deny!("g_idx and bias inputs are not supported");
        }
    }
    let bits = node.int_attr("bits", 4);
    let block = node.int_attr("block_size", 32);
    require!(bits == 4, "only 4-bit weights are supported");
    require!(matches!(block, 16 | 32 | 64 | 128), "block_size must be 16, 32, 64, or 128 (got {block})");
    Ok(())
}

fn gather_block_quantized_claim(node: &NodeView) -> ClaimResult {
    require!(node.num_outputs() >= 1, "requires at least one output");
    let out = match node.output_info(0) {
        Some(o) => o,
        None => deny!("missing output type/shape info"),
    };
    let (data, idx, scales) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(d), Some(i), Some(s)) => (d, i, s),
        _ => deny!("missing tensor type/shape info on an input"),
    };
    if !is_uint8(data.dtype)
        || !is_int_index(idx.dtype)
        || scales.dtype != out.dtype
        || !is_mlx_float(scales.dtype)
    {
        deny!("data must be uint8, indices int32/int64, and scales/output the same float dtype");
    }
    if node.input_present(3) {
        match node.input_info(3) {
            Some(zp) if is_uint8(zp.dtype) => {}
            _ => deny!("zero_points must be uint8 when present"),
        }
    }
    require!(node.int_attr("bits", 4) == 4, "only 4-bit data is supported");
    require!(node.int_attr("gather_axis", 0) == 0 && node.int_attr("quantize_axis", 1) == 1,
        "only gather_axis=0 and quantize_axis=1 are supported");
    require!(node.int_attr("block_size", 128) >= 16, "block_size must be at least 16");
    Ok(())
}

fn quantize_linear_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(nin >= 2 && nin <= 3 && node.num_outputs() >= 1,
        "expects 2 or 3 inputs and at least one output");
    let (x, s, o) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(s), Some(o)) => (x, s, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_float(x.dtype) || !is_float(s.dtype) {
        deny!("input and scale dtypes must be float32; got {} and {}",
            crate::registry::ort_dtype_name(x.dtype), crate::registry::ort_dtype_name(s.dtype));
    }
    if s.shape.len() > 1 || !is_quant_output(o.dtype) {
        deny!("scale must be scalar or rank-1 and output dtype {} must be int8, uint8, int16, or uint16",
            crate::registry::ort_dtype_name(o.dtype));
    }
    if node.input_present(2) {
        match node.input_info(2) {
            Some(z) if z.dtype == o.dtype && z.shape.len() <= 1 => {}
            _ => deny!("zero_point must match output dtype {} and be scalar or rank-1",
                crate::registry::ort_dtype_name(o.dtype)),
        }
    }
    Ok(())
}

fn dequantize_linear_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(nin >= 2 && nin <= 3 && node.num_outputs() >= 1,
        "expects 2 or 3 inputs and at least one output");
    let (x, s, o) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(s), Some(o)) => (x, s, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_dequant_input(x.dtype) || !is_float(s.dtype) || !is_float(o.dtype) {
        deny!("input dtype {} must be int8, uint8, int16, uint16, or int32; scale/output must be float32 (got {} -> {})",
            crate::registry::ort_dtype_name(x.dtype), crate::registry::ort_dtype_name(s.dtype),
            crate::registry::ort_dtype_name(o.dtype));
    }
    if s.shape.len() > 1 {
        deny!("scale must be scalar or rank-1");
    }
    if node.input_present(2) {
        match node.input_info(2) {
            Some(z) if z.dtype == x.dtype && z.shape.len() <= 1 => {}
            _ => deny!("zero_point must match input dtype {} and be scalar or rank-1",
                crate::registry::ort_dtype_name(x.dtype)),
        }
    }
    Ok(())
}

fn dynamic_quantize_linear_claim(node: &NodeView) -> ClaimResult {
    require!(node.num_inputs() == 1 && node.num_outputs() == 3,
        "expects 1 input and 3 outputs, got {}in/{}out", node.num_inputs(), node.num_outputs());
    let (x, y, sc, z) = match (
        node.input_info(0),
        node.output_info(0),
        node.output_info(1),
        node.output_info(2),
    ) {
        (Some(x), Some(y), Some(sc), Some(z)) => (x, y, sc, z),
        _ => deny!("missing tensor type/shape info on an input or output"),
    };
    require!(is_float(x.dtype) && is_uint8(y.dtype) && is_float(sc.dtype) && is_uint8(z.dtype),
        "input and scale must be float32; quantized output and zero point must be uint8");
    Ok(())
}

fn matmul_integer_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(nin >= 2 && nin <= 4 && node.num_outputs() >= 1,
        "expects 2 to 4 inputs and at least one output");
    let (a, b, o) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_int8or(a.dtype)
        || !is_int8or(b.dtype)
        || o.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    {
        deny!("input dtypes {} and {} must be int8 or uint8, and output dtype {} must be int32",
            crate::registry::ort_dtype_name(a.dtype), crate::registry::ort_dtype_name(b.dtype),
            crate::registry::ort_dtype_name(o.dtype));
    }
    if a.shape.len() != 2 || b.shape.len() != 2 {
        deny!("inputs must both be rank-2 matrices");
    }
    let k = a.shape[1];
    if k <= 0 || k != b.shape[0] || k > MAX_EXACT_ACCUM {
        deny!("inner dimensions must match and K must be 1..={MAX_EXACT_ACCUM}");
    }
    if node.input_present(2) {
        match node.input_info(2) {
            Some(z) if z.dtype == a.dtype && z.shape.len() <= 1 => {}
            _ => deny!("a_zero_point must match A dtype {} and be scalar or rank-1",
                crate::registry::ort_dtype_name(a.dtype)),
        }
    }
    if node.input_present(3) {
        match node.input_info(3) {
            Some(z) if z.dtype == b.dtype && z.shape.len() <= 1 => {}
            _ => deny!("b_zero_point must match B dtype {} and be scalar or rank-1",
                crate::registry::ort_dtype_name(b.dtype)),
        }
    }
    Ok(())
}

fn conv_attrs_ok(node: &NodeView, spatial_rank: usize, w_shape: &[i64], channels: i64, group: i64) -> bool {
    if node.string_attr("auto_pad", "NOTSET") != "NOTSET" {
        return false;
    }
    if group <= 0 || channels % group != 0 || w_shape[1] != channels / group {
        return false;
    }
    if w_shape[0] % group != 0 {
        return false;
    }
    let get = |name: &str, def_len: usize, def: i64| -> Option<Vec<i64>> {
        let (present, v) = node.ints_attr(name);
        if !present {
            Some(vec![def; def_len])
        } else if v.len() == def_len {
            Some(v)
        } else {
            None
        }
    };
    let strides = match get("strides", spatial_rank, 1) {
        Some(v) => v,
        None => return false,
    };
    let dilations = match get("dilations", spatial_rank, 1) {
        Some(v) => v,
        None => return false,
    };
    let pads = match get("pads", 2 * spatial_rank, 0) {
        Some(v) => v,
        None => return false,
    };
    if strides.iter().any(|&v| v <= 0) || dilations.iter().any(|&v| v <= 0) || pads.iter().any(|&v| v < 0) {
        return false;
    }
    for i in 0..spatial_rank {
        if pads[i] != pads[i + spatial_rank] {
            return false; // mlx conv takes a symmetric pad only
        }
    }
    let (kpresent, kernel) = node.ints_attr("kernel_shape");
    if kpresent {
        if kernel.len() != spatial_rank {
            return false;
        }
        for i in 0..spatial_rank {
            if kernel[i] != w_shape[i + 2] {
                return false;
            }
        }
    }
    true
}

fn conv_accum_exact(w_shape: &[i64]) -> bool {
    let mut n_acc = w_shape[1];
    for &d in &w_shape[2..] {
        n_acc *= d;
    }
    n_acc >= 1 && n_acc <= MAX_EXACT_ACCUM
}

fn conv_integer_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(nin >= 2 && nin <= 4 && node.num_outputs() == 1,
        "expects 2 to 4 inputs and 1 output");
    let (x, w, o) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(w), Some(o)) => (x, w, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_int8or(x.dtype)
        || !is_int8or(w.dtype)
        || o.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    {
        deny!("input dtypes {} and {} must be int8 or uint8, and output dtype {} must be int32",
            crate::registry::ort_dtype_name(x.dtype), crate::registry::ort_dtype_name(w.dtype),
            crate::registry::ort_dtype_name(o.dtype));
    }
    if (x.shape.len() != 3 && x.shape.len() != 4) || w.shape.len() != x.shape.len() {
        deny!("input/weight must have matching rank, with input rank 3 or 4");
    }
    if !static_positive(&x.shape) || !static_positive(&w.shape) {
        deny!("input and weight shapes must be static and positive");
    }
    let spatial_rank = x.shape.len() - 2;
    let group = node.int_attr("group", 1);
    require!(conv_attrs_ok(node, spatial_rank, &w.shape, x.shape[1], group),
        "convolution attributes, channels, group, or kernel shape are unsupported");
    require!(conv_accum_exact(&w.shape), "convolution accumulation size must be 1..={MAX_EXACT_ACCUM}");
    require!(param_ok(node, 2, x.dtype, -1),
        "x_zero_point must match input dtype {} and be scalar", crate::registry::ort_dtype_name(x.dtype));
    require!(param_ok(node, 3, w.dtype, w.shape[0]),
        "w_zero_point must match weight dtype {} and be scalar or per-output-channel",
        crate::registry::ort_dtype_name(w.dtype));
    Ok(())
}

fn qlinear_matmul_claim(node: &NodeView) -> ClaimResult {
    require!(node.num_inputs() == 8 && node.num_outputs() == 1,
        "expects 8 inputs and 1 output, got {}in/{}out", node.num_inputs(), node.num_outputs());
    let (a, b, o) = match (node.input_info(0), node.input_info(3), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_int8or(a.dtype) || !is_int8or(b.dtype) || !is_int8or(o.dtype) {
        deny!("A, B, and output must each be int8 or uint8");
    }
    if a.shape.len() != 2 || b.shape.len() != 2 {
        deny!("A and B must be rank-2 matrices");
    }
    let (m, k, big_n) = (a.shape[0], a.shape[1], b.shape[1]);
    if k <= 0 || k != b.shape[0] || k > MAX_EXACT_ACCUM || m <= 0 || big_n <= 0 {
        deny!("matrix dimensions must be positive, inner dimensions match, and K must be 1..={MAX_EXACT_ACCUM}");
    }
    let f = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    require!(param_ok(node, 1, f, m), "a_scale must be float32 and scalar or length M");
    require!(param_ok(node, 2, a.dtype, m), "a_zero_point must match A dtype and be scalar or length M");
    require!(param_ok(node, 4, f, big_n), "b_scale must be float32 and scalar or length N");
    require!(param_ok(node, 5, b.dtype, big_n), "b_zero_point must match B dtype and be scalar or length N");
    require!(param_ok(node, 6, f, big_n), "y_scale must be float32 and scalar or length N");
    require!(param_ok(node, 7, o.dtype, big_n), "y_zero_point must match output dtype and be scalar or length N");
    Ok(())
}

fn qlinear_conv_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(nin >= 8 && nin <= 9 && node.num_outputs() == 1,
        "expects 8 or 9 inputs and 1 output");
    let (x, w, o) = match (node.input_info(0), node.input_info(3), node.output_info(0)) {
        (Some(x), Some(w), Some(o)) => (x, w, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    if !is_int8or(x.dtype) || !is_int8or(w.dtype) || !is_int8or(o.dtype) {
        deny!("input, weight, and output must each be int8 or uint8");
    }
    if (x.shape.len() != 3 && x.shape.len() != 4) || w.shape.len() != x.shape.len() {
        deny!("input/weight must have matching rank, with input rank 3 or 4");
    }
    if !static_positive(&x.shape) || !static_positive(&w.shape) {
        deny!("input and weight shapes must be static and positive");
    }
    let spatial_rank = x.shape.len() - 2;
    let big_m = w.shape[0];
    let group = node.int_attr("group", 1);
    require!(conv_attrs_ok(node, spatial_rank, &w.shape, x.shape[1], group),
        "convolution attributes, channels, group, or kernel shape are unsupported");
    require!(conv_accum_exact(&w.shape), "convolution accumulation size must be 1..={MAX_EXACT_ACCUM}");
    let f = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    require!(param_ok(node, 1, f, -1), "x_scale must be a scalar float32");
    require!(param_ok(node, 2, x.dtype, -1), "x_zero_point must match input dtype and be scalar");
    require!(param_ok(node, 4, f, big_m), "w_scale must be float32 and scalar or per-output-channel");
    require!(param_ok(node, 5, w.dtype, big_m), "w_zero_point must match weight dtype and be scalar or per-output-channel");
    require!(param_ok(node, 6, f, -1), "y_scale must be a scalar float32");
    require!(param_ok(node, 7, o.dtype, -1), "y_zero_point must match output dtype and be scalar");
    if node.input_present(8) {
        match node.input_info(8) {
            Some(bi)
                if bi.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                    && bi.shape.len() == 1
                    && bi.shape[0] == big_m => {}
            _ => deny!("bias must be int32 with one element per output channel"),
        }
    }
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn reg(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain,
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    reg(registry, "com.microsoft", "MatMulNBits", matmulnbits_op, matmulnbits_claim);
    reg(
        registry,
        "com.microsoft",
        "GatherBlockQuantized",
        gather_block_quantized_op,
        gather_block_quantized_claim,
    );
    reg(registry, "", "QuantizeLinear", quantize_linear_op, quantize_linear_claim);
    reg(registry, "", "DequantizeLinear", dequantize_linear_op, dequantize_linear_claim);
    reg(
        registry,
        "",
        "DynamicQuantizeLinear",
        dynamic_quantize_linear_op,
        dynamic_quantize_linear_claim,
    );
    reg(registry, "", "MatMulInteger", matmul_integer_op, matmul_integer_claim);
    reg(registry, "", "ConvInteger", conv_integer_op, conv_integer_claim);
    reg(registry, "", "QLinearMatMul", qlinear_matmul_op, qlinear_matmul_claim);
    reg(registry, "", "QLinearConv", qlinear_conv_op, qlinear_conv_claim);
}
