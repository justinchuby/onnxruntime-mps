//! Vision / spatial-transform op handlers. Faithful port of the C++ `ops/vision2.cc`
//! (GridSample, AffineGrid, Col2Im) and `ops/vision2b.cc` (RoiAlign, MaxRoiPool, MaxUnpool).
//!
//! Every op reduces to the same primitives the shape handlers already use — host-computed
//! gather/scatter index maps + `take`/`take_along_axis`/`scatter_add` + on-device coordinate
//! arithmetic — so they translate exactly to MLX with no bespoke kernel. Only the static, float
//! forms the C++ claim accepts are claimed; exotic forms (cubic GridSample, 5-D volumetric,
//! adaptive RoiAlign `sampling_ratio==0`, the 3-input MaxUnpool, non-float payloads) are left to CPU.

use std::os::raw::c_void;

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, TensorRef, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    is_int_index, is_mlx_float, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- translate-time helpers ---------------------------------------------------------------------

fn astype(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    t: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.astype(a, t)
}
fn reshape(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    shape: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    ctx.reshape(a, shape)
}
fn contiguous(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.contiguous(a)
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
fn mul(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_multiply, a, b)
}
fn div(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_divide, a, b)
}
fn maximum(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_maximum, a, b)
}
fn floor(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_floor, a)
}
fn ceil(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_ceil, a)
}
fn round0(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_round(res, a, 0, s) })
}
fn sign(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_sign, a)
}
fn abs_(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_abs, a)
}

fn scalar_f(ctx: &mut TranslationContext, v: f32) -> mlx::mlx_array {
    ctx.scalar_f32(v)
}
fn scalar_i(ctx: &mut TranslationContext, v: i32) -> mlx::mlx_array {
    ctx.scalar_i32(v)
}

fn host_f32(ctx: &mut TranslationContext, data: &[f32], shape: &[i32]) -> mlx::mlx_array {
    ctx.keep(Array::from_data(
        data.as_ptr() as *const c_void,
        shape,
        mlx::mlx_dtype__MLX_FLOAT32,
    ))
}
fn host_i32(ctx: &mut TranslationContext, data: &[i32], shape: &[i32]) -> mlx::mlx_array {
    ctx.keep(Array::from_data(
        data.as_ptr() as *const c_void,
        shape,
        mlx::mlx_dtype__MLX_INT32,
    ))
}

/// Clamp `a` to [lo, hi] (scalar bounds).
fn clip(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    lo: f32,
    hi: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let los = scalar_f(ctx, lo);
    let his = scalar_f(ctx, hi);
    ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, a, los, his, s) })
}

/// Round half away from zero: sign(a)*floor(|a|+0.5) (matches std::round the ORT CPU kernels use).
fn round_away(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    let sgn = sign(ctx, a)?;
    let mag = abs_(ctx, a)?;
    let half = scalar_f(ctx, 0.5);
    let shifted = add(ctx, mag, half)?;
    let fl = floor(ctx, shifted)?;
    mul(ctx, sgn, fl)
}

/// 1.0 where lo <= a <= hi else 0.0 (float32) — the zeros-padding validity mask for a coordinate.
fn in_range_mask(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    lo: f32,
    hi: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let los = scalar_f(ctx, lo);
    let his = scalar_f(ctx, hi);
    let ge = ctx.binary(mlx::mlx_greater_equal, a, los)?;
    let le = ctx.binary(mlx::mlx_less_equal, a, his)?;
    let both = ctx.binary(mlx::mlx_logical_and, ge, le)?;
    astype(ctx, both, mlx::mlx_dtype__MLX_FLOAT32)
}

fn broadcast_to(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    shape: &[i32],
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_broadcast_to(res, a, shape.as_ptr(), shape.len(), s) })
}
fn take_along_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    idx: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_take_along_axis(res, a, idx, axis, s) })
}
fn take_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    idx: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, a, idx, axis, s) })
}
fn matmul(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_matmul(res, a, b, s) })
}
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
fn max_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_max_axis(res, a, axis, false, s) })
}
fn sum_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, a, axis, false, s) })
}
fn where_(
    ctx: &mut TranslationContext,
    c: mlx::mlx_array,
    x: mlx::mlx_array,
    y: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_where(res, c, x, y, s) })
}
fn arange(ctx: &mut TranslationContext, nval: i32) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe {
        mlx::mlx_arange(res, 0.0, nval as f64, 1.0, mlx::mlx_dtype__MLX_FLOAT32, s)
    })
}

/// `mlx_scatter_add(a, [idx], updates, axes={0})` — RAII vector-array wrapper.
fn scatter_add(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    idx: mlx::mlx_array,
    updates: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let mut vec = VectorArray::new();
    vec.append(idx);
    let vraw = vec.as_raw();
    let axes0 = [0i32];
    ctx.emit(|res, s| unsafe { mlx::mlx_scatter_add(res, a, vraw, updates, axes0.as_ptr(), 1, s) })
}

fn zeros_1d(ctx: &mut TranslationContext, n: i32) -> Result<mlx::mlx_array, MlxError> {
    let shape = [n];
    ctx.emit(|res, s| unsafe {
        mlx::mlx_zeros(res, shape.as_ptr(), 1, mlx::mlx_dtype__MLX_FLOAT32, s)
    })
}

/// Read a constant int64/int32 parameter INPUT (image_shape / block_shape / size) at translate time.
fn read_const_ints(ctx: &TranslationContext, r: &TensorRef) -> Result<Vec<i64>, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() {
        return Ok(Vec::new());
    }
    if h.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
        let p = h.data as *const i64;
        Ok(unsafe { std::slice::from_raw_parts(p, h.count) }.to_vec())
    } else {
        let p = h.data as *const i32;
        Ok(unsafe { std::slice::from_raw_parts(p, h.count) }
            .iter()
            .map(|&v| v as i64)
            .collect())
    }
}

fn out_dtype(n: &NodeDesc) -> mlx::mlx_dtype {
    mlx_dtype_from_onnx(n.outputs[0].otype)
}

// =================================================================================================
// GridSample (2D)
// =================================================================================================

#[allow(clippy::too_many_arguments)]
fn gs_sample(
    ctx: &mut TranslationContext,
    xf: mlx::mlx_array,
    xf_coord: mlx::mlx_array,
    yf_coord: mlx::mlx_array,
    w: mlx::mlx_array,
    zeros_pad: bool,
    dims: (i32, i32, i32, i32, i32), // N, C, P, W, H
) -> Result<mlx::mlx_array, MlxError> {
    let (n, c, p, w_dim, h_dim) = dims;
    let mut weight = w;
    if zeros_pad {
        let mx = in_range_mask(ctx, xf_coord, 0.0, w_dim as f32 - 1.0)?;
        weight = mul(ctx, weight, mx)?;
        let my = in_range_mask(ctx, yf_coord, 0.0, h_dim as f32 - 1.0)?;
        weight = mul(ctx, weight, my)?;
    }
    let xc = clip(ctx, xf_coord, 0.0, w_dim as f32 - 1.0)?;
    let xi = astype(ctx, xc, mlx::mlx_dtype__MLX_INT32)?;
    let yc = clip(ctx, yf_coord, 0.0, h_dim as f32 - 1.0)?;
    let yi = astype(ctx, yc, mlx::mlx_dtype__MLX_INT32)?;
    let wsc = scalar_i(ctx, w_dim);
    let ymul = mul(ctx, yi, wsc)?;
    let flat = add(ctx, ymul, xi)?; // [N,P] int32
    let flat3 = reshape(ctx, flat, &[n, 1, p])?;
    let idx = broadcast_to(ctx, flat3, &[n, c, p])?;
    let g = take_along_axis(ctx, xf, idx, 2)?; // [N,C,P]
    let w3 = reshape(ctx, weight, &[n, 1, p])?;
    mul(ctx, g, w3)
}

fn grid_sample_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let x = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(x);
    let (nn, cc, hh, ww) = (xs[0], xs[1], xs[2], xs[3]);

    let g0 = ctx.resolve(&n.inputs[1])?;
    let grid = astype(ctx, g0, mlx::mlx_dtype__MLX_FLOAT32)?;
    let gs = ctx.shape_of(grid);
    let (hout, wout) = (gs[1], gs[2]);
    let p = hout * wout;

    let mode = n
        .strings
        .get("mode")
        .map(String::as_str)
        .unwrap_or("linear");
    let padding = n
        .strings
        .get("padding_mode")
        .map(String::as_str)
        .unwrap_or("zeros");
    let align = n.ints.get("align_corners").copied().unwrap_or(0) != 0;
    let nearest = mode == "nearest";
    let zeros = padding == "zeros";

    let ax = if align {
        (ww - 1) as f32 / 2.0
    } else {
        ww as f32 / 2.0
    };
    let bx = (ww - 1) as f32 / 2.0;
    let ay = if align {
        (hh - 1) as f32 / 2.0
    } else {
        hh as f32 / 2.0
    };
    let by = (hh - 1) as f32 / 2.0;

    let xf = reshape(ctx, x, &[nn, cc, hh * ww])?;
    let gflat = reshape(ctx, grid, &[nn, p, 2])?;
    let gx_s = slice(ctx, gflat, &[0, 0, 0], &[nn, p, 1])?;
    let gx = reshape(ctx, gx_s, &[nn, p])?;
    let gy_s = slice(ctx, gflat, &[0, 0, 1], &[nn, p, 2])?;
    let gy = reshape(ctx, gy_s, &[nn, p])?;

    let axs = scalar_f(ctx, ax);
    let gxa = mul(ctx, gx, axs)?;
    let bxs = scalar_f(ctx, bx);
    let ix = add(ctx, gxa, bxs)?; // [N,P]
    let ays = scalar_f(ctx, ay);
    let gya = mul(ctx, gy, ays)?;
    let bys = scalar_f(ctx, by);
    let iy = add(ctx, gya, bys)?; // [N,P]

    let dims = (nn, cc, p, ww, hh);
    let acc = if nearest {
        let one = scalar_f(ctx, 1.0);
        let w1 = broadcast_to(ctx, one, &[nn, p])?;
        let rx = round0(ctx, ix)?;
        let ry = round0(ctx, iy)?;
        gs_sample(ctx, xf, rx, ry, w1, zeros, dims)?
    } else {
        let x0f = floor(ctx, ix)?;
        let y0f = floor(ctx, iy)?;
        let one = scalar_f(ctx, 1.0);
        let x1f = add(ctx, x0f, one)?;
        let y1f = add(ctx, y0f, one)?;
        let wx1 = sub(ctx, ix, x0f)?;
        let one2 = scalar_f(ctx, 1.0);
        let wx0 = sub(ctx, one2, wx1)?;
        let wy1 = sub(ctx, iy, y0f)?;
        let one3 = scalar_f(ctx, 1.0);
        let wy0 = sub(ctx, one3, wy1)?;

        let w00 = mul(ctx, wx0, wy0)?;
        let mut acc = gs_sample(ctx, xf, x0f, y0f, w00, zeros, dims)?;
        let w10 = mul(ctx, wx1, wy0)?;
        let c10 = gs_sample(ctx, xf, x1f, y0f, w10, zeros, dims)?;
        acc = add(ctx, acc, c10)?;
        let w01 = mul(ctx, wx0, wy1)?;
        let c01 = gs_sample(ctx, xf, x0f, y1f, w01, zeros, dims)?;
        acc = add(ctx, acc, c01)?;
        let w11 = mul(ctx, wx1, wy1)?;
        let c11 = gs_sample(ctx, xf, x1f, y1f, w11, zeros, dims)?;
        add(ctx, acc, c11)?
    };

    let out = reshape(ctx, acc, &[nn, cc, hout, wout])?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn grid_sample_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, g, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(g), Some(o)) => (x, g, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype && is_mlx_float(g.dtype),
        "input/output must share one float dtype and grid must be float, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(g.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4 && g.shape.len() == 4 && g.shape.last() == Some(&2),
        "input must be rank 4 and grid must have shape [N,H,W,2], got {:?} and {:?}",
        x.shape,
        g.shape
    );
    require!(
        x.shape.iter().all(|&d| d >= 0) && g.shape.iter().all(|&d| d >= 0),
        "input and grid shapes must be static, got {:?} and {:?}",
        x.shape,
        g.shape
    );
    let mode = node.string_attr("mode", "linear");
    require!(
        mode == "linear" || mode == "bilinear" || mode == "nearest",
        "mode must be \"linear\", \"bilinear\", or \"nearest\", got {mode:?}"
    );
    let padding = node.string_attr("padding_mode", "zeros");
    require!(
        padding == "zeros" || padding == "border",
        "padding_mode must be \"zeros\" or \"border\", got {padding:?}"
    );
    Ok(())
}

// =================================================================================================
// AffineGrid (2D)
// =================================================================================================

fn affine_grid_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let size = read_const_ints(ctx, &n.inputs[1])?; // (N, C, H, W)
    let nd = size[0] as i32;
    let hd = size[2] as i32;
    let wd = size[3] as i32;
    let align = n.ints.get("align_corners").copied().unwrap_or(0) != 0;

    let coord = |i: i32, l: i32| -> f32 {
        if align {
            if l <= 1 {
                -1.0
            } else {
                -1.0 + i as f32 * (2.0 / (l - 1) as f32)
            }
        } else {
            ((2.0 * i as f64 + 1.0) / l as f64 - 1.0) as f32
        }
    };
    let mut base = vec![0f32; (hd as usize) * (wd as usize) * 3];
    for h in 0..hd {
        let yh = coord(h, hd);
        for w in 0..wd {
            let r = ((h as usize * wd as usize) + w as usize) * 3;
            base[r] = coord(w, wd);
            base[r + 1] = yh;
            base[r + 2] = 1.0;
        }
    }
    let base_arr = host_f32(ctx, &base, &[1, hd * wd, 3]);

    let t0 = ctx.resolve(&n.inputs[0])?;
    let theta = astype(ctx, t0, mlx::mlx_dtype__MLX_FLOAT32)?; // [N,2,3]
    let theta_t = ctx.transpose(theta, &[0, 2, 1])?; // [N,3,2]
    let grid = matmul(ctx, base_arr, theta_t)?; // [N,H*W,2]
    let grid = reshape(ctx, grid, &[nd, hd, wd, 2])?;
    let grid = astype(ctx, grid, out_dtype(n))?;
    let y = contiguous(ctx, grid)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn affine_grid_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (t, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(t), Some(o)) => (t, o),
        _ => deny!("missing tensor type/shape info on theta or output"),
    };
    require!(
        is_mlx_float(t.dtype) && is_mlx_float(out.dtype),
        "theta and output must be float, got {} -> {}",
        crate::registry::ort_dtype_name(t.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        node.is_const_int_tensor(1),
        "size must be a constant integer initializer"
    );
    let size = match node.read_const_ints_any(1) {
        Some(v) => v,
        None => deny!("size must be a constant integer initializer"),
    };
    require!(
        size.len() == 4,
        "size must contain 4 dimensions, got {size:?}"
    );
    require!(
        t.shape.len() == 3 && t.shape[1] == 2 && t.shape[2] == 3 && t.shape.iter().all(|&d| d >= 0),
        "theta must have static shape [N,2,3], got {:?}",
        t.shape
    );
    require!(
        size.iter().all(|&d| d >= 1),
        "all size dimensions must be positive, got {size:?}"
    );
    Ok(())
}

// =================================================================================================
// Col2Im
// =================================================================================================

fn col2im_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let in0 = ctx.resolve(&n.inputs[0])?;
    let input = contiguous(ctx, in0)?;
    let is = ctx.shape_of(input);
    let nn = is[0];
    let l = is[2];

    let image_shape = read_const_ints(ctx, &n.inputs[1])?;
    let block_shape = read_const_ints(ctx, &n.inputs[2])?;
    let r = image_shape.len();

    let dil = n
        .int_arrays
        .get("dilations")
        .cloned()
        .unwrap_or_else(|| vec![1; r]);
    let stride = n
        .int_arrays
        .get("strides")
        .cloned()
        .unwrap_or_else(|| vec![1; r]);
    let pads = n
        .int_arrays
        .get("pads")
        .cloned()
        .unwrap_or_else(|| vec![0; 2 * r]);

    let mut kk: i64 = 1;
    for d in 0..r {
        kk *= block_shape[d];
    }
    let c = is[1] / kk as i32;
    let mut ss: i64 = 1;
    for d in 0..r {
        ss *= image_shape[d];
    }

    let mut npos = vec![0i64; r];
    let mut img_stride = vec![1i64; r];
    for d in 0..r {
        npos[d] = (image_shape[d] + pads[d] + pads[r + d] - dil[d] * (block_shape[d] - 1) - 1)
            / stride[d]
            + 1;
    }
    for d in (0..r.saturating_sub(1)).rev() {
        img_stride[d] = img_stride[d + 1] * image_shape[d + 1];
    }

    // Host-compute the (column, output-spatial) map, dropping out-of-range positions.
    let mut col_of: Vec<i32> = Vec::new();
    let mut spatial: Vec<i32> = Vec::new();
    let mut kc = vec![0i64; r];
    let mut lc = vec![0i64; r];
    for k in 0..kk {
        let mut rem = k;
        for d in (0..r).rev() {
            kc[d] = rem % block_shape[d];
            rem /= block_shape[d];
        }
        for lv in 0..(l as i64) {
            let mut r2 = lv;
            for d in (0..r).rev() {
                lc[d] = r2 % npos[d];
                r2 /= npos[d];
            }
            let mut pos: i64 = 0;
            let mut ok = true;
            for d in 0..r {
                let op = lc[d] * stride[d] + kc[d] * dil[d] - pads[d];
                if op < 0 || op >= image_shape[d] {
                    ok = false;
                    break;
                }
                pos += op * img_stride[d];
            }
            if ok {
                col_of.push((k * l as i64 + lv) as i32);
                spatial.push(pos as i32);
            }
        }
    }
    let m = col_of.len() as i32;

    let total = nn * c * ss as i32;
    let mut out_acc = zeros_1d(ctx, total)?;

    if m > 0 {
        let in_reshaped = reshape(ctx, input, &[nn, c, kk as i32 * l])?;
        let in_cols = astype(ctx, in_reshaped, mlx::mlx_dtype__MLX_FLOAT32)?;
        let col_idx = host_i32(ctx, &col_of, &[m]);
        let gathered = take_axis(ctx, in_cols, col_idx, 2)?; // [N,C,M]

        // Flat scatter index [N*C*M]: element (nn,c,j) -> (nn*C+c)*S + spatial[j].
        let mut scatter_idx = vec![0i32; (nn as i64 * c as i64 * m as i64) as usize];
        let mut w = 0usize;
        for nv in 0..nn {
            for cv in 0..c {
                let base = ((nv as i64 * c as i64 + cv as i64) * ss) as i32;
                for j in 0..(m as usize) {
                    scatter_idx[w] = base + spatial[j];
                    w += 1;
                }
            }
        }
        let ilen = scatter_idx.len() as i32;
        let idx = host_i32(ctx, &scatter_idx, &[ilen]);
        let updates = reshape(ctx, gathered, &[ilen, 1])?;
        out_acc = scatter_add(ctx, out_acc, idx, updates)?;
    }

    let mut out_shape = vec![nn, c];
    for d in 0..r {
        out_shape.push(image_shape[d] as i32);
    }
    let out = reshape(ctx, out_acc, &out_shape)?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn col2im_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "Col2Im expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input[0] or output"),
    };
    require!(
        is_mlx_float(i.dtype) && out.dtype == i.dtype,
        "Col2Im input/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        i.shape.len() == 3 && i.shape.iter().all(|&d| d >= 0),
        "Col2Im input[0] must be a static rank-3 [N,C*prod(block),L] tensor, got shape {:?}",
        i.shape
    );
    require!(
        node.is_const_int_tensor(1) && node.is_const_int_tensor(2),
        "Col2Im `image_shape` (input 1) and `block_shape` (input 2) must be constant integer \
         initializers; runtime values stay on CPU"
    );
    let image_shape = match node.read_const_ints_any(1) {
        Some(v) => v,
        None => deny!("Col2Im `image_shape` (input 1) is not a readable constant int tensor"),
    };
    let block_shape = match node.read_const_ints_any(2) {
        Some(v) => v,
        None => deny!("Col2Im `block_shape` (input 2) is not a readable constant int tensor"),
    };
    let r = image_shape.len();
    require!(
        r >= 1 && r <= 3 && block_shape.len() == r,
        "Col2Im supports 1-3 spatial dims with matching image_shape/block_shape ranks \
         (got image rank {r}, block rank {})",
        block_shape.len()
    );
    require!(
        image_shape.iter().all(|&d| d >= 1),
        "Col2Im `image_shape` dims must all be >= 1, got {image_shape:?}"
    );
    let mut kk: i64 = 1;
    for &b in &block_shape {
        require!(b >= 1, "Col2Im `block_shape` dims must all be >= 1, got {block_shape:?}");
        kk *= b;
    }
    require!(
        kk != 0 && i.shape[1] % kk == 0 && i.shape[1] / kk >= 1,
        "Col2Im input channels {} must be a positive multiple of prod(block_shape)={kk}",
        i.shape[1]
    );
    let ints_ok = |name: &str, want: usize| -> bool {
        let (present, v) = node.ints_attr(name);
        !present || v.len() == want
    };
    require!(
        ints_ok("dilations", r) && ints_ok("strides", r) && ints_ok("pads", 2 * r),
        "Col2Im `dilations`/`strides` must have {r} entries and `pads` {} entries (per spatial dim)",
        2 * r
    );
    Ok(())
}

// =================================================================================================
// RoiAlign (2D)
// =================================================================================================

/// A single [R,1] ROI column reshaped to [R].
fn roi_column(
    ctx: &mut TranslationContext,
    rois: mlx::mlx_array,
    r: i32,
    col: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let s = slice(ctx, rois, &[0, col], &[r, col + 1])?;
    reshape(ctx, s, &[r])
}

fn roi_align_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let xin = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(xin);
    let (cc, hh, ww) = (xs[1], xs[2], xs[3]);

    let r0 = ctx.resolve(&n.inputs[1])?;
    let rois = astype(ctx, r0, mlx::mlx_dtype__MLX_FLOAT32)?; // [R,4]
    let rr = ctx.shape_of(rois)[0];
    let b0 = ctx.resolve(&n.inputs[2])?;
    let batch_idx = astype(ctx, b0, mlx::mlx_dtype__MLX_INT32)?; // [R]

    let mode = n.strings.get("mode").map(String::as_str).unwrap_or("avg");
    let ctm = n
        .strings
        .get("coordinate_transformation_mode")
        .map(String::as_str)
        .unwrap_or("half_pixel");
    let half_pixel = ctm == "half_pixel";
    let is_max = mode == "max";
    let oh = n.ints.get("output_height").copied().unwrap_or(1) as i32;
    let ow = n.ints.get("output_width").copied().unwrap_or(1) as i32;
    let sr = n.ints.get("sampling_ratio").copied().unwrap_or(0) as i32;
    let scale = n.floats.get("spatial_scale").copied().unwrap_or(1.0);
    let (gh, gw) = (sr, sr);
    let offset = if half_pixel { 0.5 } else { 0.0 };
    let count = (gh * gw).max(1) as f32;

    // Per-ROI start / bin size on-device: coord = rois[:,k]*scale - offset.
    let denorm = |ctx: &mut TranslationContext, col: i32| -> Result<mlx::mlx_array, MlxError> {
        let c = roi_column(ctx, rois, rr, col)?;
        let sc = scalar_f(ctx, scale);
        let m = mul(ctx, c, sc)?;
        let off = scalar_f(ctx, offset);
        sub(ctx, m, off)
    };
    let rsw = denorm(ctx, 0)?;
    let rsh = denorm(ctx, 1)?;
    let rew = denorm(ctx, 2)?;
    let reh = denorm(ctx, 3)?;
    let mut roi_w = sub(ctx, rew, rsw)?;
    let mut roi_h = sub(ctx, reh, rsh)?;
    if !half_pixel {
        let one = scalar_f(ctx, 1.0);
        roi_w = maximum(ctx, roi_w, one)?;
        let one2 = scalar_f(ctx, 1.0);
        roi_h = maximum(ctx, roi_h, one2)?;
    }
    let owf = scalar_f(ctx, ow as f32);
    let binw = div(ctx, roi_w, owf)?;
    let ohf = scalar_f(ctx, oh as f32);
    let binh = div(ctx, roi_h, ohf)?;

    let rsh5 = reshape(ctx, rsh, &[rr, 1, 1, 1, 1])?;
    let rsw5 = reshape(ctx, rsw, &[rr, 1, 1, 1, 1])?;
    let binh5 = reshape(ctx, binh, &[rr, 1, 1, 1, 1])?;
    let binw5 = reshape(ctx, binw, &[rr, 1, 1, 1, 1])?;
    let ah = arange(ctx, oh)?;
    let ph_idx = reshape(ctx, ah, &[1, oh, 1, 1, 1])?;
    let aw = arange(ctx, ow)?;
    let pw_idx = reshape(ctx, aw, &[1, 1, ow, 1, 1])?;
    let agh = arange(ctx, gh)?;
    let iy_idx = reshape(ctx, agh, &[1, 1, 1, gh, 1])?;
    let agw = arange(ctx, gw)?;
    let ix_idx = reshape(ctx, agw, &[1, 1, 1, 1, gw])?;

    // y = rsh + ph*binh + (iy+0.5)*binh/gh ; x = rsw + pw*binw + (ix+0.5)*binw/gw
    let half = scalar_f(ctx, 0.5);
    let iy_h = add(ctx, iy_idx, half)?;
    let ghf = scalar_f(ctx, gh as f32);
    let iy_frac = div(ctx, iy_h, ghf)?;
    let iy_term = mul(ctx, iy_frac, binh5)?;
    let ph_term = mul(ctx, ph_idx, binh5)?;
    let yc0 = add(ctx, rsh5, ph_term)?;
    let yc = add(ctx, yc0, iy_term)?;

    let half2 = scalar_f(ctx, 0.5);
    let ix_h = add(ctx, ix_idx, half2)?;
    let gwf = scalar_f(ctx, gw as f32);
    let ix_frac = div(ctx, ix_h, gwf)?;
    let ix_term = mul(ctx, ix_frac, binw5)?;
    let pw_term = mul(ctx, pw_idx, binw5)?;
    let xc0 = add(ctx, rsw5, pw_term)?;
    let xc = add(ctx, xc0, ix_term)?;

    let ps = oh * ow * gh * gw;
    let yc_b = broadcast_to(ctx, yc, &[rr, oh, ow, gh, gw])?;
    let yc = reshape(ctx, yc_b, &[rr, ps])?;
    let xc_b = broadcast_to(ctx, xc, &[rr, oh, ow, gh, gw])?;
    let xc = reshape(ctx, xc_b, &[rr, ps])?;

    let vy = in_range_mask(ctx, yc, -1.0, hh as f32)?;
    let vx = in_range_mask(ctx, xc, -1.0, ww as f32)?;
    let valid = mul(ctx, vy, vx)?;
    let cy = clip(ctx, yc, 0.0, (hh - 1) as f32)?;
    let cx = clip(ctx, xc, 0.0, (ww - 1) as f32)?;
    let y0 = floor(ctx, cy)?;
    let x0 = floor(ctx, cx)?;
    let one = scalar_f(ctx, 1.0);
    let y0p1 = add(ctx, y0, one)?;
    let y1 = clip(ctx, y0p1, 0.0, (hh - 1) as f32)?;
    let one2 = scalar_f(ctx, 1.0);
    let x0p1 = add(ctx, x0, one2)?;
    let x1 = clip(ctx, x0p1, 0.0, (ww - 1) as f32)?;
    let ly = sub(ctx, cy, y0)?;
    let lx = sub(ctx, cx, x0)?;
    let one3 = scalar_f(ctx, 1.0);
    let hy = sub(ctx, one3, ly)?;
    let one4 = scalar_f(ctx, 1.0);
    let hx = sub(ctx, one4, lx)?;
    let hyhx = mul(ctx, hy, hx)?;
    let w1 = mul(ctx, hyhx, valid)?;
    let hylx = mul(ctx, hy, lx)?;
    let w2 = mul(ctx, hylx, valid)?;
    let lyhx = mul(ctx, ly, hx)?;
    let w3 = mul(ctx, lyhx, valid)?;
    let lylx = mul(ctx, ly, lx)?;
    let w4 = mul(ctx, lylx, valid)?;

    let xb = take_axis(ctx, xin, batch_idx, 0)?; // [R,C,H,W]
    let xbf = reshape(ctx, xb, &[rr, cc, hh * ww])?;

    // corner(yy, xx, w): gather bilinear neighbour, weighted.
    let corner = |ctx: &mut TranslationContext,
                  yy: mlx::mlx_array,
                  xx: mlx::mlx_array,
                  w: mlx::mlx_array|
     -> Result<mlx::mlx_array, MlxError> {
        let yi = astype(ctx, yy, mlx::mlx_dtype__MLX_INT32)?;
        let wsc = scalar_i(ctx, ww);
        let ymul = mul(ctx, yi, wsc)?;
        let xi = astype(ctx, xx, mlx::mlx_dtype__MLX_INT32)?;
        let flat = add(ctx, ymul, xi)?;
        let flat3 = reshape(ctx, flat, &[rr, 1, ps])?;
        let idx = broadcast_to(ctx, flat3, &[rr, cc, ps])?;
        let g = take_along_axis(ctx, xbf, idx, 2)?; // [R,C,PS]
        let w3d = reshape(ctx, w, &[rr, 1, ps])?;
        mul(ctx, g, w3d)
    };
    let c1 = corner(ctx, y0, x0, w1)?;
    let c2 = corner(ctx, y0, x1, w2)?;
    let c3 = corner(ctx, y1, x0, w3)?;
    let c4 = corner(ctx, y1, x1, w4)?;

    let out = if is_max {
        let m12 = maximum(ctx, c1, c2)?;
        let m34 = maximum(ctx, c3, c4)?;
        let v = maximum(ctx, m12, m34)?; // [R,C,PS]
        let vr = reshape(ctx, v, &[rr, cc, oh * ow, gh * gw])?;
        max_axis(ctx, vr, 3)?
    } else {
        let s12 = add(ctx, c1, c2)?;
        let s34 = add(ctx, c3, c4)?;
        let s = add(ctx, s12, s34)?;
        let sr2 = reshape(ctx, s, &[rr, cc, oh * ow, gh * gw])?;
        let summed = sum_axis(ctx, sr2, 3)?;
        let cnt = scalar_f(ctx, count);
        div(ctx, summed, cnt)?
    };
    let out = reshape(ctx, out, &[rr, cc, oh, ow])?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn roi_align_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, r, b, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(x), Some(r), Some(b), Some(o)) => (x, r, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype)
            && out.dtype == x.dtype
            && is_mlx_float(r.dtype)
            && is_int_index(b.dtype),
        "input/output must share one float dtype, rois must be float, and batch_indices int32/int64; got {} / {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(r.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4 && x.shape.iter().all(|&d| d >= 0),
        "input must have static rank-4 shape, got {:?}",
        x.shape
    );
    require!(
        r.shape.len() == 2 && r.shape[1] == 4 && r.shape.iter().all(|&d| d >= 0),
        "rois must have static shape [R,4], got {:?}",
        r.shape
    );
    require!(
        b.shape.len() == 1 && b.shape.iter().all(|&d| d >= 0),
        "batch_indices must have static rank-1 shape, got {:?}",
        b.shape
    );
    let sampling_ratio = node.int_attr("sampling_ratio", 0);
    require!(
        sampling_ratio > 0,
        "sampling_ratio must be positive, got {sampling_ratio}"
    );
    let mode = node.string_attr("mode", "avg");
    require!(
        mode == "avg" || mode == "max",
        "mode must be \"avg\" or \"max\", got {mode:?}"
    );
    let ctm = node.string_attr("coordinate_transformation_mode", "half_pixel");
    require!(
        ctm == "half_pixel" || ctm == "output_half_pixel",
        "coordinate_transformation_mode must be \"half_pixel\" or \"output_half_pixel\", got {ctm:?}"
    );
    Ok(())
}

// =================================================================================================
// MaxRoiPool
// =================================================================================================

fn max_roi_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let xin = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(xin);
    let (cc, hh, ww) = (xs[1], xs[2], xs[3]);

    let r0 = ctx.resolve(&n.inputs[1])?;
    let rois = astype(ctx, r0, mlx::mlx_dtype__MLX_FLOAT32)?; // [R,5]
    let rr = ctx.shape_of(rois)[0];

    let pooled = n
        .int_arrays
        .get("pooled_shape")
        .cloned()
        .unwrap_or_default();
    let ph = pooled[0] as i32;
    let pw = pooled[1] as i32;
    let scale = n.floats.get("spatial_scale").copied().unwrap_or(1.0);

    let bc = roi_column(ctx, rois, rr, 0)?;
    let batch_idx = astype(ctx, bc, mlx::mlx_dtype__MLX_INT32)?; // [R]

    let rounded = |ctx: &mut TranslationContext, col: i32| -> Result<mlx::mlx_array, MlxError> {
        let c = roi_column(ctx, rois, rr, col)?;
        let sc = scalar_f(ctx, scale);
        let m = mul(ctx, c, sc)?;
        round_away(ctx, m)
    };
    let rsw = rounded(ctx, 1)?;
    let rsh = rounded(ctx, 2)?;
    let rew = rounded(ctx, 3)?;
    let reh = rounded(ctx, 4)?;
    let one = scalar_f(ctx, 1.0);
    let dw = sub(ctx, rew, rsw)?;
    let dw1 = add(ctx, dw, one)?;
    let one_a = scalar_f(ctx, 1.0);
    let roi_w = maximum(ctx, dw1, one_a)?;
    let one_b = scalar_f(ctx, 1.0);
    let dh = sub(ctx, reh, rsh)?;
    let dh1 = add(ctx, dh, one_b)?;
    let one_c = scalar_f(ctx, 1.0);
    let roi_h = maximum(ctx, dh1, one_c)?;
    let pwf = scalar_f(ctx, pw as f32);
    let binw = div(ctx, roi_w, pwf)?;
    let phf = scalar_f(ctx, ph as f32);
    let binh = div(ctx, roi_h, phf)?;

    // Integer window bounds per (ROI, output bin), clamped to [0,limit]. lo/hi shapes [R,P].
    let bounds = |ctx: &mut TranslationContext,
                  p: i32,
                  bin: mlx::mlx_array,
                  start: mlx::mlx_array,
                  limit: i32|
     -> Result<(mlx::mlx_array, mlx::mlx_array), MlxError> {
        let ap = arange(ctx, p)?;
        let p_idx = reshape(ctx, ap, &[1, p])?;
        let bin2 = reshape(ctx, bin, &[rr, 1])?;
        let start2 = reshape(ctx, start, &[rr, 1])?;
        let lo_m = mul(ctx, p_idx, bin2)?;
        let lo_f = floor(ctx, lo_m)?;
        let lo_a = add(ctx, lo_f, start2)?;
        let lo = clip(ctx, lo_a, 0.0, limit as f32)?;
        let one_h = scalar_f(ctx, 1.0);
        let p1 = add(ctx, p_idx, one_h)?;
        let hi_m = mul(ctx, p1, bin2)?;
        let hi_c = ceil(ctx, hi_m)?;
        let hi_a = add(ctx, hi_c, start2)?;
        let hi = clip(ctx, hi_a, 0.0, limit as f32)?;
        Ok((lo, hi))
    };
    let (hstart, hend) = bounds(ctx, ph, binh, rsh, hh)?; // [R,PH]
    let (wstart, wend) = bounds(ctx, pw, binw, rsw, ww)?; // [R,PW]

    let ahh = arange(ctx, hh)?;
    let h_idx = reshape(ctx, ahh, &[1, 1, hh])?;
    let aww = arange(ctx, ww)?;
    let w_idx = reshape(ctx, aww, &[1, 1, ww])?;
    let hstart3 = reshape(ctx, hstart, &[rr, ph, 1])?;
    let ge_h = ctx.binary(mlx::mlx_greater_equal, h_idx, hstart3)?;
    let hend3 = reshape(ctx, hend, &[rr, ph, 1])?;
    let lt_h = ctx.binary(mlx::mlx_less, h_idx, hend3)?;
    let mask_h = ctx.binary(mlx::mlx_logical_and, ge_h, lt_h)?; // [R,PH,H]
    let wstart3 = reshape(ctx, wstart, &[rr, pw, 1])?;
    let ge_w = ctx.binary(mlx::mlx_greater_equal, w_idx, wstart3)?;
    let wend3 = reshape(ctx, wend, &[rr, pw, 1])?;
    let lt_w = ctx.binary(mlx::mlx_less, w_idx, wend3)?;
    let mask_w = ctx.binary(mlx::mlx_logical_and, ge_w, lt_w)?; // [R,PW,W]

    let neg = scalar_f(ctx, -3.0e38);
    let xb = take_axis(ctx, xin, batch_idx, 0)?; // [R,C,H,W]

    // Masked max over H: [R,PH,C,H,W] -> [R,PH,C,W].
    let mask_h5 = reshape(ctx, mask_h, &[rr, ph, 1, hh, 1])?;
    let xb5 = reshape(ctx, xb, &[rr, 1, cc, hh, ww])?;
    let sel_h = where_(ctx, mask_h5, xb5, neg)?;
    let maxh = max_axis(ctx, sel_h, 3)?; // [R,PH,C,W]

    // Masked max over W: [R,PH,PW,C,W] -> [R,PH,PW,C].
    let mask_w5 = reshape(ctx, mask_w, &[rr, 1, pw, 1, ww])?;
    let maxh5 = reshape(ctx, maxh, &[rr, ph, 1, cc, ww])?;
    let sel_w = where_(ctx, mask_w5, maxh5, neg)?;
    let mut out = max_axis(ctx, sel_w, 4)?; // [R,PH,PW,C]

    // Empty windows (hend<=hstart or wend<=wstart) emit 0.
    let le_h = ctx.binary(mlx::mlx_less_equal, hend, hstart)?;
    let le_h3 = reshape(ctx, le_h, &[rr, ph, 1])?;
    let le_w = ctx.binary(mlx::mlx_less_equal, wend, wstart)?;
    let le_w3 = reshape(ctx, le_w, &[rr, 1, pw])?;
    let empty = ctx.binary(mlx::mlx_logical_or, le_h3, le_w3)?;
    let empty4 = reshape(ctx, empty, &[rr, ph, pw, 1])?;
    let zero = scalar_f(ctx, 0.0);
    out = where_(ctx, empty4, zero, out)?;

    let out_t = ctx.transpose(out, &[0, 3, 1, 2])?;
    let out_t = astype(ctx, out_t, out_dtype(n))?;
    let y = contiguous(ctx, out_t)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn max_roi_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, r, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(r), Some(o)) => (x, r, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype && is_mlx_float(r.dtype),
        "input/output must share one float dtype and rois must be float, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(r.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4 && x.shape.iter().all(|&d| d >= 0),
        "input must have static rank-4 shape, got {:?}",
        x.shape
    );
    require!(
        r.shape.len() == 2 && r.shape[1] == 5 && r.shape.iter().all(|&d| d >= 0),
        "rois must have static shape [R,5], got {:?}",
        r.shape
    );
    let (present, pooled) = node.ints_attr("pooled_shape");
    require!(
        present && pooled.len() == 2,
        "pooled_shape must contain exactly 2 values, got {:?}",
        pooled
    );
    require!(
        pooled[0] >= 1 && pooled[1] >= 1,
        "pooled_shape dimensions must be positive, got {:?}",
        pooled
    );
    Ok(())
}

// =================================================================================================
// MaxUnpool (2-input form)
// =================================================================================================

fn max_unpool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let xc = contiguous(ctx, x0)?;
    let xin = astype(ctx, xc, mlx::mlx_dtype__MLX_FLOAT32)?;
    let xs = ctx.shape_of(xin);
    let nn = xs[0];
    let cc = xs[1];
    let r = xs.len() - 2;

    let kernel = n
        .int_arrays
        .get("kernel_shape")
        .cloned()
        .unwrap_or_default();
    let strides = n
        .int_arrays
        .get("strides")
        .cloned()
        .unwrap_or_else(|| vec![1; r]);
    let pads = n
        .int_arrays
        .get("pads")
        .cloned()
        .unwrap_or_else(|| vec![0; 2 * r]);

    let mut total: i64 = nn as i64 * cc as i64;
    let mut s_out: i64 = 1;
    let mut out_shape = vec![nn, cc];
    for d in 0..r {
        total *= xs[2 + d] as i64;
        let od = (xs[2 + d] as i64 - 1) * strides[d] - (pads[d] + pads[r + d]) + kernel[d];
        out_shape.push(od as i32);
        s_out *= od;
    }

    let total_out = nn * cc * s_out as i32;
    let out_acc = zeros_1d(ctx, total_out)?;

    let idx0 = ctx.resolve(&n.inputs[1])?;
    let idx_i = astype(ctx, idx0, mlx::mlx_dtype__MLX_INT32)?;
    let idx = reshape(ctx, idx_i, &[total as i32])?;
    let updates = reshape(ctx, xin, &[total as i32, 1])?;

    let scattered = scatter_add(ctx, out_acc, idx, updates)?;
    let out = reshape(ctx, scattered, &out_shape)?;
    let out = astype(ctx, out, out_dtype(n))?;
    let y = contiguous(ctx, out)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn max_unpool_claim(node: &NodeView) -> ClaimResult {
    // 2-input form only: an explicit output_shape (input 2) may crop/pad -> leave to CPU.
    require!(
        node.num_inputs() == 2 && !node.input_present(2) && node.num_outputs() == 1,
        "only the 2-input form with 1 output is supported, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, i, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(i), Some(o)) => (x, i, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype && is_int_index(i.dtype),
        "input/output must share one float dtype and indices must be int32/int64, got {} / {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 3 && x.shape.len() <= 5 && x.shape.iter().all(|&d| d >= 0),
        "input must have static rank 3-5 shape, got {:?}",
        x.shape
    );
    require!(
        i.shape == x.shape,
        "indices shape must match input shape, got {:?} vs {:?}",
        i.shape,
        x.shape
    );
    let r = x.shape.len() - 2;
    let (kp, kernel) = node.ints_attr("kernel_shape");
    require!(
        kp && kernel.len() == r,
        "kernel_shape must contain {r} values, got {:?}",
        kernel
    );
    let (hs, strides) = node.ints_attr("strides");
    require!(
        !hs || strides.len() == r,
        "strides must contain {r} values, got {:?}",
        strides
    );
    let (hp, pads) = node.ints_attr("pads");
    require!(
        !hp || pads.len() == 2 * r,
        "pads must contain {} values, got {:?}",
        2 * r,
        pads
    );
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn reg(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain: "",
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register_vision(registry: &mut OpRegistry) {
    reg(registry, "GridSample", grid_sample_op, grid_sample_claim);
    reg(registry, "AffineGrid", affine_grid_op, affine_grid_claim);
    reg(registry, "Col2Im", col2im_op, col2im_claim);
    reg(registry, "RoiAlign", roi_align_op, roi_align_claim);
    reg(registry, "MaxRoiPool", max_roi_pool_op, max_roi_pool_claim);
    reg(registry, "MaxUnpool", max_unpool_op, max_unpool_claim);
}
