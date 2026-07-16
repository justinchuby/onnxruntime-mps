//! Convolution and pooling op handlers. Faithful port of the C++ `ops/conv.cc` (Conv, ConvTranspose,
//! AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool) plus the pooling extras from
//! `ops/normpool.cc` (LpPool, GlobalLpPool).
//!
//! ONNX conv/pool tensors are NCHW (channels-first); MLX's `mlx_conv*` and the strided-window pooling
//! path expect NHWC (channels-last). Every handler therefore transposes the input to channels-last
//! (`to_channels_last`), runs the MLX op, and transposes the result back (`from_channels_last`);
//! conv weights are repacked from ONNX `[O, I/g, kH, kW]` to MLX `[O, kH, kW, I/g]`. Only the exact
//! attribute/shape forms the C++ claim accepts are claimed (NOTSET auto_pad, per-dim asymmetric pads
//! via `mlx_conv_general`, unit dilations where MLX cannot express them, no ceil_mode); everything
//! else is left to ORT CPU.

use std::os::raw::c_char;

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    is_mlx_float, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::{deny, require};

// ---- small arithmetic/movement helpers ----------------------------------------------------------

fn add(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_add, a, b)
}
fn abs_(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_abs, a)
}
fn power(ctx: &mut TranslationContext, a: mlx::mlx_array, b: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_power, a, b)
}

/// A 0-d scalar of dtype `dt` holding `v` (kept), so no unwanted upcast happens in the compute path.
fn scalar_for_dtype(ctx: &mut TranslationContext, v: f32, dt: mlx::mlx_dtype) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(v);
    ctx.astype(s, dt)
}

/// ONNX NCHW -> MLX NHWC (channels last), forced contiguous.
fn to_channels_last(ctx: &mut TranslationContext, x: mlx::mlx_array, spatial_rank: i32) -> Result<mlx::mlx_array, MlxError> {
    let axes: Vec<i32> = if spatial_rank == 1 { vec![0, 2, 1] } else { vec![0, 2, 3, 1] };
    let t = ctx.transpose(x, &axes)?;
    ctx.contiguous(t)
}

/// MLX NHWC -> ONNX NCHW, forced contiguous.
fn from_channels_last(ctx: &mut TranslationContext, x: mlx::mlx_array, spatial_rank: i32) -> Result<mlx::mlx_array, MlxError> {
    let axes: Vec<i32> = if spatial_rank == 1 { vec![0, 2, 1] } else { vec![0, 3, 1, 2] };
    let t = ctx.transpose(x, &axes)?;
    ctx.contiguous(t)
}

/// ONNX conv weight `[O, I/g, k...]` -> MLX `[O, k..., I/g]`, forced contiguous.
fn conv_weight_to_mlx(ctx: &mut TranslationContext, w: mlx::mlx_array, spatial_rank: i32) -> Result<mlx::mlx_array, MlxError> {
    let axes: Vec<i32> = if spatial_rank == 1 { vec![0, 2, 1] } else { vec![0, 2, 3, 1] };
    let t = ctx.transpose(w, &axes)?;
    ctx.contiguous(t)
}

/// `int_arrays[name]` or a default vector of `value` repeated `size` times.
fn attr_or(n: &NodeDesc, name: &str, size: usize, value: i64) -> Vec<i64> {
    n.int_arrays.get(name).cloned().unwrap_or_else(|| vec![value; size])
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

// ---- Conv ---------------------------------------------------------------------------------------

fn conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x0 = ctx.resolve(&n.inputs[0])?;
    let spatial_rank = ctx.ndim(x0) as i32 - 2;
    // The claim permits dynamic (symbolic) spatial dims; the general compiled path is shape-keyed so
    // the concrete extent is known here. Guard against a genuinely-unresolvable (<=0) spatial dim so
    // the EP declines gracefully to CPU rather than emitting a mis-sized conv.
    {
        let xs = ctx.shape_of(x0);
        if (0..spatial_rank as usize).any(|i| xs[i + 2] <= 0) {
            return Err(format!("Conv: non-positive spatial dim at trace time: {xs:?}"));
        }
    }
    let strides = attr_or(n, "strides", spatial_rank as usize, 1);
    let dilations = attr_or(n, "dilations", spatial_rank as usize, 1);
    let group = n.ints.get("group").copied().unwrap_or(1) as i32;

    let auto_pad = n.strings.get("auto_pad").map(String::as_str).unwrap_or("NOTSET");
    let x = to_channels_last(ctx, x0, spatial_rank)?;
    let w0 = ctx.resolve(&n.inputs[1])?;
    let pads: Vec<i64> = if auto_pad == "NOTSET" {
        attr_or(n, "pads", 2 * spatial_rank as usize, 0)
    } else {
        let sr = spatial_rank as usize;
        let xs = ctx.shape_of(x0);
        let ws = ctx.shape_of(w0);
        let in_sp: Vec<i64> = (0..sr).map(|i| xs[i + 2] as i64).collect();
        let kernel: Vec<i64> = (0..sr).map(|i| ws[i + 2] as i64).collect();
        auto_pad_pads(auto_pad, &in_sp, &kernel, &strides, &dilations)
            .ok_or_else(|| format!("Conv: unsupported auto_pad '{auto_pad}'"))?
    };
    let weight = conv_weight_to_mlx(ctx, w0, spatial_rank)?;

    // Symmetric pads (`pads[i] == pads[i + spatial_rank]`) use the fast symmetric conv1d/conv2d
    // primitives. ONNX also permits per-dim asymmetric pads (begin != end); MLX cannot express those
    // through conv1d/conv2d, so route them through `mlx_conv_general`, which takes separate
    // `padding_lo`/`padding_hi` vectors.
    let symmetric = (0..spatial_rank as usize).all(|i| pads[i] == pads[i + spatial_rank as usize]);

    let mut out = if symmetric && spatial_rank == 1 {
        let (st, pa, di) = (strides[0] as i32, pads[0] as i32, dilations[0] as i32);
        ctx.emit(|res, s| unsafe { mlx::mlx_conv1d(res, x, weight, st, pa, di, group, s) })?
    } else if symmetric && spatial_rank == 2 {
        let (s0, s1) = (strides[0] as i32, strides[1] as i32);
        let (p0, p1) = (pads[0] as i32, pads[1] as i32);
        let (d0, d1) = (dilations[0] as i32, dilations[1] as i32);
        ctx.emit(|res, s| unsafe {
            mlx::mlx_conv2d(res, x, weight, s0, s1, p0, p1, d0, d1, group, s)
        })?
    } else {
        let sr = spatial_rank as usize;
        let stride_i: Vec<i32> = strides.iter().map(|&v| v as i32).collect();
        let pad_lo: Vec<i32> = (0..sr).map(|i| pads[i] as i32).collect();
        let pad_hi: Vec<i32> = (0..sr).map(|i| pads[i + sr] as i32).collect();
        let dil_i: Vec<i32> = dilations.iter().map(|&v| v as i32).collect();
        let input_dil: Vec<i32> = vec![1; sr];
        ctx.emit(|res, s| unsafe {
            mlx::mlx_conv_general(
                res,
                x,
                weight,
                stride_i.as_ptr(),
                stride_i.len(),
                pad_lo.as_ptr(),
                pad_lo.len(),
                pad_hi.as_ptr(),
                pad_hi.len(),
                dil_i.as_ptr(),
                dil_i.len(),
                input_dil.as_ptr(),
                input_dil.len(),
                group,
                false,
                s,
            )
        })?
    };

    if present(n, 2) {
        let bias = ctx.resolve(&n.inputs[2])?;
        out = add(ctx, out, bias)?;
    }
    let y = from_channels_last(ctx, out, spatial_rank)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- ConvTranspose (2D) -------------------------------------------------------------------------

fn conv_transpose_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let strides = attr_or(n, "strides", 2, 1);
    let pads = attr_or(n, "pads", 4, 0);
    let output_padding = attr_or(n, "output_padding", 2, 0);

    let x0 = ctx.resolve(&n.inputs[0])?;
    let x = to_channels_last(ctx, x0, 2)?;
    // ONNX ConvTranspose weight is [I, O, kH, kW]; MLX conv_transpose2d wants [O, kH, kW, I].
    let w0 = ctx.resolve(&n.inputs[1])?;
    let wt = ctx.transpose(w0, &[1, 2, 3, 0])?;
    let weight = ctx.contiguous(wt)?;

    let (s0, s1) = (strides[0] as i32, strides[1] as i32);
    let (p0, p1) = (pads[0] as i32, pads[1] as i32);
    let (op0, op1) = (output_padding[0] as i32, output_padding[1] as i32);
    let mut out = ctx.emit(|res, s| unsafe {
        mlx::mlx_conv_transpose2d(res, x, weight, s0, s1, p0, p1, 1, 1, op0, op1, 1, s)
    })?;

    if present(n, 2) {
        let bias = ctx.resolve(&n.inputs[2])?;
        out = add(ctx, out, bias)?;
    }
    let y = from_channels_last(ctx, out, 2)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- shared pooling primitives (2D, channels-last) ----------------------------------------------

/// Pad the H/W (axes 1,2) of a channels-last [N,H,W,C] array with `value`. `pads` is [t,l,b,r].
fn pad_spatial(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    pads: &[i64],
    value: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    if pads.iter().all(|&p| p == 0) {
        return Ok(x);
    }
    let axes = [1i32, 2];
    let low = [pads[0] as i32, pads[1] as i32];
    let high = [pads[2] as i32, pads[3] as i32];
    let mode = b"constant\0";
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_pad(
            res,
            x,
            axes.as_ptr(),
            2,
            low.as_ptr(),
            2,
            high.as_ptr(),
            2,
            value,
            mode.as_ptr() as *const c_char,
            s,
        )
    })?;
    ctx.contiguous(out)
}

/// Build a [N, out_h, out_w, kH, kW, C] strided window view over channels-last [N,H,W,C].
fn sliding_windows_2d(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    kernel: &[i64],
    strides: &[i64],
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(x);
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let out_h = (h - kernel[0] as i32) / strides[0] as i32 + 1;
    let out_w = (w - kernel[1] as i32) / strides[1] as i32 + 1;
    let window_shape: [i32; 6] = [n, out_h, out_w, kernel[0] as i32, kernel[1] as i32, c];
    let row_stride = w as i64 * c as i64;
    let window_strides: [i64; 6] = [
        h as i64 * row_stride,
        strides[0] * row_stride,
        strides[1] * c as i64,
        row_stride,
        c as i64,
        1,
    ];
    ctx.emit(|res, s| unsafe {
        mlx::mlx_as_strided(
            res,
            x,
            window_shape.as_ptr(),
            window_shape.len(),
            window_strides.as_ptr(),
            window_strides.len(),
            0,
            s,
        )
    })
}

fn reduce_pool_windows(ctx: &mut TranslationContext, windows: mlx::mlx_array, average: bool) -> Result<mlx::mlx_array, MlxError> {
    let axes = [3i32, 4];
    if average {
        ctx.emit(|res, s| unsafe { mlx::mlx_mean_axes(res, windows, axes.as_ptr(), 2, false, s) })
    } else {
        ctx.emit(|res, s| unsafe { mlx::mlx_max_axes(res, windows, axes.as_ptr(), 2, false, s) })
    }
}

fn sum_axes34(ctx: &mut TranslationContext, a: mlx::mlx_array, keepdims: bool) -> Result<mlx::mlx_array, MlxError> {
    let axes = [3i32, 4];
    ctx.emit(|res, s| unsafe { mlx::mlx_sum_axes(res, a, axes.as_ptr(), 2, keepdims, s) })
}

fn pool_op(ctx: &mut TranslationContext, n: &NodeDesc, average: bool) -> Result<(), MlxError> {
    let kernel = n.int_arrays.get("kernel_shape").cloned().unwrap_or_default();
    let strides = attr_or(n, "strides", 2, 1);
    let count_include_pad =
        average && n.ints.get("count_include_pad").copied().unwrap_or(0) != 0;

    let x0 = ctx.resolve(&n.inputs[0])?;
    // Guard against a genuinely-unresolvable (<=0) spatial dim at trace time (see `conv_op`).
    {
        let xs = ctx.shape_of(x0);
        if xs.len() != 4 || xs[2] <= 0 || xs[3] <= 0 {
            return Err(format!("Pool: non-positive spatial dim at trace time: {xs:?}"));
        }
    }
    // `auto_pad` (SAME_UPPER/SAME_LOWER/VALID) → explicit pads from the static input spatial shape
    // (dilations are 1 for the claimed pool forms); NOTSET reads the `pads` attribute.
    let auto_pad = n.strings.get("auto_pad").map(String::as_str).unwrap_or("NOTSET");
    let pads: Vec<i64> = if auto_pad == "NOTSET" {
        attr_or(n, "pads", 4, 0)
    } else {
        let xs = ctx.shape_of(x0);
        let in_sp = [xs[2] as i64, xs[3] as i64];
        auto_pad_pads(auto_pad, &in_sp, &kernel, &strides, &[1, 1])
            .ok_or_else(|| format!("Pool: unsupported auto_pad '{auto_pad}'"))?
    };
    let x = to_channels_last(ctx, x0, 2)?;
    let dt = ctx.dtype_of(x);
    let pad_value = if average { 0.0f32 } else { f32::NEG_INFINITY };
    let pv = scalar_for_dtype(ctx, pad_value, dt)?;
    let padded = pad_spatial(ctx, x, &pads, pv)?;
    let windows = sliding_windows_2d(ctx, padded, &kernel, &strides)?;

    let has_padding = pads.iter().any(|&p| p != 0);
    let out = if !average || count_include_pad || !has_padding {
        reduce_pool_windows(ctx, windows, average)?
    } else {
        // count_include_pad == 0 average pooling with padding: divide the window sums by the count
        // of genuine (non-pad) elements per window (a padded ones-mask, same strided reduction).
        let sums = sum_axes34(ctx, windows, false)?;
        let x_shape = ctx.shape_of(x);
        let mask_shape = [x_shape[0], x_shape[1], x_shape[2], 1];
        let mask = ctx.emit(|res, s| unsafe {
            mlx::mlx_ones(res, mask_shape.as_ptr(), 4, dt, s)
        })?;
        let zero = scalar_for_dtype(ctx, 0.0, dt)?;
        let padded_mask = pad_spatial(ctx, mask, &pads, zero)?;
        let mask_windows = sliding_windows_2d(ctx, padded_mask, &kernel, &strides)?;
        let counts = sum_axes34(ctx, mask_windows, false)?;
        ctx.binary(mlx::mlx_divide, sums, counts)?
    };
    let y = from_channels_last(ctx, out, 2)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn average_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    pool_op(ctx, n, true)
}

fn max_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    pool_op(ctx, n, false)
}

fn global_pool_op(ctx: &mut TranslationContext, n: &NodeDesc, average: bool) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let axes = [2i32, 3];
    let out = if average {
        ctx.emit(|res, s| unsafe { mlx::mlx_mean_axes(res, x, axes.as_ptr(), 2, true, s) })?
    } else {
        ctx.emit(|res, s| unsafe { mlx::mlx_max_axes(res, x, axes.as_ptr(), 2, true, s) })?
    };
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn global_average_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    global_pool_op(ctx, n, true)
}

fn global_max_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    global_pool_op(ctx, n, false)
}

// ---- LpPool / GlobalLpPool (from normpool.cc) ---------------------------------------------------

fn lp_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let kernel = n.int_arrays.get("kernel_shape").cloned().unwrap_or_default();
    let strides = attr_or(n, "strides", 2, 1);
    let pads = attr_or(n, "pads", 4, 0);
    let p = n.ints.get("p").copied().unwrap_or(2) as f32;

    let x0 = ctx.resolve(&n.inputs[0])?;
    let x = to_channels_last(ctx, x0, 2)?;
    let dt = ctx.dtype_of(x);
    let zero = scalar_for_dtype(ctx, 0.0, dt)?;
    let padded = pad_spatial(ctx, x, &pads, zero)?;
    let a = abs_(ctx, padded)?;
    let ps = scalar_for_dtype(ctx, p, dt)?;
    let powered = power(ctx, a, ps)?;
    let windows = sliding_windows_2d(ctx, powered, &kernel, &strides)?;
    let summed = sum_axes34(ctx, windows, false)?;
    let inv = scalar_for_dtype(ctx, 1.0 / p, dt)?;
    let out = power(ctx, summed, inv)?;
    let y = from_channels_last(ctx, out, 2)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn global_lp_pool_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(x) as i32;
    let p = n.ints.get("p").copied().unwrap_or(2) as f32;
    let dt = ctx.dtype_of(x);
    let axes: Vec<i32> = (2..rank).collect();

    let a = abs_(ctx, x)?;
    let ps = scalar_for_dtype(ctx, p, dt)?;
    let powered = power(ctx, a, ps)?;
    let summed = ctx.emit(|res, s| unsafe {
        mlx::mlx_sum_axes(res, powered, axes.as_ptr(), axes.len(), true, s)
    })?;
    let inv = scalar_for_dtype(ctx, 1.0 / p, dt)?;
    let out = power(ctx, summed, inv)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim helpers (port of op_claim.h + conv.cc/normpool.cc local helpers) ----------------------

fn read_spatial_attribute(node: &NodeView, name: &str, spatial_rank: usize, default: i64) -> Option<Vec<i64>> {
    let (present, mut values) = node.ints_attr(name);
    if !present {
        values = vec![default; spatial_rank];
    }
    if values.len() != spatial_rank || !values.iter().all(|&v| v > 0) {
        return None;
    }
    Some(values)
}

fn read_pads(node: &NodeView, spatial_rank: usize) -> Option<Vec<i64>> {
    let (present, mut pads) = node.ints_attr("pads");
    if !present {
        pads = vec![0; 2 * spatial_rank];
    }
    if pads.len() != 2 * spatial_rank || !pads.iter().all(|&v| v >= 0) {
        return None;
    }
    Some(pads)
}

/// Resolve an ONNX `auto_pad` mode to explicit begin/end pads from statically known spatial dims.
/// Returns ONNX pad layout `[begin_0..begin_{r-1}, end_0..end_{r-1}]`, or `None` for an
/// unrecognized mode. `VALID` yields zero pads; `SAME_UPPER`/`SAME_LOWER` size the total pad so the
/// output is `ceil(input/stride)`, putting the extra unit of an odd total at the end (`UPPER`) or
/// the begin (`LOWER`). Asymmetric SAME_* pads are fine — `conv_op` routes them through
/// `mlx_conv_general` exactly like explicit asymmetric pads.
fn auto_pad_pads(
    auto_pad: &str,
    in_spatial: &[i64],
    kernel: &[i64],
    strides: &[i64],
    dilations: &[i64],
) -> Option<Vec<i64>> {
    let r = in_spatial.len();
    match auto_pad {
        "VALID" => Some(vec![0; 2 * r]),
        "SAME_UPPER" | "SAME_LOWER" => {
            let mut begin = vec![0i64; r];
            let mut end = vec![0i64; r];
            for i in 0..r {
                let eff_k = dilations[i] * (kernel[i] - 1) + 1;
                let out = (in_spatial[i] + strides[i] - 1) / strides[i]; // ceil(in/stride)
                let total = ((out - 1) * strides[i] + eff_k - in_spatial[i]).max(0);
                if auto_pad == "SAME_LOWER" {
                    end[i] = total / 2;
                    begin[i] = total - end[i];
                } else {
                    begin[i] = total / 2;
                    end[i] = total - begin[i];
                }
            }
            begin.extend(end);
            Some(begin)
        }
        _ => None,
    }
}

fn static_positive_shape(shape: &[i64], rank: usize) -> bool {
    shape.len() == rank && shape.iter().all(|&d| d > 0)
}

/// Like `static_positive_shape`, but permits a dynamic (symbolic / non-positive) leading batch dim.
/// ORT reports a symbolic dim as `-1` at GetCapability time; the batch dim is only carried through
/// (never used to size a kernel), so requiring it to be static needlessly rejects the whole conv/pool
/// backbone of any dynamic-batch model. All non-batch dims (channels + spatial) must still be
/// statically known and positive so the MLX conv/pool shapes are well-defined.
fn static_positive_shape_dyn_batch(shape: &[i64], rank: usize) -> bool {
    rank >= 1 && shape.len() == rank && shape[1..].iter().all(|&d| d > 0)
}

/// Conv/pool input-shape validity that additionally permits dynamic (symbolic / non-positive)
/// SPATIAL dims (H/W), on top of the dynamic batch dim already allowed by
/// `static_positive_shape_dyn_batch`. Only the channel dim (index 1) must be statically known and
/// positive — it drives group / channel-divisibility and kernel checks. Batch (index 0) and every
/// spatial dim (index 2..) may be `-1`: the general compiled path is shape-keyed, so at TRACE time
/// `ctx.shape_of` resolves the concrete spatial extent and the handler computes any auto_pad from it.
fn channels_static_dyn_batch_spatial(shape: &[i64], rank: usize) -> bool {
    rank >= 2 && shape.len() == rank && shape[1] > 0
}

/// Compare an actual (ONNX-declared) shape against an expected shape, treating any non-positive
/// (dynamic / symbolic) dim on EITHER side as a wildcard. Dynamic spatial convs declare `-1` output
/// dims and we cannot precompute the expected spatial extent (pads may depend on the runtime shape),
/// so both sides carry wildcards that must be skipped.
fn same_known_shape(actual: &[i64], expected: &[i64]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected)
        .all(|(&a, &e)| a <= 0 || e <= 0 || a == e)
}

fn optional_bias_is_valid(node: &NodeView, dtype: crate::sys::ort::ONNXTensorElementDataType, channels: i64) -> bool {
    if !node.input_present(2) {
        return true;
    }
    match node.input_info(2) {
        Some(info) => info.dtype == dtype && info.shape == vec![channels],
        None => false,
    }
}

// ---- claim predicates ---------------------------------------------------------------------------

fn conv_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (2..=3).contains(&ni) && node.num_outputs() == 1,
        "expects 2 or 3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (x, w, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(w), Some(o)) => (x, w, o),
        _ => deny!("missing tensor type/shape info on X, W, or output"),
    };
    require!(
        is_mlx_float(x.dtype) && w.dtype == x.dtype && out.dtype == x.dtype,
        "X/W/output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(w.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 3 || x.shape.len() == 4,
        "only rank-3 Conv1D and rank-4 Conv2D are supported (got X rank {}); dynamic batch/spatial dims are supported",
        x.shape.len()
    );
    let spatial_rank = x.shape.len() - 2;
    require!(
        channels_static_dyn_batch_spatial(&x.shape, spatial_rank + 2),
        "X channel dim must be statically known and positive (shape {:?}); dynamic batch/spatial dims are supported",
        x.shape
    );
    require!(
        static_positive_shape(&w.shape, spatial_rank + 2),
        "W must have rank {} with all dims statically known and positive (shape {:?})",
        spatial_rank + 2,
        w.shape
    );
    require!(
        out.shape.len() == spatial_rank + 2,
        "output rank must be {} to match X (got rank {})",
        spatial_rank + 2,
        out.shape.len()
    );
    let spatial_dynamic = (0..spatial_rank).any(|i| x.shape[i + 2] <= 0);
    require!(
        node.string_attr("auto_pad", "NOTSET") == "NOTSET" || !node.ints_attr("pads").0,
        "auto_pad and explicit pads cannot both be specified"
    );
    let strides = match read_spatial_attribute(node, "strides", spatial_rank, 1) {
        Some(v) => v,
        None => deny!(
            "strides must contain {} positive spatial values",
            spatial_rank
        ),
    };
    let dilations = match read_spatial_attribute(node, "dilations", spatial_rank, 1) {
        Some(v) => v,
        None => deny!(
            "dilations must contain {} positive spatial values",
            spatial_rank
        ),
    };
    // `auto_pad` (SAME_UPPER/SAME_LOWER/VALID) resolves to explicit pads from the static input
    // spatial shape; NOTSET reads the `pads` attribute. SAME_* pads may be asymmetric — `conv_op`
    // routes those through `mlx_conv_general` just like explicit asymmetric pads.
    let auto_pad = node.string_attr("auto_pad", "NOTSET");
    let pads: Option<Vec<i64>> = if auto_pad == "NOTSET" {
        // Explicit pads are static regardless of dynamic spatial dims.
        match read_pads(node, spatial_rank) {
            Some(v) => Some(v),
            None => deny!(
                "pads must contain {} non-negative values",
                spatial_rank * 2
            ),
        }
    } else if spatial_dynamic {
        // auto_pad + dynamic spatial: pads depend on the runtime spatial extent, so they cannot be
        // precomputed here. `conv_op` resolves them at trace time from the concrete `ctx.shape_of`.
        None
    } else {
        let in_sp: Vec<i64> = (0..spatial_rank).map(|i| x.shape[i + 2]).collect();
        let kernel: Vec<i64> = (0..spatial_rank).map(|i| w.shape[i + 2]).collect();
        match auto_pad_pads(&auto_pad, &in_sp, &kernel, &strides, &dilations) {
            Some(v) => Some(v),
            None => deny!("unsupported auto_pad value {auto_pad:?}"),
        }
    };
    // Asymmetric pads (`pads[i] != pads[i + spatial_rank]`) are supported: `conv_op` routes them
    // through `mlx_conv_general`, which takes separate `padding_lo`/`padding_hi` vectors. Only the
    // non-negativity checked in `read_pads` is required.
    let (kernel_present, kernel_shape) = node.ints_attr("kernel_shape");
    if kernel_present {
        require!(
            kernel_shape.len() == spatial_rank,
            "kernel_shape must contain {} values (got {:?})",
            spatial_rank,
            kernel_shape
        );
        for i in 0..spatial_rank {
            require!(
                kernel_shape[i] == w.shape[i + 2],
                "kernel_shape {:?} must match W spatial shape {:?}",
                kernel_shape,
                &w.shape[2..]
            );
        }
    }
    let group = node.int_attr("group", 1);
    let channels = x.shape[1];
    let out_channels = w.shape[0];
    require!(group > 0, "group must be positive (got {group})");
    require!(
        channels % group == 0 && out_channels % group == 0,
        "group {group} must divide both input channels {channels} and output channels {out_channels}"
    );
    require!(
        w.shape[1] == channels / group,
        "W input-channel dim must equal X channels/group (got {} vs {}/{})",
        w.shape[1],
        channels,
        group
    );
    require!(
        optional_bias_is_valid(node, x.dtype, out_channels),
        "optional bias must have dtype {} and shape [{}]",
        crate::registry::ort_dtype_name(x.dtype),
        out_channels
    );
    let mut expected = vec![x.shape[0], out_channels];
    for i in 0..spatial_rank {
        let in_dim = x.shape[i + 2];
        match &pads {
            // Static input spatial dim with resolved pads → precompute the expected output extent.
            Some(pads) if in_dim > 0 => {
                let effective_kernel = dilations[i] * (w.shape[i + 2] - 1) + 1;
                let padded = in_dim + pads[i] + pads[i + spatial_rank];
                require!(
                    padded >= effective_kernel,
                    "padded spatial dim {i} ({padded}) is smaller than effective kernel {effective_kernel}"
                );
                expected.push((padded - effective_kernel) / strides[i] + 1);
            }
            // Dynamic spatial dim (or unresolved auto_pad pads) → the output extent is dynamic too;
            // push a wildcard so `same_known_shape` skips it. The handler resolves it at trace time.
            _ => expected.push(-1),
        }
    }
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?} (dynamic batch/spatial dims are wildcards)",
        out.shape,
        expected
    );
    Ok(())
}

fn conv_transpose_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (2..=3).contains(&ni) && node.num_outputs() == 1,
        "expects 2 or 3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (x, w, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(w), Some(o)) => (x, w, o),
        _ => deny!("missing tensor type/shape info on X, W, or output"),
    };
    require!(
        is_mlx_float(x.dtype) && w.dtype == x.dtype && out.dtype == x.dtype,
        "X/W/output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(w.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        static_positive_shape_dyn_batch(&x.shape, 4),
        "ConvTranspose requires rank-4 X with static positive channel/spatial dims (shape {:?}); dynamic batch is supported",
        x.shape
    );
    require!(
        static_positive_shape(&w.shape, 4),
        "ConvTranspose requires rank-4 W with all dims statically known and positive (shape {:?})",
        w.shape
    );
    require!(
        out.shape.len() == 4,
        "ConvTranspose output must have rank 4 (got rank {})",
        out.shape.len()
    );
    require!(
        node.string_attr("auto_pad", "NOTSET") == "NOTSET",
        "ConvTranspose supports only auto_pad=NOTSET"
    );
    require!(
        node.int_attr("group", 1) == 1,
        "grouped ConvTranspose is unsupported (group must be 1)"
    );
    require!(
        w.shape[0] == x.shape[1],
        "W input-channel dim {} must match X channels {}",
        w.shape[0],
        x.shape[1]
    );
    let strides = match read_spatial_attribute(node, "strides", 2, 1) {
        Some(v) => v,
        None => deny!("strides must contain 2 positive spatial values"),
    };
    let dilations = match read_spatial_attribute(node, "dilations", 2, 1) {
        Some(v) => v,
        None => deny!("dilations must contain 2 positive spatial values"),
    };
    require!(
        dilations == [1, 1],
        "dilated ConvTranspose is unsupported (dilations must be [1, 1], got {:?})",
        dilations
    );
    let pads = match read_pads(node, 2) {
        Some(v) => v,
        None => deny!("pads must contain 4 non-negative values"),
    };
    require!(
        pads[0] == pads[2] && pads[1] == pads[3],
        "ConvTranspose requires symmetric pads (got {:?})",
        pads
    );
    let (op_present, mut output_padding) = node.ints_attr("output_padding");
    if !op_present {
        output_padding = vec![0, 0];
    }
    require!(
        output_padding.len() == 2
            && output_padding[0] >= 0
            && output_padding[1] >= 0
            && output_padding[0] < strides[0]
            && output_padding[1] < strides[1],
        "output_padding must contain 2 non-negative values smaller than strides {:?} (got {:?})",
        strides,
        output_padding
    );
    let (output_shape_present, _) = node.ints_attr("output_shape");
    require!(
        !output_shape_present,
        "the output_shape attribute is unsupported"
    );
    let (kernel_present, kernel_shape) = node.ints_attr("kernel_shape");
    require!(
        !kernel_present || kernel_shape == vec![w.shape[2], w.shape[3]],
        "kernel_shape {:?} must match W spatial shape {:?}",
        kernel_shape,
        &w.shape[2..]
    );
    let out_channels = w.shape[1];
    require!(
        optional_bias_is_valid(node, x.dtype, out_channels),
        "optional bias must have dtype {} and shape [{}]",
        crate::registry::ort_dtype_name(x.dtype),
        out_channels
    );
    let expected = vec![
        x.shape[0],
        out_channels,
        strides[0] * (x.shape[2] - 1) + output_padding[0] + w.shape[2] - pads[0] - pads[2],
        strides[1] * (x.shape[3] - 1) + output_padding[1] + w.shape[3] - pads[1] - pads[3],
    ];
    require!(
        expected[2] > 0 && expected[3] > 0,
        "derived output spatial shape must be positive (got {:?})",
        &expected[2..]
    );
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
    );
    Ok(())
}

fn pool_claim(node: &NodeView, average: bool) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() == 4,
        "only rank-4 Pool2D is supported (got input rank {}); dynamic batch/spatial dims are supported",
        x.shape.len()
    );
    require!(
        channels_static_dyn_batch_spatial(&x.shape, 4),
        "input channel dim must be statically known and positive (shape {:?}); dynamic batch/spatial dims are supported",
        x.shape
    );
    require!(
        out.shape.len() == 4,
        "Pool2D output must have rank 4 (got rank {})",
        out.shape.len()
    );
    require!(
        node.int_attr("ceil_mode", 0) == 0,
        "ceil_mode=1 is unsupported"
    );
    let spatial_dynamic = x.shape[2] <= 0 || x.shape[3] <= 0;
    let (kernel_present, kernel) = node.ints_attr("kernel_shape");
    require!(
        kernel_present && kernel.len() == 2 && kernel[0] > 0 && kernel[1] > 0,
        "kernel_shape must contain 2 positive values (got {:?})",
        kernel
    );
    let strides = match read_spatial_attribute(node, "strides", 2, 1) {
        Some(v) => v,
        None => deny!("strides must contain 2 positive spatial values"),
    };
    let dilations = match read_spatial_attribute(node, "dilations", 2, 1) {
        Some(v) => v,
        None => deny!("dilations must contain 2 positive spatial values"),
    };
    require!(
        dilations == [1, 1],
        "dilated pooling is unsupported (dilations must be [1, 1], got {:?})",
        dilations
    );
    // `auto_pad` (SAME_UPPER/SAME_LOWER/VALID) resolves to explicit pads from the static input
    // spatial shape (kernel_shape is the window); NOTSET reads the `pads` attribute.
    let auto_pad = node.string_attr("auto_pad", "NOTSET");
    let pads: Option<Vec<i64>> = if auto_pad == "NOTSET" {
        match read_pads(node, 2) {
            Some(v) => Some(v),
            None => deny!("pads must contain 4 non-negative values"),
        }
    } else if node.ints_attr("pads").0 {
        deny!("auto_pad and explicit pads cannot both be specified");
    } else if spatial_dynamic {
        // auto_pad + dynamic spatial: pads depend on the runtime spatial extent; `pool_op` resolves
        // them at trace time from the concrete `ctx.shape_of`.
        None
    } else {
        let in_sp = [x.shape[2], x.shape[3]];
        match auto_pad_pads(&auto_pad, &in_sp, &kernel, &strides, &dilations) {
            Some(v) => Some(v),
            None => deny!("unsupported auto_pad value {auto_pad:?}"),
        }
    };
    if average {
        let cip = node.int_attr("count_include_pad", 0);
        require!(
            cip == 0 || cip == 1,
            "count_include_pad must be 0 or 1 (got {cip})"
        );
    } else {
        require!(
            node.int_attr("storage_order", 0) == 0,
            "storage_order=1 is unsupported"
        );
    }
    let expected = match &pads {
        // Static spatial with resolved pads → validate the derived output extent.
        Some(pads) if !spatial_dynamic => {
            let padded_h = x.shape[2] + pads[0] + pads[2];
            let padded_w = x.shape[3] + pads[1] + pads[3];
            require!(
                padded_h >= kernel[0] && padded_w >= kernel[1],
                "padded spatial shape [{padded_h}, {padded_w}] is smaller than kernel {:?}",
                kernel
            );
            vec![
                x.shape[0],
                x.shape[1],
                (padded_h - kernel[0]) / strides[0] + 1,
                (padded_w - kernel[1]) / strides[1] + 1,
            ]
        }
        // Dynamic spatial → output spatial dims are dynamic; wildcard them. `pool_op` resolves the
        // concrete window math at trace time.
        _ => vec![x.shape[0], x.shape[1], -1, -1],
    };
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?} (dynamic batch/spatial dims are wildcards)",
        out.shape,
        expected
    );
    Ok(())
}

fn average_pool_claim(node: &NodeView) -> ClaimResult {
    pool_claim(node, true)
}

fn max_pool_claim(node: &NodeView) -> ClaimResult {
    // MaxPool's optional 2nd output (indices) has no MLX argmax-window primitive here; only the
    // single-output form is claimed (mirrors the C++ single-output PoolClaim path).
    require!(
        node.num_outputs() == 1,
        "MaxPool indices output is unsupported (expects 1 output, got {})",
        node.num_outputs()
    );
    pool_claim(node, false)
}

fn global_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(out)) => (x, out),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        static_positive_shape_dyn_batch(&x.shape, 4),
        "global Pool requires rank-4 input with static positive channel/spatial dims (shape {:?}); dynamic batch is supported",
        x.shape
    );
    let expected = [x.shape[0], x.shape[1], 1, 1];
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
    );
    Ok(())
}

fn lp_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        static_positive_shape_dyn_batch(&x.shape, 4),
        "LpPool requires rank-4 input with static positive channel/spatial dims (shape {:?}); dynamic batch is supported",
        x.shape
    );
    require!(
        out.shape.len() == 4,
        "LpPool output must have rank 4 (got rank {})",
        out.shape.len()
    );
    require!(
        node.string_attr("auto_pad", "NOTSET") == "NOTSET",
        "LpPool supports only auto_pad=NOTSET"
    );
    require!(
        node.int_attr("ceil_mode", 0) == 0,
        "ceil_mode=1 is unsupported"
    );
    require!(
        node.int_attr("p", 2) > 0,
        "p must be positive (got {})",
        node.int_attr("p", 2)
    );
    let (kernel_present, kernel) = node.ints_attr("kernel_shape");
    require!(
        kernel_present && kernel.len() == 2 && kernel[0] > 0 && kernel[1] > 0,
        "kernel_shape must contain 2 positive values (got {:?})",
        kernel
    );
    let strides = match read_spatial_attribute(node, "strides", 2, 1) {
        Some(v) => v,
        None => deny!("strides must contain 2 positive spatial values"),
    };
    let pads = match read_pads(node, 2) {
        Some(v) => v,
        None => deny!("pads must contain 4 non-negative values"),
    };
    let dilations = match read_spatial_attribute(node, "dilations", 2, 1) {
        Some(v) => v,
        None => deny!("dilations must contain 2 positive spatial values"),
    };
    require!(
        dilations == [1, 1],
        "dilated LpPool is unsupported (dilations must be [1, 1], got {:?})",
        dilations
    );
    let padded_h = x.shape[2] + pads[0] + pads[2];
    let padded_w = x.shape[3] + pads[1] + pads[3];
    require!(
        padded_h >= kernel[0] && padded_w >= kernel[1],
        "padded spatial shape [{padded_h}, {padded_w}] is smaller than kernel {:?}",
        kernel
    );
    let expected = vec![
        x.shape[0],
        x.shape[1],
        (padded_h - kernel[0]) / strides[0] + 1,
        (padded_w - kernel[1]) / strides[1] + 1,
    ];
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
    );
    Ok(())
}

fn global_lp_pool_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "input/output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        static_positive_shape_dyn_batch(&x.shape, 4),
        "GlobalLpPool requires rank-4 input with static positive channel/spatial dims (shape {:?}); dynamic batch is supported",
        x.shape
    );
    require!(
        node.int_attr("p", 2) > 0,
        "p must be positive (got {})",
        node.int_attr("p", 2)
    );
    let expected = [x.shape[0], x.shape[1], 1, 1];
    require!(
        same_known_shape(&out.shape, &expected),
        "output shape {:?} does not match expected {:?}",
        out.shape,
        expected
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

pub fn register_conv(registry: &mut OpRegistry) {
    reg(registry, "Conv", conv_op, conv_claim);
    reg(registry, "ConvTranspose", conv_transpose_op, conv_transpose_claim);
    reg(registry, "AveragePool", average_pool_op, average_pool_claim);
    reg(registry, "MaxPool", max_pool_op, max_pool_claim);
    reg(registry, "GlobalAveragePool", global_average_pool_op, global_pool_claim);
    reg(registry, "GlobalMaxPool", global_max_pool_op, global_pool_claim);
    reg(registry, "LpPool", lp_pool_op, lp_pool_claim);
    reg(registry, "GlobalLpPool", global_lp_pool_op, global_lp_pool_claim);
}
