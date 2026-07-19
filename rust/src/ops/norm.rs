//! Normalization op handlers. Faithful port of the C++ `ops/norm.cc` + `ops/norm_ext.cc`:
//!
//!   * RMSNormalization (ai.onnx opset 23+)             — mlx_fast_rms_norm
//!   * LayerNormalization (ai.onnx opset 17+)           — mlx_fast_layer_norm
//!   * SimplifiedLayerNormalization (com.microsoft)     — mlx_fast_rms_norm
//!   * SkipLayerNormalization (com.microsoft)           — residual add + mlx_fast_layer_norm
//!   * SkipSimplifiedLayerNormalization (com.microsoft) — residual add + mlx_fast_rms_norm
//!   * GroupNormalization (ai.onnx opset 21 form)       — composed mean/var/rsqrt
//!   * LpNormalization (ai.onnx)                        — composed abs/sum or square/sum/sqrt
//!   * BatchNormalization (ai.onnx, inference form)     — composed per-channel affine
//!   * LRN (ai.onnx, across-channel)                    — composed square/window-sum/power/divide
//!
//! Every handler honors the resolved input dtype (fp32/fp16/bf16) with no per-dtype branching:
//! the MLX fast norms run in whatever float dtype the input carries, and the composed paths keep a
//! matching-dtype epsilon scalar so no unwanted upcast occurs.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    is_mlx_float, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};
use std::os::raw::c_char;

// ---- small local MLX helpers -------------------------------------------------------------------

/// A null-`ctx` `mlx_array` — the mlx-c "empty/absent" sentinel (e.g. an omitted layer-norm bias).
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
fn divide(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    b: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.binary(mlx::mlx_divide, a, b)
}
fn rsqrt(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_rsqrt(res, a, s) })
}
fn sqrt(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sqrt(res, a, s) })
}
fn abs(ctx: &mut TranslationContext, a: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_abs(res, a, s) })
}
fn sum_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
    keepdims: bool,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, a, axis, keepdims, s) })
}
fn mean_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
    keepdims: bool,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_mean_axis(res, a, axis, keepdims, s) })
}
fn var_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
    keepdims: bool,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_var_axis(res, a, axis, keepdims, 0, s) })
}

/// A 0-d scalar of dtype `dt` holding `v` (eps constant), matching the compute dtype so no unwanted
/// upcast occurs.
fn scalar_like(
    ctx: &mut TranslationContext,
    v: f32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(v);
    if dt == mlx::mlx_dtype__MLX_FLOAT32 {
        Ok(s)
    } else {
        ctx.astype(s, dt)
    }
}

/// Reshape a per-channel vector `[C]` to `[1, C, 1, ..., 1]` so it broadcasts over N and spatial.
fn channel_broadcast(
    ctx: &mut TranslationContext,
    v: mlx::mlx_array,
    rank: usize,
    channels: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut shape = vec![1i32; rank];
    if rank >= 2 {
        shape[1] = channels;
    }
    ctx.reshape(v, &shape)
}

fn epsilon(n: &NodeDesc, default: f32) -> f32 {
    n.floats.get("epsilon").copied().unwrap_or(default)
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

// ---- handlers ----------------------------------------------------------------------------------

/// RMSNormalization (ai.onnx opset 23+): out = rms_norm(x) * scale over the last axis.
fn rms_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let g = ctx.resolve(&n.inputs[1])?;
    let eps = epsilon(n, 1e-6);
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_fast_rms_norm(res, x, g, eps, s) })?;
    ctx.mark_fast("mlx_fast_rms_norm");
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// SimplifiedLayerNormalization (com.microsoft): Y = rms_norm(X) * scale over the last axis.
fn simplified_layer_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let eps = epsilon(n, 1e-5);
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_fast_rms_norm(res, x, scale, eps, s) })?;
    ctx.mark_fast("mlx_fast_rms_norm");
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// LayerNormalization (ai.onnx opset 17+, last-axis form): Y = layer_norm(X, scale, bias, eps).
/// Only the single-output (Y) form is claimed; Mean/InvStdDev extra outputs are left to CPU.
fn layer_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let bias = if present(n, 2) {
        ctx.resolve(&n.inputs[2])?
    } else {
        empty_array()
    };
    let eps = epsilon(n, 1e-5);
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_fast_layer_norm(res, x, scale, bias, eps, s) })?;
    ctx.mark_fast("mlx_fast_layer_norm");
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// SkipLayerNormalization (com.microsoft): residual = input + skip (+ bias);
/// Y = layer_norm(residual, gamma, beta, eps). out[0]=Y; optional out[3]=residual sum.
fn skip_layer_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let input = ctx.resolve(&n.inputs[0])?;
    let skip = ctx.resolve(&n.inputs[1])?;
    let gamma = ctx.resolve(&n.inputs[2])?;
    let beta = if present(n, 3) {
        ctx.resolve(&n.inputs[3])?
    } else {
        empty_array()
    };
    let mut residual = add(ctx, input, skip)?;
    if present(n, 4) {
        let bias = ctx.resolve(&n.inputs[4])?;
        residual = add(ctx, residual, bias)?;
    }
    let eps = epsilon(n, 1e-5);
    let r =
        ctx.emit(|res, s| unsafe { mlx::mlx_fast_layer_norm(res, residual, gamma, beta, eps, s) })?;
    ctx.mark_fast("mlx_fast_layer_norm");
    ctx.bind(&n.outputs[0], r);
    if n.outputs.len() >= 4 && !n.outputs[3].name.is_empty() {
        ctx.bind(&n.outputs[3], residual);
    }
    Ok(())
}

/// SkipSimplifiedLayerNormalization (com.microsoft): residual = input + skip;
/// out = rms_norm(residual) * gamma. out[0]=normalized, out[last]=residual.
fn skip_rms_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let input = ctx.resolve(&n.inputs[0])?;
    let skip = ctx.resolve(&n.inputs[1])?;
    let gamma = ctx.resolve(&n.inputs[2])?;
    let eps = epsilon(n, 1e-6);
    let residual = add(ctx, input, skip)?;
    let norm =
        ctx.emit(|res, s| unsafe { mlx::mlx_fast_rms_norm(res, residual, gamma, eps, s) })?;
    ctx.mark_fast("mlx_fast_rms_norm");
    ctx.bind(&n.outputs[0], norm);
    if n.outputs.len() >= 2 {
        let last = n.outputs.len() - 1;
        ctx.bind(&n.outputs[last], residual);
    }
    Ok(())
}

/// GroupNormalization (ai.onnx opset 21 form): normalize within each of `num_groups` channel groups,
/// then apply per-channel scale/bias. X=[N,C,*S], scale/bias=[C].
fn group_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let bias = ctx.resolve(&n.inputs[2])?;
    let shape = ctx.shape_of(x);
    let rank = shape.len();
    let n_dim = shape[0];
    let c = shape[1];
    let groups = n.ints.get("num_groups").copied().unwrap_or(0) as i32;
    if groups <= 0 {
        return Err("MLX GroupNormalization requires num_groups > 0".to_string());
    }
    let eps = epsilon(n, 1e-5);

    let mut per_group: i32 = 1;
    for &d in &shape[1..rank] {
        per_group *= d;
    }
    per_group /= groups; // (C/groups) * prod(spatial)

    let grp = ctx.reshape(x, &[n_dim, groups, per_group])?;
    let mean = mean_axis(ctx, grp, 2, true)?;
    let var = var_axis(ctx, grp, 2, true)?;
    let eps_s = scalar_like(ctx, eps, ctx.dtype_of(x))?;
    let var_eps = add(ctx, var, eps_s)?;
    let inv = rsqrt(ctx, var_eps)?;
    let centered = sub(ctx, grp, mean)?;
    let normed = mul(ctx, centered, inv)?;
    let normed = ctx.reshape(normed, &shape)?;

    let sb = channel_broadcast(ctx, scale, rank, c)?;
    let bb = channel_broadcast(ctx, bias, rank, c)?;
    let scaled = mul(ctx, normed, sb)?;
    let out = add(ctx, scaled, bb)?;
    ctx.mark_composed(
        "GroupNormalization composed (mean/var/rsqrt) — no fused last-axis norm kernel",
    );
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

/// LpNormalization (ai.onnx): Y = X / ||X||_p along `axis` (p in {1,2}, default 2; axis default -1).
fn lp_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(x) as i64;
    let mut axis = n.ints.get("axis").copied().unwrap_or(-1);
    if axis < 0 {
        axis += rank;
    }
    let p = n.ints.get("p").copied().unwrap_or(2);
    let axis = axis as i32;

    let norm = if p == 1 {
        let a = abs(ctx, x)?;
        sum_axis(ctx, a, axis, true)?
    } else {
        let sq = mul(ctx, x, x)?;
        let s = sum_axis(ctx, sq, axis, true)?;
        sqrt(ctx, s)?
    };
    let quot = divide(ctx, x, norm)?;
    // ONNX LpNormalization: where the norm is 0, emit 0 rather than NaN (0/0). Matches the ONNX
    // reference `np.where(norm == 0, 0, x / norm)` and ORT's CPU kernel.
    let zero = scalar_like(ctx, 0.0, ctx.dtype_of(x))?;
    let is_zero = ctx.binary(mlx::mlx_equal, norm, zero)?;
    let out = ctx.where_(is_zero, zero, quot)?;
    ctx.mark_composed("LpNormalization composed (abs/sum/sqrt/divide) — no fused norm kernel");
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

/// BatchNormalization (ai.onnx, inference/spatial form): Y = (X - mean)/sqrt(var+eps) * scale + B,
/// per channel. Only the single-output inference form is claimed.
fn batch_norm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let scale = ctx.resolve(&n.inputs[1])?;
    let b = ctx.resolve(&n.inputs[2])?;
    let mean = ctx.resolve(&n.inputs[3])?;
    let var = ctx.resolve(&n.inputs[4])?;
    let shape = ctx.shape_of(x);
    let rank = shape.len();
    let c = if rank >= 2 { shape[1] } else { shape[0] };
    let eps = epsilon(n, 1e-5);

    let eps_s = scalar_like(ctx, eps, ctx.dtype_of(x))?;
    let var_eps = add(ctx, var, eps_s)?;
    let inv = rsqrt(ctx, var_eps)?; // [C]
    let a = mul(ctx, scale, inv)?; // [C]
    let mean_a = mul(ctx, mean, a)?;
    let shift = sub(ctx, b, mean_a)?; // [C]
    let ab = channel_broadcast(ctx, a, rank, c)?;
    let shiftb = channel_broadcast(ctx, shift, rank, c)?;
    let scaled = mul(ctx, x, ab)?;
    let out = add(ctx, scaled, shiftb)?;
    ctx.mark_composed("BatchNormalization composed (rsqrt/affine) — no fused batch-norm kernel");
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- LRN helpers -------------------------------------------------------------------------------

/// Pad `a` along the channel axis (axis 1) with `low`/`high` copies of `value`.
fn pad_channel(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    low: i32,
    high: i32,
    value: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    if low == 0 && high == 0 {
        return Ok(a);
    }
    let axes = [1i32];
    let lo = [low];
    let hi = [high];
    let mode = b"constant\0";
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_pad(
            res,
            a,
            axes.as_ptr(),
            1,
            lo.as_ptr(),
            1,
            hi.as_ptr(),
            1,
            value,
            mode.as_ptr() as *const c_char,
            s,
        )
    })?;
    ctx.contiguous(out)
}

/// Slice `a` along the channel axis (axis 1) to `[lo, hi)`, keeping all other axes intact.
fn slice_channel(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    lo: i32,
    hi: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(a);
    let rank = shape.len();
    let mut start = vec![0i32; rank];
    let mut stop = shape;
    let stride = vec![1i32; rank];
    start[1] = lo;
    stop[1] = hi;
    ctx.emit(|res, s| unsafe {
        mlx::mlx_slice(
            res,
            a,
            start.as_ptr(),
            rank,
            stop.as_ptr(),
            rank,
            stride.as_ptr(),
            rank,
            s,
        )
    })
}

/// LRN (ai.onnx, across-channel): for input X=[N,C,*S],
///   square_sum[n,c,*] = sum over the `size`-wide channel window centered at c (clamped to [0,C-1]),
///   Y[n,c,*] = X[n,c,*] / (bias + (alpha/size) * square_sum[n,c,*])^beta.
/// The window sum is computed by zero-padding X^2 along the channel axis (so out-of-range channels
/// contribute 0, matching ONNX's clamped window) then summing `size` shifted channel slices.
fn lrn_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let c = shape[1];
    let size = n.ints.get("size").copied().unwrap_or(1).max(1) as i32;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1e-4);
    let beta = n.floats.get("beta").copied().unwrap_or(0.75);
    let bias = n.floats.get("bias").copied().unwrap_or(1.0);
    let dt = ctx.dtype_of(x);

    let x2 = mul(ctx, x, x)?;
    // Window [c - floor((size-1)/2), c + ceil((size-1)/2)] clamped to [0, C-1].
    let pad_before = (size - 1) / 2;
    let pad_after = size - 1 - pad_before;
    let zero = scalar_like(ctx, 0.0, dt)?;
    let xp = pad_channel(ctx, x2, pad_before, pad_after, zero)?; // channel length C + size - 1

    // square_sum[:, c] = sum_{k=0}^{size-1} xp[:, c + k]
    let mut square_sum = slice_channel(ctx, xp, 0, c)?;
    for k in 1..size {
        let s = slice_channel(ctx, xp, k, k + c)?;
        square_sum = add(ctx, square_sum, s)?;
    }

    let scale = scalar_like(ctx, alpha / size as f32, dt)?;
    let bias_s = scalar_like(ctx, bias, dt)?;
    let beta_s = scalar_like(ctx, beta, dt)?;
    let scaled = mul(ctx, square_sum, scale)?;
    let base = add(ctx, scaled, bias_s)?;
    let denom = ctx.binary(mlx::mlx_power, base, beta_s)?;
    let out = divide(ctx, x, denom)?;
    ctx.mark_composed("LRN composed (square/window-sum/power/divide) — no fused LRN kernel");
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn tensor_dtype(node: &NodeView, i: usize) -> Option<ort::ONNXTensorElementDataType> {
    node.input_info(i).map(|s| s.dtype)
}

/// RMSNormalization (ai.onnx): X, scale, axis == -1. fp32/fp16/bf16 (mlx_fast_rms_norm is generic).
fn rms_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() > 0,
        "expects 2 inputs and at least 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, g, out) = match (
        node.input_info(0),
        tensor_dtype(node, 1),
        node.output_info(0),
    ) {
        (Some(x), Some(g), Some(o)) => (x, g, o),
        _ => deny!("missing tensor type/shape info on an input or the first output"),
    };
    require!(
        is_mlx_float(x.dtype) && g == x.dtype && out.dtype == x.dtype,
        "X, scale, and output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(g),
        crate::registry::ort_dtype_name(out.dtype)
    );
    let axis = node.int_attr("axis", -1);
    require!(axis == -1, "only axis=-1 is supported (got {axis})");
    Ok(())
}

/// SimplifiedLayerNormalization (com.microsoft): X + scale, last-axis, single output.
fn simplified_layer_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, g, out) = match (
        node.input_info(0),
        tensor_dtype(node, 1),
        node.output_info(0),
    ) {
        (Some(x), Some(g), Some(o)) => (x, g, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(x.dtype) && g == x.dtype && out.dtype == x.dtype,
        "X, scale, and output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(g),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        !x.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    let axis = node.int_attr("axis", -1);
    require!(
        axis == -1 || axis == x.shape.len() as i64 - 1,
        "only the last axis is supported (got axis={axis} for rank {})",
        x.shape.len()
    );
    Ok(())
}

/// LayerNormalization: fp32/fp16/bf16 X + scale (+ optional bias), last-axis, single output (Y).
fn layer_norm_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        (2..=3).contains(&nin) && node.num_outputs() == 1,
        "expects 2-3 inputs and 1 output, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (x, scale, out) = match (
        node.input_info(0),
        tensor_dtype(node, 1),
        node.output_info(0),
    ) {
        (Some(x), Some(scale), Some(o)) => (x, scale, o),
        _ => deny!("missing tensor type/shape info on X, scale, or output"),
    };
    require!(
        is_mlx_float(x.dtype) && scale == x.dtype && out.dtype == x.dtype,
        "X, scale, and output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(scale),
        crate::registry::ort_dtype_name(out.dtype)
    );
    if nin == 3 && node.input_present(2) {
        let bias = match tensor_dtype(node, 2) {
            Some(bias) => bias,
            None => deny!("missing tensor type/shape info on bias"),
        };
        require!(
            bias == x.dtype,
            "bias must match X dtype {}, got {}",
            crate::registry::ort_dtype_name(x.dtype),
            crate::registry::ort_dtype_name(bias)
        );
    }
    let rank = x.shape.len() as i64;
    require!(rank > 0, "input must have rank >= 1 (got a scalar)");
    let axis = node.int_attr("axis", -1);
    require!(
        axis == -1 || axis == rank - 1,
        "only the last axis is supported (got axis={axis} for rank {rank})"
    );
    Ok(())
}

/// SkipSimplifiedLayerNormalization (com.microsoft): input, skip, gamma. fp32/fp16/bf16. 3-input.
fn skip_rms_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() > 0,
        "expects 3 inputs and at least 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on X or the first output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "X and output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    // The handler produces only out[0] (normalized) and the optional out[last] (residual sum);
    // reject if mean (out[1]) or inv-std (out[2]) are requested — mlx_fast_rms_norm doesn't compute
    // them, so claiming would leave those outputs unbound (mirrors skip_layer_norm_claim).
    // mean (out[1]) / inv-std (out[2]) are optional diagnostic outputs the RMS handler does not
    // produce. When the graph actually CONSUMES them they become fused-subgraph boundary outputs and
    // ORT would flag them unbound; when they are declared-but-unused (the common transformers.js /
    // Mobius export case) they are never boundary outputs, so MLX simply DCEs them. We therefore
    // accept them here and let the framework's unused-output elision handle it — a consuming model
    // fails loudly (unbound output), never silently wrong.
    for (i, name) in [(1, "skip"), (2, "gamma")] {
        let dtype = match tensor_dtype(node, i) {
            Some(dtype) => dtype,
            None => deny!("missing tensor type/shape info on {name} input"),
        };
        require!(
            dtype == x.dtype,
            "{name} must match X dtype {}, got {}",
            crate::registry::ort_dtype_name(x.dtype),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    Ok(())
}

/// SkipLayerNormalization: input, skip, gamma (+ optional beta, bias), all same float dtype.
/// Only out[0] (Y) and optional out[3] (residual sum) are produced; mean/inv-std outputs → CPU.
fn skip_layer_norm_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        (3..=5).contains(&nin) && node.num_outputs() > 0,
        "expects 3-5 inputs and at least 1 output, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on X or the first output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "X and output must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    for i in 1..nin {
        if !node.input_present(i) {
            continue;
        }
        let dtype = match tensor_dtype(node, i) {
            Some(dtype) => dtype,
            None => deny!("missing tensor type/shape info on input {i}"),
        };
        require!(
            dtype == x.dtype,
            "input {i} must match X dtype {}, got {}",
            crate::registry::ort_dtype_name(x.dtype),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    // Reject if mean (out[1]) or inv-std (out[2]) are requested — we do not compute them.
    // mean (out[1]) / inv-std (out[2]) are optional diagnostic outputs the RMS handler does not
    // produce. When the graph actually CONSUMES them they become fused-subgraph boundary outputs and
    // ORT would flag them unbound; when they are declared-but-unused (the common transformers.js /
    // Mobius export case) they are never boundary outputs, so MLX simply DCEs them. We therefore
    // accept them here and let the framework's unused-output elision handle it — a consuming model
    // fails loudly (unbound output), never silently wrong.
    Ok(())
}

/// GroupNormalization: X=[N,C,*S] float, scale/bias=[C], static C divisible by num_groups.
fn group_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, scale, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(scale), Some(o)) => (x, scale, o),
        _ => deny!("missing tensor type/shape info on X, scale, or output"),
    };
    let bias = match tensor_dtype(node, 2) {
        Some(b) => b,
        None => deny!("missing tensor type/shape info on bias"),
    };
    require!(
        is_mlx_float(x.dtype)
            && scale.dtype == x.dtype
            && bias == x.dtype
            && out.dtype == x.dtype,
        "X, scale, bias, and output must share one float dtype (fp32/fp16/bf16), got {}, {}, {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(scale.dtype),
        crate::registry::ort_dtype_name(bias),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 2,
        "input must have rank >= 2 (got rank {})",
        x.shape.len()
    );
    let c = x.shape[1];
    require!(
        c > 0,
        "channel dimension must be static and positive (got {c})"
    );
    // Batch and spatial dims may be dynamic (-1): the group reshape only needs a
    // static channel count, and the translator builds the reshape from the
    // concrete trace-time shape (same as Conv, which is claimed on dynamic-spatial
    // UNets). Real diffusion UNets export with dynamic N/H/W, so requiring all
    // dims static here needlessly forced every GroupNorm onto CPU and fragmented
    // the graph. Only the channel dim must be known to split into groups.
    let groups = node.int_attr("num_groups", 0);
    require!(
        groups > 0 && c % groups == 0,
        "num_groups must be positive and divide channel count {c} (got {groups})"
    );
    // opset-21 per-channel scale/bias: shape [C].
    require!(
        scale.shape.len() == 1 && scale.shape[0] == c,
        "opset-21 scale must have shape [C]=[{c}] (got {:?})",
        scale.shape
    );
    Ok(())
}

/// LpNormalization: single float input/output, p in {1,2}.
fn lp_norm_claim(node: &NodeView) -> ClaimResult {
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
        "input/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        !x.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    let p = node.int_attr("p", 2);
    require!(p == 1 || p == 2, "only p=1 or p=2 is supported (got {p})");
    Ok(())
}

/// BatchNormalization: inference (single-output) form, 5 float inputs sharing X's dtype.
fn batch_norm_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 5 && node.num_outputs() == 1,
        "inference form expects 5 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on X or output"),
    };
    require!(
        is_mlx_float(x.dtype) && out.dtype == x.dtype,
        "X/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 2,
        "input must have rank >= 2 (got rank {})",
        x.shape.len()
    );
    for i in 1..5 {
        let dtype = match tensor_dtype(node, i) {
            Some(dtype) => dtype,
            None => deny!("missing tensor type/shape info on input {i}"),
        };
        require!(
            dtype == x.dtype,
            "input {i} must match X dtype {}, got {}",
            crate::registry::ort_dtype_name(x.dtype),
            crate::registry::ort_dtype_name(dtype)
        );
    }
    let training_mode = node.int_attr("training_mode", 0);
    require!(
        training_mode == 0,
        "training_mode must be 0; training outputs are unsupported (got {training_mode})"
    );
    Ok(())
}

/// LRN (ai.onnx, across-channel): single float input/output of equal dtype, static shape with a
/// channel axis, and a valid window `size >= 1`.
fn lrn_claim(node: &NodeView) -> ClaimResult {
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
        "input/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        x.shape.len() >= 2,
        "input must have rank >= 2 (got rank {})",
        x.shape.len()
    );
    require!(
        x.shape.iter().all(|&d| d > 0),
        "all input dimensions must be static and positive to build channel pad/slice (got {:?})",
        x.shape
    );
    let size = node.int_attr("size", 0);
    require!(size >= 1, "window size must be >= 1 (got {size})");
    Ok(())
}

// ---- registration ------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn reg(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    min_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain,
        op_type,
        min_opset,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register_norm(registry: &mut OpRegistry) {
    // RMSNormalization entered ai.onnx at opset 23.
    reg(
        registry,
        "",
        "RMSNormalization",
        23,
        rms_norm_op,
        rms_norm_claim,
    );
    // LayerNormalization entered ai.onnx at opset 17.
    reg(
        registry,
        "",
        "LayerNormalization",
        17,
        layer_norm_op,
        layer_norm_claim,
    );
    reg(
        registry,
        "",
        "GroupNormalization",
        K_ANY_OPSET,
        group_norm_op,
        group_norm_claim,
    );
    reg(
        registry,
        "",
        "LpNormalization",
        K_ANY_OPSET,
        lp_norm_op,
        lp_norm_claim,
    );
    reg(
        registry,
        "",
        "BatchNormalization",
        K_ANY_OPSET,
        batch_norm_op,
        batch_norm_claim,
    );
    reg(registry, "", "LRN", K_ANY_OPSET, lrn_op, lrn_claim);
    reg(
        registry,
        "com.microsoft",
        "SimplifiedLayerNormalization",
        K_ANY_OPSET,
        simplified_layer_norm_op,
        simplified_layer_norm_claim,
    );
    // SPECIAL CASE / UPSTREAM BUG WORKAROUND: SimplifiedLayerNormalization is a `com.microsoft`
    // CONTRIB op, but Microsoft's exporter accidentally stamps some graphs with it in the DEFAULT
    // ONNX domain (domain "") instead of com.microsoft (seen in gemma-3n / RoBERTa ONNX from
    // onnx-community). It is the exact same op with the exact same semantics
    // (Y = rms_norm(X) * scale over the last axis), so we register it under "" as well — otherwise
    // every one of these norms (113 in the gemma-4-E2B vision encoder) fragments the graph onto CPU.
    // This is NOT a real default-domain op; it only exists there because of that mis-stamping.
    reg(
        registry,
        "",
        "SimplifiedLayerNormalization",
        K_ANY_OPSET,
        simplified_layer_norm_op,
        simplified_layer_norm_claim,
    );
    reg(
        registry,
        "com.microsoft",
        "SkipLayerNormalization",
        K_ANY_OPSET,
        skip_layer_norm_op,
        skip_layer_norm_claim,
    );
    reg(
        registry,
        "com.microsoft",
        "SkipSimplifiedLayerNormalization",
        K_ANY_OPSET,
        skip_rms_norm_op,
        skip_rms_norm_claim,
    );
}
