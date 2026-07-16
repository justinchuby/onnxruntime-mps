//! State-space / KV-cache "misc" op handlers. Faithful port of the C++ `ops/ssm_misc.cc`:
//!   * TensorScatter (ai.onnx) — static-KV-cache scatter ("linear" mode) via slice_update /
//!     slice_update_dynamic.
//!   * CausalConvWithState (com.microsoft) — fused causal depthwise conv1d with carry state.
//!   * LinearAttention (com.microsoft) — chunked/recurrent linear attention (4 update rules,
//!     GQA) via static-length unrolling over the time axis T.
//! Only statically translatable, MLX-supported forms are claimed; the rest fall to ORT CPU.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    is_mlx_float, ClaimPredicate, ClaimResult, NodeView, OpHandler, OpRegistration, OpRegistry,
    K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

const T_INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
const I32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_INT32;

// ---- shared helpers -----------------------------------------------------------------------------

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent && !n.inputs[i].name.is_empty()
}

fn str_attr(n: &NodeDesc, name: &str, dflt: &str) -> String {
    n.strings
        .get(name)
        .cloned()
        .unwrap_or_else(|| dflt.to_string())
}

fn norm_axis(axis: i64, rank: i32) -> i32 {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    a as i32
}

/// True when an optional input is omitted in the MIDDLE of the input list (an interior gap), which
/// the shared clustering pass cannot represent; such forms are left to ORT CPU.
fn has_interior_gap(node: &NodeView) -> bool {
    let n = node.num_inputs();
    let mut last_present = 0usize;
    let mut seen = false;
    for i in 0..n {
        if node.input_present(i) {
            last_present = i;
            seen = true;
        }
    }
    if !seen {
        return false;
    }
    (0..last_present).any(|i| !node.input_present(i))
}

// =============================================================================================
// TensorScatter (ai.onnx) — write `update` into `past_cache` along `axis` at write_indices[0]/0.
// =============================================================================================
fn tensor_scatter_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let past = ctx.resolve(&n.inputs[0])?;
    let update = ctx.resolve(&n.inputs[1])?;
    let rank = ctx.ndim(past) as i32;
    let axis = norm_axis(n.ints.get("axis").copied().unwrap_or(-2), rank);

    let present_arr = if present(n, 2) {
        // Decode form: dynamic offset from write_indices along `axis` (batch_size == 1 per claim).
        let wi = ctx.resolve(&n.inputs[2])?;
        let wi = ctx.astype(wi, I32)?;
        let axes = [axis];
        ctx.emit(|res, s| unsafe {
            mlx::mlx_slice_update_dynamic(res, past, update, wi, axes.as_ptr(), axes.len(), s)
        })?
    } else {
        // Prefill form: write the update block at offset 0 along `axis` for every batch.
        let start = vec![0i32; rank as usize];
        let mut stop = vec![0i32; rank as usize];
        for i in 0..rank {
            stop[i as usize] = ctx.dim(past, i);
        }
        stop[axis as usize] = ctx.dim(update, axis);
        let strides = vec![1i32; rank as usize];
        ctx.emit(|res, s| unsafe {
            mlx::mlx_slice_update(
                res,
                past,
                update,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                s,
            )
        })?
    };
    let cont = ctx.contiguous(present_arr)?;
    ctx.bind(&n.outputs[0], cont);
    Ok(())
}

fn tensor_scatter_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        (ni == 2 || ni == 3) && node.num_outputs() == 1,
        "expects 2-3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let mode = node.string_attr("mode", "linear");
    require!(mode == "linear", "mode must be \"linear\", got {mode:?}");
    let (past, update, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(past.dtype) && update.dtype == past.dtype && out.dtype == past.dtype,
        "past/update/output must share one float dtype, got {} / {} -> {}",
        crate::registry::ort_dtype_name(past.dtype),
        crate::registry::ort_dtype_name(update.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    if ni == 3 && node.input_present(2) {
        match node.input_info(2) {
            Some(wi) if wi.dtype == T_INT64 => {}
            Some(wi) => deny!(
                "write_indices must be int64, got {}",
                crate::registry::ort_dtype_name(wi.dtype)
            ),
            None => deny!("missing tensor type/shape info on write_indices"),
        }
        // Only batch_size == 1 is expressible as one dynamic slice.
        require!(
            !past.shape.is_empty() && past.shape[0] == 1,
            "dynamic TensorScatter requires past_cache batch size 1, got shape {:?}",
            past.shape
        );
    }
    Ok(())
}

// =============================================================================================
// CausalConvWithState (com.microsoft) — stateful causal depthwise conv1d.
// =============================================================================================
fn causal_conv_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?; // (B, C, L)
    let weight = ctx.resolve(&n.inputs[1])?; // (C, 1, k)
    let dt = ctx.dtype_of(x);
    let b = ctx.dim(x, 0);
    let c = ctx.dim(x, 1);
    let k = ctx.dim(weight, 2);

    let has_bias = present(n, 2);
    let has_state = present(n, 3);

    // Left context: past_state (B, C, k-1) or k-1 zeros. For k == 1 there is no carry state.
    let mut x_pad = x;
    if k > 1 {
        let state = if has_state {
            ctx.resolve(&n.inputs[3])?
        } else {
            ctx.zeros(&[b, c, k - 1], dt)?
        };
        x_pad = ctx.concat2(state, x, 2)?; // (B, C, k-1+L)
    }

    // present_state = last k-1 columns of x_pad (boundary output -> contiguous).
    if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        if k > 1 {
            let padded = ctx.dim(x_pad, 2);
            let ps = ctx.slice(x_pad, &[0, 0, padded - (k - 1)], &[b, c, padded])?;
            let ps = ctx.contiguous(ps)?;
            ctx.bind(&n.outputs[1], ps);
        } else {
            let z = ctx.zeros(&[b, c, 0], dt)?;
            ctx.bind(&n.outputs[1], z);
        }
    }

    // Depthwise conv1d: MLX uses NLC data and (C_out, kernel, C_in/groups) weights.
    let x_t = ctx.transpose(x_pad, &[0, 2, 1])?;
    let x_nlc = ctx.contiguous(x_t)?; // (B, k-1+L, C)
    let w_t = ctx.transpose(weight, &[0, 2, 1])?;
    let w_ckc = ctx.contiguous(w_t)?; // (C, k, 1)
    let y_nlc = ctx.emit(|res, s| unsafe { mlx::mlx_conv1d(res, x_nlc, w_ckc, 1, 0, 1, c, s) })?;
    let y_t = ctx.transpose(y_nlc, &[0, 2, 1])?;
    let mut y = ctx.contiguous(y_t)?; // (B, C, L)

    if has_bias {
        let bias = ctx.resolve(&n.inputs[2])?; // (C,)
        let b3 = ctx.reshape(bias, &[1, c, 1])?;
        y = ctx.add(y, b3)?;
    }

    let activation = str_attr(n, "activation", "none");
    if activation == "silu" || activation == "swish" {
        let sig = ctx.emit(|res, s| unsafe { mlx::mlx_sigmoid(res, y, s) })?;
        y = ctx.mul(y, sig)?;
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn causal_conv_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(ni >= 2 && ni <= 4, "expects 2-4 inputs, got {ni}");
    let no = node.num_outputs();
    require!(no >= 1 && no <= 2, "expects 1-2 outputs, got {no}");
    require!(
        !has_interior_gap(node),
        "optional inputs may only be omitted from the trailing end"
    );
    let (input, weight) = match (node.input_info(0), node.input_info(1)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or weight"),
    };
    require!(
        is_mlx_float(input.dtype) && weight.dtype == input.dtype,
        "input/weight must share one float dtype, got {} / {}",
        crate::registry::ort_dtype_name(input.dtype),
        crate::registry::ort_dtype_name(weight.dtype)
    );
    require!(
        input.shape.len() == 3 && weight.shape.len() == 3,
        "input and weight must both be rank 3, got ranks {} and {}",
        input.shape.len(),
        weight.shape.len()
    );
    if node.input_present(2) {
        match node.input_info(2) {
            Some(b) if b.dtype == input.dtype => {}
            Some(b) => deny!(
                "bias dtype must match input dtype {}, got {}",
                crate::registry::ort_dtype_name(input.dtype),
                crate::registry::ort_dtype_name(b.dtype)
            ),
            None => deny!("missing tensor type/shape info on bias"),
        }
    }
    if node.input_present(3) {
        match node.input_info(3) {
            Some(p) if p.dtype == input.dtype => {}
            Some(p) => deny!(
                "past_state dtype must match input dtype {}, got {}",
                crate::registry::ort_dtype_name(input.dtype),
                crate::registry::ort_dtype_name(p.dtype)
            ),
            None => deny!("missing tensor type/shape info on past_state"),
        }
    }
    let activation = node.string_attr("activation", "none");
    require!(
        activation == "none" || activation == "silu" || activation == "swish",
        "activation must be \"none\", \"silu\", or \"swish\", got {activation:?}"
    );
    Ok(())
}

// =============================================================================================
// LinearAttention (com.microsoft) — delta-rule linear attention, static-length unroll over T.
// =============================================================================================
fn rule_uses_decay(rule: &str) -> bool {
    rule == "gated" || rule == "gated_delta"
}
fn rule_uses_beta(rule: &str) -> bool {
    rule == "delta" || rule == "gated_delta"
}
fn is_known_rule(rule: &str) -> bool {
    matches!(rule, "linear" | "gated" | "delta" | "gated_delta")
}

fn la_scalar(
    ctx: &mut TranslationContext,
    value: f32,
    dt: mlx::mlx_dtype,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(value);
    if dt == mlx::mlx_dtype__MLX_FLOAT32 {
        Ok(s)
    } else {
        ctx.astype(s, dt)
    }
}

/// From a (B, H, T, X) tensor pick time-step `t` as a (B, H, X) slab.
fn time_slab(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    t: i32,
    b: i32,
    h: i32,
    x: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.slice(a, &[0, 0, t, 0], &[b, h, t + 1, x])?;
    ctx.reshape(s, &[b, h, x])
}

/// From a (B, H, T) tensor pick time-step `t` as a (B, H) slab.
fn time_slab2(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    t: i32,
    b: i32,
    h: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.slice(a, &[0, 0, t], &[b, h, t + 1])?;
    ctx.reshape(s, &[b, h])
}

fn repeat_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    repeats: i32,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    if repeats == 1 {
        return Ok(a);
    }
    ctx.emit(|res, s| unsafe { mlx::mlx_repeat_axis(res, a, repeats, axis, s) })
}

fn linear_attention_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let rule = str_attr(n, "update_rule", "gated_delta");
    let uses_decay = rule_uses_decay(&rule);
    let uses_beta = rule_uses_beta(&rule);
    let hq = *n
        .ints
        .get("q_num_heads")
        .ok_or("MLX LinearAttention: q_num_heads missing")? as i32;
    let h = *n
        .ints
        .get("kv_num_heads")
        .ok_or("MLX LinearAttention: kv_num_heads missing")? as i32;
    let gqa = h / hq;

    let query = ctx.resolve(&n.inputs[0])?; // (B, T, Hq*d_k)
    let key = ctx.resolve(&n.inputs[1])?; // (B, T, Hq*d_k)
    let value = ctx.resolve(&n.inputs[2])?; // (B, T, H*d_v)
    let dt = ctx.dtype_of(query);

    let qsh = ctx.shape_of(query);
    let vsh = ctx.shape_of(value);
    let b = qsh[0];
    let t_len = qsh[1];
    let d_k = qsh[2] / hq;
    let d_v = vsh[2] / h;

    let scale_attr = n.floats.get("scale").copied().unwrap_or(0.0);
    let scale = if scale_attr != 0.0 {
        scale_attr
    } else {
        1.0 / (d_k as f32).sqrt()
    };

    let has_past = present(n, 3);
    let mut state = if has_past {
        ctx.resolve(&n.inputs[3])?
    } else {
        ctx.zeros(&[b, h, d_k, d_v], dt)?
    };

    // Zero-length time axis: no steps run. output empty; present_state == state.
    if t_len == 0 {
        if !n.outputs.is_empty() && !n.outputs[0].name.is_empty() {
            let z = ctx.zeros(&[b, 0, h * d_v], dt)?;
            ctx.bind(&n.outputs[0], z);
        }
        if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
            let ps = ctx.contiguous(state)?;
            ctx.bind(&n.outputs[1], ps);
        }
        return Ok(());
    }

    // 3D -> 4D (B, H, T, ·): reshape by head count, transpose heads before T, tile Q/K for GQA.
    let to_heads = |ctx: &mut TranslationContext,
                    a: mlx::mlx_array,
                    heads: i32,
                    last: i32|
     -> Result<mlx::mlx_array, MlxError> {
        let r = ctx.reshape(a, &[b, t_len, heads, last])?;
        ctx.transpose(r, &[0, 2, 1, 3])
    };
    let q_heads = to_heads(ctx, query, hq, d_k)?;
    let q4 = repeat_axis(ctx, q_heads, gqa, 1)?; // (B, H, T, d_k)
    let k_heads = to_heads(ctx, key, hq, d_k)?;
    let k4 = repeat_axis(ctx, k_heads, gqa, 1)?; // (B, H, T, d_k)
    let v4 = to_heads(ctx, value, h, d_v)?; // (B, H, T, d_v)
    let scale_s = la_scalar(ctx, scale, dt)?;
    let q4 = ctx.mul(q4, scale_s)?; // scaled query

    let decay4 = if uses_decay {
        let d = ctx.resolve(&n.inputs[4])?;
        Some(to_heads(ctx, d, h, d_k)?)
    } else {
        None
    };
    let beta3 = if uses_beta {
        let bta = ctx.resolve(&n.inputs[5])?;
        Some(ctx.transpose(bta, &[0, 2, 1])?)
    } else {
        None
    };

    let mut outs: Vec<mlx::mlx_array> = Vec::with_capacity(t_len as usize);
    for t in 0..t_len {
        if let Some(decay4) = decay4 {
            let slab = time_slab(ctx, decay4, t, b, h, d_k)?; // (B, H, d_k)
            let g = ctx.emit(|res, s| unsafe { mlx::mlx_exp(res, slab, s) })?;
            let g = ctx.expand_dims(g, 3)?; // (B,H,d_k,1)
            state = ctx.mul(state, g)?;
        }
        let k_t = time_slab(ctx, k4, t, b, h, d_k)?; // (B, H, d_k)
                                                     // retrieval = squeeze(k_row @ state)
        let k_row = ctx.expand_dims(k_t, 2)?; // (B,H,1,d_k)
        let retrieval_m = ctx.matmul(k_row, state)?; // (B,H,1,d_v)
        let retrieval = ctx.squeeze(retrieval_m, 2)?; // (B,H,d_v)

        let v_t = time_slab(ctx, v4, t, b, h, d_v)?; // (B, H, d_v)
        let delta = if let Some(beta3) = beta3 {
            let beta_t = time_slab2(ctx, beta3, t, b, h)?; // (B, H)
            let diff = ctx.sub(v_t, retrieval)?;
            let beta_e = ctx.expand_dims(beta_t, 2)?; // (B,H,1)
            ctx.mul(diff, beta_e)?
        } else {
            v_t
        };
        // outer = k_col @ delta_row: (B,H,d_k,1) @ (B,H,1,d_v) -> (B,H,d_k,d_v)
        let k_col = ctx.expand_dims(k_t, 3)?;
        let delta_row = ctx.expand_dims(delta, 2)?;
        let outer = ctx.matmul(k_col, delta_row)?;
        state = ctx.add(state, outer)?;

        let q_t = time_slab(ctx, q4, t, b, h, d_k)?; // (B, H, d_k)
        let q_row = ctx.expand_dims(q_t, 2)?; // (B,H,1,d_k)
        let out_m = ctx.matmul(q_row, state)?; // (B,H,1,d_v)
        let out_t = ctx.squeeze(out_m, 2)?; // (B,H,d_v)
        outs.push(out_t);
    }

    if !n.outputs.is_empty() && !n.outputs[0].name.is_empty() {
        // Assemble output (B, T, H*d_v): each step's (B, H, d_v) reshapes to (B, 1, H*d_v).
        let mut out = ctx.reshape(outs[0], &[b, 1, h * d_v])?;
        for t in 1..t_len as usize {
            let slab = ctx.reshape(outs[t], &[b, 1, h * d_v])?;
            out = ctx.concat2(out, slab, 1)?;
        }
        let out = ctx.contiguous(out)?;
        ctx.bind(&n.outputs[0], out);
    }
    if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        let ps = ctx.contiguous(state)?;
        ctx.bind(&n.outputs[1], ps);
    }
    Ok(())
}

fn linear_attention_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 3 && node.num_outputs() >= 1,
        "expects at least 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let rule = node.string_attr("update_rule", "gated_delta");
    require!(is_known_rule(&rule), "unsupported update_rule {rule:?}");
    let hq = node.int_attr("q_num_heads", 0);
    let h = node.int_attr("kv_num_heads", 0);
    require!(
        hq > 0 && h > 0 && h % hq == 0,
        "q_num_heads and kv_num_heads must be positive with kv divisible by q, got q={hq}, kv={h}"
    );
    let (q, k, v) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => deny!("missing tensor type/shape info on query, key, or value"),
    };
    require!(
        is_mlx_float(q.dtype) && k.dtype == q.dtype && v.dtype == q.dtype,
        "query/key/value must share one float dtype, got {} / {} / {}",
        crate::registry::ort_dtype_name(q.dtype),
        crate::registry::ort_dtype_name(k.dtype),
        crate::registry::ort_dtype_name(v.dtype)
    );
    require!(
        q.shape.len() == 3,
        "query must be rank 3, got shape {:?}",
        q.shape
    );
    require!(
        q.shape[1] >= 0,
        "query time dimension must be static, got shape {:?}",
        q.shape
    );
    let float_ok = |i: usize| -> bool {
        if !node.input_present(i) {
            return true;
        }
        matches!(node.input_info(i), Some(info) if info.dtype == q.dtype)
    };
    require!(
        float_ok(3),
        "past_state dtype must match query dtype {}",
        crate::registry::ort_dtype_name(q.dtype)
    );
    require!(
        float_ok(4),
        "decay dtype must match query dtype {}",
        crate::registry::ort_dtype_name(q.dtype)
    );
    require!(
        float_ok(5),
        "beta dtype must match query dtype {}",
        crate::registry::ort_dtype_name(q.dtype)
    );
    require!(
        !rule_uses_decay(&rule) || node.input_present(4),
        "update_rule {rule:?} requires the decay input"
    );
    require!(
        !rule_uses_beta(&rule) || node.input_present(5),
        "update_rule {rule:?} requires the beta input"
    );
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "TensorScatter",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: tensor_scatter_op as OpHandler,
        claim: tensor_scatter_claim as ClaimPredicate,
    });
    registry.register(OpRegistration {
        domain: "com.microsoft",
        op_type: "CausalConvWithState",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: causal_conv_op as OpHandler,
        claim: causal_conv_claim as ClaimPredicate,
    });
    registry.register(OpRegistration {
        domain: "com.microsoft",
        op_type: "LinearAttention",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: linear_attention_op as OpHandler,
        claim: linear_attention_claim as ClaimPredicate,
    });
}
