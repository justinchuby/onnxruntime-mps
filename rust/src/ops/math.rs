//! Math / activation op handlers (unary + binary elementwise beyond the core set). Port of the
//! wave-1 subset of the C++ `ops/math.cc`.

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, TranslationContext};
use crate::registry::{
    is_mlx_float, is_signed_integer, scalar_or_suffix_broadcast, ClaimResult, K_ANY_OPSET,
    NodeView, OpRegistration, OpRegistry,
};
use crate::sys::mlx;
use crate::{deny, require};

// ---- handlers -----------------------------------------------------------------------------------

fn div_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_divide, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// Pow: `base ** exp`. ONNX allows a differently-typed exponent (output keeps the base dtype), so we
/// cast the exponent up to the base dtype before `mlx_power`. Only float bases are claimed (see
/// `pow_claim`), which lets the EP serve type/opset combinations ORT's CPU kernel does not implement
/// (e.g. `float32 ** uint32`, legacy opset-6 `Pow-1`).
fn pow_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let mut b = ctx.resolve(&n.inputs[1])?;
    let at = ctx.dtype_of(a);
    if ctx.dtype_of(b) != at {
        b = ctx.astype(b, at)?;
    }
    let r = ctx.binary(mlx::mlx_power, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn relu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let zero = ctx.zeros_like(x)?;
    let r = ctx.binary(mlx::mlx_maximum, x, zero)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

macro_rules! unary_handler {
    ($name:ident, $mlx_op:expr) => {
        fn $name(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
            let x = ctx.resolve(&n.inputs[0])?;
            let r = ctx.unary($mlx_op, x)?;
            ctx.bind(&n.outputs[0], r);
            Ok(())
        }
    };
}

unary_handler!(tanh_op, mlx::mlx_tanh);
unary_handler!(exp_op, mlx::mlx_exp);
unary_handler!(log_op, mlx::mlx_log);
unary_handler!(sqrt_op, mlx::mlx_sqrt);
unary_handler!(neg_op, mlx::mlx_negative);
unary_handler!(abs_op, mlx::mlx_abs);

// Unary math / rounding / trig — each is a direct mlx-c primitive (dtype-preserving).
unary_handler!(sign_op, mlx::mlx_sign);
unary_handler!(reciprocal_op, mlx::mlx_reciprocal);
unary_handler!(ceil_op, mlx::mlx_ceil);
unary_handler!(floor_op, mlx::mlx_floor);
unary_handler!(erf_op, mlx::mlx_erf);
unary_handler!(sin_op, mlx::mlx_sin);
unary_handler!(cos_op, mlx::mlx_cos);
unary_handler!(tan_op, mlx::mlx_tan);
unary_handler!(sinh_op, mlx::mlx_sinh);
unary_handler!(cosh_op, mlx::mlx_cosh);
unary_handler!(asin_op, mlx::mlx_arcsin);
unary_handler!(acos_op, mlx::mlx_arccos);
unary_handler!(atan_op, mlx::mlx_arctan);

/// ONNX `Round` rounds halves to even (banker's rounding), which is exactly `mlx_round(x, 0)`.
fn round_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_round(res, x, 0, s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- composite activation handlers --------------------------------------------------------------

/// A kept scalar of value `v` cast to the same dtype as `x` (prevents MLX float-widening, which would
/// corrupt an fp16/bf16 output's byte width at CopyOut).
fn scalar_like(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    v: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let dt = ctx.dtype_of(x);
    let s = ctx.scalar_f32(v);
    ctx.astype(s, dt)
}

/// Cast the result back to the declared ONNX output dtype (a no-op when it already matches) so a
/// stray MLX promotion never widens the boundary tensor.
fn bind_as_out(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    r: mlx::mlx_array,
) -> Result<(), MlxError> {
    let r = ctx.astype(r, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// LeakyRelu: `x>0 ? x : alpha*x`, computed branch-free as `max(x,0) + alpha*min(x,0)` (correct for
/// any alpha, positive or negative).
fn leaky_relu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(0.01);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let pos = ctx.binary(mlx::mlx_maximum, x, zero)?;
    let negpart = ctx.binary(mlx::mlx_minimum, x, zero)?;
    let neg = ctx.binary(mlx::mlx_multiply, alpha_s, negpart)?;
    let r = ctx.binary(mlx::mlx_add, pos, neg)?;
    bind_as_out(ctx, n, r)
}

/// Elu: `x>0 ? x : alpha*(exp(x)-1)` via `expm1` and `where`.
fn elu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let cond = ctx.binary(mlx::mlx_greater, x, zero)?;
    let ex = ctx.unary(mlx::mlx_expm1, x)?;
    let neg = ctx.binary(mlx::mlx_multiply, alpha_s, ex)?;
    let r = ctx.where_(cond, x, neg)?;
    bind_as_out(ctx, n, r)
}

/// Selu: `gamma * (x>0 ? x : alpha*(exp(x)-1))`.
fn selu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.673_263_2);
    let gamma = n.floats.get("gamma").copied().unwrap_or(1.050_701);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let gamma_s = scalar_like(ctx, x, gamma)?;
    let cond = ctx.binary(mlx::mlx_greater, x, zero)?;
    let ex = ctx.unary(mlx::mlx_expm1, x)?;
    let neg = ctx.binary(mlx::mlx_multiply, alpha_s, ex)?;
    let sel = ctx.where_(cond, x, neg)?;
    let r = ctx.binary(mlx::mlx_multiply, gamma_s, sel)?;
    bind_as_out(ctx, n, r)
}

/// Celu: `max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
fn celu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let inv_alpha = scalar_like(ctx, x, 1.0 / alpha)?;
    let scaled = ctx.binary(mlx::mlx_multiply, x, inv_alpha)?;
    let ex = ctx.unary(mlx::mlx_expm1, scaled)?;
    let neg_inner = ctx.binary(mlx::mlx_multiply, alpha_s, ex)?;
    let pos = ctx.binary(mlx::mlx_maximum, x, zero)?;
    let neg = ctx.binary(mlx::mlx_minimum, zero, neg_inner)?;
    let r = ctx.binary(mlx::mlx_add, pos, neg)?;
    bind_as_out(ctx, n, r)
}

/// HardSigmoid: `clip(alpha*x + beta, 0, 1)`.
fn hard_sigmoid_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(0.2);
    let beta = n.floats.get("beta").copied().unwrap_or(0.5);
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let beta_s = scalar_like(ctx, x, beta)?;
    let zero = scalar_like(ctx, x, 0.0)?;
    let one = scalar_like(ctx, x, 1.0)?;
    let ax = ctx.binary(mlx::mlx_multiply, x, alpha_s)?;
    let t = ctx.binary(mlx::mlx_add, ax, beta_s)?;
    let lo = ctx.binary(mlx::mlx_maximum, t, zero)?;
    let r = ctx.binary(mlx::mlx_minimum, lo, one)?;
    bind_as_out(ctx, n, r)
}

/// ThresholdedRelu: `x > alpha ? x : 0`.
fn thresholded_relu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let zero = scalar_like(ctx, x, 0.0)?;
    let cond = ctx.binary(mlx::mlx_greater, x, alpha_s)?;
    let r = ctx.where_(cond, x, zero)?;
    bind_as_out(ctx, n, r)
}

/// Softplus: `log(1 + exp(x))`, computed stably as `logaddexp(0, x)`.
fn softplus_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let zero = ctx.zeros_like(x)?;
    let r = ctx.binary(mlx::mlx_logaddexp, zero, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// Softsign: `x / (1 + |x|)`.
fn softsign_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let one = scalar_like(ctx, x, 1.0)?;
    let ax = ctx.unary(mlx::mlx_abs, x)?;
    let denom = ctx.binary(mlx::mlx_add, one, ax)?;
    let r = ctx.binary(mlx::mlx_divide, x, denom)?;
    bind_as_out(ctx, n, r)
}

/// Gelu (`approximate` = `none` | `tanh`).
///   none: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///   tanh: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
fn gelu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let half = scalar_like(ctx, x, 0.5)?;
    let one = scalar_like(ctx, x, 1.0)?;
    let approximate = n
        .strings
        .get("approximate")
        .map(String::as_str)
        .unwrap_or("none");
    let gate = if approximate == "tanh" {
        let c0 = scalar_like(ctx, x, 0.797_884_56)?; // sqrt(2/pi)
        let c1 = scalar_like(ctx, x, 0.044_715)?;
        let x2 = ctx.binary(mlx::mlx_multiply, x, x)?;
        let x3 = ctx.binary(mlx::mlx_multiply, x2, x)?;
        let c1x3 = ctx.binary(mlx::mlx_multiply, c1, x3)?;
        let inner_sum = ctx.binary(mlx::mlx_add, x, c1x3)?;
        let inner = ctx.binary(mlx::mlx_multiply, c0, inner_sum)?;
        let t = ctx.unary(mlx::mlx_tanh, inner)?;
        ctx.binary(mlx::mlx_add, one, t)?
    } else {
        let inv_sqrt2 = scalar_like(ctx, x, 0.707_106_77)?; // 1/sqrt(2)
        let scaled = ctx.binary(mlx::mlx_multiply, x, inv_sqrt2)?;
        let e = ctx.unary(mlx::mlx_erf, scaled)?;
        ctx.binary(mlx::mlx_add, one, e)?
    };
    let hx = ctx.binary(mlx::mlx_multiply, half, x)?;
    let r = ctx.binary(mlx::mlx_multiply, hx, gate)?;
    bind_as_out(ctx, n, r)
}

/// Clip: bound `x` below/above by `min`/`max`. Opset>=11 passes them as optional inputs 1/2; opset<11
/// as `min`/`max` float attributes. Absent bounds are skipped.
fn clip_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    use crate::engine::Src;
    let mut r = ctx.resolve(&n.inputs[0])?;
    let dt = ctx.dtype_of(r);
    let present = |i: usize| i < n.inputs.len() && n.inputs[i].source != Src::Absent;
    // min bound
    let min_arr = if present(1) {
        let m = ctx.resolve(&n.inputs[1])?;
        Some(ctx.astype(m, dt)?)
    } else {
        n.floats.get("min").copied().map(|v| scalar_like(ctx, r, v)).transpose()?
    };
    if let Some(mn) = min_arr {
        r = ctx.binary(mlx::mlx_maximum, r, mn)?;
    }
    // max bound
    let max_arr = if present(2) {
        let m = ctx.resolve(&n.inputs[2])?;
        Some(ctx.astype(m, dt)?)
    } else {
        n.floats.get("max").copied().map(|v| scalar_like(ctx, r, v)).transpose()?
    };
    if let Some(mx) = max_arr {
        r = ctx.binary(mlx::mlx_minimum, r, mx)?;
    }
    bind_as_out(ctx, n, r)
}

// ---- claim predicates ---------------------------------------------------------------------------

fn unary_same_type_claim(node: &NodeView, allow_signed_int: bool) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        i.dtype == o.dtype,
        "input/output must share one dtype (got {} -> {})",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    require!(
        is_mlx_float(i.dtype) || (allow_signed_int && is_signed_integer(i.dtype)),
        "dtype {} not supported here ({})",
        crate::registry::ort_dtype_name(i.dtype),
        if allow_signed_int {
            "float fp32/fp16/bf16 or signed integer only"
        } else {
            "float fp32/fp16/bf16 only"
        }
    );
    Ok(())
}

fn float_unary_claim(node: &NodeView) -> ClaimResult {
    unary_same_type_claim(node, false)
}

fn signed_numeric_unary_claim(node: &NodeView) -> ClaimResult {
    unary_same_type_claim(node, true)
}

/// Div: fp32/fp16/bf16, same dtype in/out, scalar-or-suffix broadcast.
fn div_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        a.dtype == b.dtype && b.dtype == out.dtype,
        "inputs/output must share one dtype (got {}, {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        is_mlx_float(a.dtype),
        "dtype {} not supported (float fp32/fp16/bf16 only)",
        crate::registry::ort_dtype_name(a.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

fn relu_claim(node: &NodeView) -> ClaimResult {
    float_unary_claim(node)
}

fn tanh_claim(node: &NodeView) -> ClaimResult {
    float_unary_claim(node)
}

/// Pow: float base (fp32/fp16/bf16), output keeps the base dtype, exponent may be any numeric type
/// (cast to the base dtype in the handler), scalar-or-suffix broadcast. Integer bases are left to ORT
/// CPU (which serves them correctly).
fn pow_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(a.dtype),
        "base dtype {} not supported (float fp32/fp16/bf16 only)",
        crate::registry::ort_dtype_name(a.dtype)
    );
    require!(
        a.dtype == out.dtype,
        "output dtype must match base dtype (got {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

/// Clip: fp32/fp16/bf16 input/output; any present `min`/`max` inputs must share the input dtype.
fn clip_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 1 && node.num_outputs() == 1,
        "expects 1+ inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(i.dtype) && i.dtype == o.dtype,
        "input/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    for b in [1usize, 2] {
        if node.input_present(b) {
            match node.input_info(b) {
                Some(bi) if bi.dtype == i.dtype => {}
                Some(bi) => deny!(
                    "bound input[{b}] dtype {} must match data dtype {}",
                    crate::registry::ort_dtype_name(bi.dtype),
                    crate::registry::ort_dtype_name(i.dtype)
                ),
                None => deny!("bound input[{b}] has no tensor type/shape info"),
            }
        }
    }
    Ok(())
}

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

fn reg_dom(
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
    reg(registry, "Div", div_op, div_claim);
    reg(registry, "Pow", pow_op, pow_claim);
    reg(registry, "Relu", relu_op, relu_claim);
    reg(registry, "Tanh", tanh_op, tanh_claim);
    reg(registry, "Exp", exp_op, float_unary_claim);
    reg(registry, "Log", log_op, float_unary_claim);
    reg(registry, "Sqrt", sqrt_op, float_unary_claim);
    reg(registry, "Neg", neg_op, signed_numeric_unary_claim);
    reg(registry, "Abs", abs_op, signed_numeric_unary_claim);

    // Unary math / rounding.
    reg(registry, "Sign", sign_op, signed_numeric_unary_claim);
    reg(registry, "Reciprocal", reciprocal_op, float_unary_claim);
    reg(registry, "Ceil", ceil_op, float_unary_claim);
    reg(registry, "Floor", floor_op, float_unary_claim);
    reg(registry, "Round", round_op, float_unary_claim);
    reg(registry, "Erf", erf_op, float_unary_claim);

    // Trigonometric / hyperbolic.
    reg(registry, "Sin", sin_op, float_unary_claim);
    reg(registry, "Cos", cos_op, float_unary_claim);
    reg(registry, "Tan", tan_op, float_unary_claim);
    reg(registry, "Sinh", sinh_op, float_unary_claim);
    reg(registry, "Cosh", cosh_op, float_unary_claim);
    reg(registry, "Asin", asin_op, float_unary_claim);
    reg(registry, "Acos", acos_op, float_unary_claim);
    reg(registry, "Atan", atan_op, float_unary_claim);

    // Activations (unary + attrs).
    reg(registry, "LeakyRelu", leaky_relu_op, float_unary_claim);
    reg(registry, "Elu", elu_op, float_unary_claim);
    reg(registry, "Selu", selu_op, float_unary_claim);
    reg(registry, "Celu", celu_op, float_unary_claim);
    reg(registry, "HardSigmoid", hard_sigmoid_op, float_unary_claim);
    reg(registry, "ThresholdedRelu", thresholded_relu_op, float_unary_claim);
    reg(registry, "Softplus", softplus_op, float_unary_claim);
    reg(registry, "Softsign", softsign_op, float_unary_claim);
    reg(registry, "Gelu", gelu_op, float_unary_claim);
    // Gelu also ships in the com.microsoft fused-activation domain.
    reg_dom(registry, "com.microsoft", "Gelu", gelu_op, float_unary_claim);

    // Clip (min/max as optional inputs or opset<11 attrs).
    reg(registry, "Clip", clip_op, clip_claim);
}
