//! Elementwise + activation + cast op handlers (dtype-generic: each resolves inputs wrapped with
//! their ACTUAL dtype, and MLX carries fp32/fp16/bf16 through unchanged). Port of the wave-1 subset
//! of the C++ `ops/elementwise.cc`.

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, TranslationContext};
use crate::registry::{
    is_int_index, is_mlx_float, is_mlx_numeric, is_signed_integer, is_unsigned_integer,
    scalar_or_suffix_broadcast, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry,
};
use crate::sys::mlx;
use crate::sys::ort;

// ---- handlers -----------------------------------------------------------------------------------

fn add_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_add, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn mul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_multiply, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn sub_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_subtract, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn sigmoid_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.unary(mlx::mlx_sigmoid, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn softmax_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.softmax_last_axis(x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn cast_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.astype(x, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- variadic (1..N elementwise, numpy-broadcasting) --------------------------------------------

/// Cast the produced array to the declared ONNX output dtype (no-op when it already matches) so a
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

/// Fold the variadic inputs with `op` (`Max`/`Min`/`Sum`).
fn fold_variadic(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    op: unsafe extern "C" fn(
        *mut mlx::mlx_array,
        mlx::mlx_array,
        mlx::mlx_array,
        mlx::mlx_stream,
    ) -> i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut acc = ctx.resolve(&n.inputs[0])?;
    for i in 1..n.inputs.len() {
        let next = ctx.resolve(&n.inputs[i])?;
        acc = ctx.binary(op, acc, next)?;
    }
    Ok(acc)
}

fn max_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let r = fold_variadic(ctx, n, mlx::mlx_maximum)?;
    bind_as_out(ctx, n, r)
}

fn min_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let r = fold_variadic(ctx, n, mlx::mlx_minimum)?;
    bind_as_out(ctx, n, r)
}

fn sum_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let r = fold_variadic(ctx, n, mlx::mlx_add)?;
    bind_as_out(ctx, n, r)
}

/// Mean = Sum / N (the divisor is cast to the accumulator dtype to avoid float widening).
fn mean_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let acc = fold_variadic(ctx, n, mlx::mlx_add)?;
    let dt = ctx.dtype_of(acc);
    let count = ctx.scalar_f32(n.inputs.len() as f32);
    let count = ctx.astype(count, dt)?;
    let r = ctx.binary(mlx::mlx_divide, acc, count)?;
    bind_as_out(ctx, n, r)
}

// ---- comparisons / logical (bool output) --------------------------------------------------------

macro_rules! binary_bool_handler {
    ($name:ident, $mlx_op:expr) => {
        fn $name(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
            let a = ctx.resolve(&n.inputs[0])?;
            let b = ctx.resolve(&n.inputs[1])?;
            let r = ctx.binary($mlx_op, a, b)?;
            ctx.bind(&n.outputs[0], r);
            Ok(())
        }
    };
}

binary_bool_handler!(equal_op, mlx::mlx_equal);
binary_bool_handler!(greater_op, mlx::mlx_greater);
binary_bool_handler!(less_op, mlx::mlx_less);
binary_bool_handler!(greater_equal_op, mlx::mlx_greater_equal);
binary_bool_handler!(less_equal_op, mlx::mlx_less_equal);
binary_bool_handler!(and_op, mlx::mlx_logical_and);
binary_bool_handler!(or_op, mlx::mlx_logical_or);
// ONNX Xor over bools == elementwise not-equal.
binary_bool_handler!(xor_op, mlx::mlx_not_equal);

fn not_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.unary(mlx::mlx_logical_not, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- Mod / BitShift -----------------------------------------------------------------------------

/// Mod: `fmod=0` → Python modulo (sign of divisor), served by `mlx_remainder`; `fmod=1` → C `fmod`
/// (sign of dividend), computed as `a - trunc(a/b)*b`.
fn mod_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let fmod = n.ints.get("fmod").copied().unwrap_or(0) != 0;
    let r = if !fmod {
        ctx.binary(mlx::mlx_remainder, a, b)?
    } else {
        let q = ctx.binary(mlx::mlx_divide, a, b)?;
        let fl = ctx.unary(mlx::mlx_floor, q)?;
        let cl = ctx.unary(mlx::mlx_ceil, q)?;
        let dt = ctx.dtype_of(q);
        let zero = ctx.scalar_f32(0.0);
        let zero = ctx.astype(zero, dt)?;
        let nonneg = ctx.binary(mlx::mlx_greater_equal, q, zero)?;
        let trunc = ctx.where_(nonneg, fl, cl)?;
        let prod = ctx.binary(mlx::mlx_multiply, trunc, b)?;
        ctx.binary(mlx::mlx_subtract, a, prod)?
    };
    bind_as_out(ctx, n, r)
}

/// BitShift: `direction` = `LEFT` | `RIGHT` → `mlx_left_shift` / `mlx_right_shift`.
fn bitshift_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let left = n
        .strings
        .get("direction")
        .map(String::as_str)
        .unwrap_or("LEFT")
        == "LEFT";
    let r = if left {
        ctx.binary(mlx::mlx_left_shift, a, b)?
    } else {
        ctx.binary(mlx::mlx_right_shift, a, b)?
    };
    bind_as_out(ctx, n, r)
}

// ---- claim predicates ---------------------------------------------------------------------------

/// Binary same-dtype with scalar-or-suffix broadcast. Floats (fp32/fp16/bf16) are always accepted;
/// `int_ok` decides which integer dtypes are additionally admitted (MLX `mlx_add`/`mlx_multiply`/
/// `mlx_subtract` carry these element-wise, matching ORT CPU including two's-complement wraparound).
fn binary_same_type_claim(
    node: &NodeView,
    int_ok: fn(ort::ONNXTensorElementDataType) -> bool,
) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    if a.dtype != b.dtype || b.dtype != out.dtype {
        return false;
    }
    if !scalar_or_suffix_broadcast(&a.shape, &b.shape) {
        return false;
    }
    is_mlx_float(a.dtype) || int_ok(a.dtype)
}

/// Add: fp32/fp16/bf16 or int32/int64 (index/shape/loop-counter arithmetic in detector subgraphs).
fn add_claim(node: &NodeView) -> bool {
    binary_same_type_claim(node, is_int_index)
}

/// Mul: fp32/fp16/bf16 or int32/int64 (same integer index/shape arithmetic as Add).
fn mul_claim(node: &NodeView) -> bool {
    binary_same_type_claim(node, is_int_index)
}

/// Sub: fp32/fp16/bf16 or signed-integer (the seqlens-prep chain uses int64).
fn sub_claim(node: &NodeView) -> bool {
    binary_same_type_claim(node, is_signed_integer)
}

/// Single fp32/fp16/bf16 input, same dtype out.
fn float_unary_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => i.dtype == o.dtype && is_mlx_float(i.dtype),
        _ => false,
    }
}

fn sigmoid_claim(node: &NodeView) -> bool {
    float_unary_claim(node)
}

/// Softmax over the last axis (axis == -1 or rank-1), fp32/fp16/bf16.
fn softmax_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() < 1 {
        return false;
    }
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => return false,
    };
    if !is_mlx_float(i.dtype) || i.dtype != o.dtype {
        return false;
    }
    let rank = i.shape.len() as i64;
    let axis = node.int_attr("axis", -1);
    rank > 0 && (axis == -1 || axis == rank - 1)
}

/// Cast conversions MLX's `mlx_astype` produces bit-identically to ORT CPU:
///   * float<->float among fp32/fp16/bf16 (distinct pair);
///   * int32<->int64 (exact within range);
///   * int32/int64 -> fp32/fp16 (round-to-nearest, matching CPU static_cast/convert);
///   * fp32/fp16 -> int32/int64 (truncation toward zero, matching ONNX Cast + CPU static_cast).
/// float64/bool/uint are intentionally excluded (not part of the audited detector subgraphs and not
/// all verified against CPU).
fn cast_pair_claimable(
    src: ort::ONNXTensorElementDataType,
    dst: ort::ONNXTensorElementDataType,
) -> bool {
    if is_mlx_float(src) && is_mlx_float(dst) && src != dst {
        return true;
    }
    // int32 <-> int64 (exact).
    if is_int_index(src) && is_int_index(dst) && src != dst {
        return true;
    }
    // int32/int64 -> fp32/fp16.
    if is_int_index(src) && is_cast_float(dst) {
        return true;
    }
    // fp32/fp16 -> int32/int64 (truncation toward zero).
    if is_cast_float(src) && is_int_index(dst) {
        return true;
    }
    false
}

/// fp32/fp16 — the float side of the claimable integer<->float casts (bf16 is not feedable/readable
/// through the ORT Python binding and its CPU-match is covered separately via the float<->float path).
fn is_cast_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
}

/// Cast: the dtype-agnostic handler just calls `astype` to the output dtype, so the predicate is the
/// only gate. See `cast_pair_claimable` for the exact set of conversions verified against ORT CPU.
fn cast_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => return false,
    };
    cast_pair_claimable(i.dtype, o.dtype)
}

fn is_bool(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

/// Variadic `Max`/`Min`/`Sum`/`Mean`: 1..N inputs of one dtype, each numpy-broadcasting to the output
/// shape. `allow_int` also admits signed/unsigned integers (Mean stays float-only since it divides).
fn variadic_claim(node: &NodeView, allow_int: bool) -> bool {
    if node.num_inputs() < 1 || node.num_outputs() != 1 {
        return false;
    }
    let out = match node.output_info(0) {
        Some(o) => o,
        None => return false,
    };
    let ok_dtype = is_mlx_float(out.dtype)
        || (allow_int && (is_signed_integer(out.dtype) || is_unsigned_integer(out.dtype)));
    if !ok_dtype {
        return false;
    }
    for i in 0..node.num_inputs() {
        match node.input_info(i) {
            Some(inf)
                if inf.dtype == out.dtype
                    && scalar_or_suffix_broadcast(&inf.shape, &out.shape) => {}
            _ => return false,
        }
    }
    true
}

fn float_variadic_claim(node: &NodeView) -> bool {
    variadic_claim(node, false)
}

fn numeric_variadic_claim(node: &NodeView) -> bool {
    variadic_claim(node, true)
}

/// Comparison (`Equal`/`Greater`/`Less`/`GreaterOrEqual`/`LessOrEqual`): two same-dtype numeric (or,
/// for Equal/bool, boolean) inputs, boolean output, scalar-or-suffix broadcast.
fn comparison_claim(node: &NodeView, allow_bool: bool) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    if a.dtype != b.dtype || !is_bool(out.dtype) {
        return false;
    }
    (is_mlx_numeric(a.dtype) || (allow_bool && is_bool(a.dtype)))
        && scalar_or_suffix_broadcast(&a.shape, &b.shape)
}

/// Ordered comparisons (Greater/Less/…): numeric inputs only.
fn ordered_comparison_claim(node: &NodeView) -> bool {
    comparison_claim(node, false)
}

/// Equal: numeric OR boolean inputs.
fn equal_claim(node: &NodeView) -> bool {
    comparison_claim(node, true)
}

/// Logical And/Or/Xor: two boolean inputs, boolean output, scalar-or-suffix broadcast.
fn logical_binary_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    is_bool(a.dtype) && is_bool(b.dtype) && is_bool(out.dtype)
        && scalar_or_suffix_broadcast(&a.shape, &b.shape)
}

/// Not: single boolean input/output.
fn not_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => is_bool(i.dtype) && is_bool(o.dtype),
        _ => false,
    }
}

/// Mod: two same-dtype inputs, scalar-or-suffix broadcast. `fmod=0` (Python modulo) serves float and
/// integer; `fmod=1` (C fmod) is float-only (the truncation composition needs float floor/ceil).
fn mod_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    if a.dtype != b.dtype || b.dtype != out.dtype {
        return false;
    }
    if !scalar_or_suffix_broadcast(&a.shape, &b.shape) {
        return false;
    }
    let fmod = node.int_attr("fmod", 0) != 0;
    if fmod {
        is_mlx_float(a.dtype)
    } else {
        is_mlx_float(a.dtype) || is_signed_integer(a.dtype) || is_unsigned_integer(a.dtype)
    }
}

/// BitShift: two same-dtype unsigned-integer inputs/output (excluding uint64, which has no CopyOut
/// path), scalar-or-suffix broadcast.
fn bitshift_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    let u64_t = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    a.dtype == b.dtype
        && b.dtype == out.dtype
        && is_unsigned_integer(a.dtype)
        && a.dtype != u64_t
        && scalar_or_suffix_broadcast(&a.shape, &b.shape)
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

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "Add",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: add_op,
        claim: add_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Mul",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: mul_op,
        claim: mul_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Sub",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sub_op,
        claim: sub_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Sigmoid",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sigmoid_op,
        claim: sigmoid_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Softmax",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: softmax_op,
        claim: softmax_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Cast",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: cast_op,
        claim: cast_claim,
    });
    // Sigmoid is also claimed in the com.microsoft domain (fused activation).
    registry.register(OpRegistration {
        domain: "com.microsoft",
        op_type: "Sigmoid",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sigmoid_op,
        claim: sigmoid_claim,
    });

    // Variadic elementwise.
    reg(registry, "Max", max_op, numeric_variadic_claim);
    reg(registry, "Min", min_op, numeric_variadic_claim);
    reg(registry, "Sum", sum_op, float_variadic_claim);
    reg(registry, "Mean", mean_op, float_variadic_claim);

    // Comparisons (bool output).
    reg(registry, "Equal", equal_op, equal_claim);
    reg(registry, "Greater", greater_op, ordered_comparison_claim);
    reg(registry, "Less", less_op, ordered_comparison_claim);
    reg(registry, "GreaterOrEqual", greater_equal_op, ordered_comparison_claim);
    reg(registry, "LessOrEqual", less_equal_op, ordered_comparison_claim);

    // Logical (bool).
    reg(registry, "And", and_op, logical_binary_claim);
    reg(registry, "Or", or_op, logical_binary_claim);
    reg(registry, "Xor", xor_op, logical_binary_claim);
    reg(registry, "Not", not_op, not_claim);

    // Misc elementwise.
    reg(registry, "Mod", mod_op, mod_claim);
    reg(registry, "BitShift", bitshift_op, bitshift_claim);
}
