//! Reduction op handlers: ReduceSum/Mean/Max/Min/Prod/SumSquare/L1/L2/LogSum/LogSumExp, ArgMax,
//! ArgMin, CumSum and (multi-output) TopK. Faithful port of the C++ `ops/reduction.cc` +
//! `ops/reduction2.cc` + the ArgMin/ArgMax handlers from `ops/math.cc`.
//!
//! Both opset forms are handled: axes as the legacy INTS attribute (opset-13) AND as the opset-18
//! `axes` INPUT tensor (read at translate time via `RawHost`), plus `keepdims` and
//! `noop_with_empty_axes`. Zero-size inputs are handled ON MLX (Max/Min/LogSumExp fill a
//! correctly-shaped identity array instead of calling the aborting kernel).

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    is_mlx_float, is_mlx_numeric, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Sum,
    Max,
    Mean,
    Min,
    Prod,
    LogSumExp,
}

#[derive(Clone, Copy, PartialEq)]
enum PreOp {
    None,
    Abs,
    Square,
}

#[derive(Clone, Copy, PartialEq)]
enum PostOp {
    None,
    Log,
    Sqrt,
}

fn has_axes_input(n: &NodeDesc) -> bool {
    n.inputs.len() >= 2 && n.inputs[1].source != Src::Absent
}

fn read_axes(ctx: &TranslationContext, n: &NodeDesc) -> Result<Vec<i64>, MlxError> {
    if has_axes_input(n) {
        return ctx.read_ints(&n.inputs[1]);
    }
    Ok(n.int_arrays.get("axes").cloned().unwrap_or_default())
}

fn normalize_axes(axes: &[i64], rank: i32) -> Result<Vec<i32>, MlxError> {
    let mut out: Vec<i32> = Vec::with_capacity(axes.len());
    for &raw in axes {
        let axis = if raw < 0 { raw + rank as i64 } else { raw };
        if axis < 0 || axis >= rank as i64 {
            return Err("MLX reduction axis is out of range".to_string());
        }
        let v = axis as i32;
        if out.contains(&v) {
            return Err("MLX reduction axes contain a duplicate".to_string());
        }
        out.push(v);
    }
    Ok(out)
}

/// Fill a correctly-shaped identity array for a zero-size Max/Min/LogSumExp reduction (the MLX kernel
/// aborts at construction on an empty input, so synthesise the result directly).
fn empty_reduce(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    axes: &[i32],
    reduce_all: bool,
    keepdims: bool,
    identity_float: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let in_shape = ctx.shape_of(x);
    let rank = in_shape.len();
    let mut reduced = vec![false; rank];
    if reduce_all {
        reduced.iter_mut().for_each(|r| *r = true);
    } else {
        for &a in axes {
            if (a as usize) < rank {
                reduced[a as usize] = true;
            }
        }
    }
    let mut out_shape: Vec<i32> = Vec::new();
    for i in 0..rank {
        if reduced[i] {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(in_shape[i]);
        }
    }
    let dt = ctx.dtype_of(x);
    let is_float = dt == mlx::mlx_dtype__MLX_FLOAT32
        || dt == mlx::mlx_dtype__MLX_FLOAT16
        || dt == mlx::mlx_dtype__MLX_BFLOAT16;
    let ident = if is_float { identity_float } else { 0.0 };
    let mut scalar = ctx.scalar_f32(ident);
    if dt != mlx::mlx_dtype__MLX_FLOAT32 {
        scalar = ctx.astype(scalar, dt)?;
    }
    ctx.emit(|res, s| unsafe {
        mlx::mlx_full(res, out_shape.as_ptr(), out_shape.len(), scalar, dt, s)
    })
}

fn apply_reduction(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    axes: &[i32],
    reduce_all: bool,
    keepdims: bool,
    kind: Kind,
) -> Result<mlx::mlx_array, MlxError> {
    if matches!(kind, Kind::Max | Kind::Min | Kind::LogSumExp) && ctx.size_of(x) == 0 {
        let ident = f32::NEG_INFINITY; // Max/LogSumExp identity; Min uses +inf below
        let ident = if kind == Kind::Min {
            f32::INFINITY
        } else {
            ident
        };
        return empty_reduce(ctx, x, axes, reduce_all, keepdims, ident);
    }
    if reduce_all {
        match kind {
            Kind::Sum => ctx.emit(|res, s| unsafe { mlx::mlx_sum(res, x, keepdims, s) }),
            Kind::Max => ctx.emit(|res, s| unsafe { mlx::mlx_max(res, x, keepdims, s) }),
            Kind::Mean => ctx.emit(|res, s| unsafe { mlx::mlx_mean(res, x, keepdims, s) }),
            Kind::Min => ctx.emit(|res, s| unsafe { mlx::mlx_min(res, x, keepdims, s) }),
            Kind::Prod => ctx.emit(|res, s| unsafe { mlx::mlx_prod(res, x, keepdims, s) }),
            Kind::LogSumExp => {
                ctx.emit(|res, s| unsafe { mlx::mlx_logsumexp(res, x, keepdims, s) })
            }
        }
    } else {
        let n = axes.len();
        let p = axes.as_ptr();
        match kind {
            Kind::Sum => ctx.emit(|res, s| unsafe { mlx::mlx_sum_axes(res, x, p, n, keepdims, s) }),
            Kind::Max => ctx.emit(|res, s| unsafe { mlx::mlx_max_axes(res, x, p, n, keepdims, s) }),
            Kind::Mean => {
                ctx.emit(|res, s| unsafe { mlx::mlx_mean_axes(res, x, p, n, keepdims, s) })
            }
            Kind::Min => ctx.emit(|res, s| unsafe { mlx::mlx_min_axes(res, x, p, n, keepdims, s) }),
            Kind::Prod => {
                ctx.emit(|res, s| unsafe { mlx::mlx_prod_axes(res, x, p, n, keepdims, s) })
            }
            Kind::LogSumExp => {
                ctx.emit(|res, s| unsafe { mlx::mlx_logsumexp_axes(res, x, p, n, keepdims, s) })
            }
        }
    }
}

fn reduce(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    kind: Kind,
    pre: PreOp,
    post: PostOp,
) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let body = match pre {
        PreOp::None => x,
        PreOp::Abs => ctx.emit(|res, s| unsafe { mlx::mlx_abs(res, x, s) })?,
        PreOp::Square => ctx.emit(|res, s| unsafe { mlx::mlx_square(res, x, s) })?,
    };

    let has_axes = has_axes_input(n) || n.int_arrays.contains_key("axes");
    let raw_axes = read_axes(ctx, n)?;
    let noop = n.ints.get("noop_with_empty_axes").copied().unwrap_or(0) != 0;

    let apply_post =
        |ctx: &mut TranslationContext, v: mlx::mlx_array| -> Result<mlx::mlx_array, MlxError> {
            match post {
                PostOp::None => Ok(v),
                PostOp::Log => ctx.emit(|res, s| unsafe { mlx::mlx_log(res, v, s) }),
                PostOp::Sqrt => ctx.emit(|res, s| unsafe { mlx::mlx_sqrt(res, v, s) }),
            }
        };

    if has_axes && raw_axes.is_empty() && noop {
        let out = apply_post(ctx, body)?;
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    }

    let rank = ctx.ndim(x) as i32;
    let axes = if raw_axes.is_empty() {
        Vec::new()
    } else {
        normalize_axes(&raw_axes, rank)?
    };
    let keepdims = n.ints.get("keepdims").copied().unwrap_or(1) != 0;
    let reduced = apply_reduction(ctx, body, &axes, raw_axes.is_empty(), keepdims, kind)?;
    let out = apply_post(ctx, reduced)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

macro_rules! reduce_handler {
    ($name:ident, $kind:expr, $pre:expr, $post:expr) => {
        fn $name(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
            reduce(ctx, n, $kind, $pre, $post)
        }
    };
}

reduce_handler!(reduce_sum_op, Kind::Sum, PreOp::None, PostOp::None);
reduce_handler!(reduce_mean_op, Kind::Mean, PreOp::None, PostOp::None);
reduce_handler!(reduce_max_op, Kind::Max, PreOp::None, PostOp::None);
reduce_handler!(reduce_min_op, Kind::Min, PreOp::None, PostOp::None);
reduce_handler!(reduce_prod_op, Kind::Prod, PreOp::None, PostOp::None);
reduce_handler!(reduce_sumsquare_op, Kind::Sum, PreOp::Square, PostOp::None);
reduce_handler!(reduce_l1_op, Kind::Sum, PreOp::Abs, PostOp::None);
reduce_handler!(reduce_l2_op, Kind::Sum, PreOp::Square, PostOp::Sqrt);
reduce_handler!(reduce_logsum_op, Kind::Sum, PreOp::None, PostOp::Log);
reduce_handler!(
    reduce_logsumexp_op,
    Kind::LogSumExp,
    PreOp::None,
    PostOp::None
);

// ---- ArgMin / ArgMax ---------------------------------------------------------------------------

type ArgOp =
    unsafe extern "C" fn(*mut mlx::mlx_array, mlx::mlx_array, i32, bool, mlx::mlx_stream) -> i32;

fn argminmax(ctx: &mut TranslationContext, n: &NodeDesc, op: ArgOp) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(x) as i32;
    let mut axis = n.ints.get("axis").copied().unwrap_or(0) as i32;
    if axis < 0 {
        axis += rank;
    }
    let keepdims = n.ints.get("keepdims").copied().unwrap_or(1) != 0;
    let select_last = n.ints.get("select_last_index").copied().unwrap_or(0) != 0;
    let dim = ctx.dim(x, axis);

    let arg_input = if select_last {
        let rev = ctx.emit(|res, s| unsafe {
            mlx::mlx_arange(
                res,
                (dim - 1) as f64,
                -1.0,
                -1.0,
                mlx::mlx_dtype__MLX_INT32,
                s,
            )
        })?;
        ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, x, rev, axis, s) })?
    } else {
        x
    };

    let result = ctx.emit(|res, s| unsafe { op(res, arg_input, axis, keepdims, s) })?;
    let mut result = ctx.astype(result, mlx::mlx_dtype__MLX_INT64)?;
    if select_last {
        let base = ctx.scalar_i64((dim - 1) as i64);
        result = ctx.binary(mlx::mlx_subtract, base, result)?;
    }
    ctx.bind(&n.outputs[0], result);
    Ok(())
}

fn argmax_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    argminmax(ctx, n, mlx::mlx_argmax_axis)
}

fn argmin_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    argminmax(ctx, n, mlx::mlx_argmin_axis)
}

// ---- CumSum ------------------------------------------------------------------------------------

fn read_scalar_int(
    ctx: &TranslationContext,
    r: &crate::engine::TensorRef,
) -> Result<i64, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.count != 1 || h.data.is_null() {
        return Err("MLX expected a scalar integer input".to_string());
    }
    match h.dtype {
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            Ok(unsafe { *(h.data as *const i32) } as i64)
        }
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            Ok(unsafe { *(h.data as *const i64) })
        }
        _ => Err("MLX expected an int32 or int64 scalar input".to_string()),
    }
}

fn cumsum_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(x) as i64;
    let mut axis = read_scalar_int(ctx, &n.inputs[1])?;
    if axis < 0 {
        axis += rank;
    }
    if axis < 0 || axis >= rank {
        return Err("MLX CumSum axis is out of range".to_string());
    }
    let reverse = n.ints.get("reverse").copied().unwrap_or(0) != 0;
    let inclusive = n.ints.get("exclusive").copied().unwrap_or(0) == 0;
    let axis = axis as i32;
    let out = ctx.emit(|res, s| unsafe { mlx::mlx_cumsum(res, x, axis, reverse, inclusive, s) })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- TopK (multi-output) -----------------------------------------------------------------------

fn topk_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let mut axis = n.ints.get("axis").copied().unwrap_or(-1) as i32;
    if axis < 0 {
        axis += shape.len() as i32;
    }
    if axis < 0 || axis as usize >= shape.len() {
        return Err("MLX TopK axis is out of range".to_string());
    }
    let k64 = read_scalar_int(ctx, &n.inputs[1])?;
    if k64 <= 0 || k64 > shape[axis as usize] as i64 {
        return Err("MLX TopK K is out of range".to_string());
    }
    let k = k64 as i32;
    let largest = n.ints.get("largest").copied().unwrap_or(1) != 0;

    let sort_input = if largest {
        ctx.emit(|res, s| unsafe { mlx::mlx_negative(res, x, s) })?
    } else {
        x
    };
    let sorted_indices =
        ctx.emit(|res, s| unsafe { mlx::mlx_argsort_axis(res, sort_input, axis, s) })?;
    let selector = ctx.emit(|res, s| unsafe {
        mlx::mlx_arange(res, 0.0, k as f64, 1.0, mlx::mlx_dtype__MLX_INT32, s)
    })?;
    let top_indices =
        ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, sorted_indices, selector, axis, s) })?;
    let values =
        ctx.emit(|res, s| unsafe { mlx::mlx_take_along_axis(res, x, top_indices, axis, s) })?;
    let cvalues = ctx.contiguous(values)?;
    let cindices = ctx.contiguous(top_indices)?;
    ctx.bind(&n.outputs[0], cvalues);
    let idx64 = ctx.astype(cindices, mlx::mlx_dtype__MLX_INT64)?;
    ctx.bind(&n.outputs[1], idx64);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn axes_are_valid(axes: &[i64], rank: i64) -> bool {
    let mut seen: Vec<i64> = Vec::new();
    for &a in axes {
        let axis = if a < 0 { a + rank } else { a };
        if axis < 0 || axis >= rank || seen.contains(&axis) {
            return false;
        }
        seen.push(axis);
    }
    true
}

fn reduction_claim(node: &NodeView, float_only: bool) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        (1..=2).contains(&nin) && node.num_outputs() == 1,
        "expects 1-2 inputs and 1 output, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (x, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(o)) => (x, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        !x.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    require!(
        x.dtype == out.dtype,
        "input/output dtypes must match, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    if float_only {
        require!(
            is_mlx_float(x.dtype),
            "dtype must be float32, float16, or bfloat16, got {}",
            crate::registry::ort_dtype_name(x.dtype)
        );
    } else {
        require!(
            is_mlx_numeric(x.dtype),
            "dtype must be an MLX-supported numeric type, got {}",
            crate::registry::ort_dtype_name(x.dtype)
        );
    }
    if nin == 2 && node.input_present(1) {
        let axes = match node.input_info(1) {
            Some(axes) => axes,
            None => deny!("missing tensor type/shape info on axes input"),
        };
        require!(
            axes.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            "axes input must be int64, got {}",
            crate::registry::ort_dtype_name(axes.dtype)
        );
        require!(
            axes.shape.len() <= 1,
            "axes input must be a scalar or 1-D tensor, got rank {}",
            axes.shape.len()
        );
    }
    let (present, axes) = node.ints_attr("axes");
    require!(
        !present || axes_are_valid(&axes, x.shape.len() as i64),
        "axes attribute {:?} contains an out-of-range or duplicate axis for rank {}",
        axes,
        x.shape.len()
    );
    let keepdims = node.int_attr("keepdims", 1);
    let noop = node.int_attr("noop_with_empty_axes", 0);
    require!(
        keepdims == 0 || keepdims == 1,
        "keepdims must be 0 or 1 (got {keepdims})"
    );
    require!(
        noop == 0 || noop == 1,
        "noop_with_empty_axes must be 0 or 1 (got {noop})"
    );
    Ok(())
}

fn reduce_numeric_claim(node: &NodeView) -> ClaimResult {
    reduction_claim(node, false)
}

fn reduce_float_claim(node: &NodeView) -> ClaimResult {
    reduction_claim(node, true)
}

fn argminmax_claim(node: &NodeView) -> ClaimResult {
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
        !i.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    require!(
        is_mlx_numeric(i.dtype)
            && i.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        "input dtype {} is unsupported (must be MLX numeric, excluding uint64)",
        crate::registry::ort_dtype_name(i.dtype)
    );
    require!(
        o.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "output dtype must be int64, got {}",
        crate::registry::ort_dtype_name(o.dtype)
    );
    let mut axis = node.int_attr("axis", 0);
    let raw_axis = axis;
    if axis < 0 {
        axis += i.shape.len() as i64;
    }
    let keepdims = node.int_attr("keepdims", 1);
    let select_last = node.int_attr("select_last_index", 0);
    require!(
        axis >= 0 && axis < i.shape.len() as i64,
        "axis {raw_axis} is out of range for rank {}",
        i.shape.len()
    );
    require!(
        keepdims == 0 || keepdims == 1,
        "keepdims must be 0 or 1 (got {keepdims})"
    );
    require!(
        select_last == 0 || select_last == 1,
        "select_last_index must be 0 or 1 (got {select_last})"
    );
    Ok(())
}

fn cumsum_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, axis, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(a), Some(o)) => (x, a, o),
        _ => deny!("missing tensor type/shape info on input, axis, or output"),
    };
    require!(
        !x.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    require!(
        x.dtype == out.dtype && is_mlx_numeric(x.dtype),
        "input/output must share an MLX numeric dtype, got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        axis.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            || axis.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "axis input must be int32 or int64, got {}",
        crate::registry::ort_dtype_name(axis.dtype)
    );
    require!(
        axis.shape.is_empty() || (axis.shape.len() == 1 && axis.shape[0] == 1),
        "axis input must be scalar or shape [1], got {:?}",
        axis.shape
    );
    let exclusive = node.int_attr("exclusive", 0);
    let reverse = node.int_attr("reverse", 0);
    require!(
        exclusive == 0 || exclusive == 1,
        "exclusive must be 0 or 1 (got {exclusive})"
    );
    require!(
        reverse == 0 || reverse == 1,
        "reverse must be 0 or 1 (got {reverse})"
    );
    Ok(())
}

fn topk_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 2,
        "expects 2 inputs and 2 outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, k, values, indices) = match (
        node.input_info(0),
        node.input_info(1),
        node.output_info(0),
        node.output_info(1),
    ) {
        (Some(x), Some(k), Some(v), Some(i)) => (x, k, v, i),
        _ => deny!("missing tensor type/shape info on X, K, values, or indices"),
    };
    require!(
        !x.shape.is_empty(),
        "input must have rank >= 1 (got a scalar)"
    );
    require!(
        is_mlx_float(x.dtype) && values.dtype == x.dtype,
        "X/values must share one float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(values.dtype)
    );
    require!(
        k.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "K must be an int64 scalar read at translation time, got {}",
        crate::registry::ort_dtype_name(k.dtype)
    );
    require!(
        indices.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "indices output must be int64, got {}",
        crate::registry::ort_dtype_name(indices.dtype)
    );
    require!(
        k.shape.is_empty() || (k.shape.len() == 1 && k.shape[0] == 1),
        "K must be a scalar or shape [1], read at translation time and constrained to 1..=axis dimension (got shape {:?})",
        k.shape
    );
    let mut axis = node.int_attr("axis", -1);
    let raw_axis = axis;
    if axis < 0 {
        axis += x.shape.len() as i64;
    }
    require!(
        axis >= 0 && axis < x.shape.len() as i64,
        "axis {raw_axis} is out of range for rank {}; K is limited to 1..=that axis dimension",
        x.shape.len()
    );
    let largest = node.int_attr("largest", 1);
    let sorted = node.int_attr("sorted", 1);
    require!(
        largest == 0 || largest == 1,
        "largest must be 0 or 1 (got {largest})"
    );
    require!(
        sorted == 1,
        "only sorted=1 is supported (got {sorted}); K is read at translation time and must be within the selected axis"
    );
    Ok(())
}

fn reg(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain: "",
        op_type,
        min_opset,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    reg(
        registry,
        "ReduceSum",
        K_ANY_OPSET,
        reduce_sum_op,
        reduce_numeric_claim,
    );
    reg(
        registry,
        "ReduceMax",
        K_ANY_OPSET,
        reduce_max_op,
        reduce_numeric_claim,
    );
    reg(
        registry,
        "ReduceMean",
        K_ANY_OPSET,
        reduce_mean_op,
        reduce_float_claim,
    );
    reg(
        registry,
        "ReduceMin",
        K_ANY_OPSET,
        reduce_min_op,
        reduce_numeric_claim,
    );
    reg(
        registry,
        "ReduceProd",
        K_ANY_OPSET,
        reduce_prod_op,
        reduce_numeric_claim,
    );
    reg(
        registry,
        "ReduceSumSquare",
        K_ANY_OPSET,
        reduce_sumsquare_op,
        reduce_numeric_claim,
    );
    reg(
        registry,
        "ReduceL1",
        K_ANY_OPSET,
        reduce_l1_op,
        reduce_numeric_claim,
    );
    reg(
        registry,
        "ReduceL2",
        K_ANY_OPSET,
        reduce_l2_op,
        reduce_float_claim,
    );
    reg(
        registry,
        "ReduceLogSum",
        K_ANY_OPSET,
        reduce_logsum_op,
        reduce_float_claim,
    );
    reg(
        registry,
        "ReduceLogSumExp",
        K_ANY_OPSET,
        reduce_logsumexp_op,
        reduce_float_claim,
    );
    reg(registry, "ArgMax", K_ANY_OPSET, argmax_op, argminmax_claim);
    reg(registry, "ArgMin", K_ANY_OPSET, argmin_op, argminmax_claim);
    reg(registry, "CumSum", 11, cumsum_op, cumsum_claim);
    reg(registry, "TopK", 10, topk_op, topk_claim);
}
