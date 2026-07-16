//! Random / miscellaneous op handlers (ai.onnx): RandomNormal(+Like), RandomUniform(+Like),
//! Bernoulli, Multinomial, Einsum. Faithful port of the C++ `ops/randommisc.cc`. NonZero / Unique are
//! deliberately left to ORT CPU (mlx-c has no nonzero/unique primitive).

use std::collections::{HashMap, HashSet};

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    is_mlx_float, is_mlx_supported, ClaimPredicate, ClaimResult, NodeView, OpHandler,
    OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- handlers -----------------------------------------------------------------------------------

/// The MLX PRNG key from a `seed` float attribute, or an empty array (default key) when absent.
fn random_key(ctx: &mut TranslationContext, n: &NodeDesc) -> mlx::mlx_array {
    match n.floats.get("seed") {
        Some(&seed) => {
            let raw = unsafe {
                let mut r = mlx::mlx_array_new();
                mlx::mlx_random_key(&mut r, seed as u64);
                r
            };
            ctx.keep(Array::from_raw(raw))
        }
        None => ctx.keep(Array::new()),
    }
}

fn attr_shape(n: &NodeDesc) -> Vec<i32> {
    n.int_arrays
        .get("shape")
        .map(|v| v.iter().map(|&d| d as i32).collect())
        .unwrap_or_default()
}

fn random_normal_with_shape(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    shape: Vec<i32>,
) -> Result<(), MlxError> {
    let key = random_key(ctx, n);
    let dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    let mean = n.floats.get("mean").copied().unwrap_or(0.0);
    let scale = n.floats.get("scale").copied().unwrap_or(1.0);
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_normal(res, shape.as_ptr(), shape.len(), dtype, mean, scale, key, s)
    })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn random_normal_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    random_normal_with_shape(ctx, n, attr_shape(n))
}

fn random_normal_like_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    random_normal_with_shape(ctx, n, shape)
}

fn random_uniform_with_shape(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    shape: Vec<i32>,
) -> Result<(), MlxError> {
    let low = ctx.scalar_f32(n.floats.get("low").copied().unwrap_or(0.0));
    let high = ctx.scalar_f32(n.floats.get("high").copied().unwrap_or(1.0));
    let key = random_key(ctx, n);
    let dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_uniform(res, low, high, shape.as_ptr(), shape.len(), dtype, key, s)
    })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn random_uniform_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    random_uniform_with_shape(ctx, n, attr_shape(n))
}

fn random_uniform_like_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    random_uniform_with_shape(ctx, n, shape)
}

fn bernoulli_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let probs = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(probs);
    let key = random_key(ctx, n);
    let sampled = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_bernoulli(res, probs, shape.as_ptr(), shape.len(), key, s)
    })?;
    let out = ctx.astype(sampled, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn multinomial_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let logits = ctx.resolve(&n.inputs[0])?;
    let sample_size = n.ints.get("sample_size").copied().unwrap_or(1) as i32;
    let key = random_key(ctx, n);
    let sampled = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_categorical_num_samples(res, logits, -1, sample_size, key, s)
    })?;
    let out = ctx.astype(sampled, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn einsum_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut operands = VectorArray::new();
    for input in &n.inputs {
        let a = ctx.resolve(input)?;
        operands.append(a);
    }
    let equation: String = n
        .strings
        .get("equation")
        .cloned()
        .unwrap_or_default()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let ceq = std::ffi::CString::new(equation).map_err(|_| "einsum: bad equation".to_string())?;
    let operands_raw = operands.as_raw();
    let out = ctx.emit(|res, s| unsafe { mlx::mlx_einsum(res, ceq.as_ptr(), operands_raw, s) })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim predicates ---------------------------------------------------------------------------

/// Optional-seed validation: absent is fine; a present seed must be a finite non-negative float.
fn optional_seed_supported(node: &NodeView) -> bool {
    if !node.has_attr("seed") {
        return true;
    }
    match node.float_attr_opt("seed") {
        Some(seed) => seed.is_finite() && seed >= 0.0 && (seed as f64) < 2f64.powi(64),
        None => false, // present but not a float
    }
}

fn is_random_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
}

fn is_boundary_type(t: ort::ONNXTensorElementDataType) -> bool {
    use ort::*;
    is_mlx_float(t)
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
}

fn valid_shape(shape: &[i64]) -> bool {
    shape.iter().all(|&d| d >= 0 && d <= i32::MAX as i64)
}

fn shapes_compatible(a: &[i64], b: &[i64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| x < 0 || y < 0 || x == y)
}

fn random_shape_claim(node: &NodeView, normal: bool) -> ClaimResult {
    require!(
        node.num_inputs() == 0 && node.num_outputs() == 1,
        "expects 0 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        optional_seed_supported(node),
        "seed must be a finite non-negative float representable as u64"
    );
    let out = match node.output_info(0) {
        Some(o) => o,
        None => deny!("missing tensor type/shape info on output"),
    };
    require!(
        is_random_float(out.dtype),
        "output dtype must be float32 or float16, got {}",
        crate::registry::ort_dtype_name(out.dtype)
    );
    let dtype_attr = node.int_attr(
        "dtype",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as i64,
    );
    require!(
        dtype_attr == out.dtype as i64,
        "dtype attribute {} must match output dtype {}",
        crate::registry::ort_dtype_name(dtype_attr as ort::ONNXTensorElementDataType),
        crate::registry::ort_dtype_name(out.dtype)
    );
    let (present, attr_shape) = node.ints_attr("shape");
    require!(present, "shape attribute is required");
    require!(
        valid_shape(&attr_shape),
        "shape dimensions must be in 0..=i32::MAX (got {:?})",
        attr_shape
    );
    require!(
        out.shape == attr_shape,
        "shape attribute {:?} must match output shape {:?}",
        attr_shape,
        out.shape
    );
    if normal {
        let mean = node.float_attr_opt("mean").unwrap_or(0.0);
        let scale = node.float_attr_opt("scale").unwrap_or(1.0);
        require!(
            mean.is_finite() && scale.is_finite() && scale >= 0.0,
            "mean must be finite and scale finite/non-negative (got mean={mean}, scale={scale})"
        );
    } else {
        let low = node.float_attr_opt("low").unwrap_or(0.0);
        let high = node.float_attr_opt("high").unwrap_or(1.0);
        require!(
            low.is_finite() && high.is_finite() && low < high,
            "low/high must be finite with low < high (got low={low}, high={high})"
        );
    }
    Ok(())
}

fn random_normal_claim(node: &NodeView) -> ClaimResult {
    random_shape_claim(node, true)
}

fn random_uniform_claim(node: &NodeView) -> ClaimResult {
    random_shape_claim(node, false)
}

fn random_like_claim(node: &NodeView, normal: bool) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        optional_seed_supported(node),
        "seed must be a finite non-negative float representable as u64"
    );
    let (inp, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_supported(inp.dtype),
        "input dtype {} is not supported by MLX",
        crate::registry::ort_dtype_name(inp.dtype)
    );
    require!(
        is_random_float(out.dtype),
        "output dtype must be float32 or float16, got {}",
        crate::registry::ort_dtype_name(out.dtype)
    );
    let dtype_attr = node.int_attr("dtype", inp.dtype as i64);
    require!(
        dtype_attr == out.dtype as i64,
        "dtype attribute {} must match output dtype {}",
        crate::registry::ort_dtype_name(dtype_attr as ort::ONNXTensorElementDataType),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        shapes_compatible(&inp.shape, &out.shape),
        "input/output shapes must be compatible, got {:?} -> {:?}",
        inp.shape,
        out.shape
    );
    if normal {
        let mean = node.float_attr_opt("mean").unwrap_or(0.0);
        let scale = node.float_attr_opt("scale").unwrap_or(1.0);
        require!(
            mean.is_finite() && scale.is_finite() && scale >= 0.0,
            "mean must be finite and scale finite/non-negative (got mean={mean}, scale={scale})"
        );
    } else {
        let low = node.float_attr_opt("low").unwrap_or(0.0);
        let high = node.float_attr_opt("high").unwrap_or(1.0);
        require!(
            low.is_finite() && high.is_finite() && low < high,
            "low/high must be finite with low < high (got low={low}, high={high})"
        );
    }
    Ok(())
}

fn random_normal_like_claim(node: &NodeView) -> ClaimResult {
    random_like_claim(node, true)
}

fn random_uniform_like_claim(node: &NodeView) -> ClaimResult {
    random_like_claim(node, false)
}

fn bernoulli_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        optional_seed_supported(node),
        "seed must be a finite non-negative float representable as u64"
    );
    let (inp, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_random_float(inp.dtype),
        "probability input dtype must be float32 or float16, got {}",
        crate::registry::ort_dtype_name(inp.dtype)
    );
    require!(
        is_boundary_type(out.dtype),
        "output dtype {} is unsupported",
        crate::registry::ort_dtype_name(out.dtype)
    );
    let dtype_attr = node.int_attr("dtype", inp.dtype as i64);
    require!(
        dtype_attr == out.dtype as i64,
        "dtype attribute {} must match output dtype {}",
        crate::registry::ort_dtype_name(dtype_attr as ort::ONNXTensorElementDataType),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        shapes_compatible(&inp.shape, &out.shape),
        "input/output shapes must be compatible, got {:?} -> {:?}",
        inp.shape,
        out.shape
    );
    Ok(())
}

fn multinomial_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        optional_seed_supported(node),
        "seed must be a finite non-negative float representable as u64"
    );
    let (inp, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    let sample_size = node.int_attr("sample_size", 1);
    require!(
        is_random_float(inp.dtype),
        "input dtype must be float32 or float16, got {}",
        crate::registry::ort_dtype_name(inp.dtype)
    );
    require!(
        out.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            || out.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        "output dtype must be int32 or int64, got {}",
        crate::registry::ort_dtype_name(out.dtype)
    );
    let dtype_attr = node.int_attr(
        "dtype",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 as i64,
    );
    require!(
        dtype_attr == out.dtype as i64,
        "dtype attribute {} must match output dtype {}",
        crate::registry::ort_dtype_name(dtype_attr as ort::ONNXTensorElementDataType),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        inp.shape.len() == 2 && out.shape.len() == 2,
        "input/output must both have rank 2, got rank {} -> {}",
        inp.shape.len(),
        out.shape.len()
    );
    require!(
        inp.shape[1] > 0,
        "class dimension must be static and positive (got {})",
        inp.shape[1]
    );
    require!(
        sample_size > 0 && sample_size <= i32::MAX as i64,
        "sample_size must be in 1..=i32::MAX (got {sample_size})"
    );
    require!(
        inp.shape[0] < 0 || out.shape[0] < 0 || inp.shape[0] == out.shape[0],
        "batch dimensions must match, got {} -> {}",
        inp.shape[0],
        out.shape[0]
    );
    require!(
        out.shape[1] < 0 || out.shape[1] == sample_size,
        "output sample dimension must equal sample_size {sample_size}, got {}",
        out.shape[1]
    );
    Ok(())
}

fn parse_einsum(raw: &str) -> Option<(Vec<String>, String)> {
    let eq: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let arrow = eq.find("->")?;
    if eq[arrow + 2..].contains("->") {
        return None;
    }
    let lhs = &eq[..arrow];
    let output = eq[arrow + 2..].to_string();
    if lhs.is_empty() || output.is_empty() {
        return None;
    }
    let mut terms = Vec::new();
    for term in lhs.split(',') {
        if term.is_empty() {
            return None;
        }
        terms.push(term.to_string());
    }
    let simple = |t: &str| -> bool {
        let mut seen = HashSet::new();
        t.chars()
            .all(|c| ('a'..='z').contains(&c) && seen.insert(c))
    };
    if !simple(&output) || !terms.iter().all(|t| simple(t)) {
        return None;
    }
    Some((terms, output))
}

fn einsum_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        ni > 0 && node.num_outputs() == 1,
        "expects at least 1 input and exactly 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let equation = node.string_attr("equation", "");
    require!(
        node.has_attr("equation") && !equation.is_empty(),
        "non-empty equation attribute is required"
    );
    let (input_terms, output_term) = match parse_einsum(&equation) {
        Some(v) => v,
        None => deny!(
            "equation must use explicit -> output, lowercase labels only, and no repeated label within a term (got {equation:?})"
        ),
    };
    require!(
        input_terms.len() == ni,
        "equation has {} input terms but node has {ni} inputs",
        input_terms.len()
    );
    let (in0, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on first input or output"),
    };
    let dtype = in0.dtype;
    require!(
        is_random_float(dtype) && out.dtype == dtype,
        "all tensors must share float32 or float16 dtype, got first input {} and output {}",
        crate::registry::ort_dtype_name(dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        out.shape.len() == output_term.len(),
        "output rank {} must match equation output term length {}",
        out.shape.len(),
        output_term.len()
    );
    let mut dims: HashMap<char, i64> = HashMap::new();
    for i in 0..ni {
        let info = match node.input_info(i) {
            Some(x) => x,
            None => deny!("missing tensor type/shape info on input {i}"),
        };
        require!(
            info.dtype == dtype,
            "input {i} dtype must match {}, got {}",
            crate::registry::ort_dtype_name(dtype),
            crate::registry::ort_dtype_name(info.dtype)
        );
        require!(
            info.shape.len() == input_terms[i].len(),
            "input {i} rank {} must match equation term length {}",
            info.shape.len(),
            input_terms[i].len()
        );
        for (axis, label) in input_terms[i].chars().enumerate() {
            let d = info.shape[axis];
            match dims.get(&label).copied() {
                None => {
                    dims.insert(label, d);
                }
                Some(existing) => {
                    require!(
                        existing < 0 || d < 0 || existing == d,
                        "label {label:?} has incompatible dimensions {existing} and {d}"
                    );
                    if existing < 0 && d >= 0 {
                        dims.insert(label, d);
                    }
                }
            }
        }
    }
    for (axis, label) in output_term.chars().enumerate() {
        match dims.get(&label).copied() {
            Some(d) => require!(
                d < 0 || out.shape[axis] < 0 || d == out.shape[axis],
                "output label {label:?} dimension {} does not match inferred dimension {d}",
                out.shape[axis]
            ),
            None => deny!("output label {label:?} does not appear in any input term"),
        }
    }
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn reg(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    handler: OpHandler,
    claim: ClaimPredicate,
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
        "RandomNormal",
        1,
        random_normal_op,
        random_normal_claim,
    );
    reg(
        registry,
        "RandomNormalLike",
        1,
        random_normal_like_op,
        random_normal_like_claim,
    );
    reg(
        registry,
        "RandomUniform",
        1,
        random_uniform_op,
        random_uniform_claim,
    );
    reg(
        registry,
        "RandomUniformLike",
        1,
        random_uniform_like_op,
        random_uniform_like_claim,
    );
    reg(registry, "Bernoulli", 15, bernoulli_op, bernoulli_claim);
    reg(
        registry,
        "Multinomial",
        7,
        multinomial_op,
        multinomial_claim,
    );
    reg(registry, "Einsum", 12, einsum_op, einsum_claim);
}
