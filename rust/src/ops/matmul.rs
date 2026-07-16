//! Dense linear-algebra op handlers (MatMul, Gemm). Faithful port of the C++ `ops/matmul.cc`.
//!
//! Both map onto MLX's dense GEMM (`mlx_matmul`), which carries numpy/ONNX batch-dim broadcasting and
//! the resolved dtype (fp32/fp16/bf16) through with no per-dtype code. An empty product is
//! re-materialised as a clean, correctly-shaped zeros array (mlx_matmul leaves an empty result with
//! no backing buffer, which the boundary CopyOut cannot memcpy).

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    is_mlx_float, ClaimResult, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::{deny, require};

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

/// A dtype-matched scalar (float value cast to `dt`) so alpha/beta scaling keeps the GEMM dtype.
fn scalar_like(ctx: &mut TranslationContext, value: f32, dt: mlx::mlx_dtype) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(value);
    ctx.astype(s, dt)
}

fn matmul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let y = ctx.binary(mlx::mlx_matmul, a, b)?;
    ctx.mark_fast("mlx_matmul");
    if ctx.size_of(y) == 0 {
        let shp = ctx.shape_of(y);
        let dt = ctx.dtype_of(y);
        let z = ctx.zeros(&shp, dt)?;
        ctx.bind(&n.outputs[0], z);
        return Ok(());
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn gemm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut a = ctx.resolve(&n.inputs[0])?;
    let mut b = ctx.resolve(&n.inputs[1])?;
    let dt = ctx.dtype_of(a);

    let trans_a = n.ints.get("transA").copied().unwrap_or(0) != 0;
    let trans_b = n.ints.get("transB").copied().unwrap_or(0) != 0;
    if trans_a {
        a = ctx.transpose(a, &[1, 0])?;
    }
    if trans_b {
        b = ctx.transpose(b, &[1, 0])?;
    }

    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let beta = n.floats.get("beta").copied().unwrap_or(1.0);

    let mm = ctx.binary(mlx::mlx_matmul, a, b)?;
    let mut y = if alpha != 1.0 {
        let s = scalar_like(ctx, alpha, dt)?;
        ctx.binary(mlx::mlx_multiply, mm, s)?
    } else {
        mm
    };
    if present(n, 2) {
        let mut c = ctx.resolve(&n.inputs[2])?;
        if beta != 1.0 {
            let s = scalar_like(ctx, beta, dt)?;
            c = ctx.binary(mlx::mlx_multiply, c, s)?;
        }
        y = ctx.binary(mlx::mlx_add, y, c)?;
    }
    ctx.mark_fast("mlx_matmul");
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn matmul_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() >= 1,
        "expects 2 inputs and 1+ outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(a.dtype) && b.dtype == a.dtype && out.dtype == a.dtype,
        "inputs/output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        a.shape.len() >= 2 && b.shape.len() >= 2,
        "both inputs must have rank >= 2 (got ranks {} and {})",
        a.shape.len(),
        b.shape.len()
    );
    Ok(())
}

fn gemm_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        (nin == 2 || nin == 3) && node.num_outputs() >= 1,
        "expects 2 or 3 inputs and 1+ outputs, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_mlx_float(a.dtype) && b.dtype == a.dtype && out.dtype == a.dtype,
        "A/B/output must share one float dtype (fp32/fp16/bf16), got {}, {} -> {}",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        a.shape.len() == 2 && b.shape.len() == 2,
        "A and B must both have rank 2 (got ranks {} and {})",
        a.shape.len(),
        b.shape.len()
    );
    if nin == 3 && node.input_present(2) {
        match node.input_info(2) {
            Some(c) if c.dtype == a.dtype && c.shape.len() <= 2 => {}
            Some(c) => deny!(
                "C must match dtype {} and have rank <= 2 (got dtype {}, rank {})",
                crate::registry::ort_dtype_name(a.dtype),
                crate::registry::ort_dtype_name(c.dtype),
                c.shape.len()
            ),
            None => deny!("C input has no tensor type/shape info"),
        }
    }
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "MatMul",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: matmul_op,
        claim: matmul_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Gemm",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: gemm_op,
        claim: gemm_claim,
    });
}
