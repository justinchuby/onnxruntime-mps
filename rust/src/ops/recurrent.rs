//! Recurrent op handlers (ai.onnx opset-17+): RNN, GRU, LSTM. Faithful port of the C++
//! `ops/recurrent.cc` (see docs/OP_ARCHITECTURE.md §5).
//!
//! Static-length unrolling: RNN/GRU/LSTM are SINGLE nodes carrying weight inputs W/R/(B)/(P) and a
//! `hidden_size` attribute; the recurrence runs over the sequence axis of X ([S, B, I]). When S is
//! STATICALLY KNOWN we UNROLL into a fixed MLX graph — a host loop over t = 0..S-1 builds S steps,
//! each computing gate pre-activations via matmuls (Xt·Wᵀ + H_{t-1}·Rᵀ + bias) then activations.
//!
//! Directions: forward, reverse, bidirectional. `hidden_size`, `clip`, optional B / initial_h /
//! (LSTM) initial_c / P supported. Only DEFAULT activations are translatable (RNN=Tanh,
//! GRU=Sigmoid/Tanh, LSTM=Sigmoid/Tanh/Tanh); since the STRINGS `activations` attribute is not
//! carried into NodeDesc, a node that carries ANY `activations` attribute is conservatively left to
//! ORT CPU. Every other unclaimed form (dynamic S, sequence_lens, non-default layout) → CPU.

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{
    is_mlx_float, ClaimPredicate, NodeView, OpHandler, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;

// ---- small handler helpers ----------------------------------------------------------------------

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

fn gate_count(op: &str) -> i32 {
    match op {
        "LSTM" => 4,
        "GRU" => 3,
        _ => 1,
    }
}

/// A dtype-matched scalar (float value cast to `dt`).
fn scalar(ctx: &mut TranslationContext, value: f32, dt: mlx::mlx_dtype) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(value);
    if dt == mlx::mlx_dtype__MLX_FLOAT32 {
        Ok(s)
    } else {
        ctx.astype(s, dt)
    }
}

/// Drop leading axis 0 by selecting index d: arr[d] (rank R -> rank R-1).
fn dir_slab(ctx: &mut TranslationContext, arr: mlx::mlx_array, d: i32) -> Result<mlx::mlx_array, MlxError> {
    let sh = ctx.shape_of(arr);
    let mut start = vec![0i32; sh.len()];
    let mut stop = sh.clone();
    start[0] = d;
    stop[0] = d + 1;
    let s = ctx.slice(arr, &start, &stop)?;
    let ns: Vec<i32> = sh[1..].to_vec();
    ctx.reshape(s, &ns)
}

/// Columns [g*H, (g+1)*H) of a [rows, gates*H] gate block.
fn gate_col(ctx: &mut TranslationContext, m: mlx::mlx_array, g: i32, h: i32) -> Result<mlx::mlx_array, MlxError> {
    let sh = ctx.shape_of(m);
    ctx.slice(m, &[0, g * h], &[sh[0], (g + 1) * h])
}

/// Sub-vector [a, b) of a 1-D array.
fn vec1d(ctx: &mut TranslationContext, v: mlx::mlx_array, a: i32, b: i32) -> Result<mlx::mlx_array, MlxError> {
    ctx.slice(v, &[a], &[b])
}

/// Rows [a, b) of a 2-D [rows, cols] array.
fn rows(ctx: &mut TranslationContext, m: mlx::mlx_array, a: i32, b: i32) -> Result<mlx::mlx_array, MlxError> {
    let sh = ctx.shape_of(m);
    ctx.slice(m, &[a, 0], &[b, sh[1]])
}

fn sigmoid(ctx: &mut TranslationContext, x: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_sigmoid(res, x, s) })
}

fn tanh_(ctx: &mut TranslationContext, x: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_tanh(res, x, s) })
}

/// Per-timestep slab Xproj[t] -> [batch, gates*H] from a [S, batch, gates*H] tensor.
fn step_slab(ctx: &mut TranslationContext, xproj: mlx::mlx_array, t: i32, b: i32, gh: i32) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.slice(xproj, &[t, 0, 0], &[t + 1, b, gh])?;
    ctx.reshape(s, &[b, gh])
}

/// Clamp `pre` to [-clip, clip] when clip > 0.
fn clip_(ctx: &mut TranslationContext, pre: mlx::mlx_array, clip: f32, dt: mlx::mlx_dtype) -> Result<mlx::mlx_array, MlxError> {
    if clip <= 0.0 {
        return Ok(pre);
    }
    let hi_s = scalar(ctx, clip, dt)?;
    let hi = ctx.binary(mlx::mlx_minimum, pre, hi_s)?;
    let lo_s = scalar(ctx, -clip, dt)?;
    ctx.binary(mlx::mlx_maximum, hi, lo_s)
}

/// One direction's weight / state slabs (the [num_directions, ...] tensors, un-sliced).
struct DirInputs {
    w: mlx::mlx_array,
    r: mlx::mlx_array,
    has_bias: bool,
    b: Option<mlx::mlx_array>,
    has_init_h: bool,
    init_h: Option<mlx::mlx_array>,
    has_init_c: bool,
    init_c: Option<mlx::mlx_array>,
    has_peephole: bool,
    p: Option<mlx::mlx_array>,
}

struct DirResult {
    ys: Vec<mlx::mlx_array>,
    final_h: mlx::mlx_array,
    final_c: Option<mlx::mlx_array>,
}

#[allow(clippy::too_many_arguments)]
fn run_direction(
    ctx: &mut TranslationContext,
    op: &str,
    din: &DirInputs,
    x: mlx::mlx_array,
    d: i32,
    reverse: bool,
    s_len: i32,
    b_sz: i32,
    h: i32,
    gh: i32,
    clip: f32,
    linear_before_reset: i32,
    input_forget: i32,
    dt: mlx::mlx_dtype,
) -> Result<DirResult, MlxError> {
    let wd = dir_slab(ctx, din.w, d)?; // [G*H, I]
    let rd = dir_slab(ctx, din.r, d)?; // [G*H, H]
    let wdt = ctx.transpose(wd, &[1, 0])?; // [I, G*H]
    let rdt = ctx.transpose(rd, &[1, 0])?; // [H, G*H]

    // Xproj = X · Wᵀ (+ Wb) once for the whole sequence: [S,B,I] @ [I,G*H] -> [S,B,G*H].
    let mut xproj = ctx.matmul(x, wdt)?;
    let mut rb: Option<mlx::mlx_array> = None;
    let has_bias = din.has_bias;
    if has_bias {
        let bd = dir_slab(ctx, din.b.ok_or_else(|| "RNN: bias flagged but missing".to_string())?, d)?; // [2*G*H]
        let wb = vec1d(ctx, bd, 0, gh)?;
        rb = Some(vec1d(ctx, bd, gh, 2 * gh)?);
        xproj = ctx.add(xproj, wb)?;
    }

    // Initial states.
    let mut h_prev = if din.has_init_h {
        dir_slab(ctx, din.init_h.ok_or_else(|| "RNN: initial_h flagged but missing".to_string())?, d)?
    } else {
        ctx.zeros(&[b_sz, h], dt)?
    };
    let mut c_prev = if op == "LSTM" {
        if din.has_init_c {
            Some(dir_slab(ctx, din.init_c.ok_or_else(|| "LSTM: initial_c flagged but missing".to_string())?, d)?)
        } else {
            Some(ctx.zeros(&[b_sz, h], dt)?)
        }
    } else {
        None
    };

    // Peepholes (LSTM).
    let (mut pi, mut po, mut pf) = (None, None, None);
    if op == "LSTM" && din.has_peephole {
        let pd = dir_slab(ctx, din.p.ok_or_else(|| "LSTM: peephole flagged but missing".to_string())?, d)?; // [3*H]
        pi = Some(vec1d(ctx, pd, 0, h)?);
        po = Some(vec1d(ctx, pd, h, 2 * h)?);
        pf = Some(vec1d(ctx, pd, 2 * h, 3 * h)?);
    }

    let mut ys: Vec<mlx::mlx_array> = vec![x; s_len as usize]; // placeholder handles
    for step in 0..s_len {
        let t = if reverse { s_len - 1 - step } else { step };
        let xg = step_slab(ctx, xproj, t, b_sz, gh)?; // [B, G*H] (already carries Wb)
        let mut hf = ctx.matmul(h_prev, rdt)?; // [B, G*H]
        if has_bias {
            hf = ctx.add(hf, rb.ok_or_else(|| "RNN: recurrent bias missing".to_string())?)?;
        }

        if op == "RNN" {
            let sum = ctx.add(xg, hf)?;
            let pre = clip_(ctx, sum, clip, dt)?;
            let h_new = tanh_(ctx, pre)?;
            ys[t as usize] = h_new;
            h_prev = h_new;
        } else if op == "GRU" {
            // gate order z, r, h
            let xz = gate_col(ctx, xg, 0, h)?;
            let xr = gate_col(ctx, xg, 1, h)?;
            let xh = gate_col(ctx, xg, 2, h)?;
            let hz = gate_col(ctx, hf, 0, h)?;
            let hr = gate_col(ctx, hf, 1, h)?;
            let zsum = ctx.add(xz, hz)?;
            let zc = clip_(ctx, zsum, clip, dt)?;
            let zt = sigmoid(ctx, zc)?;
            let rsum = ctx.add(xr, hr)?;
            let rc = clip_(ctx, rsum, clip, dt)?;
            let rt = sigmoid(ctx, rc)?;
            let htpre = if linear_before_reset != 0 {
                // ht = g(xh + rt ∘ (H_{t-1}·Rhᵀ + Rbh)); Hf's h column already carries Rbh.
                let hh = gate_col(ctx, hf, 2, h)?;
                let rhh = ctx.mul(rt, hh)?;
                ctx.add(xh, rhh)?
            } else {
                // ht = g(xh + (rt ∘ H_{t-1})·Rhᵀ + Rbh)
                let rh_rows = rows(ctx, rd, 2 * h, 3 * h)?;
                let rht = ctx.transpose(rh_rows, &[1, 0])?; // [H,H]
                let rh_state = ctx.mul(rt, h_prev)?;
                let mut hh = ctx.matmul(rh_state, rht)?;
                if has_bias {
                    let rbh = vec1d(ctx, rb.ok_or_else(|| "RNN: recurrent bias missing".to_string())?, 2 * h, 3 * h)?;
                    hh = ctx.add(hh, rbh)?;
                }
                ctx.add(xh, hh)?
            };
            let htc = clip_(ctx, htpre, clip, dt)?;
            let ht = tanh_(ctx, htc)?;
            // Ht = (1 - zt) ∘ ht + zt ∘ H_{t-1}
            let one = scalar(ctx, 1.0, dt)?;
            let one_minus_z = ctx.sub(one, zt)?;
            let a = ctx.mul(one_minus_z, ht)?;
            let bb = ctx.mul(zt, h_prev)?;
            let h_new = ctx.add(a, bb)?;
            ys[t as usize] = h_new;
            h_prev = h_new;
        } else {
            // LSTM, gate order i, o, f, c
            let xi = gate_col(ctx, xg, 0, h)?;
            let xo = gate_col(ctx, xg, 1, h)?;
            let xf = gate_col(ctx, xg, 2, h)?;
            let xc = gate_col(ctx, xg, 3, h)?;
            let hi = gate_col(ctx, hf, 0, h)?;
            let ho = gate_col(ctx, hf, 1, h)?;
            let hfg = gate_col(ctx, hf, 2, h)?;
            let hc = gate_col(ctx, hf, 3, h)?;

            let mut ipre = ctx.add(xi, hi)?;
            let mut fpre = ctx.add(xf, hfg)?;
            if din.has_peephole {
                let cp = c_prev.ok_or_else(|| "LSTM: cell state missing".to_string())?;
                let pic = ctx.mul(pi.ok_or_else(|| "LSTM: peephole pi missing".to_string())?, cp)?;
                ipre = ctx.add(ipre, pic)?;
                let pfc = ctx.mul(pf.ok_or_else(|| "LSTM: peephole pf missing".to_string())?, cp)?;
                fpre = ctx.add(fpre, pfc)?;
            }
            let ipc = clip_(ctx, ipre, clip, dt)?;
            let it = sigmoid(ctx, ipc)?;
            // Couple input/forget gates: ft = 1 - it when input_forget != 0.
            let ft = if input_forget != 0 {
                let one = scalar(ctx, 1.0, dt)?;
                ctx.sub(one, it)?
            } else {
                let fpc = clip_(ctx, fpre, clip, dt)?;
                sigmoid(ctx, fpc)?
            };
            let csum = ctx.add(xc, hc)?;
            let cc = clip_(ctx, csum, clip, dt)?;
            let ct = tanh_(ctx, cc)?;
            let cp = c_prev.ok_or_else(|| "LSTM: cell state missing".to_string())?;
            let fc = ctx.mul(ft, cp)?;
            let ic = ctx.mul(it, ct)?;
            let c_new = ctx.add(fc, ic)?;

            let mut opre = ctx.add(xo, ho)?;
            if din.has_peephole {
                let poc = ctx.mul(po.ok_or_else(|| "LSTM: peephole po missing".to_string())?, c_new)?;
                opre = ctx.add(opre, poc)?;
            }
            let opc = clip_(ctx, opre, clip, dt)?;
            let ot = sigmoid(ctx, opc)?;
            let tc = tanh_(ctx, c_new)?;
            let h_new = ctx.mul(ot, tc)?;

            ys[t as usize] = h_new;
            h_prev = h_new;
            c_prev = Some(c_new);
        }
    }

    Ok(DirResult {
        ys,
        final_h: h_prev,
        final_c: c_prev,
    })
}

// ---- the single handler shared by RNN / GRU / LSTM ----------------------------------------------

fn recurrent_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let op = n.op_type.clone();
    let g = gate_count(&op);

    let h = *n
        .ints
        .get("hidden_size")
        .ok_or_else(|| "MLX recurrent: missing hidden_size".to_string())? as i32;
    let direction = n
        .strings
        .get("direction")
        .cloned()
        .unwrap_or_else(|| "forward".to_string());
    let bidir = direction == "bidirectional";
    let reverse = direction == "reverse";
    let clip = n.floats.get("clip").copied().unwrap_or(0.0);
    let linear_before_reset = n.ints.get("linear_before_reset").copied().unwrap_or(0) as i32;
    let input_forget = n.ints.get("input_forget").copied().unwrap_or(0) as i32;

    let x = ctx.resolve(&n.inputs[0])?; // [S, B, I]
    let xsh = ctx.shape_of(x);
    let s_len = xsh[0];
    let b_sz = xsh[1];
    let gh = g * h;
    let dt = ctx.dtype_of(x);

    let mut din = DirInputs {
        w: ctx.resolve(&n.inputs[1])?,
        r: ctx.resolve(&n.inputs[2])?,
        has_bias: false,
        b: None,
        has_init_h: false,
        init_h: None,
        has_init_c: false,
        init_c: None,
        has_peephole: false,
        p: None,
    };
    if present(n, 3) {
        din.has_bias = true;
        din.b = Some(ctx.resolve(&n.inputs[3])?);
    }
    // input index 4 is sequence_lens (claimed only when absent).
    if present(n, 5) {
        din.has_init_h = true;
        din.init_h = Some(ctx.resolve(&n.inputs[5])?);
    }
    if op == "LSTM" {
        if present(n, 6) {
            din.has_init_c = true;
            din.init_c = Some(ctx.resolve(&n.inputs[6])?);
        }
        if present(n, 7) {
            din.has_peephole = true;
            din.p = Some(ctx.resolve(&n.inputs[7])?);
        }
    }

    let mut results: Vec<DirResult> = Vec::new();
    if bidir {
        results.push(run_direction(ctx, &op, &din, x, 0, false, s_len, b_sz, h, gh, clip, linear_before_reset, input_forget, dt)?);
        results.push(run_direction(ctx, &op, &din, x, 1, true, s_len, b_sz, h, gh, clip, linear_before_reset, input_forget, dt)?);
    } else {
        results.push(run_direction(ctx, &op, &din, x, 0, reverse, s_len, b_sz, h, gh, clip, linear_before_reset, input_forget, dt)?);
    }
    let num_dir = results.len();

    // Y : [S, num_dir, B, H]. Per direction: stack Ys along axis 0 -> [S,B,H], expand -> [S,1,B,H];
    // concat directions along axis 1.
    if !n.outputs.is_empty() && !n.outputs[0].name.is_empty() {
        let mut y: Option<mlx::mlx_array> = None;
        for d in 0..num_dir {
            let stacked = ctx.stack(&results[d].ys, 0)?; // [S,B,H]
            let ydir = ctx.expand_dims(stacked, 1)?; // [S,1,B,H]
            y = Some(match y {
                None => ydir,
                Some(prev) => ctx.concat2(prev, ydir, 1)?,
            });
        }
        ctx.bind(&n.outputs[0], y.ok_or_else(|| "RNN: no directions produced Y".to_string())?);
    }

    // Y_h : [num_dir, B, H].
    if n.outputs.len() >= 2 && !n.outputs[1].name.is_empty() {
        let mut yh: Option<mlx::mlx_array> = None;
        for d in 0..num_dir {
            let hd = ctx.expand_dims(results[d].final_h, 0)?; // [1,B,H]
            yh = Some(match yh {
                None => hd,
                Some(prev) => ctx.concat2(prev, hd, 0)?,
            });
        }
        ctx.bind(&n.outputs[1], yh.ok_or_else(|| "RNN: no directions produced Y_h".to_string())?);
    }

    // Y_c : [num_dir, B, H] (LSTM only).
    if op == "LSTM" && n.outputs.len() >= 3 && !n.outputs[2].name.is_empty() {
        let mut yc: Option<mlx::mlx_array> = None;
        for d in 0..num_dir {
            let cd = ctx.expand_dims(results[d].final_c.ok_or_else(|| "LSTM: final cell state missing".to_string())?, 0)?; // [1,B,H]
            yc = Some(match yc {
                None => cd,
                Some(prev) => ctx.concat2(prev, cd, 0)?,
            });
        }
        ctx.bind(&n.outputs[2], yc.ok_or_else(|| "LSTM: no directions produced Y_c".to_string())?);
    }
    Ok(())
}

// ---- claim-time helpers -------------------------------------------------------------------------

fn float_ok(node: &NodeView, i: usize, xt: crate::sys::ort::ONNXTensorElementDataType) -> bool {
    if !node.input_present(i) {
        return true;
    }
    matches!(node.input_info(i), Some(info) if info.dtype == xt)
}

fn ieq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// True iff every activation matches the op's default pattern (cyclically). ORT may surface the
/// schema default (e.g. RNN `["Tanh","Tanh"]`) with more entries than `num_directions`, so we do not
/// require an exact count — only that each entry equals the corresponding default. A genuinely
/// non-default activation (e.g. Relu) mismatches and the node is left to ORT CPU.
fn activations_are_default(op: &str, acts: &[String]) -> bool {
    let base: &[&str] = match op {
        "RNN" => &["tanh"],
        "GRU" => &["sigmoid", "tanh"],
        _ => &["sigmoid", "tanh", "tanh"], // LSTM
    };
    !acts.is_empty()
        && acts
            .iter()
            .enumerate()
            .all(|(i, a)| ieq(a, base[i % base.len()]))
}

/// Shared claim predicate for RNN / GRU / LSTM.
fn recurrent_claim(node: &NodeView, op: &str) -> bool {
    let ninputs = node.num_inputs();
    if ninputs < 3 || node.num_outputs() == 0 {
        return false;
    }

    // X (rank-3, static positive seq length), W, R — all the same float dtype.
    let (xinfo, winfo, rinfo) = match (node.input_info(0), node.input_info(1), node.input_info(2)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return false,
    };
    let xt = xinfo.dtype;
    if !is_mlx_float(xt) || winfo.dtype != xt || rinfo.dtype != xt {
        return false;
    }
    if xinfo.shape.len() != 3 || xinfo.shape[0] <= 0 {
        return false; // dynamic / symbolic seq length -> CPU
    }

    // Optional float inputs must share the float dtype when present.
    if !float_ok(node, 3, xt) || !float_ok(node, 5, xt) || !float_ok(node, 6, xt) || !float_ok(node, 7, xt) {
        return false;
    }

    // sequence_lens (index 4) present => variable-length masking; leave to CPU.
    if node.input_present(4) {
        return false;
    }

    // Default layout only.
    if node.int_attr("layout", 0) != 0 {
        return false;
    }

    // hidden_size is required for the unroll.
    if !node.has_attr("hidden_size") {
        return false;
    }

    // Direction determines num_directions; only forward/reverse/bidirectional are supported.
    let direction = node.string_attr("direction", "forward");
    if direction != "forward" && direction != "reverse" && direction != "bidirectional" {
        return false;
    }

    // Only DEFAULT activations are translatable (the STRINGS attribute is not carried into NodeDesc,
    // so the handler hard-codes the defaults). ORT surfaces the schema-default activations for these
    // nodes, so validate the set against the per-op defaults; any non-default set -> ORT CPU.
    if let Some(acts) = node.strings_attr("activations") {
        if !activations_are_default(op, &acts) {
            return false;
        }
    }

    true
}

fn rnn_claim(node: &NodeView) -> bool {
    recurrent_claim(node, "RNN")
}
fn gru_claim(node: &NodeView) -> bool {
    recurrent_claim(node, "GRU")
}
fn lstm_claim(node: &NodeView) -> bool {
    recurrent_claim(node, "LSTM")
}

fn reg(registry: &mut OpRegistry, op_type: &'static str, handler: OpHandler, claim: ClaimPredicate) {
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
    reg(registry, "RNN", recurrent_op, rnn_claim);
    reg(registry, "GRU", recurrent_op, gru_claim);
    reg(registry, "LSTM", recurrent_op, lstm_claim);
}
