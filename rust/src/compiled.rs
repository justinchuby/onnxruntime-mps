//! Unified compiled fast path (`mlx_compile`) — ONE parameterised core for every compiled strategy.
//!
//! The eager translator (`engine::TranslationContext::execute`) rebuilds and dispatches EVERY node
//! of a claimed subgraph as separate unfused MLX primitive launches on every Compute call. For a
//! decoder that is ~393 kernels per token; for a CNN/audio graph it is hundreds per inference —
//! dominating runtime. This module traces the WHOLE claimed subgraph translation into an
//! `mlx_closure` over its dynamic (non-constant) ctx inputs ONCE, compiles it with `mlx_compile`
//! (kernel fusion), caches the compiled closure on the plan, and on each subsequent call just
//! applies the compiled closure to the freshly-wrapped inputs.
//!
//! Historically this lived in two modules — a decode-only shapeless path and a general static-shape
//! path. They are now a single [`crate::engine::CompiledSubgraph`] core parameterised by a
//! [`crate::engine::CompiledConfig`] (`shape_mode` + the opt-in `kv_alias` / `rope_as_data` /
//! `delta_copyout` / `contiguous_outputs` features). The strategies are just configurations:
//!   * **decode**  = `{ Shapeless,  kv_alias, rope_as_data, delta_copyout }` — shapeless so a growing
//!     KV length never retraces; RoPE position + valid-past fed as DATA; delta KV copy-out.
//!   * **general** = `{ ShapeKeyed, contiguous_outputs }` — retraces on a shape change; no attention.
//!   * **prefill** = `{ ShapeKeyed, kv_alias, rope_as_data, delta_copyout }` (Phase 2) — the SAME
//!     decoder subgraph as decode but with a variable query length S>1, so it is shape-keyed.
//!
//! Two shapeless-compile tricks carried by `rope_as_data` (ported verbatim from the C++ blueprint):
//!   * **RoPE rotate-half via a `[hd,hd]` matmul** (only when `rotary_dim == head_dim`) so the
//!     compiled graph carries no Slice (which shapeless compile cannot shape-infer).
//!   * **Pre-sliced cos/sin rows fed as synthetic closure inputs**, so the position offset is DATA
//!     (not baked) and the compiled graph never slices a cos/sin cache at a runtime position.
//!
//! CORRECTNESS: every path here falls back to the eager translator on any doubt (ineligible plan,
//! missing cache, trace/apply/eval error) — the compiled path never crashes and never diverges.

use std::collections::HashSet;
use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;

use crate::engine::{
    copy_out_raw_delta, dim_i32, mlx_dtype_from_onnx, read_ctx_input_raw, rope_row_key, DeltaWrite,
    DynInput, MlxError, NodeDesc, OutRef, Plan, ShapeMode, Slot, Src, SynthRope, TracePayload,
    TranslationContext,
};
use crate::mlx::{self, Array, Closure, VectorArray};
use crate::sys::mlx as mlxsys;
use crate::sys::ort;

/// Decide whether any compiled fast path is allowed for this plan. Disabled by the
/// `ONNX_GENAI_MLX_NO_COMPILE` kill-switch (forces eager for debugging / numerical A-B) or when the
/// subgraph contains a control-flow node (its graph structure depends on runtime data).
pub fn compile_enabled(has_control_flow: bool) -> bool {
    let killed = std::env::var_os("ONNX_GENAI_MLX_NO_COMPILE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    !has_control_flow && !killed
}

/// Op types whose subgraphs keep their existing (eager / compiled-decode) routes because they are
/// incompatible with a single static-shape fused trace:
///   * attention ops carry KV-cache aliasing + delta copy-out semantics not modelled by the general
///     config (they take the decode/prefill configs instead);
///   * host-computed ops (`Det`/`NonZero`/`Unique`) GPU-eval their DYNAMIC input DATA mid-translate
///     (`contiguous_eval`) and/or emit a data-dependent output shape — both illegal inside an
///     `mlx_compile` trace (the placeholder has no data), so a subgraph containing one is never
///     general-compiled.
fn is_general_compile_unsafe(op_type: &str) -> bool {
    matches!(
        op_type,
        "GroupQueryAttention"
            | "Attention"
            | "MultiHeadAttention"
            | "SparseAttention"
            | "Det"
            | "NonZero"
            | "Unique"
    )
}

/// Decide whether the general compiled fast path is allowed for this plan. Shares the compile
/// kill-switch (`ONNX_GENAI_MLX_NO_COMPILE`), and is additionally disabled for control-flow or any
/// subgraph containing an op that is unsafe to trace once (see [`is_general_compile_unsafe`]). An
/// extra kill-switch `MLX_EP_NO_GENERAL_COMPILE` forces eager for A/B numerical validation without
/// touching the decode path.
pub fn general_enabled(has_control_flow: bool, nodes: &[NodeDesc]) -> bool {
    if !compile_enabled(has_control_flow) {
        return false;
    }
    if std::env::var_os("MLX_EP_NO_GENERAL_COMPILE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
    {
        return false;
    }
    !nodes.iter().any(|n| is_general_compile_unsafe(&n.op_type))
}

/// Decide whether the compiled-PREFILL fast path (Phase 2) is allowed for this plan. Prefill is the
/// same decoder subgraph as decode but at query length S>1, so it needs a `GroupQueryAttention` node
/// (the RoPE/KV machinery build declines otherwise) and no control-flow.
///
/// OPT-IN by default (`MLX_EP_PREFILL_COMPILE=1` to enable). The compiled shared-buffer path runs
/// SDPA over the full KV capacity with a causal mask, whereas the eager prefill attends only the
/// valid prefix `[0, S)`. For decode (S=1, launch-bound) kernel fusion dominates and wins; for
/// prefill (S large, compute-bound) the full-cap attention outweighs the fusion saving and is a
/// measured ~15-20% TTFT regression. Correct (byte-identical to eager) but not yet a win, so it
/// stays behind an opt-in flag to preserve the eager-fallback discipline until the attention is
/// narrowed to the valid prefix. Also honours the global compile kill-switch. The build itself
/// falls back to eager for any non-decoder / partial-rotary shape.
pub fn prefill_enabled(has_control_flow: bool, nodes: &[NodeDesc]) -> bool {
    if !compile_enabled(has_control_flow) {
        return false;
    }
    if !std::env::var_os("MLX_EP_PREFILL_COMPILE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
    {
        return false;
    }
    nodes.iter().any(|n| n.op_type == "GroupQueryAttention")
}

/// Query sequence length S = trailing dim of the `input_ids` dynamic ctx input (decode => 1, prefill
/// => prompt length). Scans the plan nodes directly so it works before the compiled closure's
/// `dyn_inputs` list is built. Returns `None` if the input is not found / has no shape.
pub fn detect_seq_len(
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    plan: &Plan,
) -> Option<i32> {
    for node in &plan.nodes {
        for inp in &node.inputs {
            if inp.source == Src::CtxInput && !inp.constant && inp.name == "input_ids" {
                if let Ok((_d, shape, _t)) = read_ctx_input_raw(api, kctx, inp.ctx_index) {
                    return shape.last().map(|&d| d as i32);
                }
            }
        }
    }
    None
}

/// Detect whether this session drives a fixed-capacity SHARED KV buffer (present aliased onto past at
/// a runtime-owned max length) as opposed to the growing past/present contract. Reads live ctx once
/// at compiled-closure build time: `cap` = a GQA past-KV cache's seq-axis (2) length, and `total` =
/// valid keys after this step (= `total_sequence_length[0]`, or the `attention_mask` width when that
/// scalar is computed in-subgraph). Shared iff `cap > total - S` (the buffer holds more than the
/// valid past). Returns `(shared, mask_ctx_index)` — `mask_ctx_index` is the `attention_mask` ctx
/// input used to recover the valid-past RoPE start at apply time (`-1` if unknown). `None` when the
/// shape is not a recognizable decoder GQA.
fn detect_shared_kv(
    plan: &Plan,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
) -> Option<(bool, i32)> {
    let s = detect_seq_len(api, kctx, plan).unwrap_or(1);
    let mask_ctx_index = plan
        .nodes
        .iter()
        .flat_map(|n| n.inputs.iter())
        .find(|inp| inp.source == Src::CtxInput && inp.name == "attention_mask")
        .map(|inp| inp.ctx_index as i32)
        .unwrap_or(-1);
    for node in &plan.nodes {
        if node.op_type != "GroupQueryAttention" || node.inputs.len() < 7 {
            continue;
        }
        // Capacity from the past-KV cache (input[3], a ctx input) seq axis.
        let past_k = &node.inputs[3];
        if past_k.source != Src::CtxInput {
            continue;
        }
        let cap = match read_ctx_input_raw(api, kctx, past_k.ctx_index) {
            Ok((_d, shape, _t)) => *shape.get(2)? as i32,
            Err(_) => continue,
        };
        // Valid keys after this step. Prefer total_sequence_length (input[6]) when it arrives as a
        // ctx input; otherwise recover it from the attention_mask width (the in-subgraph scalar is
        // Cast(Gather(Shape(attention_mask),1))).
        let ts = &node.inputs[6];
        let total = if ts.source == Src::CtxInput && !ts.constant {
            match read_ctx_input_raw(api, kctx, ts.ctx_index) {
                Ok((data, _shape, _t)) if !data.is_null() => unsafe { *(data as *const i32) },
                _ => continue,
            }
        } else if mask_ctx_index >= 0 {
            match read_ctx_input_raw(api, kctx, mask_ctx_index as usize) {
                Ok((_d, shape, _t)) => *shape.get(1)? as i32,
                Err(_) => continue,
            }
        } else {
            continue;
        };
        return Some((cap > total - s, mask_ctx_index));
    }
    None
}

/// A contiguous unit-stride slice `[start, stop)` -> owning [`Array`].
fn mk_slice(
    a: mlxsys::mlx_array,
    start: &[i32],
    stop: &[i32],
    stream: mlxsys::mlx_stream,
) -> Result<Array, MlxError> {
    let stride = vec![1i32; start.len()];
    let mut res = unsafe { mlxsys::mlx_array_new() };
    let rc = unsafe {
        mlxsys::mlx_slice(
            &mut res,
            a,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            stride.as_ptr(),
            stride.len(),
            stream,
        )
    };
    if rc != 0 {
        unsafe { mlxsys::mlx_array_free(res) };
        return Err("mlx_slice failed".to_string());
    }
    Ok(Array::from_raw(res))
}

/// Concatenate two arrays along `axis` -> owning [`Array`].
fn mk_concat(
    a: mlxsys::mlx_array,
    b: mlxsys::mlx_array,
    axis: i32,
    stream: mlxsys::mlx_stream,
) -> Result<Array, MlxError> {
    let mut v = VectorArray::new();
    v.append(a);
    v.append(b);
    let mut res = unsafe { mlxsys::mlx_array_new() };
    let rc = unsafe { mlxsys::mlx_concatenate_axis(&mut res, v.as_raw(), axis, stream) };
    if rc != 0 {
        unsafe { mlxsys::mlx_array_free(res) };
        return Err("mlx_concatenate_axis failed".to_string());
    }
    Ok(Array::from_raw(res))
}

/// Shape (as `i32`) of a borrowed raw array handle.
fn shape_of(a: mlxsys::mlx_array) -> Vec<i32> {
    let nd = unsafe { mlxsys::mlx_array_ndim(a) };
    let sh = unsafe { mlxsys::mlx_array_shape(a) };
    (0..nd).map(|i| unsafe { *sh.add(i) }).collect()
}

/// The compiled fast path for the given `slot`. Returns:
///   * `Ok(true)`  — the call was handled by the compiled closure (outputs already copied out).
///   * `Ok(false)` — the plan is not compile-eligible (caller must fall back to the eager path).
///   * `Err(_)`    — a hard failure copying results out.
///
/// The plan is accessed through `plan_ptr` (raw) so that NO Rust `&mut Plan` is alive across
/// `mlx_closure_apply` — the trace thunk (invoked synchronously on the first apply / a retrace)
/// needs its own `&mut Plan`, so only one mutable access is ever live at a time (single-threaded,
/// reentrant via FFI, exactly like the C++ EP).
pub fn try_compiled(
    plan_ptr: *mut Plan,
    slot: Slot,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
) -> Result<bool, MlxError> {
    let cfg = slot.get(unsafe { &*plan_ptr }).config;
    if !slot.get(unsafe { &*plan_ptr }).enabled {
        return Ok(false);
    }
    // One-time discovery + compile. `build_closure` returns `false` for a TRANSIENT decline (the
    // constant cos/sin cache is not resident yet — it is only populated during the first eager
    // translation, which for prefill is the very first Compute call); in that case we leave
    // `attempted` unset so a later call retries once the cache is warm. Every other outcome (compiled,
    // or a permanent ineligibility / compile failure) is terminal and marks `attempted`.
    if !slot.get(unsafe { &*plan_ptr }).attempted {
        let terminal = build_closure(plan_ptr, slot, api, kctx, stream);
        if terminal {
            slot.get_mut(unsafe { &mut *plan_ptr }).attempted = true;
        }
    }
    if !slot.get(unsafe { &*plan_ptr }).valid {
        return Ok(false);
    }

    // Shape-keyed guard: skip (=> eager) when any dynamic input is an empty tensor (a zero-size dim).
    // Empty arrays are a degenerate edge the eager translator handles directly, but tracing /
    // `mlx_compile` over a zero-size shape can abort inside MLX. Checked on EVERY call (a shape-keyed
    // closure would otherwise retrace into the same abort). Shapeless decode never sees empty inputs.
    if cfg.shape_mode == ShapeMode::ShapeKeyed {
        let plan = unsafe { &*plan_ptr };
        for di in &slot.get(plan).dyn_inputs {
            let (_data, shape, _dtype) = read_ctx_input_raw(api, kctx, di.ctx_index)?;
            if shape.iter().any(|&d| d == 0) {
                return Ok(false);
            }
        }
    }

    // Gather the dynamic ctx inputs (zero-copy wrap of the live ORT buffers) in closure order.
    let mut arena: Vec<Array> = Vec::new();
    let mut input = VectorArray::new();
    {
        // Immutable snapshot of the closure input schema — no `&mut Plan` alive here.
        let plan = unsafe { &*plan_ptr };
        for di in &slot.get(plan).dyn_inputs {
            let (data, shape, dtype) = read_ctx_input_raw(api, kctx, di.ctx_index)?;
            let ishape: Vec<i32> = shape.iter().map(|&d| dim_i32(d)).collect::<Result<_, _>>()?;
            // Zero-copy wrap of the live ORT input buffer: MLX borrows it (no-op deallocator) for
            // this apply only. `arena` keeps the wrapper alive until after eval + copy-out, and the
            // ORT tensor is valid for the whole Compute call, so the borrow never dangles. In
            // shared-buffer mode this is the per-token past K/V full [B,kv,cap,hd] buffer whose
            // memcpy dominated decode. (ORT KV buffers are 16 KB page-aligned, so Metal takes the
            // true no-copy path; small unaligned inputs fall back to MLX's internal copy.)
            let arr = Array::from_data_managed(data, &ishape, mlx_dtype_from_onnx(dtype));
            input.append(arr.as_raw());
            arena.push(arr);
        }
    }

    // ---- rope_as_data prologue (decode / prefill): position + valid_past fed as DATA -------------
    // `past` = RoPE start = position of the new query rows, and `s` = number of new rows (1 for
    // decode, S for prefill). Both the synth cos/sin rows and the KV delta below use them.
    let mut past: i32 = 0;
    let mut s_new: i32 = 0;
    if cfg.rope_as_data {
        let s = detect_seq_len(api, kctx, unsafe { &*plan_ptr }).unwrap_or(1);
        s_new = s;
        // Current RoPE start. In the growing contract this is the past-KV sequence length (past_k's
        // seq axis). In the shared-buffer contract past_k's seq axis is the fixed capacity, so the
        // true position is the VALID past = attention_mask width - S. Fed as DATA via the rows below.
        past = {
            let plan = unsafe { &*plan_ptr };
            let c = slot.get(plan);
            if c.shared_kv && c.mask_ctx_index >= 0 {
                let (_d, shape, _t) =
                    read_ctx_input_raw(api, kctx, c.mask_ctx_index as usize)?;
                (*shape.get(1).unwrap_or(&0) as i32) - s
            } else {
                let idx = c.rope_past_ctx_index as usize;
                let axis = c.rope_past_axis as usize;
                let (_d, shape, _t) = read_ctx_input_raw(api, kctx, idx)?;
                if axis < shape.len() {
                    shape[axis] as i32
                } else {
                    0
                }
            }
        };

        // Pre-slice each RoPE cos/sin cache at [past, past+S) and feed the FULL-width rows (the
        // half-width slice duplicated across both halves) in — keeping the compiled graph static and
        // free of any Slice primitive. Done OUTSIDE the compiled graph so the position stays data.
        let synth = slot.get(unsafe { &*plan_ptr }).synth_ropes.clone();
        for sr in &synth {
            let cache_raw = match unsafe { &*plan_ptr }.cache.get(&sr.cache_name) {
                Some(a) => a.as_raw(),
                None => return Ok(false), // cache not resident yet -> eager
            };
            let half = shape_of(cache_raw).get(1).copied().unwrap_or(0);
            if half == 0 {
                return Ok(false);
            }
            let row = mk_slice(cache_raw, &[past, 0], &[past + s, half], stream)?; // [S,half]
            let full = mk_concat(row.as_raw(), row.as_raw(), 1, stream)?; // [S,2*half]
            input.append(full.as_raw());
            arena.push(row);
            arena.push(full);
        }

        // Shared-buffer contract: append the live `valid_past` (= `past`) as a [1] int32 closure
        // input AFTER the synth RoPE rows. The compiled GQA op reads it to place the in-place K/V
        // write and build the causal mask — as pure per-step data, so shapeless compile never freezes
        // the growing offset. Kept in the arena for the apply's life.
        if slot.get(unsafe { &*plan_ptr }).shared_kv {
            let vp = [past];
            let arr = Array::from_data(
                vp.as_ptr() as *const std::os::raw::c_void,
                &[1],
                mlxsys::mlx_dtype__MLX_INT32,
            );
            input.append(arr.as_raw());
            arena.push(arr);
        }
    }

    // Refresh the live kernel context on the trace payload (only read during a (re)trace).
    unsafe {
        if let Some(p) = slot.get_mut(&mut *plan_ptr).payload.as_mut() {
            p.kctx = kctx;
        }
    }

    // Take the compiled closure OUT of the plan so no `&mut Plan` field is borrowed across apply
    // (the trace thunk mutates the plan through `plan_ptr`).
    let closure = match slot.get_mut(unsafe { &mut *plan_ptr }).closure.take() {
        Some(c) => c,
        None => return Ok(false),
    };
    let apply_res = closure.apply(&input);
    slot.get_mut(unsafe { &mut *plan_ptr }).closure = Some(closure);

    let outs = match apply_res {
        Ok(v) => v,
        Err(_) => {
            slot.get_mut(unsafe { &mut *plan_ptr }).valid = false; // disable, fall back to eager
            return Ok(false);
        }
    };
    if mlx::eval(&outs).is_err() {
        slot.get_mut(unsafe { &mut *plan_ptr }).valid = false;
        return Ok(false);
    }

    let ext_len = slot.get(unsafe { &*plan_ptr }).ext_outputs.len();
    if outs.size() != ext_len {
        slot.get_mut(unsafe { &mut *plan_ptr }).valid = false;
        return Ok(false);
    }

    // Shared-buffer KV `present` outputs alias `past` in ORT memory, so their per-call copy-out is a
    // delta write of only the `S` new rows at axis-2 offset `past` instead of the whole
    // [B,kv,cap,hd] buffer (the O(1) copy-out win). The write is gated on the present buffer actually
    // matching the recorded `past` pointer, so a non-aliasing runtime safely takes the full copy.
    // Empty set (growing / non-delta paths) => every output takes the full copy.
    let kv_present: Vec<(String, usize)> = if cfg.delta_copyout {
        slot.get(unsafe { &*plan_ptr }).kv_present_names.clone()
    } else {
        Vec::new()
    };
    for i in 0..ext_len {
        let a = outs.get(i);
        // Clone the small OutRef so we don't hold a plan borrow across the copy-out FFI.
        let o: OutRef = slot.get(unsafe { &*plan_ptr }).ext_outputs[i].clone();
        let delta = kv_present
            .iter()
            .find(|(name, _)| *name == o.name)
            .and_then(|(_, past_ctx)| {
                read_ctx_input_raw(api, kctx, *past_ctx)
                    .ok()
                    .map(|(p, _, _)| DeltaWrite {
                        axis: 2,
                        offset: past as i64,
                        count: s_new as i64,
                        alias_ptr: p as usize,
                    })
            });
        copy_out_raw_delta(api, kctx, &o, a.as_raw(), delta)?;
    }
    // `arena` (transient per-call inputs) drops here — after eval + copy-out have consumed them.
    Ok(true)
}

/// One-time discovery + compile of the closure for `slot`. Populates the slot's `dyn_inputs`
/// (ordered dynamic ctx inputs = closure inputs) and `ext_outputs` (boundary outputs in append
/// order); for `rope_as_data` slots it also discovers `synth_ropes`, the RoPE-start source, and the
/// shared-KV contract. Compiles the closure (shapeless iff `shape_mode == Shapeless`). Leaves
/// `valid = false` (=> caller falls back to eager) if the plan is not eligible or the compile fails.
///
/// Returns whether the outcome is TERMINAL: `true` after a successful compile or a permanent
/// ineligibility, `false` for a transient decline (the constant cos/sin cache is not resident yet)
/// so the caller retries on a later call once the constants have been warmed by an eager translation.
fn build_closure(
    plan_ptr: *mut Plan,
    slot: Slot,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
) -> bool {
    let cfg = slot.get(unsafe { &*plan_ptr }).config;

    let mut dyn_inputs: Vec<DynInput> = Vec::new();
    let mut ext_outputs: Vec<OutRef> = Vec::new();
    let mut synth_ropes: Vec<SynthRope> = Vec::new();
    let mut rope_past: i32 = -1;
    {
        let plan = unsafe { &*plan_ptr };
        // Ordered, de-duplicated dynamic (non-constant) ctx inputs = the closure input vector.
        let mut seen: HashSet<String> = HashSet::new();
        for node in &plan.nodes {
            for inp in &node.inputs {
                if inp.source == Src::CtxInput && !inp.constant && seen.insert(inp.name.clone()) {
                    dyn_inputs.push(DynInput {
                        name: inp.name.clone(),
                        ctx_index: inp.ctx_index,
                    });
                    // Read the RoPE start (past length) from a past-KV key input's sequence axis.
                    if cfg.rope_as_data && rope_past < 0 && inp.name.contains(".key") {
                        rope_past = inp.ctx_index as i32;
                    }
                }
            }
        }
        // External boundary outputs, in stable node/output order.
        for node in &plan.nodes {
            for o in &node.outputs {
                if o.external {
                    ext_outputs.push(o.clone());
                }
            }
        }
        // Distinct RoPE cos/sin caches (GQA inputs[7]/[8] when do_rotary). Their per-step rows are
        // fed as synthetic closure inputs so the compiled graph never slices a cache at runtime.
        if cfg.rope_as_data {
            let mut synth_seen: HashSet<String> = HashSet::new();
            for node in &plan.nodes {
                if node.op_type != "GroupQueryAttention" || node.inputs.len() < 9 {
                    continue;
                }
                let do_rotary = !node.ints.contains_key("do_rotary")
                    || node.ints.get("do_rotary").copied() != Some(0);
                if !do_rotary {
                    continue;
                }
                for idx in [7usize, 8usize] {
                    let nm = &node.inputs[idx].name;
                    if synth_seen.insert(nm.clone()) {
                        synth_ropes.push(SynthRope {
                            key: rope_row_key(nm),
                            cache_name: nm.clone(),
                        });
                    }
                }
            }
        }
    }

    // Need at least one dynamic input (else the graph is fully constant — cheap to leave eager) and
    // at least one boundary output. `rope_as_data` slots additionally need the decoder RoPE shape.
    if dyn_inputs.is_empty() || ext_outputs.is_empty() {
        return true;
    }
    if cfg.rope_as_data && (rope_past < 0 || synth_ropes.is_empty()) {
        return true; // not the expected decoder shape; stay on the eager path
    }

    let mut shared_kv = false;
    let mut mask_ctx_index = -1;
    if cfg.rope_as_data {
        // The compiled RoPE uses a [hd,hd] rotate-half matmul, which requires rotary_dim == head_dim
        // (rot == hd). Validate from the live ctx (head dim = last axis of the past-KV cache) vs the
        // cos cache width (half); fall back to eager for a partial-rotary head.
        let hd = match read_ctx_input_raw(api, kctx, rope_past as usize) {
            Ok((_d, shape, _t)) => shape.last().copied().unwrap_or(0) as i32,
            Err(_) => 0,
        };
        // The cos/sin cache is a constant only loaded into `plan.cache` during an eager translation.
        // On the very first Compute call (prefill) it is not resident yet — a TRANSIENT condition, so
        // return non-terminal and retry once the constants are warm (the apply prologue needs them
        // too). A resident cache with the wrong width is a PERMANENT decline (partial-rotary head).
        let (half, cache_resident) = {
            let plan = unsafe { &*plan_ptr };
            match plan.cache.get(&synth_ropes[0].cache_name) {
                Some(a) => {
                    let sh = a.shape();
                    (if sh.len() >= 2 { sh[1] as i32 } else { 0 }, true)
                }
                None => (0, false),
            }
        };
        if !cache_resident {
            return false; // transient — constants not warmed yet
        }
        if hd == 0 || half == 0 || 2 * half != hd {
            return true; // partial-rotary head not supported by the compiled path
        }
    }
    if cfg.kv_alias {
        // Detect a fixed-capacity shared KV buffer once from live ctx. In that contract GQA writes
        // the new K/V in place at the (data-dependent) valid-past offset and emits present at the
        // buffer's full capacity — the trace handles both via a data start index + a static-shape
        // additive mask, so the compiled fast path still applies.
        let detected = detect_shared_kv(unsafe { &*plan_ptr }, api, kctx).unwrap_or((false, -1));
        shared_kv = detected.0;
        mask_ctx_index = detected.1;
    }

    // Publish the discovered schema + a stable trace payload onto the slot.
    unsafe {
        let c = slot.get_mut(&mut *plan_ptr);
        c.shared_kv = shared_kv;
        c.mask_ctx_index = mask_ctx_index;
        c.dyn_inputs = dyn_inputs;
        c.ext_outputs = ext_outputs;
        c.synth_ropes = synth_ropes;
        c.rope_past_ctx_index = rope_past;
        c.rope_past_axis = 2;
        c.payload = Some(Box::new(TracePayload {
            plan: plan_ptr,
            ort_api: api,
            kctx,
            stream,
            slot,
        }));
    }

    let payload_ptr: *mut c_void = unsafe {
        slot.get_mut(&mut *plan_ptr).payload.as_mut().unwrap().as_mut() as *mut TracePayload
            as *mut c_void
    };
    // Shapeless (decode) so the growing KV length never triggers a recompile; shape-keyed (general /
    // prefill) so a changed input shape safely retraces (re-invokes the thunk) rather than
    // miscomputes. The trace thunk runs lazily on the first `apply`.
    let shapeless = matches!(cfg.shape_mode, ShapeMode::Shapeless);
    let base = Closure::new_func_payload(trace_thunk, payload_ptr);
    match Closure::compile(&base, shapeless) {
        Ok(compiled) => {
            let c = slot.get_mut(unsafe { &mut *plan_ptr });
            c.closure = Some(compiled);
            c.valid = true;
        }
        Err(_) => { /* stay eager */ }
    }
    true
}

/// `mlx_closure` trace thunk (payload = [`TracePayload`]): seed each dynamic ctx input (+ pre-sliced
/// RoPE row / valid_past for `rope_as_data`) from the closure input vector, translate the whole
/// subgraph, and return the cast external boundary outputs. Invoked lazily by mlx on the first
/// `apply` and again after any input shape change (shape-keyed slots). Never unwinds across the FFI
/// boundary.
extern "C" fn trace_thunk(
    out: *mut mlxsys::mlx_vector_array,
    input: mlxsys::mlx_vector_array,
    payload: *mut c_void,
) -> std::os::raw::c_int {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| trace_body(out, input, payload)));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("[rust-mlx-ep] compiled trace failed ({e}); falling back to eager");
            1
        }
        Err(_) => {
            eprintln!("[rust-mlx-ep] compiled trace panicked; falling back to eager");
            1
        }
    }
}

fn trace_body(
    out: *mut mlxsys::mlx_vector_array,
    input: mlxsys::mlx_vector_array,
    payload: *mut c_void,
) -> Result<(), MlxError> {
    let pl = unsafe { &mut *(payload as *mut TracePayload) };
    let plan_ptr = pl.plan;
    let api = pl.ort_api;
    let kctx = pl.kctx;
    let stream = pl.stream;
    let slot = pl.slot;

    // Snapshot the closure schema (clone the small metadata) so no plan borrow overlaps translation.
    let (cfg, dyn_inputs, synth_ropes, ext_outputs, shared_kv) = {
        let c = slot.get(unsafe { &*plan_ptr });
        (
            c.config,
            c.dyn_inputs.clone(),
            c.synth_ropes.clone(),
            c.ext_outputs.clone(),
            c.shared_kv,
        )
    };

    let (res_raw, arena, kv_present) = {
        let plan = unsafe { &mut *plan_ptr };
        let mut tc = TranslationContext::new(plan, api, kctx, stream);
        if cfg.rope_as_data {
            // RoPE uses the pre-sliced cos/sin ROW placeholders + a matmul rotate-half, so the graph
            // carries no dynamic Slice (which shapeless `mlx_compile` cannot shape-infer).
            tc.set_compiled_trace(shared_kv);
        } else {
            // General trace: dynamic inputs are shapeless placeholders with no host data, so any
            // mid-graph host eval must fail the trace cleanly (=> eager) rather than eval a tracer.
            tc.set_general_trace();
        }

        // Seed the dynamic ctx input placeholders (closure inputs [0..ndyn)).
        let ndyn = dyn_inputs.len();
        for (i, di) in dyn_inputs.iter().enumerate() {
            let mut a = unsafe { mlxsys::mlx_array_new() };
            unsafe { mlxsys::mlx_vector_array_get(&mut a, input, i) };
            let raw = tc.keep(Array::from_raw(a));
            tc.seed(di.name.clone(), raw);
        }
        if cfg.rope_as_data {
            // Seed the pre-sliced RoPE cos/sin row placeholders (closure inputs [ndyn..ndyn+nsynth)).
            for (j, sr) in synth_ropes.iter().enumerate() {
                let mut a = unsafe { mlxsys::mlx_array_new() };
                unsafe { mlxsys::mlx_vector_array_get(&mut a, input, ndyn + j) };
                let raw = tc.keep(Array::from_raw(a));
                tc.seed(sr.key.clone(), raw);
            }
            // Shared-buffer: seed the live `valid_past` scalar (closure input [ndyn+nsynth]).
            if shared_kv {
                let mut a = unsafe { mlxsys::mlx_array_new() };
                unsafe { mlxsys::mlx_vector_array_get(&mut a, input, ndyn + synth_ropes.len()) };
                let raw = tc.keep(Array::from_raw(a));
                tc.seed(crate::engine::GQA_VALID_PAST_KEY.to_string(), raw);
            }
        }

        let res = tc.run_trace(&ext_outputs, cfg.contiguous_outputs)?;
        let kv_present = tc.take_compiled_kv_present();
        let arena = tc.take_arena();
        (res.into_raw(), arena, kv_present)
        // `tc` dropped here (its arena already taken) — releases the plan borrow.
    };

    // Hand the trace's transient handles + discovered KV-present set to the slot so the compiled
    // graph (walked after this thunk returns) keeps its inputs alive; they are freed once with the
    // plan.
    unsafe {
        let c = slot.get_mut(&mut *plan_ptr);
        c.trace_transient.extend(arena);
        if cfg.delta_copyout {
            c.kv_present_names = kv_present;
        }
        *out = res_raw;
    }
    Ok(())
}
