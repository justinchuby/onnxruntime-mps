//! Compiled-decode fast path (`mlx_compile`) — the #1 decode perf lever.
//!
//! For decode (query seq-len S==1) the graph STRUCTURE is invariant across steps: only input DATA
//! and the KV length grow. The eager path (`engine::TranslationContext::execute`) rebuilds and
//! dispatches all ~393 graph kernels every token. Instead we trace the whole subgraph into an
//! `mlx_closure` over its dynamic inputs ONCE, compile it *shapeless* (so a growing KV length never
//! triggers a recompile), cache the compiled closure on the plan, and on each subsequent decode step
//! just apply the compiled closure to the new inputs — fusing the per-token kernels into far fewer
//! launches. This is a faithful port of the C++ EP's `ExecuteCompiledDecode` / `BuildCompiledClosure`
//! / `TraceThunk`.
//!
//! Two shapeless-compile tricks are ported verbatim (see the C++ blueprint):
//!   * **RoPE rotate-half via a `[hd,hd]` matmul** (only when `rotary_dim == head_dim`) so the
//!     compiled graph carries no Slice (which shapeless compile cannot shape-infer). Driven by
//!     `TranslationContext::rope_dynamic`; the eager path's `mlx_fast_rope` is left untouched.
//!   * **Pre-sliced cos/sin rows fed as synthetic closure inputs**, so the position offset is DATA
//!     (not baked) and the compiled graph never slices a cos/sin cache at a runtime position.
//!
//! CORRECTNESS: every path here falls back to the eager translator on any doubt (ineligible plan,
//! missing cache, apply/eval error) — the compiled path never crashes and never diverges.

use std::collections::HashSet;
use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;

use crate::engine::{
    copy_out_raw, dim_i32, mlx_dtype_from_onnx, read_ctx_input_raw, rope_row_key, DynInput, MlxError, OutRef,
    Plan, Src, SynthRope, TracePayload, TranslationContext,
};
use crate::mlx::{self, Array, Closure, VectorArray};
use crate::sys::mlx as mlxsys;
use crate::sys::ort;

/// Decide whether the compiled-decode fast path is allowed for this plan. Disabled by the
/// `ONNX_GENAI_MLX_NO_COMPILE` kill-switch (forces eager for debugging / numerical A-B) or when the
/// subgraph contains a control-flow node (its graph structure depends on runtime data).
pub fn compile_enabled(has_control_flow: bool) -> bool {
    let killed = std::env::var_os("ONNX_GENAI_MLX_NO_COMPILE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    !has_control_flow && !killed
}

/// Query sequence length S = trailing dim of the `input_ids` dynamic ctx input (decode => 1). Scans
/// the plan nodes directly so it works before the compiled closure's `dyn_inputs` list is built.
/// Returns `None` if the input is not found / has no shape.
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

/// Detect whether this decode session drives a fixed-capacity SHARED KV buffer (present aliased onto
/// past at a runtime-owned max length) as opposed to the growing past/present contract. Reads live
/// ctx once at compiled-closure build time: `cap` = a GQA past-KV cache's seq-axis (2) length, and
/// `total` = valid keys after this step (= `total_sequence_length[0]`, or the `attention_mask` width
/// when that scalar is computed in-subgraph). Shared iff `cap > total - S` (the buffer holds more
/// than the valid past). Returns `None` when the shape is not a recognizable decoder GQA.
fn detect_shared_kv(
    plan: &Plan,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
) -> Option<bool> {
    let s = detect_seq_len(api, kctx, plan).unwrap_or(1);
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
        } else {
            let mask_ci = plan.nodes.iter().flat_map(|n| n.inputs.iter()).find(|inp| {
                inp.source == Src::CtxInput && inp.name == "attention_mask"
            })?;
            match read_ctx_input_raw(api, kctx, mask_ci.ctx_index) {
                Ok((_d, shape, _t)) => *shape.get(1)? as i32,
                Err(_) => continue,
            }
        };
        return Some(cap > total - s);
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

/// Compiled-decode fast path. Returns:
///   * `Ok(true)`  — the decode step was handled by the compiled closure (outputs copied out).
///   * `Ok(false)` — the plan is not compile-eligible (caller must fall back to the eager path).
///   * `Err(_)`    — a hard failure copying results out.
///
/// The plan is accessed through `plan_ptr` (raw) so that NO Rust `&mut Plan` is alive across
/// `mlx_closure_apply` — the trace thunk (invoked synchronously on the first apply) needs its own
/// `&mut Plan`, so only one mutable access is ever live at a time (single-threaded, reentrant via
/// FFI, exactly like the C++ EP).
pub fn try_compiled_decode(
    plan_ptr: *mut Plan,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
) -> Result<bool, MlxError> {
    if !unsafe { &*plan_ptr }.compiled.enabled {
        return Ok(false);
    }
    // One-time discovery + shapeless compile.
    if !unsafe { &*plan_ptr }.compiled.attempted {
        unsafe { &mut *plan_ptr }.compiled.attempted = true;
        build_compiled_closure(plan_ptr, api, kctx, stream);
    }
    if !unsafe { &*plan_ptr }.compiled.valid {
        return Ok(false);
    }

    // Gather the dynamic ctx inputs (wrap live ORT data) in closure order.
    let mut arena: Vec<Array> = Vec::new();
    let mut input = VectorArray::new();
    {
        // Immutable snapshot of the closure input schema — no `&mut Plan` alive here.
        let plan = unsafe { &*plan_ptr };
        for di in &plan.compiled.dyn_inputs {
            let (data, shape, dtype) = read_ctx_input_raw(api, kctx, di.ctx_index)?;
            let ishape: Vec<i32> = shape.iter().map(|&d| dim_i32(d)).collect::<Result<_, _>>()?;
            let arr = Array::from_data(data, &ishape, mlx_dtype_from_onnx(dtype));
            input.append(arr.as_raw());
            arena.push(arr);
        }
    }

    // Current RoPE start = past-KV sequence length (grows each step; fed as DATA via the rows below).
    let past = {
        let plan = unsafe { &*plan_ptr };
        let idx = plan.compiled.rope_past_ctx_index as usize;
        let axis = plan.compiled.rope_past_axis as usize;
        let (_d, shape, _t) = read_ctx_input_raw(api, kctx, idx)?;
        if axis < shape.len() {
            shape[axis] as i32
        } else {
            0
        }
    };

    // Pre-slice each RoPE cos/sin cache at [past, past+1) and feed the FULL-width row (the half-width
    // slice duplicated across both halves) in — keeping the compiled graph static-shaped and free of
    // any Slice primitive. Done OUTSIDE the compiled graph so the growing position stays pure data.
    {
        let synth = unsafe { &*plan_ptr }.compiled.synth_ropes.clone();
        for sr in &synth {
            let cache_raw = match unsafe { &*plan_ptr }.cache.get(&sr.cache_name) {
                Some(a) => a.as_raw(),
                None => return Ok(false), // cache not resident yet -> eager
            };
            let half = shape_of(cache_raw).get(1).copied().unwrap_or(0);
            if half == 0 {
                return Ok(false);
            }
            let row = mk_slice(cache_raw, &[past, 0], &[past + 1, half], stream)?; // [1,half]
            let full = mk_concat(row.as_raw(), row.as_raw(), 1, stream)?; // [1,2*half]
            input.append(full.as_raw());
            arena.push(row);
            arena.push(full);
        }
    }

    // Refresh the live kernel context on the trace payload (only read during the one-time trace).
    unsafe {
        if let Some(p) = (&mut *plan_ptr).compiled.payload.as_mut() {
            p.kctx = kctx;
        }
    }

    // Take the compiled closure OUT of the plan so no `&mut Plan` field is borrowed across apply
    // (the trace thunk mutates the plan through `plan_ptr`).
    let closure = match unsafe { &mut *plan_ptr }.compiled.closure.take() {
        Some(c) => c,
        None => return Ok(false),
    };
    let apply_res = closure.apply(&input);
    unsafe { &mut *plan_ptr }.compiled.closure = Some(closure);

    let outs = match apply_res {
        Ok(v) => v,
        Err(_) => {
            unsafe { &mut *plan_ptr }.compiled.valid = false; // disable, fall back to eager
            return Ok(false);
        }
    };
    if mlx::eval(&outs).is_err() {
        unsafe { &mut *plan_ptr }.compiled.valid = false;
        return Ok(false);
    }

    let ext_len = unsafe { &*plan_ptr }.compiled.ext_outputs.len();
    if outs.size() != ext_len {
        unsafe { &mut *plan_ptr }.compiled.valid = false;
        return Ok(false);
    }
    for i in 0..ext_len {
        let a = outs.get(i);
        // Clone the small OutRef so we don't hold a plan borrow across the copy-out FFI.
        let o: OutRef = unsafe { &*plan_ptr }.compiled.ext_outputs[i].clone();
        copy_out_raw(api, kctx, &o, a.as_raw())?;
    }
    // `arena` (transient per-step inputs) drops here — after eval + copy-out have consumed them.
    Ok(true)
}

/// One-time discovery + shapeless compile of the decode closure. Populates the plan's `dyn_inputs`
/// (ordered dynamic ctx inputs = closure inputs), `ext_outputs` (boundary outputs in append order),
/// `synth_ropes` (pre-sliced cos/sin caches), and the RoPE-start source, then compiles the closure.
/// Leaves `plan.compiled.valid = false` (=> caller falls back to eager) if the plan is not eligible.
fn build_compiled_closure(
    plan_ptr: *mut Plan,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
) {
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
                    if rope_past < 0 && inp.name.contains(".key") {
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
        let mut synth_seen: HashSet<String> = HashSet::new();
        for node in &plan.nodes {
            if node.op_type != "GroupQueryAttention" || node.inputs.len() < 9 {
                continue;
            }
            let do_rotary =
                !node.ints.contains_key("do_rotary") || node.ints.get("do_rotary").copied() != Some(0);
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

    if dyn_inputs.is_empty() || ext_outputs.is_empty() || rope_past < 0 || synth_ropes.is_empty() {
        return; // not the expected decoder shape; stay on the eager path
    }

    // The compiled RoPE uses a [hd,hd] rotate-half matmul, which requires rotary_dim == head_dim
    // (rot == hd). Validate from the live ctx (head dim = last axis of the past-KV cache) vs the cos
    // cache width (half); fall back to eager for a partial-rotary head.
    let hd = match read_ctx_input_raw(api, kctx, rope_past as usize) {
        Ok((_d, shape, _t)) => shape.last().copied().unwrap_or(0) as i32,
        Err(_) => 0,
    };
    let half = {
        let plan = unsafe { &*plan_ptr };
        match plan.cache.get(&synth_ropes[0].cache_name) {
            Some(a) => {
                let sh = a.shape();
                if sh.len() >= 2 {
                    sh[1] as i32
                } else {
                    0
                }
            }
            None => 0, // cos/sin cache not resident yet (prefill should have populated it)
        }
    };
    if hd == 0 || half == 0 || 2 * half != hd {
        return; // partial-rotary head not supported by the compiled path
    }

    // Disable the compiled fast path for a fixed-capacity shared KV buffer. In that contract the
    // present output must be written IN PLACE at the (data-dependent) valid-past offset and emitted
    // at the buffer's full capacity, which the shapeless compiled trace cannot express (its concat
    // form would produce a cap+S present and fail ORT's pre-bound output-size check). Such sessions
    // run the eager shared-buffer path instead — still O(1)/token, just not fused. Detected once here
    // from live ctx: shared iff the past-KV capacity exceeds the valid-past length (= total keys - S).
    if detect_shared_kv(unsafe { &*plan_ptr }, api, kctx).unwrap_or(false) {
        return;
    }

    // Publish the discovered schema + a stable trace payload onto the plan.
    unsafe {
        let plan = &mut *plan_ptr;
        plan.compiled.dyn_inputs = dyn_inputs;
        plan.compiled.ext_outputs = ext_outputs;
        plan.compiled.synth_ropes = synth_ropes;
        plan.compiled.rope_past_ctx_index = rope_past;
        plan.compiled.rope_past_axis = 2;
        plan.compiled.payload = Some(Box::new(TracePayload {
            plan: plan_ptr,
            ort_api: api,
            kctx,
            stream,
        }));
    }

    let payload_ptr: *mut c_void = unsafe {
        (&mut *plan_ptr).compiled.payload.as_mut().unwrap().as_mut() as *mut TracePayload
            as *mut c_void
    };
    // Shapeless so the growing KV length never triggers a recompile. The trace thunk runs lazily on
    // the first `apply`.
    let base = Closure::new_func_payload(trace_thunk, payload_ptr);
    match Closure::compile(&base, true) {
        Ok(compiled) => {
            let plan = unsafe { &mut *plan_ptr };
            plan.compiled.closure = Some(compiled);
            plan.compiled.valid = true;
        }
        Err(_) => { /* stay eager */ }
    }
}

/// `mlx_closure` trace thunk (payload = [`TracePayload`]): seed each dynamic ctx input + pre-sliced
/// RoPE row as a placeholder from the closure input vector, translate the whole subgraph (RoPE in
/// its slice-free matmul form via `rope_dynamic`), and return the cast external boundary outputs.
/// Invoked lazily by mlx on the first `apply`. Never unwinds across the FFI boundary.
extern "C" fn trace_thunk(
    out: *mut mlxsys::mlx_vector_array,
    input: mlxsys::mlx_vector_array,
    payload: *mut c_void,
) -> std::os::raw::c_int {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| trace_body(out, input, payload)));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("[rust-mlx-ep] compiled-decode trace failed ({e}); falling back to eager");
            1
        }
        Err(_) => {
            eprintln!("[rust-mlx-ep] compiled-decode trace panicked; falling back to eager");
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

    // Snapshot the closure schema (clone the small metadata) so no plan borrow overlaps translation.
    let (dyn_inputs, synth_ropes, ext_outputs) = {
        let plan = unsafe { &*plan_ptr };
        (
            plan.compiled.dyn_inputs.clone(),
            plan.compiled.synth_ropes.clone(),
            plan.compiled.ext_outputs.clone(),
        )
    };

    let (res_raw, arena) = {
        let plan = unsafe { &mut *plan_ptr };
        let mut tc = TranslationContext::new_trace(plan, api, kctx, stream);

        // Seed the dynamic ctx input placeholders (closure inputs [0..ndyn)).
        let ndyn = dyn_inputs.len();
        for (i, di) in dyn_inputs.iter().enumerate() {
            let mut a = unsafe { mlxsys::mlx_array_new() };
            unsafe { mlxsys::mlx_vector_array_get(&mut a, input, i) };
            let raw = tc.keep(Array::from_raw(a));
            tc.seed(di.name.clone(), raw);
        }
        // Seed the pre-sliced RoPE cos/sin row placeholders (closure inputs [ndyn..)).
        for (j, sr) in synth_ropes.iter().enumerate() {
            let mut a = unsafe { mlxsys::mlx_array_new() };
            unsafe { mlxsys::mlx_vector_array_get(&mut a, input, ndyn + j) };
            let raw = tc.keep(Array::from_raw(a));
            tc.seed(sr.key.clone(), raw);
        }

        let res = tc.run_trace(&ext_outputs)?;
        let arena = tc.take_arena();
        (res.into_raw(), arena)
        // `tc` dropped here (its arena already taken) — releases the plan borrow.
    };

    // Hand the trace's transient handles to the plan so the compiled graph (walked after this thunk
    // returns) keeps its inputs alive; they are freed once with the plan.
    unsafe {
        (&mut *plan_ptr).compiled.trace_transient.extend(arena);
        *out = res_raw;
    }
    Ok(())
}
