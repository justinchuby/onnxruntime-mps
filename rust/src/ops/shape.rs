//! Shape / data-movement op handlers. Faithful port of the C++ `ops/shape.cc` (+ `WhereOp` from
//! `ops/math.cc`): Gather, GatherElements, ScatterElements, Concat, Reshape, Transpose, Unsqueeze,
//! Squeeze, Flatten, Expand, Slice, Split (multi-output), Tile, Pad, Identity, Range, Shape, Size,
//! ConstantOfShape and Where.
//!
//! Shape/axes/indices operands (Reshape shape, Slice starts/ends/axes/steps, Pad pads, Expand/Tile
//! shape, ConstantOfShape/Range scalars) are read from their constant inputs at translate time via
//! `RawHost`/`read_ints`. View-op boundary outputs (transpose/slice/expand/split/gather) are forced
//! contiguous before the shared CopyOut memcpy. Zero-size results (Pad/Expand) are re-materialised as
//! clean zeros arrays rather than rejected to CPU.

use crate::engine::{dim_i32, mlx_dtype_from_onnx, MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    is_int_index, is_mlx_float, is_mlx_numeric, is_movable, is_range_type, ClaimResult, NodeView,
    OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- small helpers -----------------------------------------------------------------------------

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

fn norm_axis(axis: i64, rank: i32) -> i32 {
    (if axis < 0 { axis + rank as i64 } else { axis }) as i32
}

fn clamp_i64(v: i64, lo: i64, hi: i64) -> i64 {
    v.max(lo).min(hi)
}

fn int_attr(n: &NodeDesc, name: &str, default: i64) -> i64 {
    n.ints.get(name).copied().unwrap_or(default)
}

/// Read a constant int64 parameter input (shape/axes/starts/...) at translate time.
fn read_ints(ctx: &mut TranslationContext, n: &NodeDesc, i: usize) -> Result<Vec<i64>, MlxError> {
    ctx.read_ints_eval(&n.inputs[i])
}

/// Read an axes/split-style list from either the opset-13 input or the opset<13 INTS attribute.
fn read_list_input_or_attr(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    input_idx: usize,
    attr: &str,
) -> Result<(bool, Vec<i64>), MlxError> {
    if present(n, input_idx) {
        return Ok((true, read_ints(ctx, n, input_idx)?));
    }
    if let Some(v) = n.int_arrays.get(attr) {
        return Ok((true, v.clone()));
    }
    Ok((false, Vec::new()))
}

fn where_op(
    ctx: &mut TranslationContext,
    cond: mlx::mlx_array,
    x: mlx::mlx_array,
    y: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_where(res, cond, x, y, s) })
}

fn resize_src_coord(
    mode: &str,
    output_index: i64,
    scale: f64,
    input_len: i64,
    output_len: i64,
) -> f64 {
    match mode {
        "align_corners" => {
            if output_len == 1 {
                0.0
            } else {
                output_index as f64 * (input_len - 1) as f64 / (output_len - 1) as f64
            }
        }
        "asymmetric" => output_index as f64 / scale,
        "pytorch_half_pixel" if output_len == 1 => 0.0,
        _ => (output_index as f64 + 0.5) / scale - 0.5,
    }
}

fn resize_nearest_index(mode: &str, source: f64, input_len: i64) -> i32 {
    let index = match mode {
        "floor" => source.floor(),
        "ceil" => source.ceil(),
        "round_prefer_ceil" => (source + 0.5).floor(),
        _ => (source - 0.5).ceil(),
    } as i64;
    clamp_i64(index, 0, input_len - 1) as i32
}

/// Normalize + wrap negative gather indices into [0, dim) as int32 (the take/gather index dtype).
fn normalize_indices(
    ctx: &mut TranslationContext,
    indices: mlx::mlx_array,
    dim: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let idx = ctx.astype(indices, mlx::mlx_dtype__MLX_INT32)?;
    let dim_s = ctx.scalar_i32(dim);
    let zero_s = ctx.scalar_i32(0);
    let neg = ctx.binary(mlx::mlx_less, idx, zero_s)?;
    let wrapped = ctx.binary(mlx::mlx_add, idx, dim_s)?;
    where_op(ctx, neg, wrapped, idx)
}

// ---- handlers ----------------------------------------------------------------------------------

fn gather_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let indices = ctx.resolve(&n.inputs[1])?;
    let rank = ctx.ndim(data) as i32;
    let axis = norm_axis(int_attr(n, "axis", 0), rank);
    let dim = ctx.dim(data, axis);
    let idx = normalize_indices(ctx, indices, dim)?;
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, data, idx, axis, s) })?;
    let r = ctx.contiguous(r)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn gather_elements_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let indices = ctx.resolve(&n.inputs[1])?;
    let rank = ctx.ndim(data) as i32;
    let axis = norm_axis(int_attr(n, "axis", 0), rank);
    let dim = ctx.dim(data, axis);
    let idx = normalize_indices(ctx, indices, dim)?;
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_take_along_axis(res, data, idx, axis, s) })?;
    let r = ctx.contiguous(r)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn scatter_elements_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let indices = ctx.resolve(&n.inputs[1])?;
    let updates = ctx.resolve(&n.inputs[2])?;
    let rank = ctx.ndim(data) as i32;
    let axis = norm_axis(int_attr(n, "axis", 0), rank);
    let dim = ctx.dim(data, axis);
    let idx = normalize_indices(ctx, indices, dim)?;
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_put_along_axis(res, data, idx, updates, axis, s) })?;
    let r = ctx.contiguous(r)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn concat_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut vec = VectorArray::new();
    let mut rank = 0i32;
    for (i, inp) in n.inputs.iter().enumerate() {
        let a = ctx.resolve(inp)?;
        if i == 0 {
            rank = ctx.ndim(a) as i32;
        }
        vec.append(a);
    }
    let axis = norm_axis(int_attr(n, "axis", 0), rank);
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_concatenate_axis(res, vec.as_raw(), axis, s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn reshape_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let shape = read_ints(ctx, n, 1)?;
    let in_shape = ctx.shape_of(data);
    let target: Vec<i32> = shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 && i < in_shape.len() {
                Ok(in_shape[i]) // allowzero=0: copy the input dim
            } else {
                dim_i32(d) // preserves -1 (infer); errors on >i32 dims
            }
        })
        .collect::<Result<_, _>>()?;
    let r = ctx.reshape(data, &target)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn transpose_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(data) as i32;
    let perm: Vec<i32> = match n.int_arrays.get("perm") {
        Some(p) => p.iter().map(|&x| norm_axis(x, rank)).collect(),
        None => (0..rank).rev().collect(),
    };
    let t = ctx.transpose(data, &perm)?;
    // Zero-copy strided view; materialised to contiguous only if it reaches the output boundary.
    ctx.bind(&n.outputs[0], t);
    Ok(())
}

fn unsqueeze_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let (_, axes) = read_list_input_or_attr(ctx, n, 1, "axes")?;
    let out_rank = ctx.ndim(data) as i32 + axes.len() as i32;
    let mut a: Vec<i32> = axes.iter().map(|&x| norm_axis(x, out_rank)).collect();
    a.sort_unstable();
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_expand_dims_axes(res, data, a.as_ptr(), a.len(), s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn squeeze_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(data) as i32;
    let (have, axes) = read_list_input_or_attr(ctx, n, 1, "axes")?;
    let r = if have {
        let a: Vec<i32> = axes.iter().map(|&x| norm_axis(x, rank)).collect();
        ctx.emit(|res, s| unsafe { mlx::mlx_squeeze_axes(res, data, a.as_ptr(), a.len(), s) })?
    } else {
        ctx.emit(|res, s| unsafe { mlx::mlx_squeeze(res, data, s) })?
    };
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn flatten_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(data);
    let rank = shape.len() as i64;
    let mut axis = int_attr(n, "axis", 1);
    if axis < 0 {
        axis += rank;
    }
    let mut outer: i32 = 1;
    let mut inner: i32 = 1;
    for (i, &d) in shape.iter().enumerate() {
        if (i as i64) < axis {
            outer *= d;
        } else {
            inner *= d;
        }
    }
    let r = ctx.reshape(data, &[outer, inner])?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn expand_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let target = read_ints(ctx, n, 1)?;
    let in_shape = ctx.shape_of(data);
    let out_rank = in_shape.len().max(target.len());
    let mut result = vec![0i32; out_rank];
    let mut incompatible = false;
    for i in 0..out_rank {
        let d_in = if i + in_shape.len() < out_rank {
            1i64
        } else {
            in_shape[i - (out_rank - in_shape.len())] as i64
        };
        let d_t = if i + target.len() < out_rank {
            1i64
        } else {
            target[i - (out_rank - target.len())]
        };
        let d_out = if d_in == 1 {
            d_t
        } else if d_t == 1 {
            d_in
        } else if d_in == d_t {
            d_in
        } else {
            incompatible = true;
            d_in.max(d_t)
        };
        result[i] = dim_i32(d_out)?;
    }
    let dt = ctx.dtype_of(data);
    let r = if incompatible {
        ctx.zeros(&result, dt)?
    } else {
        ctx.emit(|res, s| unsafe { mlx::mlx_broadcast_to(res, data, result.as_ptr(), result.len(), s) })?
    };
    // Broadcast is a zero-copy stride-0 view; contiguous is deferred to the output boundary.
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn slice_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(data);
    let rank = shape.len();
    let starts = read_ints(ctx, n, 1)?;
    let ends = read_ints(ctx, n, 2)?;
    let axes: Vec<i64> = if present(n, 3) {
        read_ints(ctx, n, 3)?
    } else {
        (0..starts.len() as i64).collect()
    };
    let steps: Vec<i64> = if present(n, 4) {
        read_ints(ctx, n, 4)?
    } else {
        vec![1; starts.len()]
    };

    let mut start = vec![0i32; rank];
    let mut stop = shape.clone();
    let mut stride = vec![1i32; rank];
    for i in 0..starts.len() {
        let ax = norm_axis(axes[i], rank as i32) as usize;
        let dim = shape[ax] as i64;
        let s = if starts[i] < 0 { starts[i] + dim } else { starts[i] };
        let e = if ends[i] < 0 { ends[i] + dim } else { ends[i] };
        start[ax] = clamp_i64(s, 0, dim) as i32;
        stop[ax] = clamp_i64(e, 0, dim) as i32;
        stride[ax] = steps[i] as i32;
    }
    let r = ctx.emit(|res, s| unsafe {
        mlx::mlx_slice(
            res,
            data,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            stride.as_ptr(),
            stride.len(),
            s,
        )
    })?;
    // Slice is a zero-copy strided view; contiguous is deferred to the output boundary.
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn split_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(data) as i32;
    let axis = norm_axis(int_attr(n, "axis", 0), rank);
    let num_out = n.outputs.len();

    let sizes: Vec<i64> = if present(n, 1) {
        read_ints(ctx, n, 1)?
    } else {
        n.int_arrays.get("split").cloned().unwrap_or_default()
    };

    let parts = if !sizes.is_empty() {
        // Cumulative boundary indices (exclusive of the final section) for mlx_split_sections.
        let mut indices: Vec<i32> = Vec::new();
        let mut acc = 0i32;
        for i in 0..sizes.len().saturating_sub(1) {
            acc += sizes[i] as i32;
            indices.push(acc);
        }
        let mut pv = VectorArray::new();
        let rc = unsafe {
            mlx::mlx_split_sections(
                pv.as_mut_ptr(),
                data,
                indices.as_ptr(),
                indices.len(),
                axis,
                ctx.stream(),
            )
        };
        if rc != 0 {
            return Err("mlx_split_sections failed".to_string());
        }
        pv
    } else {
        let mut pv = VectorArray::new();
        let rc = unsafe { mlx::mlx_split(pv.as_mut_ptr(), data, num_out as i32, axis, ctx.stream()) };
        if rc != 0 {
            return Err("mlx_split failed".to_string());
        }
        pv
    };

    let count = parts.size();
    for i in 0..count.min(num_out) {
        let part = ctx.keep(parts.get(i));
        // Split parts are zero-copy strided views; contiguous is deferred to the output boundary.
        ctx.bind(&n.outputs[i], part);
    }
    Ok(())
}

fn tile_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let repeats = read_ints(ctx, n, 1)?;
    let reps: Vec<i32> = repeats.iter().map(|&x| x as i32).collect();
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_tile(res, data, reps.as_ptr(), reps.len(), s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn pad_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(data) as i32;
    let pads = read_ints(ctx, n, 1)?;
    let axes: Vec<i64> = if present(n, 3) {
        read_ints(ctx, n, 3)?
    } else {
        (0..rank as i64).collect()
    };
    let naxes = axes.len();
    let mut ax = vec![0i32; naxes];
    let mut low = vec![0i32; naxes];
    let mut high = vec![0i32; naxes];
    for i in 0..naxes {
        ax[i] = norm_axis(axes[i], rank);
        low[i] = pads[i] as i32;
        high[i] = pads[i + naxes] as i32;
    }
    let dt = ctx.dtype_of(data);
    let pad_value = if present(n, 2) {
        let cv = ctx.resolve(&n.inputs[2])?;
        ctx.astype(cv, dt)?
    } else {
        let z = ctx.scalar_i64(0);
        ctx.astype(z, dt)?
    };
    let mode = b"constant\0";
    let r = ctx.emit(|res, s| unsafe {
        mlx::mlx_pad(
            res,
            data,
            ax.as_ptr(),
            ax.len(),
            low.as_ptr(),
            low.len(),
            high.as_ptr(),
            high.len(),
            pad_value,
            mode.as_ptr() as *const std::os::raw::c_char,
            s,
        )
    })?;
    // A zero-sized pad result has no backing buffer; re-materialise as clean zeros for CopyOut.
    if ctx.size_of(r) == 0 {
        let shp = ctx.shape_of(r);
        let rdt = ctx.dtype_of(r);
        let z = ctx.zeros(&shp, rdt)?;
        ctx.bind(&n.outputs[0], z);
        return Ok(());
    }
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn identity_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    ctx.bind(&n.outputs[0], a);
    Ok(())
}

fn range_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let start = read_range_scalar(ctx, n, 0)?;
    let limit = read_range_scalar(ctx, n, 1)?;
    let delta = read_range_scalar(ctx, n, 2)?;
    let dt = mlx_dtype_from_onnx(n.outputs[0].otype);
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_arange(res, start, limit, delta, dt, s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn read_range_scalar(ctx: &TranslationContext, n: &NodeDesc, i: usize) -> Result<f64, MlxError> {
    let h = ctx.raw_host(&n.inputs[i])?;
    if h.count != 1 || h.data.is_null() {
        return Err("Range expected a scalar initializer".to_string());
    }
    match h.dtype {
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            Ok(unsafe { *(h.data as *const i16) } as f64)
        }
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            Ok(unsafe { *(h.data as *const i32) } as f64)
        }
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            Ok(unsafe { *(h.data as *const i64) } as f64)
        }
        _ => Err("Range initializer dtype is not supported".to_string()),
    }
}

fn shape_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let input_shape = ctx.shape_of(data);
    let rank = input_shape.len() as i64;
    let start_attr = int_attr(n, "start", 0);
    let end_attr = int_attr(n, "end", rank);
    let (start, end) = shape_interval(rank, start_attr, end_attr);
    let result: Vec<i64> = (start..end).map(|i| input_shape[i as usize] as i64).collect();
    let out = ctx.from_host_i64(&result, &[result.len() as i32]);
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn size_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let data = ctx.resolve(&n.inputs[0])?;
    let size = ctx.size_of(data) as i64;
    let out = ctx.from_host_i64(&[size], &[]);
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn shape_interval(rank: i64, start: i64, end: i64) -> (i64, i64) {
    let mut s = if start < 0 { start + rank } else { start };
    let mut e = if end < 0 { end + rank } else { end };
    s = clamp_i64(s, 0, rank);
    e = clamp_i64(e, 0, rank);
    if e < s {
        e = s;
    }
    (s, e)
}

fn constant_of_shape_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let shape_i64 = read_ints(ctx, n, 0)?;
    let s: Vec<i32> = shape_i64.iter().map(|&x| x as i32).collect();
    let out_type = n.outputs[0].otype;
    let r = if out_type == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
        let value = ctx.scalar_i64(-1);
        ctx.emit(|res, st| unsafe { mlx::mlx_full(res, s.as_ptr(), s.len(), value, mlx::mlx_dtype__MLX_INT64, st) })?
    } else {
        ctx.zeros(&s, mlx::mlx_dtype__MLX_FLOAT32)?
    };
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn where_handler(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let cond = ctx.resolve(&n.inputs[0])?;
    let x = ctx.resolve(&n.inputs[1])?;
    let y = ctx.resolve(&n.inputs[2])?;
    let r = where_op(ctx, cond, x, y)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn resize_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut data = ctx.resolve(&n.inputs[0])?;
    let input_shape = ctx.shape_of(data);
    let rank = input_shape.len();
    let output_dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    let mode = n
        .strings
        .get("mode")
        .map(String::as_str)
        .unwrap_or("nearest");
    let coordinate_mode = n
        .strings
        .get("coordinate_transformation_mode")
        .map(String::as_str)
        .unwrap_or("half_pixel");
    let nearest_mode = n
        .strings
        .get("nearest_mode")
        .map(String::as_str)
        .unwrap_or("round_prefer_floor");

    let (output_lengths, scales): (Vec<i32>, Vec<f64>) = if present(n, 3) {
        let sizes = read_ints(ctx, n, 3)?;
        (
            sizes.iter().map(|&size| size as i32).collect(),
            sizes
                .iter()
                .enumerate()
                .map(|(i, &size)| size as f64 / input_shape[i] as f64)
                .collect(),
        )
    } else {
        let scales_host = ctx.raw_host(&n.inputs[2])?;
        if scales_host.data.is_null() || scales_host.count != rank {
            return Err("MLX Resize requires constant float32 scales".to_string());
        }
        let scales = unsafe {
            std::slice::from_raw_parts(scales_host.data as *const f32, scales_host.count)
        };
        (
            scales
                .iter()
                .enumerate()
                .map(|(i, &scale)| (scale as f64 * input_shape[i] as f64).floor() as i32)
                .collect(),
            scales.iter().map(|&scale| scale as f64).collect(),
        )
    };

    let restore_dtype = ctx.dtype_of(data);
    let use_f32 = mode == "linear" || restore_dtype == mlx::mlx_dtype__MLX_BFLOAT16;
    if use_f32 {
        data = ctx.astype(data, mlx::mlx_dtype__MLX_FLOAT32)?;
    }
    for axis in 0..rank {
        let input_len = input_shape[axis] as i64;
        let output_len = output_lengths[axis] as i64;
        if input_len == output_len {
            continue;
        }
        if mode == "nearest" {
            let indices: Vec<i32> = (0..output_len)
                .map(|i| {
                    resize_nearest_index(
                        nearest_mode,
                        resize_src_coord(coordinate_mode, i, scales[axis], input_len, output_len),
                        input_len,
                    )
                })
                .collect();
            let index = ctx.keep(Array::from_data(
                indices.as_ptr() as *const std::os::raw::c_void,
                &[output_len as i32],
                mlx::mlx_dtype__MLX_INT32,
            ));
            data =
                ctx.emit(|res, s| unsafe { mlx::mlx_take_axis(res, data, index, axis as i32, s) })?;
            continue;
        }

        let mut lower = Vec::with_capacity(output_len as usize);
        let mut upper = Vec::with_capacity(output_len as usize);
        let mut lower_weight = Vec::with_capacity(output_len as usize);
        let mut upper_weight = Vec::with_capacity(output_len as usize);
        for i in 0..output_len {
            let source = resize_src_coord(coordinate_mode, i, scales[axis], input_len, output_len);
            let floor = source.floor();
            lower.push(clamp_i64(floor as i64, 0, input_len - 1) as i32);
            upper.push(clamp_i64(floor as i64 + 1, 0, input_len - 1) as i32);
            upper_weight.push((source - floor) as f32);
            lower_weight.push((1.0 - (source - floor)) as f32);
        }
        let lower_index = ctx.keep(Array::from_data(
            lower.as_ptr() as *const std::os::raw::c_void,
            &[output_len as i32],
            mlx::mlx_dtype__MLX_INT32,
        ));
        let upper_index = ctx.keep(Array::from_data(
            upper.as_ptr() as *const std::os::raw::c_void,
            &[output_len as i32],
            mlx::mlx_dtype__MLX_INT32,
        ));
        let mut weight_shape = vec![1i32; rank];
        weight_shape[axis] = output_len as i32;
        let lw = ctx.keep(Array::from_data(
            lower_weight.as_ptr() as *const std::os::raw::c_void,
            &weight_shape,
            mlx::mlx_dtype__MLX_FLOAT32,
        ));
        let uw = ctx.keep(Array::from_data(
            upper_weight.as_ptr() as *const std::os::raw::c_void,
            &weight_shape,
            mlx::mlx_dtype__MLX_FLOAT32,
        ));
        let lo = ctx
            .emit(|res, s| unsafe { mlx::mlx_take_axis(res, data, lower_index, axis as i32, s) })?;
        let hi = ctx
            .emit(|res, s| unsafe { mlx::mlx_take_axis(res, data, upper_index, axis as i32, s) })?;
        let lo = ctx.mul(lo, lw)?;
        let hi = ctx.mul(hi, uw)?;
        data = ctx.add(lo, hi)?;
    }
    if use_f32 {
        data = ctx.astype(data, output_dtype)?;
    }
    let data = ctx.contiguous(data)?;
    ctx.bind(&n.outputs[0], data);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn movable_io(node: &NodeView) -> Option<(ort::ONNXTensorElementDataType, ort::ONNXTensorElementDataType)> {
    match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => Some((i.dtype, o.dtype)),
        _ => None,
    }
}

fn gather_like_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "Gather/GatherElements expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    let idx = match node.input_info(1) {
        Some(i) => i.dtype,
        None => deny!("missing tensor type/shape info on the indices input"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    require!(
        is_int_index(idx),
        "indices dtype {} must be int32/int64",
        crate::registry::ort_dtype_name(idx)
    );
    Ok(())
}

fn scatter_elements_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "ScatterElements expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        node.string_attr("reduction", "none") == "none",
        "ScatterElements: only reduction=none is claimed (add/mul/min/max reductions stay on CPU)"
    );
    let (data, indices, updates, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    // mlx_put_along_axis's GPU kernel aborts on int64 payloads → keep to MLX float dtypes.
    require!(
        is_mlx_float(data.dtype),
        "data dtype {} unsupported: mlx_put_along_axis's GPU kernel needs an MLX float (fp32/fp16/bf16)",
        crate::registry::ort_dtype_name(data.dtype)
    );
    require!(
        is_int_index(indices.dtype),
        "indices dtype {} must be int32/int64",
        crate::registry::ort_dtype_name(indices.dtype)
    );
    require!(
        updates.dtype == data.dtype && out.dtype == data.dtype,
        "updates/output dtype must match data dtype {} (got updates {}, out {})",
        crate::registry::ort_dtype_name(data.dtype),
        crate::registry::ort_dtype_name(updates.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        !data.shape.is_empty(),
        "data must have rank >= 1 (scalar data is unsupported)"
    );
    require!(
        indices.shape == updates.shape,
        "indices shape {:?} must equal updates shape {:?}",
        indices.shape,
        updates.shape
    );
    require!(
        indices.shape.len() == data.shape.len(),
        "indices rank {} must equal data rank {}",
        indices.shape.len(),
        data.shape.len()
    );
    require!(
        out.shape == data.shape,
        "output shape {:?} must equal data shape {:?}",
        out.shape,
        data.shape
    );
    let rank = data.shape.len() as i64;
    let axis = node.int_attr("axis", 0);
    require!(
        axis >= -rank && axis < rank,
        "axis {axis} is out of range for rank {rank}"
    );
    let ax = norm_axis(axis, rank as i32) as usize;
    for i in 0..data.shape.len() {
        require!(
            i == ax || indices.shape[i] <= data.shape[i],
            "on non-axis dim {i}, indices extent {} must be <= data extent {}",
            indices.shape[i],
            data.shape[i]
        );
    }
    Ok(())
}

fn concat_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() != 0 && node.num_outputs() == 1,
        "Concat expects 1+ inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let out = match node.output_info(0) {
        Some(o) if is_movable(o.dtype) => o.dtype,
        Some(o) => deny!(
            "output dtype {} is not a movable dtype supported on GPU",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("missing output tensor type/shape info"),
    };
    for i in 0..node.num_inputs() {
        match node.input_info(i) {
            Some(info) if info.dtype == out => {}
            Some(info) => deny!(
                "input[{i}] dtype {} must match output dtype {}",
                crate::registry::ort_dtype_name(info.dtype),
                crate::registry::ort_dtype_name(out)
            ),
            None => deny!("input[{i}] has no tensor type/shape info"),
        }
    }
    Ok(())
}

fn reshape_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "Reshape expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    require!(
        node.input_info(1).map(|i| i.dtype)
            == Some(ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
        "Reshape `shape` input must be int64 (runtime shape-derived values resolve at trace time; a data-dependent shape falls back to eager, and a cyclic partition is dropped to CPU)"
    );
    require!(
        node.int_attr("allowzero", 0) == 0,
        "Reshape: allowzero=1 is unsupported (only the default copy-input-dim-on-zero behavior is claimed)"
    );
    Ok(())
}

fn transpose_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the input or output"),
    };
    require!(
        is_movable(data),
        "input dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match input dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    Ok(())
}

fn unsqueeze_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() != 0 && node.num_outputs() == 1,
        "Unsqueeze/Squeeze expects 1+ inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    if node.num_inputs() == 2 {
        require!(
            node.is_const_int64(1),
            "Unsqueeze/Squeeze: only a constant int64 `axes` initializer is claimed; this node's is a runtime value — stays on CPU"
        );
        return Ok(());
    }
    require!(
        node.num_inputs() == 1,
        "Unsqueeze/Squeeze: expected 1 or 2 inputs, got {}",
        node.num_inputs()
    );
    Ok(())
}

fn squeeze_claim(node: &NodeView) -> ClaimResult {
    unsqueeze_claim(node)
}

fn flatten_claim(node: &NodeView) -> ClaimResult {
    transpose_claim(node)
}

fn expand_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "Expand expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    require!(
        node.is_const_int64(1),
        "Expand: only a constant int64 `shape` initializer is claimed; this node's is a runtime value — stays on CPU"
    );
    Ok(())
}

fn slice_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        nin >= 3 && nin <= 5 && node.num_outputs() == 1,
        "Slice expects 3-5 inputs and 1 output, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    require!(
        node.is_const_int64(1) && node.is_const_int64(2),
        "Slice: `starts` and `ends` must be constant int64 initializers; this node's are runtime values — stays on CPU"
    );
    if nin >= 4 && node.input_present(3) {
        require!(
            node.is_const_int64(3),
            "Slice: `axes` must be a constant int64 initializer; this node's is a runtime value — stays on CPU"
        );
    }
    if nin >= 5 && node.input_present(4) {
        match node.read_const_int64(4) {
            Some(steps) => {
                require!(
                    steps.iter().all(|&st| st >= 1),
                    "Slice: negative/zero `steps` (reverse or strided slicing) are unsupported — left to CPU"
                );
            }
            None => deny!(
                "Slice: `steps` must be a constant int64 initializer; this node's is a runtime value — stays on CPU"
            ),
        }
    }
    Ok(())
}

fn split_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        nin != 0 && nin <= 2 && node.num_outputs() != 0,
        "Split expects 1-2 inputs and 1+ outputs, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let data = match node.input_info(0) {
        Some(d) if is_movable(d.dtype) && !d.shape.is_empty() => d,
        Some(d) => deny!(
            "Split: data dtype {} must be a movable GPU dtype and data must be non-scalar (shape {:?})",
            crate::registry::ort_dtype_name(d.dtype),
            d.shape
        ),
        None => deny!("missing data tensor type/shape info"),
    };
    for i in 0..node.num_outputs() {
        match node.output_info(i) {
            Some(o) if o.dtype == data.dtype => {}
            Some(o) => deny!(
                "Split: output[{i}] dtype {} must match data dtype {}",
                crate::registry::ort_dtype_name(o.dtype),
                crate::registry::ort_dtype_name(data.dtype)
            ),
            None => deny!("Split: output[{i}] has no tensor type/shape info"),
        }
    }
    let rank = data.shape.len() as i64;
    let axis = node.int_attr("axis", 0);
    require!(
        axis >= -rank && axis < rank,
        "axis {axis} is out of range for rank {rank}"
    );
    let axis_size = data.shape[norm_axis(axis, rank as i32) as usize];

    // Explicit per-section sizes: opset-13 `split` input or opset<13 `split` INTS attribute.
    let (have_sizes, sizes) = if nin == 2 && node.input_present(1) {
        match node.read_const_int64(1) {
            Some(s) => (true, s),
            None => deny!(
                "Split: dynamic `split` sizes are unsupported; only a constant int64 `split` initializer is claimed — stays on CPU"
            ),
        }
    } else {
        let (present, s) = node.ints_attr("split");
        (present, s)
    };
    if have_sizes {
        let mut total = 0i64;
        for &s in &sizes {
            require!(s >= 0, "Split: `split` sizes must be non-negative (got {s})");
            total += s;
        }
        require!(
            sizes.len() == node.num_outputs(),
            "Split: `split` count {} must equal the number of outputs {}",
            sizes.len(),
            node.num_outputs()
        );
        require!(
            total == axis_size,
            "Split: `split` sizes sum {total} must equal axis extent {axis_size}"
        );
        return Ok(());
    }
    // Equal split: MLX requires the axis to divide evenly by the output count.
    require!(
        axis_size % node.num_outputs() as i64 == 0,
        "Split: equal split requires axis extent {axis_size} to divide evenly by output count {}",
        node.num_outputs()
    );
    Ok(())
}

fn tile_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "Tile expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    require!(
        node.is_const_int64(1),
        "Tile: only a constant `repeats` initializer is claimed; this node's is a runtime value — stays on CPU"
    );
    Ok(())
}

fn pad_claim(node: &NodeView) -> ClaimResult {
    let nin = node.num_inputs();
    require!(
        nin >= 2 && nin <= 4 && node.num_outputs() == 1,
        "Pad expects 2-4 inputs and 1 output, got {}in/{}out",
        nin,
        node.num_outputs()
    );
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => deny!("missing tensor type/shape info on the data input or output"),
    };
    require!(
        is_movable(data),
        "data dtype {} is not a movable dtype supported on GPU",
        crate::registry::ort_dtype_name(data)
    );
    require!(
        out == data,
        "output dtype {} must match data dtype {}",
        crate::registry::ort_dtype_name(out),
        crate::registry::ort_dtype_name(data)
    );
    require!(
        node.string_attr("mode", "constant") == "constant",
        "Pad: only constant, non-negative `pads` with mode=constant are claimed (runtime pads, negative/cropping pads, or reflect/edge modes stay on CPU)"
    );
    match node.read_const_int64(1) {
        Some(pads) => {
            require!(
                pads.iter().all(|&p| p >= 0),
                "Pad: only constant, non-negative `pads` with mode=constant are claimed (runtime pads, negative/cropping pads, or reflect/edge modes stay on CPU)"
            );
        }
        None => deny!(
            "Pad: only constant, non-negative `pads` with mode=constant are claimed (runtime pads, negative/cropping pads, or reflect/edge modes stay on CPU)"
        ),
    }
    if nin >= 3 && node.input_present(2) {
        match node.input_info(2) {
            Some(cv) if cv.dtype == data => {}
            Some(cv) => deny!(
                "Pad: constant_value dtype {} must match data dtype {}",
                crate::registry::ort_dtype_name(cv.dtype),
                crate::registry::ort_dtype_name(data)
            ),
            None => deny!("Pad: constant_value input has no tensor type/shape info"),
        }
    }
    if nin >= 4 && node.input_present(3) {
        require!(
            node.is_const_int64(3),
            "Pad: `axes` must be a constant int64 initializer; this node's is a runtime value — stays on CPU"
        );
    }
    Ok(())
}

fn identity_claim(node: &NodeView) -> ClaimResult {
    transpose_claim(node)
}

fn range_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "Range expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let ty = match node.input_info(0) {
        Some(i) if is_range_type(i.dtype) => i.dtype,
        Some(i) => deny!(
            "Range: start dtype {} unsupported (only int32/int64/fp32 range types are claimed)",
            crate::registry::ort_dtype_name(i.dtype)
        ),
        None => deny!("missing `start` tensor type/shape info"),
    };
    let out = match node.output_info(0) {
        Some(o) if o.dtype == ty && o.shape.len() == 1 => o,
        Some(o) => deny!(
            "Range: output dtype {} must equal start dtype {} and output must be 1-D (shape {:?})",
            crate::registry::ort_dtype_name(o.dtype),
            crate::registry::ort_dtype_name(ty),
            o.shape
        ),
        None => deny!("missing output tensor type/shape info"),
    };
    let (start, limit, delta) = match (
        node.read_const_scalar_f64(0),
        node.read_const_scalar_f64(1),
        node.read_const_scalar_f64(2),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => deny!(
            "Range: `start`/`limit`/`delta` must be constant scalar initializers; this node's are runtime values — stays on CPU"
        ),
    };
    require!(delta != 0.0, "Range: `delta` must be non-zero");
    let count = ((limit - start) / delta).ceil().max(0.0);
    require!(
        count.is_finite() && count <= i32::MAX as f64,
        "Range: computed element count {count} is not finite or exceeds i32::MAX"
    );
    require!(
        out.shape[0] == count as i64,
        "Range: output extent {} must equal computed element count {}",
        out.shape[0],
        count as i64
    );
    Ok(())
}

fn shape_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "Shape expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let data = match node.input_info(0) {
        Some(d) if is_movable(d.dtype) => d,
        Some(d) => deny!(
            "data dtype {} is not a movable dtype supported on GPU",
            crate::registry::ort_dtype_name(d.dtype)
        ),
        None => deny!("missing data tensor type/shape info"),
    };
    let out = match node.output_info(0) {
        Some(o)
            if o.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && o.shape.len() == 1 =>
        {
            o
        }
        Some(o) => deny!(
            "Shape: output must be a 1-D int64 tensor (got {} shape {:?})",
            crate::registry::ort_dtype_name(o.dtype),
            o.shape
        ),
        None => deny!("missing output tensor type/shape info"),
    };
    let rank = data.shape.len() as i64;
    let (s, e) = shape_interval(rank, node.int_attr("start", 0), node.int_attr("end", rank));
    require!(
        out.shape[0] == e - s,
        "Shape: output extent {} must equal the sliced rank interval {} (start..end)",
        out.shape[0],
        e - s
    );
    Ok(())
}

fn size_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "Size expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        matches!(node.input_info(0), Some(d) if is_movable(d.dtype)),
        "Size: data must have a movable GPU dtype with known tensor info"
    );
    require!(
        matches!(node.output_info(0), Some(o)
            if o.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && o.shape.is_empty()),
        "Size: output must be a scalar int64 tensor"
    );
    Ok(())
}

fn constant_of_shape_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "ConstantOfShape expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let out = match node.output_info(0) {
        Some(o) if is_movable(o.dtype) => o.dtype,
        Some(o) => deny!(
            "output dtype {} is not a movable dtype supported on GPU",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("missing output tensor type/shape info"),
    };
    let shape = match node.read_const_int64(0) {
        Some(s) => s,
        None => deny!(
            "ConstantOfShape: `input` shape must be a constant int64 initializer; this node's is a runtime value — stays on CPU"
        ),
    };
    require!(
        shape.iter().all(|&d| d >= 0 && d <= i32::MAX as i64),
        "ConstantOfShape: shape dims must be in [0, i32::MAX] (got {:?})",
        shape
    );
    // The `value` TENSOR attribute is not carried through the NodeDesc, so only the no-value-attr
    // fp32-zeros form is claimed; ORT CPU constant-folds / evaluates the explicit-value forms.
    require!(
        !node.has_attr("value"),
        "ConstantOfShape: an explicit `value` attribute is not claimed (only the default fp32-zeros form is); ORT CPU evaluates the explicit-value forms"
    );
    require!(
        out == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        "ConstantOfShape: only fp32 output is claimed (the default-value zeros form), got {}",
        crate::registry::ort_dtype_name(out)
    );
    Ok(())
}

fn resize_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 3 && node.num_inputs() <= 4 && node.num_outputs() == 1,
        "Resize expects 3-4 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(input), Some(output)) => (input, output),
        _ => deny!("missing tensor type/shape info on the input or output"),
    };
    let rank = input.shape.len();
    // Spatial dims MAY be dynamic (<=0): the concrete input size is resolved at trace time from
    // `ctx.shape_of`, so we only require the resize TARGET to be statically determinable (constant
    // `scales`/`sizes`) and the batch/channel dims to be untouched. Rank must be known and match.
    require!(
        is_mlx_float(input.dtype),
        "Resize: input dtype {} must be an MLX float (fp32/fp16/bf16)",
        crate::registry::ort_dtype_name(input.dtype)
    );
    require!(
        output.dtype == input.dtype,
        "Resize: output dtype {} must match input dtype {}",
        crate::registry::ort_dtype_name(output.dtype),
        crate::registry::ort_dtype_name(input.dtype)
    );
    require!(
        rank != 0 && rank <= 4,
        "Resize: input rank {rank} unsupported (only 1-D..4-D tensors are claimed)"
    );
    require!(
        output.shape.len() == rank,
        "Resize: output rank {} must equal input rank {rank}",
        output.shape.len()
    );
    let mode = node.string_attr("mode", "nearest");
    require!(
        mode == "nearest" || mode == "linear",
        "Resize: only mode=nearest|linear are claimed (got mode={mode}; cubic stays on CPU)"
    );
    let coordinate_mode = node.string_attr("coordinate_transformation_mode", "half_pixel");
    require!(
        matches!(
            coordinate_mode.as_str(),
            "half_pixel" | "asymmetric" | "align_corners" | "pytorch_half_pixel"
        ),
        "Resize: coordinate_transformation_mode={coordinate_mode} is unsupported (only half_pixel|asymmetric|align_corners|pytorch_half_pixel are claimed)"
    );
    require!(
        mode != "nearest"
            || matches!(
                node.string_attr("nearest_mode", "round_prefer_floor").as_str(),
                "round_prefer_floor" | "round_prefer_ceil" | "floor" | "ceil"
            ),
        "Resize: nearest_mode={} is unsupported (only round_prefer_floor|round_prefer_ceil|floor|ceil are claimed)",
        node.string_attr("nearest_mode", "round_prefer_floor")
    );
    require!(
        node.int_attr("exclude_outside", 0) == 0,
        "Resize: exclude_outside=1 is unsupported — stays on CPU"
    );
    require!(
        node.int_attr("antialias", 0) == 0,
        "Resize: antialias=1 is unsupported — stays on CPU"
    );
    require!(
        !node.has_attr("axes"),
        "Resize: the `axes` attribute is unsupported (full-rank scales/sizes only) — stays on CPU"
    );
    require!(
        node.string_attr("keep_aspect_ratio_policy", "stretch") == "stretch",
        "Resize: only keep_aspect_ratio_policy=stretch is claimed — stays on CPU"
    );
    require!(
        !node.input_present(1),
        "Resize: the `roi` input (tf_crop_and_resize) is unsupported — stays on CPU"
    );
    let has_scales = node.input_present(2);
    let has_sizes = node.input_present(3);
    require!(
        has_scales != has_sizes,
        "Resize: exactly one of `scales` or `sizes` must be provided (not both/neither)"
    );
    // For rank>=3 the batch/channel axes (0,1) must NOT be resized — MLX resize here is
    // spatial-only, and a batch/channel target can't be verified against a dynamic input dim.
    let bc = if rank >= 3 { 2 } else { 0 };
    if has_sizes {
        // Constant `sizes` give the exact (static) output extents directly.
        let sizes = match node.read_const_int64(3) {
            Some(s) if s.len() == rank && s.iter().all(|&v| v >= 1) => s,
            Some(s) => deny!(
                "Resize: `sizes` must have rank {rank} with all entries >= 1 (got {:?})",
                s
            ),
            None => deny!(
                "Resize: only constant (initializer) `scales`/`sizes` are claimed; this node's are runtime/dynamic — precompute them or export a static-shape model"
            ),
        };
        for ax in 0..bc {
            // Batch/channel: input dim must be static and unchanged.
            require!(
                input.shape[ax] > 0 && sizes[ax] == input.shape[ax],
                "Resize: batch/channel dim {ax} must be static and unchanged (input {}, requested size {})",
                input.shape[ax],
                sizes[ax]
            );
        }
        for ax in bc..rank {
            // Where the output spatial dim is statically known, it must equal the requested size.
            require!(
                output.shape[ax] <= 0 || output.shape[ax] == sizes[ax],
                "Resize: static output spatial dim {ax} ({}) must equal requested size {}",
                output.shape[ax],
                sizes[ax]
            );
        }
        Ok(())
    } else {
        // Constant `scales`: the concrete output is `floor(scale * input)` computed at trace time.
        let scales = match node.read_const_f32(2) {
            Some(s) if s.len() == rank && s.iter().all(|&v| v > 0.0) => s,
            Some(s) => deny!(
                "Resize: `scales` must have rank {rank} with all entries > 0 (got {:?})",
                s
            ),
            None => deny!(
                "Resize: only constant (initializer) `scales`/`sizes` are claimed; this node's are runtime/dynamic — precompute them or export a static-shape model"
            ),
        };
        for ax in 0..bc {
            // Batch/channel scale must be exactly 1 (no resize).
            require!(
                (scales[ax] - 1.0).abs() <= f32::EPSILON,
                "Resize: batch/channel dim {ax} must not be resized (scale {} != 1)",
                scales[ax]
            );
        }
        for ax in bc..rank {
            // Where both input and output spatial dims are static, verify the computed extent.
            if input.shape[ax] > 0 && output.shape[ax] > 0 {
                let computed = (scales[ax] as f64 * input.shape[ax] as f64).floor() as i64;
                require!(
                    computed == output.shape[ax],
                    "Resize: spatial dim {ax} computed extent {computed} (floor(scale*input)) must equal static output {}",
                    output.shape[ax]
                );
            }
        }
        Ok(())
    }
}

fn where_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 3 && node.num_outputs() == 1,
        "Where expects 3 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (cond, x, y, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    let is_bool = |t| t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    require!(
        is_bool(cond.dtype),
        "Where: `condition` must be bool (got {})",
        crate::registry::ort_dtype_name(cond.dtype)
    );
    require!(
        x.dtype == y.dtype && y.dtype == out.dtype,
        "Where: X/Y/output must share one dtype (got {}, {} -> {})",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(y.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        is_mlx_numeric(x.dtype) || is_bool(x.dtype),
        "Where: X/Y dtype {} must be numeric or bool",
        crate::registry::ort_dtype_name(x.dtype)
    );
    require!(
        crate::registry::scalar_or_suffix_broadcast(&cond.shape, &out.shape)
            && crate::registry::scalar_or_suffix_broadcast(&x.shape, &out.shape)
            && crate::registry::scalar_or_suffix_broadcast(&y.shape, &out.shape),
        "Where: only scalar or trailing-suffix broadcast to the output shape is supported (cond {:?}, X {:?}, Y {:?} -> {:?})",
        cond.shape,
        x.shape,
        y.shape,
        out.shape
    );
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

pub fn register(registry: &mut OpRegistry) {
    reg(registry, "Gather", gather_op, gather_like_claim);
    reg(registry, "GatherElements", gather_elements_op, gather_like_claim);
    reg(registry, "ScatterElements", scatter_elements_op, scatter_elements_claim);
    reg(registry, "Concat", concat_op, concat_claim);
    reg(registry, "Reshape", reshape_op, reshape_claim);
    reg(registry, "Transpose", transpose_op, transpose_claim);
    reg(registry, "Unsqueeze", unsqueeze_op, unsqueeze_claim);
    reg(registry, "Squeeze", squeeze_op, squeeze_claim);
    reg(registry, "Flatten", flatten_op, flatten_claim);
    reg(registry, "Expand", expand_op, expand_claim);
    reg(registry, "Slice", slice_op, slice_claim);
    reg(registry, "Split", split_op, split_claim);
    reg(registry, "Tile", tile_op, tile_claim);
    reg(registry, "Pad", pad_op, pad_claim);
    reg(registry, "Identity", identity_op, identity_claim);
    reg(registry, "Range", range_op, range_claim);
    reg(registry, "Shape", shape_op, shape_claim);
    reg(registry, "Size", size_op, size_claim);
    reg(registry, "ConstantOfShape", constant_of_shape_op, constant_of_shape_claim);
    reg(registry, "Where", where_handler, where_claim);
    reg(registry, "Resize", resize_op, resize_claim);
}
