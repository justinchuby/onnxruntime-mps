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

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, Src, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    is_int_index, is_mlx_float, is_mlx_numeric, is_movable, is_range_type, NodeView, OpRegistration,
    OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;

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
fn read_ints(ctx: &TranslationContext, n: &NodeDesc, i: usize) -> Result<Vec<i64>, MlxError> {
    ctx.read_ints(&n.inputs[i])
}

/// Read an axes/split-style list from either the opset-13 input or the opset<13 INTS attribute.
fn read_list_input_or_attr(
    ctx: &TranslationContext,
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
                in_shape[i] // allowzero=0: copy the input dim
            } else {
                d as i32
            }
        })
        .collect();
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
        result[i] = d_out as i32;
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

fn gather_like_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => return false,
    };
    let idx = match node.input_info(1) {
        Some(i) => i.dtype,
        None => return false,
    };
    is_movable(data) && out == data && is_int_index(idx)
}

fn scatter_elements_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 3
        || node.num_outputs() != 1
        || node.string_attr("reduction", "none") != "none"
    {
        return false;
    }
    let (data, indices, updates, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return false,
    };
    // mlx_put_along_axis's GPU kernel aborts on int64 payloads → keep to MLX float dtypes.
    if !is_mlx_float(data.dtype)
        || !is_int_index(indices.dtype)
        || updates.dtype != data.dtype
        || out.dtype != data.dtype
        || data.shape.is_empty()
        || indices.shape != updates.shape
        || indices.shape.len() != data.shape.len()
        || out.shape != data.shape
    {
        return false;
    }
    let rank = data.shape.len() as i64;
    let axis = node.int_attr("axis", 0);
    if axis < -rank || axis >= rank {
        return false;
    }
    let ax = norm_axis(axis, rank as i32) as usize;
    for i in 0..data.shape.len() {
        if i != ax && indices.shape[i] > data.shape[i] {
            return false;
        }
    }
    true
}

fn concat_claim(node: &NodeView) -> bool {
    if node.num_inputs() == 0 || node.num_outputs() != 1 {
        return false;
    }
    let out = match node.output_info(0) {
        Some(o) if is_movable(o.dtype) => o.dtype,
        _ => return false,
    };
    for i in 0..node.num_inputs() {
        match node.input_info(i) {
            Some(info) if info.dtype == out => {}
            _ => return false,
        }
    }
    true
}

fn reshape_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => return false,
    };
    is_movable(data) && out == data && node.is_const_int64(1) && node.int_attr("allowzero", 0) == 0
}

fn transpose_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    match movable_io(node) {
        Some((data, out)) => is_movable(data) && out == data,
        None => false,
    }
}

fn unsqueeze_claim(node: &NodeView) -> bool {
    if node.num_inputs() == 0 || node.num_outputs() != 1 {
        return false;
    }
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => return false,
    };
    if !is_movable(data) || out != data {
        return false;
    }
    if node.num_inputs() == 2 {
        return node.is_const_int64(1);
    }
    node.num_inputs() == 1
}

fn squeeze_claim(node: &NodeView) -> bool {
    unsqueeze_claim(node)
}

fn flatten_claim(node: &NodeView) -> bool {
    transpose_claim(node)
}

fn expand_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    match movable_io(node) {
        Some((data, out)) => is_movable(data) && out == data && node.is_const_int64(1),
        None => false,
    }
}

fn slice_claim(node: &NodeView) -> bool {
    let nin = node.num_inputs();
    if nin < 3 || nin > 5 || node.num_outputs() != 1 {
        return false;
    }
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => return false,
    };
    if !is_movable(data) || out != data {
        return false;
    }
    if !node.is_const_int64(1) || !node.is_const_int64(2) {
        return false;
    }
    if nin >= 4 && node.input_present(3) && !node.is_const_int64(3) {
        return false;
    }
    if nin >= 5 && node.input_present(4) {
        match node.read_const_int64(4) {
            Some(steps) => {
                if steps.iter().any(|&st| st < 1) {
                    return false; // negative/zero strides left to CPU
                }
            }
            None => return false,
        }
    }
    true
}

fn split_claim(node: &NodeView) -> bool {
    let nin = node.num_inputs();
    if nin == 0 || nin > 2 || node.num_outputs() == 0 {
        return false;
    }
    let data = match node.input_info(0) {
        Some(d) if is_movable(d.dtype) && !d.shape.is_empty() => d,
        _ => return false,
    };
    for i in 0..node.num_outputs() {
        match node.output_info(i) {
            Some(o) if o.dtype == data.dtype => {}
            _ => return false,
        }
    }
    let rank = data.shape.len() as i64;
    let axis = node.int_attr("axis", 0);
    if axis < -rank || axis >= rank {
        return false;
    }
    let axis_size = data.shape[norm_axis(axis, rank as i32) as usize];

    // Explicit per-section sizes: opset-13 `split` input or opset<13 `split` INTS attribute.
    let (have_sizes, sizes) = if nin == 2 && node.input_present(1) {
        match node.read_const_int64(1) {
            Some(s) => (true, s),
            None => return false, // dynamic split sizes → CPU
        }
    } else {
        let (present, s) = node.ints_attr("split");
        (present, s)
    };
    if have_sizes {
        let mut total = 0i64;
        for &s in &sizes {
            if s < 0 {
                return false;
            }
            total += s;
        }
        return sizes.len() == node.num_outputs() && total == axis_size;
    }
    // Equal split: MLX requires the axis to divide evenly by the output count.
    axis_size % node.num_outputs() as i64 == 0
}

fn tile_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    match movable_io(node) {
        Some((data, out)) => is_movable(data) && out == data && node.is_const_int64(1),
        None => false,
    }
}

fn pad_claim(node: &NodeView) -> bool {
    let nin = node.num_inputs();
    if nin < 2 || nin > 4 || node.num_outputs() != 1 {
        return false;
    }
    let (data, out) = match movable_io(node) {
        Some(v) => v,
        None => return false,
    };
    if !is_movable(data) || out != data || node.string_attr("mode", "constant") != "constant" {
        return false;
    }
    match node.read_const_int64(1) {
        Some(pads) => {
            if pads.iter().any(|&p| p < 0) {
                return false; // negative pads (cropping) left to CPU
            }
        }
        None => return false,
    }
    if nin >= 3 && node.input_present(2) {
        match node.input_info(2) {
            Some(cv) if cv.dtype == data => {}
            _ => return false,
        }
    }
    if nin >= 4 && node.input_present(3) && !node.is_const_int64(3) {
        return false;
    }
    true
}

fn identity_claim(node: &NodeView) -> bool {
    transpose_claim(node)
}

fn range_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 3 || node.num_outputs() != 1 {
        return false;
    }
    let ty = match node.input_info(0) {
        Some(i) if is_range_type(i.dtype) => i.dtype,
        _ => return false,
    };
    let out = match node.output_info(0) {
        Some(o) if o.dtype == ty && o.shape.len() == 1 => o,
        _ => return false,
    };
    let (start, limit, delta) = match (
        node.read_const_scalar_f64(0),
        node.read_const_scalar_f64(1),
        node.read_const_scalar_f64(2),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return false,
    };
    if delta == 0.0 {
        return false;
    }
    let count = ((limit - start) / delta).ceil().max(0.0);
    count.is_finite() && count <= i32::MAX as f64 && out.shape[0] == count as i64
}

fn shape_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    let data = match node.input_info(0) {
        Some(d) if is_movable(d.dtype) => d,
        _ => return false,
    };
    let out = match node.output_info(0) {
        Some(o)
            if o.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                && o.shape.len() == 1 =>
        {
            o
        }
        _ => return false,
    };
    let rank = data.shape.len() as i64;
    let (s, e) = shape_interval(rank, node.int_attr("start", 0), node.int_attr("end", rank));
    out.shape[0] == e - s
}

fn size_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    let data_ok = matches!(node.input_info(0), Some(d) if is_movable(d.dtype));
    let out_ok = matches!(node.output_info(0), Some(o)
        if o.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && o.shape.is_empty());
    data_ok && out_ok
}

fn constant_of_shape_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    let out = match node.output_info(0) {
        Some(o) if is_movable(o.dtype) => o.dtype,
        _ => return false,
    };
    let shape = match node.read_const_int64(0) {
        Some(s) => s,
        None => return false,
    };
    if shape.iter().any(|&d| d < 0 || d > i32::MAX as i64) {
        return false;
    }
    // The `value` TENSOR attribute is not carried through the NodeDesc, so only the no-value-attr
    // fp32-zeros form is claimed; ORT CPU constant-folds / evaluates the explicit-value forms.
    if node.has_attr("value") {
        return false;
    }
    out == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
}

fn resize_claim(node: &NodeView) -> bool {
    if node.num_inputs() < 3 || node.num_inputs() > 4 || node.num_outputs() != 1 {
        return false;
    }
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(input), Some(output)) => (input, output),
        _ => return false,
    };
    if !is_mlx_float(input.dtype)
        || output.dtype != input.dtype
        || input.shape.is_empty()
        || input.shape.len() > 4
        || input.shape.len() != output.shape.len()
        || input
            .shape
            .iter()
            .chain(output.shape.iter())
            .any(|&d| d < 1)
    {
        return false;
    }
    let mode = node.string_attr("mode", "nearest");
    if mode != "nearest" && mode != "linear" {
        return false;
    }
    let coordinate_mode = node.string_attr("coordinate_transformation_mode", "half_pixel");
    if !matches!(
        coordinate_mode.as_str(),
        "half_pixel" | "asymmetric" | "align_corners" | "pytorch_half_pixel"
    ) {
        return false;
    }
    if mode == "nearest"
        && !matches!(
            node.string_attr("nearest_mode", "round_prefer_floor")
                .as_str(),
            "round_prefer_floor" | "round_prefer_ceil" | "floor" | "ceil"
        )
    {
        return false;
    }
    if node.int_attr("exclude_outside", 0) != 0
        || node.int_attr("antialias", 0) != 0
        || node.has_attr("axes")
        || node.string_attr("keep_aspect_ratio_policy", "stretch") != "stretch"
        || node.input_present(1)
    {
        return false;
    }
    let has_scales = node.input_present(2);
    let has_sizes = node.input_present(3);
    if has_scales == has_sizes {
        return false;
    }
    let computed = if has_sizes {
        match node.read_const_int64(3) {
            Some(sizes) if sizes.len() == input.shape.len() => sizes,
            _ => return false,
        }
    } else {
        match node.read_const_f32(2) {
            Some(scales)
                if scales.len() == input.shape.len() && scales.iter().all(|&s| s > 0.0) =>
            {
                scales
                    .iter()
                    .zip(input.shape.iter())
                    .map(|(&scale, &size)| (scale as f64 * size as f64).floor() as i64)
                    .collect()
            }
            _ => return false,
        }
    };
    if computed
        .iter()
        .zip(output.shape.iter())
        .any(|(&a, &b)| a < 1 || a != b)
    {
        return false;
    }
    input.shape.len() < 3 || (computed[0] == input.shape[0] && computed[1] == input.shape[1])
}

fn where_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 3 || node.num_outputs() != 1 {
        return false;
    }
    let (cond, x, y, out) = match (
        node.input_info(0),
        node.input_info(1),
        node.input_info(2),
        node.output_info(0),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return false,
    };
    let is_bool = |t| t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    if !is_bool(cond.dtype)
        || x.dtype != y.dtype
        || y.dtype != out.dtype
        || !(is_mlx_numeric(x.dtype) || is_bool(x.dtype))
    {
        return false;
    }
    crate::registry::scalar_or_suffix_broadcast(&cond.shape, &out.shape)
        && crate::registry::scalar_or_suffix_broadcast(&x.shape, &out.shape)
        && crate::registry::scalar_or_suffix_broadcast(&y.shape, &out.shape)
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
