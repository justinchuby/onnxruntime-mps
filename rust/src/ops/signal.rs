//! Signal / FFT op handlers (ai.onnx opset-17+): DFT, STFT, Hann/Hamming/Blackman windows and
//! MelWeightMatrix. Faithful port of the C++ `ops/signal.cc`. Only statically translatable forms are
//! claimed (constant dft/frame lengths, constant axis, real STFT input, non-(inverse&&onesided) DFT,
//! fp32 in/out); every other form is left to ORT CPU.

use std::f64::consts::PI;

use crate::engine::{MlxError, NodeDesc, TensorRef, TranslationContext};
use crate::registry::{
    is_mlx_float, ClaimPredicate, ClaimResult, NodeView, OpHandler, OpRegistration, OpRegistry,
    K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

const NORM_BACKWARD: mlx::mlx_fft_norm = mlx::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;

// ---- translate-time helpers ---------------------------------------------------------------------

/// Read a constant scalar integer input (int32/int64) at translate time.
fn read_scalar_int(ctx: &TranslationContext, r: &TensorRef) -> Result<i64, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() || h.count < 1 {
        return Err("MLX signal: expected a scalar int".to_string());
    }
    match h.dtype {
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            Ok(unsafe { *(h.data as *const i64) })
        }
        t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            Ok(unsafe { *(h.data as *const i32) } as i64)
        }
        _ => Err("MLX signal: scalar int input has an unsupported dtype".to_string()),
    }
}

/// Read a constant scalar float32 input at translate time.
fn read_scalar_float(ctx: &TranslationContext, r: &TensorRef) -> Result<f64, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() || h.count < 1 {
        return Err("MLX signal: expected a scalar float".to_string());
    }
    if h.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT {
        return Err("MLX signal: scalar float input has an unsupported dtype".to_string());
    }
    Ok(unsafe { *(h.data as *const f32) } as f64)
}

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != crate::engine::Src::Absent
}

/// x[..., idx] dropping the trailing components axis (rank shrinks by 1).
fn take_last_index(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    idx: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let shape = ctx.shape_of(x);
    let rank = shape.len();
    let mut start = vec![0i32; rank];
    let stop = shape.clone();
    let mut stop2 = stop;
    start[rank - 1] = idx;
    stop2[rank - 1] = idx + 1;
    let sliced = ctx.slice(x, &start, &stop2)?;
    let squeezed: Vec<i32> = shape[..rank - 1].to_vec();
    ctx.reshape(sliced, &squeezed)
}

/// Stack real+imag of a complex array into a new trailing axis of size 2 (ONNX (real, imag) form).
fn stack_real_imag(
    ctx: &mut TranslationContext,
    cx: mlx::mlx_array,
    append_axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let re = ctx.emit(|res, s| unsafe { mlx::mlx_real(res, cx, s) })?;
    let im = ctx.emit(|res, s| unsafe { mlx::mlx_imag(res, cx, s) })?;
    ctx.stack(&[re, im], append_axis)
}

// ---- DFT ----------------------------------------------------------------------------------------

fn dft_axis(ctx: &TranslationContext, n: &NodeDesc, rank: i32) -> Result<i32, MlxError> {
    let mut axis: i64 = if n.since_version >= 20 {
        if present(n, 2) {
            read_scalar_int(ctx, &n.inputs[2])?
        } else {
            -2
        }
    } else {
        *n.ints.get("axis").unwrap_or(&1)
    };
    if axis < 0 {
        axis += rank as i64;
    }
    if axis < 0 || axis >= rank as i64 - 1 {
        return Err("MLX DFT: axis out of range".to_string());
    }
    Ok(axis as i32)
}

fn dft_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let rank = shape.len() as i32;
    let last = shape[shape.len() - 1];
    let axis = dft_axis(ctx, n, rank)?;

    let inverse = n.ints.get("inverse").copied().unwrap_or(0) != 0;
    let onesided = n.ints.get("onesided").copied().unwrap_or(0) != 0;

    let dft_length: i32 = if present(n, 1) {
        read_scalar_int(ctx, &n.inputs[1])? as i32
    } else {
        shape[axis as usize]
    };

    // Lift the real part into complex64 by multiplying with (1+0j); add i*imag when complex input.
    let one_c = ctx.scalar_complex(1.0, 0.0);
    let real = take_last_index(ctx, x, 0)?;
    let mut signal = ctx.mul(real, one_c)?;
    if last == 2 {
        let i_unit = ctx.scalar_complex(0.0, 1.0);
        let imag = take_last_index(ctx, x, 1)?;
        let imag_c = ctx.mul(imag, i_unit)?;
        signal = ctx.add(signal, imag_c)?;
    }

    let mut spectrum = if inverse {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_ifft(res, signal, dft_length, axis, NORM_BACKWARD, s)
        })?
    } else {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_fft(res, signal, dft_length, axis, NORM_BACKWARD, s)
        })?
    };

    if onesided && !inverse {
        let res_shape = ctx.shape_of(spectrum);
        let start = vec![0i32; res_shape.len()];
        let mut stop = res_shape;
        stop[axis as usize] = dft_length / 2 + 1;
        spectrum = ctx.slice(spectrum, &start, &stop)?;
    }

    let out = stack_real_imag(ctx, spectrum, rank - 1)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn is_int_type(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

fn is_scalar_shape(shape: &[i64]) -> bool {
    shape.is_empty() || (shape.len() == 1 && shape[0] == 1)
}

fn const_scalar_int(node: &NodeView, i: usize) -> bool {
    match node.input_info(i) {
        Some(info) => {
            is_int_type(info.dtype)
                && is_scalar_shape(&info.shape)
                && node.is_constant_initializer(i)
        }
        None => false,
    }
}

fn const_scalar_float(node: &NodeView, i: usize) -> bool {
    match node.input_info(i) {
        Some(info) => {
            info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                && is_scalar_shape(&info.shape)
                && node.is_constant_initializer(i)
        }
        None => false,
    }
}

fn output_datatype_ok(node: &NodeView) -> bool {
    let dt = node.int_attr("output_datatype", 1);
    dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as i64
        || dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 as i64
        || dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as i64
}

fn dft_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        ni >= 1 && ni <= 3 && node.num_outputs() == 1,
        "expects 1-3 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (in_info, out_info) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        in_info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && out_info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        "input/output must both be fp32, got {} -> {}",
        crate::registry::ort_dtype_name(in_info.dtype),
        crate::registry::ort_dtype_name(out_info.dtype)
    );
    let rank = in_info.shape.len();
    require!(rank >= 2, "input rank must be at least 2, got {rank}");
    let last = in_info.shape[rank - 1];
    require!(
        last == 1 || last == 2,
        "input trailing real/imag dimension must be 1 or 2, got {last}"
    );
    let since = node.since_version();
    let inverse = node.int_attr("inverse", 0);
    let onesided = node.int_attr("onesided", 0);
    require!(
        (inverse == 0 || inverse == 1) && (onesided == 0 || onesided == 1),
        "inverse and onesided must each be 0 or 1, got inverse={inverse}, onesided={onesided}"
    );
    require!(
        inverse != 1 || onesided != 1,
        "inverse DFT does not support onesided=1"
    );
    let mut axis: i64 = if since >= 20 {
        if node.input_present(2) {
            require!(
                const_scalar_int(node, 2),
                "axis must be a constant scalar initializer"
            );
            match node.read_const_scalar_f64(2) {
                Some(v) => v as i64,
                None => deny!("axis must be a constant scalar initializer"),
            }
        } else {
            -2
        }
    } else {
        node.int_attr("axis", 1)
    };
    if axis < 0 {
        axis += rank as i64;
    }
    require!(
        axis >= 0 && axis < rank as i64 - 1,
        "axis is out of range for rank {rank} or selects the trailing real/imag dimension"
    );
    if node.input_present(1) {
        require!(
            const_scalar_int(node, 1),
            "dft_length must be a constant scalar initializer"
        );
    } else {
        require!(
            in_info.shape[axis as usize] >= 0,
            "DFT axis dimension must be static when dft_length is omitted"
        );
    }
    Ok(())
}

// ---- STFT ---------------------------------------------------------------------------------------

fn stft_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let signal = ctx.resolve(&n.inputs[0])?;
    let in_shape = ctx.shape_of(signal);
    let batch = in_shape[0];
    let signal_len = in_shape[1];
    let frame_step = read_scalar_int(ctx, &n.inputs[1])? as i32;

    let has_window = present(n, 2);
    let window = if has_window {
        Some(ctx.resolve(&n.inputs[2])?)
    } else {
        None
    };
    let frame_length: i32 = if let Some(w) = window {
        ctx.shape_of(w)[0]
    } else {
        read_scalar_int(ctx, &n.inputs[3])? as i32
    };
    let onesided = n.ints.get("onesided").copied().unwrap_or(1) != 0;
    let n_frames = 1 + (signal_len - frame_length) / frame_step;

    let flat = ctx.reshape(signal, &[batch, signal_len])?;
    let flat = ctx.contiguous(flat)?;
    let frame_shape = [batch, n_frames, frame_length];
    let strides: [i64; 3] = [signal_len as i64, frame_step as i64, 1];
    let mut frames = ctx.emit(|res, s| unsafe {
        mlx::mlx_as_strided(
            res,
            flat,
            frame_shape.as_ptr(),
            frame_shape.len(),
            strides.as_ptr(),
            strides.len(),
            0,
            s,
        )
    })?;
    if let Some(w) = window {
        frames = ctx.mul(frames, w)?;
    }

    let spectrum = if onesided {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_rfft(res, frames, frame_length, 2, NORM_BACKWARD, s)
        })?
    } else {
        ctx.emit(|res, s| unsafe {
            mlx::mlx_fft_fft(res, frames, frame_length, 2, NORM_BACKWARD, s)
        })?
    };

    let out = stack_real_imag(ctx, spectrum, 3)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn stft_claim(node: &NodeView) -> ClaimResult {
    let ni = node.num_inputs();
    require!(
        ni >= 2 && ni <= 4 && node.num_outputs() == 1,
        "expects 2-4 inputs and 1 output, got {}in/{}out",
        ni,
        node.num_outputs()
    );
    let (sig, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => deny!("missing tensor type/shape info on signal or output"),
    };
    require!(
        sig.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && out.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        "signal/output must both be fp32, got {} -> {}",
        crate::registry::ort_dtype_name(sig.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        sig.shape.len() == 3 && sig.shape[1] >= 0 && sig.shape[2] == 1,
        "signal must have static shape [batch, length, 1], got {:?}",
        sig.shape
    );
    require!(
        node.input_present(1) && const_scalar_int(node, 1),
        "frame_step must be a constant scalar initializer"
    );
    let has_window = node.input_present(2);
    let has_frame_length = node.input_present(3);
    if has_window {
        match node.input_info(2) {
            Some(w)
                if w.dtype
                    == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    && w.shape.len() == 1
                    && w.shape[0] >= 0 => {}
            Some(w) => deny!(
                "window must be a static rank-1 fp32 tensor, got dtype {} shape {:?}",
                crate::registry::ort_dtype_name(w.dtype),
                w.shape
            ),
            None => deny!("missing tensor type/shape info on window"),
        }
    } else {
        require!(
            has_frame_length && const_scalar_int(node, 3),
            "frame_length must be a constant scalar initializer"
        );
    }
    if has_frame_length {
        require!(
            const_scalar_int(node, 3),
            "frame_length must be a constant scalar initializer"
        );
    }
    let onesided = node.int_attr("onesided", 1);
    require!(
        onesided == 0 || onesided == 1,
        "onesided must be 0 or 1, got {onesided}"
    );
    Ok(())
}

// ---- Cosine-sum windows -------------------------------------------------------------------------

fn cosine_window(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    a0: f64,
    a1: f64,
    a2: f64,
) -> Result<(), MlxError> {
    let size = read_scalar_int(ctx, &n.inputs[0])?;
    let periodic = n.ints.get("periodic").copied().unwrap_or(1) != 0;
    let mut denom = if periodic {
        size as f64
    } else {
        (size - 1) as f64
    };
    if denom <= 0.0 {
        denom = 1.0;
    }
    let idx = ctx.emit(|res, s| unsafe {
        mlx::mlx_arange(res, 0.0, size as f64, 1.0, mlx::mlx_dtype__MLX_FLOAT32, s)
    })?;
    let two_pi = ctx.scalar_f32((2.0 * PI / denom) as f32);
    let arg = ctx.mul(idx, two_pi)?;
    let cos1 = ctx.emit(|res, s| unsafe { mlx::mlx_cos(res, arg, s) })?;
    let a1s = ctx.scalar_f32(a1 as f32);
    let cos1a1 = ctx.mul(cos1, a1s)?;
    let a0s = ctx.scalar_f32(a0 as f32);
    let mut y = ctx.sub(a0s, cos1a1)?;
    if a2 != 0.0 {
        let two = ctx.scalar_f32(2.0);
        let arg2 = ctx.mul(arg, two)?;
        let cos2 = ctx.emit(|res, s| unsafe { mlx::mlx_cos(res, arg2, s) })?;
        let a2s = ctx.scalar_f32(a2 as f32);
        let cos2a2 = ctx.mul(cos2, a2s)?;
        y = ctx.add(y, cos2a2)?;
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn hann_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    cosine_window(ctx, n, 0.5, 0.5, 0.0)
}

fn hamming_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    cosine_window(ctx, n, 0.54347826086, 0.45652173913, 0.0)
}

fn blackman_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    cosine_window(ctx, n, 0.42, 0.5, 0.08)
}

fn window_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        const_scalar_int(node, 0),
        "size must be a constant scalar initializer"
    );
    match node.output_info(0) {
        Some(o) if is_mlx_float(o.dtype) => {}
        Some(o) => deny!(
            "output dtype {} is not supported; expected fp32/fp16/bf16",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("missing tensor type/shape info on output"),
    }
    require!(
        output_datatype_ok(node),
        "output_datatype {} is not supported; expected fp32, fp16, or bf16",
        crate::registry::ort_dtype_name(
            node.int_attr("output_datatype", 1) as ort::ONNXTensorElementDataType
        )
    );
    let periodic = node.int_attr("periodic", 1);
    require!(
        periodic == 0 || periodic == 1,
        "periodic must be 0 or 1, got {periodic}"
    );
    Ok(())
}

// ---- MelWeightMatrix ----------------------------------------------------------------------------

fn mel_weight_matrix_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_mel_bins = read_scalar_int(ctx, &n.inputs[0])? as i32;
    let dft_length = read_scalar_int(ctx, &n.inputs[1])? as i32;
    let sample_rate = read_scalar_int(ctx, &n.inputs[2])? as i32;
    let lower_hz = read_scalar_float(ctx, &n.inputs[3])?;
    let upper_hz = read_scalar_float(ctx, &n.inputs[4])?;

    let num_spectrogram_bins = dft_length / 2 + 1;
    let num_edges = num_mel_bins + 2;

    let low_mel = 2595.0 * (1.0 + lower_hz / 700.0).log10();
    let high_mel = 2595.0 * (1.0 + upper_hz / 700.0).log10();
    let mel_step = (high_mel - low_mel) / num_edges as f64;

    let mut bins = vec![0i32; num_edges as usize];
    for i in 0..num_edges {
        let mel = i as f64 * mel_step + low_mel;
        let hz = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
        bins[i as usize] = ((dft_length + 1) as f64 * hz / sample_rate as f64).floor() as i32;
    }

    let ncols = num_mel_bins as usize;
    let mut out = vec![0.0f32; num_spectrogram_bins as usize * ncols];
    let mut put = |row: i32, col: i32, value: f32| {
        if row >= 0 && row < num_spectrogram_bins {
            out[row as usize * ncols + col as usize] = value;
        }
    };
    for i in 0..num_mel_bins {
        let lower_bin = bins[i as usize];
        let center_bin = bins[i as usize + 1];
        let higher_bin = bins[i as usize + 2];
        let low_to_center = center_bin - lower_bin;
        if low_to_center == 0 {
            put(center_bin, i, 1.0);
        } else {
            for j in lower_bin..=center_bin {
                put(j, i, (j - lower_bin) as f32 / low_to_center as f32);
            }
        }
        let center_to_high = higher_bin - center_bin;
        if center_to_high > 0 {
            for j in center_bin..higher_bin {
                put(j, i, (higher_bin - j) as f32 / center_to_high as f32);
            }
        }
    }

    let mat_shape = [num_spectrogram_bins, num_mel_bins];
    let arr = ctx.from_host(
        out.as_ptr() as *const std::os::raw::c_void,
        &mat_shape,
        mlx::mlx_dtype__MLX_FLOAT32,
    );
    ctx.bind(&n.outputs[0], arr);
    Ok(())
}

fn mel_weight_matrix_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 5 && node.num_outputs() == 1,
        "expects 5 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(
        const_scalar_int(node, 0),
        "num_mel_bins must be a constant scalar initializer"
    );
    require!(
        const_scalar_int(node, 1),
        "dft_length must be a constant scalar initializer"
    );
    require!(
        const_scalar_int(node, 2),
        "sample_rate must be a constant scalar initializer"
    );
    require!(
        const_scalar_float(node, 3),
        "lower_edge_hertz must be a constant scalar initializer"
    );
    require!(
        const_scalar_float(node, 4),
        "upper_edge_hertz must be a constant scalar initializer"
    );
    match node.output_info(0) {
        Some(o) if is_mlx_float(o.dtype) => {}
        Some(o) => deny!(
            "output dtype {} is not supported; expected fp32/fp16/bf16",
            crate::registry::ort_dtype_name(o.dtype)
        ),
        None => deny!("missing tensor type/shape info on output"),
    }
    require!(
        output_datatype_ok(node),
        "output_datatype {} is not supported; expected fp32, fp16, or bf16",
        crate::registry::ort_dtype_name(
            node.int_attr("output_datatype", 1) as ort::ONNXTensorElementDataType
        )
    );
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn reg(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: OpHandler,
    claim: ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain: "",
        op_type,
        min_opset: 17,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    reg(registry, "DFT", dft_op, dft_claim);
    reg(registry, "STFT", stft_op, stft_claim);
    reg(registry, "HannWindow", hann_op, window_claim);
    reg(registry, "HammingWindow", hamming_op, window_claim);
    reg(registry, "BlackmanWindow", blackman_op, window_claim);
    reg(
        registry,
        "MelWeightMatrix",
        mel_weight_matrix_op,
        mel_weight_matrix_claim,
    );
}
