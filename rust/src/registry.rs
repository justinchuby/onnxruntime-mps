//! The (domain, op_type, [min,max] opset) -> { handler, claim predicate } registry — the single
//! source of truth for which ops the MLX EP can translate. Both the claim-time membership check
//! (GetCapability) and the run-time translator dispatch through the SAME table, so "claimed" and
//! "translatable" can never disagree (faithful port of `op_registry.{h,cc}`).

use std::os::raw::c_char;
use std::sync::OnceLock;

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::sys::ort;

/// A translation handler: reads a NodeDesc, emits MLX ops through the context, binds the outputs.
pub type OpHandler = fn(&mut TranslationContext, &NodeDesc) -> Result<(), MlxError>;

/// A claim-time predicate: given the concrete ONNX node, decide whether MLX can translate it exactly
/// (dtypes / shapes / attributes / input form). The (domain, op_type, opset) key is matched first.
pub type ClaimPredicate = fn(&NodeView) -> bool;

/// Sentinel for an unbounded opset endpoint.
pub const K_ANY_OPSET: i32 = -1;

/// One registry entry: match (domain, op_type) with since_version in [min_opset, max_opset].
pub struct OpRegistration {
    pub domain: &'static str,
    pub op_type: &'static str,
    pub min_opset: i32,
    pub max_opset: i32,
    pub handler: OpHandler,
    pub claim: ClaimPredicate,
}

/// The opset-aware (domain, op) -> entry table (process-wide singleton).
pub struct OpRegistry {
    table: Vec<OpRegistration>,
}

impl OpRegistry {
    fn new() -> Self {
        OpRegistry { table: Vec::new() }
    }

    pub fn register(&mut self, entry: OpRegistration) {
        self.table.push(entry);
    }

    /// The matching entry for (domain, op_type, since_version), or None.
    pub fn find_entry(
        &self,
        domain: &str,
        op_type: &str,
        since_version: i32,
    ) -> Option<&OpRegistration> {
        self.table.iter().find(|e| {
            e.domain == domain
                && e.op_type == op_type
                && (e.min_opset == K_ANY_OPSET || since_version >= e.min_opset)
                && (e.max_opset == K_ANY_OPSET || since_version <= e.max_opset)
        })
    }
}

fn registry() -> &'static OpRegistry {
    static REGISTRY: OnceLock<OpRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut r = OpRegistry::new();
        register_builtin_ops(&mut r);
        r
    })
}

/// Populate the table with every built-in op module (wave-1: elementwise + math).
fn register_builtin_ops(registry: &mut OpRegistry) {
    crate::ops::elementwise::register(registry);
    crate::ops::math::register(registry);
    crate::ops::reduction::register(registry);
    crate::ops::shape::register(registry);
    crate::ops::matmul::register(registry);
    // norm+attention
    crate::ops::norm::register_norm(registry);
    crate::ops::attention::register_attention(registry);
}

/// Run-time dispatch: find the handler for a node and translate it.
pub fn translate(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let handler = registry()
        .find_entry(&n.domain, &n.op_type, n.since_version)
        .map(|e| e.handler)
        .ok_or_else(|| {
            format!(
                "MLX: no translation for op {}::{}",
                if n.domain.is_empty() { "ai.onnx" } else { &n.domain },
                n.op_type
            )
        })?;
    handler(ctx, n)
}

/// Claim-time node predicate consulted from GetCapability. True iff the registry has a matching
/// (domain, op, opset) entry AND that entry's claim predicate accepts this concrete node.
pub fn claimable(node: &NodeView) -> bool {
    match registry().find_entry(&node.domain(), &node.op_type(), node.since_version()) {
        Some(entry) => (entry.claim)(node),
        None => false,
    }
}

// ---- Claim-time node view -----------------------------------------------------------------------

/// A light read-only view over an `OrtNode` used by claim predicates (mirrors `Ort::ConstNode` plus
/// the `op_claim.h` helpers). All FFI is confined here.
pub struct NodeView {
    api: *const ort::OrtApi,
    node: *const ort::OrtNode,
}

/// Tensor element type + shape of a node value slot; `None` for an omitted optional / non-tensor.
pub struct SlotInfo {
    pub dtype: ort::ONNXTensorElementDataType,
    pub shape: Vec<i64>,
}

impl NodeView {
    pub fn new(api: *const ort::OrtApi, node: *const ort::OrtNode) -> Self {
        NodeView { api, node }
    }

    fn api(&self) -> &ort::OrtApi {
        unsafe { &*self.api }
    }

    fn cstr(&self, p: *const c_char) -> String {
        if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        }
    }

    pub fn op_type(&self) -> String {
        unsafe {
            let mut p: *const c_char = std::ptr::null();
            (self.api().Node_GetOperatorType.unwrap())(self.node, &mut p);
            self.cstr(p)
        }
    }

    pub fn domain(&self) -> String {
        unsafe {
            let mut p: *const c_char = std::ptr::null();
            (self.api().Node_GetDomain.unwrap())(self.node, &mut p);
            self.cstr(p)
        }
    }

    pub fn since_version(&self) -> i32 {
        unsafe {
            let mut v: i32 = 0;
            (self.api().Node_GetSinceVersion.unwrap())(self.node, &mut v);
            v
        }
    }

    pub fn num_inputs(&self) -> usize {
        unsafe {
            let mut n: usize = 0;
            (self.api().Node_GetNumInputs.unwrap())(self.node, &mut n);
            n
        }
    }

    pub fn num_outputs(&self) -> usize {
        unsafe {
            let mut n: usize = 0;
            (self.api().Node_GetNumOutputs.unwrap())(self.node, &mut n);
            n
        }
    }

    fn inputs_raw(&self) -> Vec<*const ort::OrtValueInfo> {
        let n = self.num_inputs();
        let mut v: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); n];
        if n > 0 {
            unsafe { (self.api().Node_GetInputs.unwrap())(self.node, v.as_mut_ptr(), n) };
        }
        v
    }

    fn outputs_raw(&self) -> Vec<*const ort::OrtValueInfo> {
        let n = self.num_outputs();
        let mut v: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); n];
        if n > 0 {
            unsafe { (self.api().Node_GetOutputs.unwrap())(self.node, v.as_mut_ptr(), n) };
        }
        v
    }

    fn slot_info(&self, vi: *const ort::OrtValueInfo) -> Option<SlotInfo> {
        if vi.is_null() {
            return None;
        }
        unsafe {
            let api = self.api();
            let mut ti: *const ort::OrtTypeInfo = std::ptr::null();
            let st = (api.GetValueInfoTypeInfo.unwrap())(vi, &mut ti);
            if !st.is_null() || ti.is_null() {
                return None;
            }
            let mut onnx_type: ort::ONNXType = 0;
            (api.GetOnnxTypeFromTypeInfo.unwrap())(ti, &mut onnx_type);
            if onnx_type != ort::ONNXType_ONNX_TYPE_TENSOR {
                return None;
            }
            let mut tsi: *const ort::OrtTensorTypeAndShapeInfo = std::ptr::null();
            (api.CastTypeInfoToTensorInfo.unwrap())(ti, &mut tsi);
            if tsi.is_null() {
                return None;
            }
            let mut dtype: ort::ONNXTensorElementDataType = 0;
            (api.GetTensorElementType.unwrap())(tsi, &mut dtype);
            let mut nd: usize = 0;
            (api.GetDimensionsCount.unwrap())(tsi, &mut nd);
            let mut dims = vec![0i64; nd];
            if nd > 0 {
                (api.GetDimensions.unwrap())(tsi, dims.as_mut_ptr(), nd);
            }
            Some(SlotInfo { dtype, shape: dims })
        }
    }

    /// Element type + shape of input `i` (None if omitted / non-tensor).
    pub fn input_info(&self, i: usize) -> Option<SlotInfo> {
        let ins = self.inputs_raw();
        ins.get(i).and_then(|&vi| self.slot_info(vi))
    }

    /// Element type + shape of output `i` (None if omitted / non-tensor).
    pub fn output_info(&self, i: usize) -> Option<SlotInfo> {
        let outs = self.outputs_raw();
        outs.get(i).and_then(|&vi| self.slot_info(vi))
    }

    /// Release a non-null `OrtStatus` returned on an error/not-found path (the OrtApi allocates a
    /// status object even for "attribute not found", which the caller owns and must free).
    #[inline]
    unsafe fn release_status(&self, st: *mut ort::OrtStatus) {
        if !st.is_null() {
            (self.api().ReleaseStatus.unwrap())(st);
        }
    }

    /// Read a scalar INT attribute by name, or `default` when absent / of another type.
    pub fn int_attr(&self, name: &str, default: i64) -> i64 {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return default,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st =
                (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            if attr.is_null() {
                return default;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_INT {
                return default;
            }
            let mut value: i64 = default;
            let mut out_len: usize = 0;
            let st = (api.ReadOpAttr.unwrap())(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_INT,
                &mut value as *mut i64 as *mut std::os::raw::c_void,
                std::mem::size_of::<i64>(),
                &mut out_len,
            );
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            value
        }
    }

    /// Read a scalar FLOAT attribute by name, or `default` when absent / of another type
    /// (mirrors `FloatAttribute`).
    pub fn float_attr(&self, name: &str, default: f32) -> f32 {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return default,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st =
                (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            if attr.is_null() {
                return default;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT {
                return default;
            }
            let mut value: f32 = default;
            let mut out_len: usize = 0;
            let st = (api.ReadOpAttr.unwrap())(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT,
                &mut value as *mut f32 as *mut std::os::raw::c_void,
                std::mem::size_of::<f32>(),
                &mut out_len,
            );
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            value
        }
    }

    /// True iff output `i` is present (declared, non-null value info with a non-empty name).
    pub fn output_present(&self, i: usize) -> bool {
        let outs = self.outputs_raw();
        match outs.get(i) {
            Some(&vi) if !vi.is_null() => {
                let mut p: *const c_char = std::ptr::null();
                unsafe { (self.api().GetValueInfoName.unwrap())(vi, &mut p) };
                !p.is_null() && unsafe { !std::ffi::CStr::from_ptr(p).to_bytes().is_empty() }
            }
            _ => false,
        }
    }

    /// Read an INTS attribute. Returns `(present, values)`: `present` is whether the node carries a
    /// genuine INTS attribute of that name (mirrors `IntsAttribute`).
    pub fn ints_attr(&self, name: &str) -> (bool, Vec<i64>) {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return (false, Vec::new()),
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return (false, Vec::new());
            }
            if attr.is_null() {
                return (false, Vec::new());
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_INTS {
                return (false, Vec::new());
            }
            let read = api.ReadOpAttr.unwrap();
            let mut needed: usize = 0;
            let st0 = read(
                attr,
                atype,
                std::ptr::null_mut(),
                0,
                &mut needed,
            );
            self.release_status(st0);
            if needed == 0 {
                return (true, Vec::new());
            }
            let count = needed / std::mem::size_of::<i64>();
            let mut buf = vec![0i64; count];
            let mut out: usize = 0;
            let st = read(
                attr,
                atype,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                needed,
                &mut out,
            );
            if st.is_null() {
                (true, buf)
            } else {
                self.release_status(st);
                (false, Vec::new())
            }
        }
    }

    /// Read a STRING attribute, or `default` when absent / of another type.
    pub fn string_attr(&self, name: &str, default: &str) -> String {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return default.to_string(),
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return default.to_string();
            }
            if attr.is_null() {
                return default.to_string();
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_STRING {
                return default.to_string();
            }
            let read = api.ReadOpAttr.unwrap();
            let mut needed: usize = 0;
            let st0 = read(attr, atype, std::ptr::null_mut(), 0, &mut needed);
            self.release_status(st0);
            if needed == 0 {
                return String::new();
            }
            let mut buf = vec![0u8; needed];
            let mut out: usize = 0;
            let st = read(
                attr,
                atype,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                needed,
                &mut out,
            );
            if !st.is_null() {
                self.release_status(st);
                return default.to_string();
            }
            buf.truncate(out.min(needed));
            String::from_utf8(buf).unwrap_or_else(|_| default.to_string())
        }
    }

    /// True iff the node carries a genuine (non-UNDEFINED) attribute of `name`.
    pub fn has_attr(&self, name: &str) -> bool {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return false,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return false;
            }
            if attr.is_null() {
                return false;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            atype != ort::OrtOpAttrType_ORT_OP_ATTR_UNDEFINED
        }
    }

    /// True iff input `i` is present (non-null value info with a non-empty name).
    pub fn input_present(&self, i: usize) -> bool {
        let ins = self.inputs_raw();
        match ins.get(i) {
            Some(&vi) if !vi.is_null() => {
                let mut p: *const c_char = std::ptr::null();
                unsafe { (self.api().GetValueInfoName.unwrap())(vi, &mut p) };
                !p.is_null() && unsafe { !std::ffi::CStr::from_ptr(p).to_bytes().is_empty() }
            }
            _ => false,
        }
    }

    /// True iff input `i` is a constant initializer (readable at translate time).
    pub fn is_constant_initializer(&self, i: usize) -> bool {
        let ins = self.inputs_raw();
        let vi = match ins.get(i) {
            Some(&vi) if !vi.is_null() => vi,
            _ => return false,
        };
        unsafe {
            let mut is_const = false;
            let st =
                (self.api().ValueInfo_IsConstantInitializer.unwrap())(vi, &mut is_const);
            if !st.is_null() {
                self.release_status(st);
                return false;
            }
            is_const
        }
    }

    /// True iff input `i` is a tensor(int64) constant initializer.
    pub fn is_const_int64(&self, i: usize) -> bool {
        matches!(self.input_info(i), Some(info)
            if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            && self.is_constant_initializer(i)
    }

    /// Read the int64 values of a constant-initializer input `i` AT CLAIM TIME. Returns None when the
    /// input is not a readable int64 constant initializer (→ node left to CPU).
    pub fn read_const_int64(&self, i: usize) -> Option<Vec<i64>> {
        if !self.is_const_int64(i) {
            return None;
        }
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        unsafe {
            let api = self.api();
            let mut value: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            let mut data: *const std::os::raw::c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            if data.is_null() {
                return if count == 0 { Some(Vec::new()) } else { None };
            }
            Some(std::slice::from_raw_parts(data as *const i64, count).to_vec())
        }
    }

    /// Read a scalar (count-1) constant-initializer integer input `i` as f64 at CLAIM time, honoring
    /// int16/int32/int64 element types (the Range element dtypes). Returns None when the input is not
    /// a readable scalar integer constant initializer.
    pub fn read_const_scalar_f64(&self, i: usize) -> Option<f64> {
        if !self.is_constant_initializer(i) {
            return None;
        }
        let dtype = self.input_info(i)?.dtype;
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        unsafe {
            let api = self.api();
            let mut value: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            if count != 1 {
                return None;
            }
            let mut data: *const std::os::raw::c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            if data.is_null() {
                return None;
            }
            match dtype {
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
                    Some(*(data as *const i16) as f64)
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                    Some(*(data as *const i32) as f64)
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                    Some(*(data as *const i64) as f64)
                }
                _ => None,
            }
        }
    }
}

// ---- Shared claim helpers (port of op_claim.h) --------------------------------------------------

use ort::*;

/// Float dtypes the dtype-generic MLX paths handle: fp32, fp16, bf16.
pub fn is_mlx_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
}

pub fn is_signed_integer(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

pub fn is_unsigned_integer(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
}

/// The most relaxed dtype set the MLX Metal backend can carry: bool, all int/uint widths (8-64),
/// and fp16/bf16/fp32. EXCLUDES float64/complex/string/fp8. Port of `IsMlxSupportedType`.
pub fn is_mlx_supported(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t) || is_signed_integer(t) || is_unsigned_integer(t)
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

/// Numeric (non-bool) MLX types: the reductions/argmin-max/cumsum dtype set.
pub fn is_mlx_numeric(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t) || is_signed_integer(t) || is_unsigned_integer(t)
}

/// Dtypes the pure data-movement ops carry end-to-end (every case CopyOut can memcpy). Excludes
/// float64 and uint64 (no CopyOut case). Port of `IsMovableType`.
pub fn is_movable(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t)
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

/// int32/int64 — the gather/scatter index dtype.
pub fn is_int_index(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

/// int16/int32/int64 — the Range element dtype.
pub fn is_range_type(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

/// Strict elementwise-or-trailing-suffix broadcast (rejects mismatched non-suffix shapes). A scalar
/// operand is allowed only via `scalar_or_suffix`.
pub fn suffix_broadcast(a: &[i64], b: &[i64]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    // The longer shape's trailing dims must match the shorter shape's dims (numpy suffix rule),
    // requiring equal or 1 on the broadcast side.
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let off = long.len() - short.len();
    for i in 0..short.len() {
        let l = long[off + i];
        let s = short[i];
        if l != s && s != 1 && l != 1 {
            return false;
        }
    }
    true
}

/// Lenient variant that also accepts a genuine scalar operand (empty shape).
pub fn scalar_or_suffix_broadcast(a: &[i64], b: &[i64]) -> bool {
    if a.is_empty() || b.is_empty() {
        return true;
    }
    suffix_broadcast(a, b)
}
