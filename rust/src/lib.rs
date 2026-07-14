//! Rust spike: an MLX-native ONNX Runtime plugin execution provider.
//!
//! Proves the two boundaries of a full Rust rewrite end-to-end:
//!   1. the ORT plugin-EP C ABI, implemented from Rust (`extern "C"` vtables), and
//!   2. mlx-c, bound DIRECTLY (bindgen, no mlx-rs crate) and driven from Rust.
//!
//! Scope: claims `Add` (fp32) and runs it through `mlx_add`. The existing
//! `tests/ops` pytest harness is the oracle (`test_binary_fp32[Add-...]`).

mod engine;
mod ep;
mod factory;
mod mlx;
mod ops;
mod registry;
mod sys;
mod trace;

use std::ffi::c_char;
use std::ptr;

use factory::MlxEpFactory;
use sys::ort;

/// ORT resolves this symbol via `dlsym` when a session calls
/// `register_execution_provider_library`.
///
/// # Safety
/// Called by ORT with valid ABI pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    registration_name: *const c_char,
    ort_api_base: *const ort::OrtApiBase,
    _default_logger: *const ort::OrtLogger,
    factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> *mut ort::OrtStatus {
    let api_base = &*ort_api_base;
    let get_api = api_base.GetApi.unwrap();
    let ort_api = get_api(factory::ORT_API_VERSION);
    if ort_api.is_null() {
        let legacy = get_api(1);
        return ((*legacy).CreateStatus.unwrap())(
            ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
            c"MLXExecutionProvider requires ONNX Runtime with ORT_API_VERSION >= 27".as_ptr(),
        );
    }
    let ep_api = ((*ort_api).GetEpApi.unwrap())();

    if max_factories < 1 {
        return ((*ort_api).CreateStatus.unwrap())(
            ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
            c"MLXExecutionProvider needs room for one OrtEpFactory".as_ptr(),
        );
    }

    let factory = MlxEpFactory::new(registration_name, ort_api, ep_api);
    *factories.add(0) = factory.as_ptr();
    *num_factories = 1;
    ptr::null_mut()
}

/// Free a factory created by `CreateEpFactories`.
///
/// # Safety
/// `factory` must have come from `CreateEpFactories`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(factory: *mut ort::OrtEpFactory) -> *mut ort::OrtStatus {
    factory::release_factory(factory);
    ptr::null_mut()
}
