//! `MlxEpFactory` — our implementation of the ORT `OrtEpFactory` C-ABI vtable.
//!
//! The ORT struct is embedded as the FIRST field so a `*OrtEpFactory` handed to
//! ORT is pointer-identical to our `*MlxEpFactory` (repr(C), offset 0). We fill
//! only the entry points we need; everything else stays `None` (zeroed).

use std::ffi::{c_char, c_void, CString};
use std::ptr;

use crate::ep::MlxEp;
use crate::sys::ort;

pub const ORT_API_VERSION: u32 = 27;

#[repr(C)]
pub struct MlxEpFactory {
    base: ort::OrtEpFactory,
    pub ort_api: *const ort::OrtApi,
    pub ep_api: *const ort::OrtEpApi,
    name: CString,
    vendor: CString,
    version: CString,
}

impl MlxEpFactory {
    pub fn new(
        registration_name: *const c_char,
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
    ) -> Box<MlxEpFactory> {
        let name = unsafe {
            if registration_name.is_null() || *registration_name == 0 {
                c"MLXExecutionProvider".to_owned()
            } else {
                std::ffi::CStr::from_ptr(registration_name).to_owned()
            }
        };
        let mut base: ort::OrtEpFactory = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetVendor = Some(get_vendor);
        base.GetVendorId = Some(get_vendor_id);
        base.GetVersion = Some(get_version);
        base.GetSupportedDevices = Some(get_supported_devices);
        base.CreateEp = Some(create_ep);
        base.ReleaseEp = Some(release_ep);
        base.CreateAllocator = Some(create_allocator);
        base.ReleaseAllocator = Some(release_allocator);
        base.CreateDataTransfer = Some(create_data_transfer);
        base.CreateSyncStreamForDevice = Some(create_sync_stream_for_device);
        base.IsStreamAware = Some(is_stream_aware);

        Box::new(MlxEpFactory {
            base,
            ort_api,
            ep_api,
            name,
            vendor: c"onnxruntime-mlx".to_owned(),
            version: c"0.1.0".to_owned(),
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEpFactory {
        Box::into_raw(self) as *mut ort::OrtEpFactory
    }
}

#[inline]
unsafe fn this(p: *const ort::OrtEpFactory) -> *const MlxEpFactory {
    p as *const MlxEpFactory
}

unsafe extern "C" fn get_name(p: *const ort::OrtEpFactory) -> *const c_char {
    unsafe { (*this(p)).name.as_ptr() }
}
unsafe extern "C" fn get_vendor(p: *const ort::OrtEpFactory) -> *const c_char {
    unsafe { (*this(p)).vendor.as_ptr() }
}
unsafe extern "C" fn get_vendor_id(_p: *const ort::OrtEpFactory) -> u32 {
    0x106B // Apple
}
unsafe extern "C" fn get_version(p: *const ort::OrtEpFactory) -> *const c_char {
    unsafe { (*this(p)).version.as_ptr() }
}
unsafe extern "C" fn is_stream_aware(_p: *const ort::OrtEpFactory) -> bool {
    false
}

unsafe extern "C" fn get_supported_devices(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut ort::OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> *mut ort::OrtStatus {
    unsafe {
        let f = &*this(p);
        let ort_api = &*f.ort_api;
        let ep_api = &*f.ep_api;
        *num_ep_devices = 0;

        let hw_type = ort_api.HardwareDevice_Type.unwrap();
        let mut gpu: *const ort::OrtHardwareDevice = ptr::null();
        let mut cpu: *const ort::OrtHardwareDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *devices.add(i);
            let t = hw_type(dev);
            if t == ort::OrtHardwareDeviceType_OrtHardwareDeviceType_GPU && gpu.is_null() {
                gpu = dev;
            } else if t == ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU && cpu.is_null() {
                cpu = dev;
            }
        }
        let selected = if !gpu.is_null() { gpu } else { cpu };
        if selected.is_null() || max_ep_devices < 1 {
            return ptr::null_mut();
        }

        let mut ep_device: *mut ort::OrtEpDevice = ptr::null_mut();
        let create = ep_api.CreateEpDevice.unwrap();
        let st = create(
            p,
            selected,
            ptr::null(),
            ptr::null(),
            &mut ep_device,
        );
        if !st.is_null() {
            return st;
        }
        *ep_devices.add(0) = ep_device;
        *num_ep_devices = 1;
        eprintln!("[rust-mlx-ep] GetSupportedDevices: bound to {} device", if !gpu.is_null() { "GPU" } else { "CPU" });
        ptr::null_mut()
    }
}

unsafe extern "C" fn create_ep(
    p: *mut ort::OrtEpFactory,
    _devices: *const *const ort::OrtHardwareDevice,
    _ep_metadata: *const *const ort::OrtKeyValuePairs,
    num_devices: usize,
    _session_options: *const ort::OrtSessionOptions,
    logger: *const ort::OrtLogger,
    ep: *mut *mut ort::OrtEp,
) -> *mut ort::OrtStatus {
    unsafe {
        let f = &*this(p);
        *ep = ptr::null_mut();
        if num_devices != 1 {
            return ((*f.ort_api).CreateStatus.unwrap())(
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                c"MLXExecutionProvider expects exactly one device".as_ptr(),
            );
        }
        let mlx_ep = MlxEp::new(f.ort_api, f.ep_api, &f.name, logger);
        *ep = mlx_ep.as_ptr();
        ptr::null_mut()
    }
}

unsafe extern "C" fn release_ep(_p: *mut ort::OrtEpFactory, ep: *mut ort::OrtEp) {
    if !ep.is_null() {
        drop(unsafe { Box::from_raw(ep as *mut MlxEp) });
    }
}

// The spike advertises no device memory (I/O stays on the CPU allocator), so these
// slots are never exercised for real — but ORT calls some of them during library
// registration, so they must be non-NULL. Safe stubs: hand back nothing.
unsafe extern "C" fn create_allocator(
    _p: *mut ort::OrtEpFactory,
    _memory_info: *const ort::OrtMemoryInfo,
    _options: *const ort::OrtKeyValuePairs,
    allocator: *mut *mut ort::OrtAllocator,
) -> *mut ort::OrtStatus {
    unsafe { *allocator = ptr::null_mut() };
    ptr::null_mut()
}
unsafe extern "C" fn release_allocator(
    _p: *mut ort::OrtEpFactory,
    _allocator: *mut ort::OrtAllocator,
) {
}
unsafe extern "C" fn create_data_transfer(
    _p: *mut ort::OrtEpFactory,
    data_transfer: *mut *mut ort::OrtDataTransferImpl,
) -> *mut ort::OrtStatus {
    unsafe { *data_transfer = ptr::null_mut() };
    ptr::null_mut()
}
unsafe extern "C" fn create_sync_stream_for_device(
    _p: *mut ort::OrtEpFactory,
    _memory_device: *const ort::OrtMemoryDevice,
    _options: *const ort::OrtKeyValuePairs,
    stream: *mut *mut ort::OrtSyncStreamImpl,
) -> *mut ort::OrtStatus {
    unsafe { *stream = ptr::null_mut() };
    ptr::null_mut()
}

/// Free a factory created in `CreateEpFactories`.
///
/// # Safety
/// `p` must be a pointer returned by `MlxEpFactory::as_ptr`.
pub unsafe fn release_factory(p: *mut ort::OrtEpFactory) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p as *mut MlxEpFactory) });
    }
}

// Silence "field never read directly" — accessed via the ABI base pointer.
const _: fn() = || {
    let _ = std::mem::size_of::<*mut c_void>();
};
