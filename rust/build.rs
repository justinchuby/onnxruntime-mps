use std::env;
use std::path::PathBuf;
use std::process::Command;

fn brew_prefix(pkg: &str) -> String {
    let out = Command::new("brew")
        .arg("--prefix")
        .arg(pkg)
        .output()
        .unwrap_or_else(|_| panic!("failed to run `brew --prefix {pkg}`"));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

/// Discover the ONNX Runtime C-API include directory for a standalone checkout.
///
/// Resolution order:
///   1. `ORT_INCLUDE_DIR` — an explicit include dir (highest precedence).
///   2. `$ORT_HOME/include` — the standard layout of an ORT release tarball
///      (`onnxruntime-osx-arm64-<ver>/include`), which is what CI provisions.
///
/// Whichever we pick, we verify `onnxruntime_c_api.h` is actually present so the
/// error surfaces here (with a clear message) rather than deep inside bindgen.
fn resolve_ort_include() -> String {
    let candidate = env::var("ORT_INCLUDE_DIR").ok().or_else(|| {
        env::var("ORT_HOME")
            .ok()
            .map(|home| format!("{home}/include"))
    });

    match candidate {
        Some(dir) if PathBuf::from(&dir).join("onnxruntime_c_api.h").is_file() => dir,
        Some(dir) => panic!(
            "ORT include dir '{dir}' does not contain onnxruntime_c_api.h. \
             Set ORT_INCLUDE_DIR to the ONNX Runtime C-API include directory, \
             or set ORT_HOME to an ONNX Runtime release root (expects \
             $ORT_HOME/include/onnxruntime_c_api.h)."
        ),
        None => panic!(
            "Could not locate the ONNX Runtime headers. Set ORT_INCLUDE_DIR to \
             the ORT C-API include directory, or set ORT_HOME to an ONNX Runtime \
             release root (expects $ORT_HOME/include/onnxruntime_c_api.h)."
        ),
    }
}

fn main() {
    let ort_inc = resolve_ort_include();
    let mlxc = brew_prefix("mlx-c");
    let mlx = brew_prefix("mlx");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=wrapper_ort.h");
    println!("cargo:rerun-if-changed=wrapper_mlx.h");
    println!("cargo:rerun-if-env-changed=ORT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=ORT_HOME");

    // --- ORT plugin-EP C ABI bindings (pure C header; pulls in onnxruntime_ep_c_api.h) ---
    bindgen::Builder::default()
        .header("wrapper_ort.h")
        .clang_arg(format!("-I{ort_inc}"))
        .allowlist_type("Ort.*")
        .allowlist_type("ONNX.*")
        .allowlist_function("Ort.*")
        .wrap_unsafe_ops(true)
        .generate()
        .expect("ORT bindgen failed")
        .write_to_file(out.join("ort.rs"))
        .unwrap();

    // --- mlx-c bindings (bind the C API DIRECTLY, no mlx-rs crate) ---
    bindgen::Builder::default()
        .header("wrapper_mlx.h")
        .clang_arg(format!("-I{mlxc}/include"))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .wrap_unsafe_ops(true)
        .generate()
        .expect("mlx-c bindgen failed")
        .write_to_file(out.join("mlx.rs"))
        .unwrap();

    // We call ORT purely through the OrtApi function-pointer table handed to
    // CreateEpFactories, so we do NOT link libonnxruntime. Only mlx-c + mlx + frameworks.
    println!("cargo:rustc-link-search=native={mlxc}/lib");
    println!("cargo:rustc-link-search=native={mlx}/lib");
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlxc}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx}/lib");
    for fw in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
