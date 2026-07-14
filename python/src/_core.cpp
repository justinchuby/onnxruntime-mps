// Copyright (c) 2026. Licensed under the MIT License.
//
// nanobind extension `onnxruntime_mlx._core`.
//
// It exposes just enough native surface for the Python layer to locate and
// register the bundled MLX execution-provider plugin dylib:
//   * ep_name()      -> the registered EP name ("MLXExecutionProvider")
//   * version()      -> the plugin version string
//   * library_path() -> absolute path to the bundled libonnxruntime_mlx_ep.dylib
//
// library_path() is resolved from this extension's own on-disk location (via
// dladdr) so it keeps working regardless of where the wheel is installed.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <dlfcn.h>

#include <string>

#include "onnxruntime_mlx/version.h"

namespace nb = nanobind;

namespace {

// Basename of the plugin dylib bundled next to this extension inside the wheel.
constexpr const char* kPluginDylibName = "libonnxruntime_mlx_ep.dylib";

// Directory containing this compiled extension, discovered at runtime.
std::string this_module_dir() {
  Dl_info info{};
  // Any symbol defined in this translation unit works as the dladdr anchor.
  if (dladdr(reinterpret_cast<const void*>(&this_module_dir), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::string path(info.dli_fname);
  const auto slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return path.substr(0, slash);
}

std::string library_path() {
  const std::string dir = this_module_dir();
  if (dir.empty()) {
    return kPluginDylibName;
  }
  return dir + "/" + kPluginDylibName;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "Native helpers for the MLX ONNX Runtime execution-provider plugin.";

  m.def("ep_name", []() { return std::string(ORT_MLX_EP_NAME); },
        "Registered execution-provider name (pass to register_execution_provider_library).");

  m.def("version", []() { return std::string(ORT_MLX_EP_VERSION); },
        "Version string of the bundled MLX execution-provider plugin.");

  m.def("vendor", []() { return std::string(ORT_MLX_EP_VENDOR); },
        "Vendor string of the plugin.");

  m.def("library_path", &library_path,
        "Absolute path to the bundled libonnxruntime_mlx_ep.dylib inside the installed package.");
}
