"""onnxruntime-ep-mlx: MLX-native ONNX Runtime execution provider for Apple Silicon.

This package bundles the ``libonnxruntime_mlx_ep.dylib`` plugin execution
provider and a thin registration helper. Typical usage::

    import onnxruntime as ort
    import onnxruntime_ep_mlx

    onnxruntime_ep_mlx.register_execution_provider_library()
    sess = ort.InferenceSession(
        model,
        providers=["MLXExecutionProvider", "CPUExecutionProvider"],
    )

The plugin targets the ONNX Runtime plugin-EP C ABI (ORT_API_VERSION >= 27,
i.e. onnxruntime >= 1.22 with the plugin-EP API). ``onnxruntime`` must expose
``register_execution_provider_library`` (available in onnxruntime >= 1.22).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort

__all__ = [
    "EP_NAME",
    "ep_name",
    "version",
    "vendor",
    "library_path",
    "register_execution_provider_library",
    "append_to_session_options",
    "__version__",
]

#: The registered execution-provider name understood by ONNX Runtime. Must match
#: the name the Rust EP factory reports (rust/src/factory.rs) and the vendor id
#: onnx-genai binds by; do not change without updating the Rust crate.
EP_NAME: str = "MLXExecutionProvider"

#: Vendor string carried by the plugin (the repo name).
_VENDOR: str = "onnxruntime-ep-mlx"

#: Basename of the cargo-built plugin dylib bundled inside this package.
_PLUGIN_DYLIB_NAME: str = "libonnxruntime_mlx_ep.dylib"

try:  # Python 3.8+: read the installed distribution version.
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__: str = _pkg_version("onnxruntime-ep-mlx")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        __version__ = "0.0.0+unknown"
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+unknown"


def ep_name() -> str:
    """Return the registered execution-provider name (``"MLXExecutionProvider"``)."""
    return EP_NAME


def version() -> str:
    """Return the version string of the bundled MLX execution-provider plugin."""
    return __version__


def vendor() -> str:
    """Return the vendor string of the bundled plugin."""
    return _VENDOR


def library_path() -> str:
    """Return the absolute path to the bundled ``libonnxruntime_mlx_ep.dylib``.

    The dylib is the cargo-built Rust execution provider, bundled next to this
    module in the wheel. Resolved from this package's on-disk location so it
    keeps working regardless of where the wheel is installed.

    Raises:
        FileNotFoundError: if the bundled plugin dylib cannot be found (which
            indicates a broken installation).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PLUGIN_DYLIB_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bundled MLX EP plugin not found at {path!r}. "
            "The onnxruntime-ep-mlx wheel appears to be incomplete."
        )
    return path


def _import_onnxruntime() -> "ort":
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required to register the MLX execution provider. "
            "Install it with `pip install 'onnxruntime>=1.22'`."
        ) from exc
    return ort


def register_execution_provider_library(name: str = EP_NAME) -> str:
    """Register the bundled MLX EP plugin with ONNX Runtime.

    This calls :func:`onnxruntime.register_execution_provider_library` with the
    absolute path to the plugin dylib bundled in this package. It only needs to
    be called once per process before creating an
    :class:`onnxruntime.InferenceSession` that lists ``name`` in its providers.

    Args:
        name: The name to register the provider under. Defaults to
            ``"MLXExecutionProvider"``; use the same name in the session's
            ``providers`` list.

    Returns:
        The absolute path to the registered plugin dylib.
    """
    ort = _import_onnxruntime()
    if not hasattr(ort, "register_execution_provider_library"):
        raise RuntimeError(
            "The installed onnxruntime does not expose "
            "register_execution_provider_library (plugin-EP C API). "
            "Upgrade to onnxruntime>=1.22."
        )
    lib = library_path()
    ort.register_execution_provider_library(name, lib)
    return lib


def append_to_session_options(
    session_options: Any,
    name: str = EP_NAME,
    provider_options: "dict[str, str] | None" = None,
) -> Any:
    """Append the MLX execution provider to ``session_options`` if supported.

    Newer ONNX Runtime releases expose
    ``SessionOptions.add_provider_for_devices`` / ``add_provider``; when
    available this registers the plugin (if needed) and appends it. Prefer the
    2-line pattern documented in the module docstring if this method is not
    available in your onnxruntime build.

    Args:
        session_options: An :class:`onnxruntime.SessionOptions` instance.
        name: The provider name to append. Defaults to ``"MLXExecutionProvider"``.
        provider_options: Optional provider options mapping.

    Returns:
        The same ``session_options`` object, for chaining.
    """
    register_execution_provider_library(name)
    opts = dict(provider_options or {})
    if hasattr(session_options, "add_provider"):
        session_options.add_provider(name, opts)
    else:  # pragma: no cover - depends on onnxruntime version
        raise RuntimeError(
            "This onnxruntime build has no SessionOptions.add_provider; pass "
            f'providers=["{name}", "CPUExecutionProvider"] to InferenceSession '
            "after calling register_execution_provider_library() instead."
        )
    return session_options
