"""Deprecated: ``onnxruntime-mlx`` has been renamed to ``onnxruntime-ep-mlx``.

Install and import the new package instead::

    pip install onnxruntime-ep-mlx
    import onnxruntime_ep_mlx

This module is a thin compatibility shim: it depends on ``onnxruntime-ep-mlx`` and
re-exports it (with a :class:`DeprecationWarning`) so existing ``import onnxruntime_mlx``
code keeps working. It will not receive further updates.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The 'onnxruntime-mlx' package has been renamed to 'onnxruntime-ep-mlx'. "
    "Install 'onnxruntime-ep-mlx' and use 'import onnxruntime_ep_mlx' instead; "
    "'onnxruntime-mlx' is now a compatibility shim and is no longer maintained.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the new package's public API so old imports keep working unchanged.
from onnxruntime_ep_mlx import *  # noqa: F401,F403
from onnxruntime_ep_mlx import __all__ as __all__  # noqa: F401
from onnxruntime_ep_mlx import __version__ as __version__  # noqa: F401
