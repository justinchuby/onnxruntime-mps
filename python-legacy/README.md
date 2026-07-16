# onnxruntime-mlx → renamed to `onnxruntime-ep-mlx`

> ⚠️ **This package has been renamed.** `onnxruntime-mlx` is now published as
> **[`onnxruntime-ep-mlx`](https://pypi.org/project/onnxruntime-ep-mlx/)**.

## Migrate

```bash
pip uninstall onnxruntime-mlx
pip install onnxruntime-ep-mlx
```

```python
import onnxruntime_ep_mlx        # was: import onnxruntime_mlx
onnxruntime_ep_mlx.register_execution_provider_library()
```

This final `onnxruntime-mlx` release is a thin **compatibility shim**: it depends on
`onnxruntime-ep-mlx` and re-exports it, so existing `import onnxruntime_mlx` code keeps
working (with a `DeprecationWarning`). It will not receive further updates — please
switch to **`onnxruntime-ep-mlx`**.

Source / docs: https://github.com/justinchuby/onnxruntime-mlx
