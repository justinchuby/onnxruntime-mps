"""Hatchling build hook for onnxruntime-ep-mlx.

Builds the Rust execution provider (`cargo build --release` in ../rust), then
bundles the resulting `libonnxruntime_mlx_ep.dylib` together with its mlx-c/mlx
runtime dependencies into the wheel package, relinked so they load from
``@loader_path`` (a self-contained wheel). Finally it forces a platform wheel
tag: the package ships no CPython-ABI extension, so a single
``py3-none-macosx_*_arm64`` wheel installs on 3.12, 3.13 and the free-threaded
builds alike.

The onnxruntime dependency is intentionally NOT bundled — it is resolved at
runtime from the host ``onnxruntime`` package (two-level namespace), matching the
EP's design.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PLUGIN_DYLIB = "libonnxruntime_mlx_ep.dylib"


def _brew_prefix(pkg: str) -> Path:
    out = subprocess.run(
        ["brew", "--prefix", pkg],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def _run(cmd: list[str], **kw) -> None:
    print("[onnxruntime-ep-mlx build] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def _resolve_ort_include() -> str:
    """Mirror rust/build.rs: ORT_INCLUDE_DIR, else $ORT_HOME/include."""
    inc = os.environ.get("ORT_INCLUDE_DIR")
    if not inc:
        home = os.environ.get("ORT_HOME")
        if home:
            inc = str(Path(home) / "include")
    if inc and (Path(inc) / "onnxruntime_c_api.h").is_file():
        return inc
    raise RuntimeError(
        "Could not locate the ONNX Runtime headers. Set ORT_INCLUDE_DIR to the "
        "ORT C-API include dir, or ORT_HOME to an ONNX Runtime release root "
        "(expects $ORT_HOME/include/onnxruntime_c_api.h)."
    )


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("onnxruntime-ep-mlx only builds on macOS (Apple Silicon).")

        project_root = Path(self.root)          # the python/ dir
        repo_root = project_root.parent
        rust_dir = repo_root / "rust"
        if not (rust_dir / "Cargo.toml").is_file():
            raise RuntimeError(
                f"Rust crate not found at {rust_dir}. The wheel must be built "
                "from a full repository checkout (python/ next to rust/)."
            )

        pkg_dir = project_root / "src" / "onnxruntime_ep_mlx"

        # 1) Build the Rust EP dylib.
        env = dict(os.environ)
        env["ORT_INCLUDE_DIR"] = _resolve_ort_include()
        _run(["cargo", "build", "--release"], cwd=str(rust_dir), env=env)

        built = rust_dir / "target" / "release" / PLUGIN_DYLIB
        if not built.is_file():
            raise RuntimeError(f"cargo did not produce {built}")

        # 2) Copy the plugin into the package.
        dest_plugin = pkg_dir / PLUGIN_DYLIB
        shutil.copy2(built, dest_plugin)
        os.chmod(dest_plugin, 0o755)

        # 3) Bundle + relink the mlx runtime next to the plugin (self-contained).
        self._bundle_mlx(pkg_dir, dest_plugin)

        # 4) This is a platform wheel with no Python-ABI extension: force a
        #    py3-none-macosx_*_arm64 tag so ONE wheel serves every interpreter
        #    (CPython 3.10+, free-threaded 3.13t/3.14t, ...).
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        # Honour MACOSX_DEPLOYMENT_TARGET for the platform floor: the bundled
        # dylibs (and mlx) target it, so the tag should advertise it rather than
        # whatever floor the running interpreter was built against.
        dep_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if dep_target and plat.startswith("macosx_"):
            arch = plat.rsplit("_", 1)[-1]
            major, _, minor = dep_target.partition(".")
            plat = f"macosx_{major}_{minor or '0'}_{arch}"
        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{plat}"

    # -- mlx bundling ---------------------------------------------------------
    def _bundle_mlx(self, pkg_dir: Path, plugin: Path) -> None:
        mlxc_pfx = _brew_prefix("mlx-c")
        mlx_pfx = _brew_prefix("mlx")
        mlxc_src = mlxc_pfx / "lib" / "libmlxc.dylib"
        mlx_src = mlx_pfx / "lib" / "libmlx.dylib"
        metallib_src = mlx_pfx / "lib" / "mlx.metallib"
        for f in (mlxc_src, mlx_src, metallib_src):
            if not f.is_file():
                raise RuntimeError(f"Required mlx artifact missing: {f} (brew install mlx-c)")

        mlxc_dst = pkg_dir / "libmlxc.dylib"
        mlx_dst = pkg_dir / "libmlx.dylib"
        for src, dst in ((mlxc_src, mlxc_dst), (mlx_src, mlx_dst), (metallib_src, pkg_dir / "mlx.metallib")):
            shutil.copy2(src, dst)
            os.chmod(dst, 0o644)
        os.chmod(mlxc_dst, 0o755)
        os.chmod(mlx_dst, 0o755)

        def name_tool(*args: str) -> None:
            _run(["install_name_tool", *args])

        def resign(f: Path) -> None:
            subprocess.run(["codesign", "--force", "--sign", "-", str(f)], check=False)

        # Bundled mlx install ids -> @loader_path.
        name_tool("-id", "@loader_path/libmlxc.dylib", str(mlxc_dst))
        name_tool("-id", "@loader_path/libmlx.dylib", str(mlx_dst))
        # libmlxc depends on libmlx by its absolute install name -> sibling.
        name_tool("-change", str(mlx_src), "@loader_path/libmlx.dylib", str(mlxc_dst))

        # Plugin's mlx deps -> colocated copies.
        name_tool("-change", str(mlxc_src), "@loader_path/libmlxc.dylib", str(plugin))
        name_tool("-change", str(mlx_src), "@loader_path/libmlx.dylib", str(plugin))

        # The Rust EP does NOT link libonnxruntime — it reaches ORT purely through
        # the OrtApi function-pointer table handed to CreateEpFactories (see
        # rust/build.rs). So there is no onnxruntime dependency to relink here;
        # ORT dlopen()s the plugin by the absolute path library_path() returns.

        # Re-sign everything we mutated (install_name_tool voids the ad-hoc sig;
        # dyld SIGKILLs unsigned/invalid arm64 images).
        for f in (mlxc_dst, mlx_dst, plugin):
            resign(f)
