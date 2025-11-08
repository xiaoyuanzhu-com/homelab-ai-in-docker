"""Best-effort initialization of CUDA/NVIDIA shared library search paths.

This addresses missing libcudnn/libcublas on Python 3.13 where pip's nvidia-*
packages may not set LD_LIBRARY_PATH early enough for Torch to find them.

Safe to call multiple times; no-op if paths already present.
"""

from __future__ import annotations

import glob
import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _candidate_site_packages() -> list[Path]:
    paths: list[str] = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        # venv user site
        p = site.getusersitepackages()
        if p:
            paths.append(p)
    except Exception:
        pass
    # sys.path entries that look like site-packages
    for p in sys.path:
        if p and "site-packages" in p:
            paths.append(p)
    # De-duplicate and keep order
    seen = set()
    out: list[Path] = []
    for p in paths:
        try:
            path = Path(p)
            if path.exists() and path not in seen:
                out.append(path)
                seen.add(path)
        except Exception:
            continue
    return out


def _nvidia_lib_dirs(site_pkgs: list[Path]) -> list[Path]:
    subdirs = [
        "nvidia/cuda_runtime/lib",
        "nvidia/cublas/lib",
        "nvidia/cudnn/lib",
        "nvidia/cufft/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/nvjitlink/lib",
        "nvidia/cuda_cupti/lib",
        "torch/lib",
    ]
    out: list[Path] = []
    for base in site_pkgs:
        for sub in subdirs:
            p = base / sub
            if p.exists():
                out.append(p)
    # De-duplicate
    seen = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _prepend_ld_library_path(paths: list[Path]) -> None:
    if not paths:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    add = ":".join(str(p) for p in paths)
    # Prepend so our paths win
    new_value = f"{add}:{current}" if current else add
    os.environ["LD_LIBRARY_PATH"] = new_value
    logger.debug("LD_LIBRARY_PATH updated with CUDA/NVIDIA libs: %s", add)


def _try_preload_libs(paths: list[Path]) -> None:
    # Attempt to preload critical libs explicitly so later dlopen resolves
    try:
        import ctypes  # noqa: WPS433

        # Determine RTLD_GLOBAL if available
        try:
            mode = ctypes.RTLD_GLOBAL  # type: ignore[attr-defined]
        except Exception:
            mode = None

        def _load_one(patterns: list[str]) -> None:
            for p in paths:
                for pat in patterns:
                    for lib in glob.glob(str(p / pat)):
                        try:
                            if mode is None:
                                ctypes.CDLL(lib)
                            else:
                                ctypes.CDLL(lib, mode=mode)
                            logger.debug("Preloaded shared library: %s", lib)
                            return
                        except Exception as e:  # noqa: WPS429
                            logger.debug("Failed to preload %s: %s", lib, e)

        # Try cudnn variants first (cnn/ops/core), then cublas
        _load_one(["libcudnn_cnn.so.9*", "libcudnn_ops.so.9*", "libcudnn.so.9*"])
        _load_one(["libcublasLt.so.12*", "libcublas.so.12*"])
    except Exception as e:
        logger.debug("Skipping shared library preload: %s", e)


def setup_cuda_libraries() -> None:
    """Ensure CUDA/NVIDIA libraries from pip are discoverable at runtime.

    Call this before importing torch/torchaudio/pyannote to avoid errors like:
      Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
      Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
    """
    try:
        site_pkgs = _candidate_site_packages()
        lib_dirs = _nvidia_lib_dirs(site_pkgs)
        if not lib_dirs:
            logger.debug("No NVIDIA/CUDA lib dirs found under site-packages.")
            return
        _prepend_ld_library_path(lib_dirs)
        _try_preload_libs(lib_dirs)
    except Exception as e:
        logger.debug("CUDA library path setup failed: %s", e)

