"""Utility functions for worker management."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Optional


def find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_python_for_env(env_name: Optional[str]) -> str:
    """
    Get Python interpreter path for a virtual environment.

    Args:
        env_name: Virtual environment name (e.g., "deepseek-ocr" for .venv-deepseek-ocr)
                  If None, returns the current interpreter.

    Returns:
        Path to Python interpreter
    """
    if not env_name:
        return sys.executable

    # Try .venv-{env_name}/bin/python
    env_path = Path(f".venv-{env_name}/bin/python")
    if env_path.exists():
        return str(env_path.resolve())

    # Fall back to main env
    return sys.executable


def get_gpu_memory_mb() -> float:
    """
    Get GPU memory used by this process.

    Returns:
        GPU memory in MB, or 0.0 if no GPU available
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def cleanup_gpu_memory() -> None:
    """Clear GPU memory caches."""
    try:
        import gc

        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
