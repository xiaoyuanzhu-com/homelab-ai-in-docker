"""Utility functions for worker management."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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


def get_gpu_memory_nvidia_smi(pid: Optional[int] = None) -> Optional[float]:
    """
    Get GPU memory usage via nvidia-smi for a specific process.

    Uses nvidia-smi which reports actual GPU memory allocated (more accurate
    than torch.cuda.memory_allocated() which only shows PyTorch allocations).

    Args:
        pid: Process ID to query. If None, uses current process.

    Returns:
        GPU memory in MB, or None if nvidia-smi unavailable or no GPU found.
    """
    if pid is None:
        pid = os.getpid()

    try:
        # Query process-specific GPU memory
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        # Parse output: each line is "pid, memory_mb"
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    line_pid = int(parts[0].strip())
                    memory_mb = float(parts[1].strip())
                    if line_pid == pid:
                        return memory_mb
                except (ValueError, IndexError):
                    continue

        # Process not found in GPU list (might not have allocated yet)
        return 0.0

    except FileNotFoundError:
        # nvidia-smi not installed
        return None
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return None
    except Exception as e:
        logger.warning(f"nvidia-smi failed: {e}")
        return None


class GpuMemoryTracker:
    """
    Tracks GPU memory usage by periodically sampling nvidia-smi.

    Usage:
        tracker = GpuMemoryTracker(pid=os.getpid(), interval=1.0)
        tracker.start()
        # ... do GPU work ...
        tracker.stop()
        print(f"Peak GPU: {tracker.peak_mb}MB")
    """

    def __init__(self, pid: Optional[int] = None, interval: float = 1.0):
        """
        Initialize GPU memory tracker.

        Args:
            pid: Process ID to track. Defaults to current process.
            interval: Sampling interval in seconds.
        """
        self.pid = pid or os.getpid()
        self.interval = interval

        self._before_mb: Optional[float] = None
        self._peak_mb: Optional[float] = None
        self._after_mb: Optional[float] = None
        self._samples: list[float] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def before_mb(self) -> Optional[float]:
        """GPU memory before task started."""
        return self._before_mb

    @property
    def peak_mb(self) -> Optional[float]:
        """Peak GPU memory during task."""
        return self._peak_mb

    @property
    def after_mb(self) -> Optional[float]:
        """GPU memory after task completed."""
        return self._after_mb

    @property
    def samples(self) -> list[float]:
        """All memory samples collected."""
        return self._samples.copy()

    def start(self) -> None:
        """Start tracking GPU memory."""
        if self._running:
            return

        # Sample before
        self._before_mb = get_gpu_memory_nvidia_smi(self.pid)
        self._samples = []
        if self._before_mb is not None:
            self._samples.append(self._before_mb)
            self._peak_mb = self._before_mb

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop tracking and record final memory."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Sample after
        self._after_mb = get_gpu_memory_nvidia_smi(self.pid)

        # Update peak with final sample
        if self._after_mb is not None:
            self._samples.append(self._after_mb)
            if self._peak_mb is None or self._after_mb > self._peak_mb:
                self._peak_mb = self._after_mb

    def _sample_loop(self) -> None:
        """Background thread that samples GPU memory."""
        while self._running:
            memory = get_gpu_memory_nvidia_smi(self.pid)
            if memory is not None:
                self._samples.append(memory)
                if self._peak_mb is None or memory > self._peak_mb:
                    self._peak_mb = memory
            time.sleep(self.interval)

    def get_stats(self) -> dict:
        """
        Get memory statistics.

        Returns:
            Dict with before_mb, peak_mb, after_mb, sample_count
        """
        return {
            "before_mb": self._before_mb,
            "peak_mb": self._peak_mb,
            "after_mb": self._after_mb,
            "sample_count": len(self._samples),
        }
