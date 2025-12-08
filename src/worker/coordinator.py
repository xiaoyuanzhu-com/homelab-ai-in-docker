"""WorkerCoordinator - Centralized coordinator for all AI workers.

Responsibilities:
- Serialize GPU access (only one worker active at a time)
- Manage worker lifecycle (spawn, reuse, terminate)
- Route requests to appropriate workers
- Handle worker health monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from .protocol import WorkerHandle, WorkerState
from .utils import find_free_port, get_python_for_env
from ..db import worker_logs

logger = logging.getLogger(__name__)


# Worker module paths for each task type
WORKER_MODULES: Dict[str, str] = {
    "embedding": "src.worker.workers.embedding_worker",
    "captioning": "src.worker.workers.captioning_worker",
    "text-generation": "src.worker.workers.text_generation_worker",
    "asr": "src.worker.workers.asr_worker",
    "ocr": "src.worker.workers.ocr_worker",
    "speaker-diarization": "src.worker.workers.diarization_worker",
}


@dataclass
class WorkerConfig:
    """Configuration for spawning a worker."""

    task: str
    model_id: str
    python_env: Optional[str] = None  # Custom venv name
    extra_args: Dict[str, str] = field(default_factory=dict)  # Extra CLI args


class WorkerCoordinator:
    """
    Centralized coordinator for all AI workers.

    Uses a single GPU lock to serialize all GPU operations:
    - Worker spawn (model loading)
    - Inference requests

    This prevents OOM from concurrent GPU operations.
    """

    def __init__(
        self,
        worker_idle_timeout: int = 60,
        worker_startup_timeout: int = 120,
        worker_request_timeout: int = 300,
    ):
        """
        Initialize coordinator.

        Args:
            worker_idle_timeout: Seconds before idle worker self-terminates
            worker_startup_timeout: Max seconds to wait for worker startup
            worker_request_timeout: Max seconds for inference request
        """
        self._gpu_lock = asyncio.Lock()  # Serializes ALL GPU operations
        self._workers: Dict[str, WorkerHandle] = {}  # key -> worker
        self._worker_lock = asyncio.Lock()  # Protects _workers dict

        self.worker_idle_timeout = worker_idle_timeout
        self.worker_startup_timeout = worker_startup_timeout
        self.worker_request_timeout = worker_request_timeout

    def _worker_key(self, task: str, model_id: str) -> str:
        """Generate unique key for a worker."""
        return f"{task}:{model_id}"

    async def _spawn_worker(self, config: WorkerConfig) -> WorkerHandle:
        """
        Spawn a new worker subprocess.

        Args:
            config: Worker configuration

        Returns:
            WorkerHandle for the new worker
        """
        module = WORKER_MODULES.get(config.task)
        if not module:
            raise ValueError(f"Unknown task type: {config.task}")

        port = find_free_port()
        python = get_python_for_env(config.python_env)

        cmd = [
            python,
            "-m",
            module,
            "--model-id",
            config.model_id,
            "--port",
            str(port),
            "--idle-timeout",
            str(self.worker_idle_timeout),
        ]

        # Add extra args
        for key, value in config.extra_args.items():
            cmd.append(f"--{key}={value}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        logger.info(f"Spawning worker: {config.task}:{config.model_id} on port {port}")
        proc = subprocess.Popen(cmd, env=env)

        worker = WorkerHandle(
            task=config.task,
            model_id=config.model_id,
            port=port,
            proc=proc,
            state=WorkerState.STARTING,
            python_env=config.python_env,
        )

        # Wait for worker to become ready
        await self._wait_ready(worker)
        worker.state = WorkerState.READY
        worker.last_active = time.time()

        return worker

    async def _wait_ready(self, worker: WorkerHandle) -> None:
        """
        Wait for worker to become ready (respond to /healthz).

        Args:
            worker: Worker to wait for

        Raises:
            RuntimeError: If worker exits before ready
            TimeoutError: If worker doesn't become ready in time
        """
        url = f"http://127.0.0.1:{worker.port}/healthz"
        deadline = time.time() + self.worker_startup_timeout

        while time.time() < deadline:
            if not worker.is_alive():
                raise RuntimeError(
                    f"Worker {worker.key} exited before becoming ready "
                    f"(exit code: {worker.proc.returncode})"
                )

            try:
                req = urlrequest.Request(url, method="GET")
                with urlrequest.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        if data.get("ready", False):
                            logger.info(f"Worker {worker.key} ready on port {worker.port}")
                            return
            except (URLError, HTTPError, ConnectionError, TimeoutError):
                pass

            await asyncio.sleep(0.5)

        raise TimeoutError(
            f"Worker {worker.key} failed to become ready in {self.worker_startup_timeout}s"
        )

    async def _ensure_worker(self, config: WorkerConfig) -> WorkerHandle:
        """
        Get existing worker or spawn a new one.

        Args:
            config: Worker configuration

        Returns:
            Ready WorkerHandle
        """
        key = self._worker_key(config.task, config.model_id)

        async with self._worker_lock:
            worker = self._workers.get(key)

            # Check if existing worker is still alive
            if worker is not None:
                if worker.is_alive():
                    return worker
                else:
                    # Clean up dead worker
                    logger.warning(f"Worker {key} died, removing")
                    self._workers.pop(key, None)

            # Spawn new worker
            worker = await self._spawn_worker(config)
            self._workers[key] = worker
            return worker

    async def _send_request(
        self, worker: WorkerHandle, payload: Dict[str, Any], request_id: str = ""
    ) -> Dict[str, Any]:
        """
        Send inference request to worker.

        Args:
            worker: Target worker
            payload: Task-specific request data
            request_id: Optional request ID for tracking

        Returns:
            Inference result
        """
        url = f"http://127.0.0.1:{worker.port}/infer"
        data = json.dumps({"payload": payload, "request_id": request_id}).encode()

        def _do_request():
            req = urlrequest.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=self.worker_request_timeout) as resp:
                    return json.loads(resp.read().decode())
            except HTTPError as e:
                try:
                    error_body = e.read().decode()
                    error_data = json.loads(error_body)
                    error_msg = error_data.get("detail", str(e))
                except Exception:
                    error_msg = f"Worker returned HTTP {e.code}: {e.reason}"
                raise RuntimeError(f"Worker error: {error_msg}") from e

        try:
            result = await asyncio.to_thread(_do_request)
            worker.last_active = time.time()
            return result
        except TimeoutError:
            # Kill stuck worker
            logger.error(f"Worker {worker.key} timed out, terminating")
            await self._terminate_worker(worker)
            raise TimeoutError(
                f"Worker inference timed out after {self.worker_request_timeout}s"
            )

    async def _terminate_worker(self, worker: WorkerHandle) -> None:
        """
        Terminate a worker gracefully, then forcefully if needed.

        Args:
            worker: Worker to terminate
        """
        if not worker.is_alive():
            return

        # Try graceful shutdown via HTTP
        try:
            url = f"http://127.0.0.1:{worker.port}/shutdown"
            req = urlrequest.Request(url, method="POST")
            with urlrequest.urlopen(req, timeout=1):
                pass
        except Exception:
            pass

        # Send SIGTERM
        try:
            worker.proc.terminate()
        except Exception:
            pass

        # Wait briefly then SIGKILL
        try:
            worker.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                worker.proc.kill()
                worker.proc.wait(timeout=2)
            except Exception:
                pass

        worker.state = WorkerState.TERMINATED

        # Remove from registry
        async with self._worker_lock:
            self._workers.pop(worker.key, None)

    async def infer(
        self,
        task: str,
        model_id: str,
        payload: Dict[str, Any],
        request_id: str = "",
        python_env: Optional[str] = None,
        extra_args: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a worker.

        This is the main entry point for all inference requests.
        Holds GPU lock during spawn and inference to prevent OOM.

        Args:
            task: Task type (embedding, ocr, etc.)
            model_id: Model to use
            payload: Task-specific request data
            request_id: Optional request ID for tracking
            python_env: Optional custom Python environment
            extra_args: Optional extra args for worker

        Returns:
            Inference result from worker
        """
        config = WorkerConfig(
            task=task,
            model_id=model_id,
            python_env=python_env,
            extra_args=extra_args or {},
        )

        # Calculate input size for logging
        input_size = len(json.dumps(payload).encode()) if payload else 0

        # Create log entry at start
        log_id = None
        try:
            log_id = worker_logs.create_log(
                task=task,
                model_id=model_id,
                request_id=request_id or None,
                input_size_bytes=input_size,
            )
        except Exception as e:
            logger.warning(f"Failed to create worker log: {e}")

        start_time = time.time()
        worker = None
        response = None
        error_msg = None

        try:
            async with self._gpu_lock:
                # Get or spawn worker (may load model -> uses GPU)
                worker = await self._ensure_worker(config)

                # Update log with worker info
                if log_id is not None:
                    try:
                        from ..db.db_config import get_db
                        with get_db() as conn:
                            conn.execute(
                                "UPDATE worker_logs SET worker_port = ?, worker_pid = ? WHERE id = ?",
                                (worker.port, worker.proc.pid, log_id)
                            )
                    except Exception:
                        pass

                # Send inference request (uses GPU)
                response = await self._send_request(worker, payload, request_id)

        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            # Complete the log entry
            duration_ms = int((time.time() - start_time) * 1000)
            if log_id is not None:
                try:
                    # Extract GPU memory stats from response if available
                    gpu_memory = response.get("gpu_memory", {}) if response else {}
                    output_size = len(json.dumps(response).encode()) if response else 0

                    worker_logs.complete_log(
                        log_id=log_id,
                        duration_ms=duration_ms,
                        gpu_memory_before_mb=gpu_memory.get("before_mb"),
                        gpu_memory_peak_mb=gpu_memory.get("peak_mb"),
                        gpu_memory_after_mb=gpu_memory.get("after_mb"),
                        output_size_bytes=output_size,
                        status="completed" if error_msg is None else "failed",
                        error_message=error_msg,
                    )
                except Exception as e:
                    logger.warning(f"Failed to complete worker log: {e}")

        # Extract result from response
        return response.get("result", response)

    async def get_worker_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all workers.

        Returns:
            Dict of worker key -> status info
        """
        async with self._worker_lock:
            status = {}
            for key, worker in self._workers.items():
                if worker.is_alive():
                    idle_seconds = time.time() - worker.last_active
                    status[key] = {
                        "state": worker.state.value,
                        "port": worker.port,
                        "idle_seconds": round(idle_seconds, 1),
                        "python_env": worker.python_env,
                    }
            return status

    async def shutdown_all(self) -> None:
        """Terminate all workers."""
        async with self._worker_lock:
            workers = list(self._workers.values())

        for worker in workers:
            try:
                await self._terminate_worker(worker)
            except Exception as e:
                logger.warning(f"Error terminating worker {worker.key}: {e}")

    def is_gpu_locked(self) -> bool:
        """Check if GPU lock is currently held."""
        return self._gpu_lock.locked()


# Global coordinator instance
_coordinator: Optional[WorkerCoordinator] = None


def get_coordinator() -> WorkerCoordinator:
    """
    Get or create the global WorkerCoordinator.

    Returns:
        Global coordinator instance
    """
    global _coordinator
    if _coordinator is None:
        from ..db.settings import get_setting_int

        _coordinator = WorkerCoordinator(
            worker_idle_timeout=get_setting_int("worker_idle_timeout_seconds", 60),
            worker_startup_timeout=get_setting_int("worker_startup_timeout_seconds", 120),
            worker_request_timeout=get_setting_int("worker_request_timeout_seconds", 300),
        )
    return _coordinator


async def coordinator_infer(
    task: str,
    model_id: str,
    payload: Dict[str, Any],
    request_id: str = "",
    python_env: Optional[str] = None,
    extra_args: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run inference via global coordinator.

    Args:
        task: Task type (embedding, ocr, etc.)
        model_id: Model to use
        payload: Task-specific request data
        request_id: Optional request ID for tracking
        python_env: Optional custom Python environment
        extra_args: Optional extra args for worker

    Returns:
        Inference result from worker
    """
    return await get_coordinator().infer(
        task=task,
        model_id=model_id,
        payload=payload,
        request_id=request_id,
        python_env=python_env,
        extra_args=extra_args,
    )
