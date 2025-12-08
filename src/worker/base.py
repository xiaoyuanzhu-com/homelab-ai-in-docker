"""Base class for all inference workers.

Workers are isolated subprocess servers that:
1. Load a single model at startup
2. Serve inference requests via HTTP
3. Self-terminate after idle timeout
4. Handle OOM by exiting to release GPU context
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .utils import cleanup_gpu_memory, get_gpu_memory_mb, GpuMemoryTracker

logger = logging.getLogger("worker")


class WorkerInferRequest(BaseModel):
    """Standard inference request."""

    payload: Dict[str, Any] = Field(..., description="Task-specific request data")
    request_id: str = Field(default="", description="Request tracking ID")


class GpuMemoryStats(BaseModel):
    """GPU memory statistics from inference."""

    before_mb: Optional[float] = Field(None, description="GPU memory before inference")
    peak_mb: Optional[float] = Field(None, description="Peak GPU memory during inference")
    after_mb: Optional[float] = Field(None, description="GPU memory after inference")
    sample_count: int = Field(0, description="Number of samples taken")


class WorkerInferResponse(BaseModel):
    """Standard inference response."""

    result: Dict[str, Any] = Field(..., description="Task-specific result data")
    request_id: str = Field(default="", description="Matches request ID")
    processing_time_ms: int = Field(..., description="Inference duration in ms")
    model: str = Field(..., description="Model that processed the request")
    gpu_memory: Optional[GpuMemoryStats] = Field(None, description="GPU memory stats")


class WorkerHealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="'ok' or 'error'")
    model: str = Field(..., description="Model ID being served")
    ready: bool = Field(..., description="True when ready for inference")
    task: str = Field(..., description="Task type (embedding, ocr, etc.)")


class WorkerInfoResponse(BaseModel):
    """Model and memory info response."""

    model: str
    task: str
    memory_mb: float
    load_time_ms: int


class BaseWorker(ABC):
    """
    Base class for all inference workers.

    Subclasses must implement:
    - task_name: Class attribute for task type
    - load_model(): Load model into memory
    - infer(): Run inference on loaded model
    """

    task_name: str = "unknown"  # Override in subclass

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.port = port
        self.idle_timeout = idle_timeout
        self.model_config = model_config or {}

        self._model: Any = None
        self._last_active = time.time()
        self._idle_task: Optional[asyncio.Task] = None
        self._load_time_ms: int = 0
        self._ready = False

        # Create FastAPI app
        self.app = FastAPI(title=f"{self.task_name} worker")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register HTTP endpoints."""

        @self.app.get("/healthz", response_model=WorkerHealthResponse)
        async def healthz() -> WorkerHealthResponse:
            return WorkerHealthResponse(
                status="ok" if self._ready else "starting",
                model=self.model_id,
                ready=self._ready,
                task=self.task_name,
            )

        @self.app.get("/info", response_model=WorkerInfoResponse)
        async def info() -> WorkerInfoResponse:
            return WorkerInfoResponse(
                model=self.model_id,
                task=self.task_name,
                memory_mb=get_gpu_memory_mb(),
                load_time_ms=self._load_time_ms,
            )

        @self.app.post("/infer", response_model=WorkerInferResponse)
        async def infer_endpoint(req: WorkerInferRequest) -> WorkerInferResponse:
            start = time.time()

            # Start GPU memory tracking (samples every 1s)
            gpu_tracker = GpuMemoryTracker(interval=1.0)
            gpu_tracker.start()

            try:
                result = await self._run_inference(req.payload)
            except Exception as e:
                gpu_tracker.stop()
                logger.error(f"Inference failed: {e}", exc_info=True)
                self._handle_error(e)
                raise HTTPException(status_code=500, detail=f"Inference error: {e}")

            # Stop tracking and get stats
            gpu_tracker.stop()
            gpu_stats = gpu_tracker.get_stats()

            proc_ms = int((time.time() - start) * 1000)
            self._last_active = time.time()
            self._schedule_idle_shutdown()

            return WorkerInferResponse(
                result=result,
                request_id=req.request_id,
                processing_time_ms=proc_ms,
                model=self.model_id,
                gpu_memory=GpuMemoryStats(**gpu_stats),
            )

        @self.app.post("/shutdown")
        async def shutdown_endpoint():
            asyncio.create_task(self._shutdown())
            return {"status": "shutting_down"}

    async def _run_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference in thread pool to avoid blocking.

        Subclasses can override for async inference.
        """
        return await asyncio.to_thread(self.infer, payload)

    def _handle_error(self, e: Exception) -> None:
        """Handle inference errors, exit on OOM."""
        msg = str(e).lower()
        if "out of memory" in msg:
            logger.error("CUDA OOM detected, exiting to release GPU context")
            cleanup_gpu_memory()
            os._exit(1)

    def _schedule_idle_shutdown(self) -> None:
        """Schedule shutdown after idle timeout."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

        async def _watchdog():
            try:
                await asyncio.sleep(self.idle_timeout)
                idle = time.time() - self._last_active
                if idle >= self.idle_timeout:
                    logger.info(f"Worker idle for {idle:.1f}s, shutting down")
                    await self._shutdown()
            except asyncio.CancelledError:
                pass

        self._idle_task = loop.create_task(_watchdog())

    async def _shutdown(self) -> None:
        """Cleanup and exit."""
        try:
            self.cleanup()
        finally:
            os._exit(0)

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the model into memory.

        Called once at startup. Should set self._model.

        Returns:
            The loaded model object
        """
        pass

    @abstractmethod
    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the loaded model.

        Args:
            payload: Task-specific request data

        Returns:
            Task-specific result data
        """
        pass

    def cleanup(self) -> None:
        """
        Release model resources.

        Called before shutdown. Override for custom cleanup.
        """
        if self._model is not None:
            # Try to move to CPU first
            if hasattr(self._model, "cpu"):
                try:
                    self._model.cpu()
                except Exception:
                    pass
            del self._model
            self._model = None

        gc.collect()
        cleanup_gpu_memory()

    def run(self) -> int:
        """
        Start the worker server.

        Loads model, starts uvicorn, handles signals.

        Returns:
            Exit code (0 for success)
        """
        # Load model
        logger.info(f"Loading model {self.model_id}...")
        start = time.time()
        try:
            self._model = self.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return 1

        self._load_time_ms = int((time.time() - start) * 1000)
        self._ready = True
        self._last_active = time.time()
        logger.info(
            f"Model loaded in {self._load_time_ms}ms, "
            f"GPU memory: {get_gpu_memory_mb():.1f}MB"
        )

        # Setup signal handler
        def _term_handler(signum, frame):
            try:
                asyncio.get_event_loop().create_task(self._shutdown())
            except Exception:
                os._exit(0)

        try:
            signal.signal(signal.SIGTERM, _term_handler)
        except Exception:
            pass

        # Start server
        import uvicorn

        logger.info(f"Worker started for {self.task_name}:{self.model_id} on port {self.port}")
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="info")
        return 0


def create_worker_main(worker_class: type[BaseWorker]) -> callable:
    """
    Create a main() function for a worker module.

    Usage in worker module:
        class MyWorker(BaseWorker):
            ...

        main = create_worker_main(MyWorker)

        if __name__ == "__main__":
            raise SystemExit(main(sys.argv[1:]))
    """

    def main(argv: list[str]) -> int:
        parser = argparse.ArgumentParser(description=f"{worker_class.task_name} worker")
        parser.add_argument("--model-id", required=True, help="Model identifier")
        parser.add_argument("--port", type=int, required=True, help="Port to listen on")
        parser.add_argument(
            "--idle-timeout", type=int, default=60, help="Idle seconds before exit"
        )
        # Allow extra args to be passed through to worker
        args, extra = parser.parse_known_args(argv)

        # Parse extra args as key=value pairs for model_config
        model_config = {}
        for arg in extra:
            if "=" in arg:
                key, value = arg.split("=", 1)
                key = key.lstrip("-").replace("-", "_")
                model_config[key] = value

        worker = worker_class(
            model_id=args.model_id,
            port=args.port,
            idle_timeout=args.idle_timeout,
            model_config=model_config,
        )
        return worker.run()

    return main
