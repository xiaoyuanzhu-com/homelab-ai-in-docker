"""Shared request/response models for worker protocol.

All workers expose the same HTTP interface:
- GET  /healthz  -> Health check with model info
- POST /infer    -> Task-specific inference
- POST /shutdown -> Graceful shutdown
- GET  /info     -> Model and memory info
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time


class WorkerState(str, Enum):
    """Worker lifecycle states."""

    STARTING = "starting"  # Process spawned, loading model
    READY = "ready"  # Model loaded, ready for inference
    BUSY = "busy"  # Processing inference request
    TERMINATED = "terminated"  # Shutdown complete


@dataclass
class HealthResponse:
    """Response from /healthz endpoint."""

    status: str  # "ok" or "error"
    model: str  # Model ID being served
    ready: bool  # True when model is loaded and ready
    task: str  # Task type (embedding, captioning, etc.)


@dataclass
class InfoResponse:
    """Response from /info endpoint."""

    model: str  # Model ID
    task: str  # Task type
    memory_mb: float  # GPU memory used in MB
    load_time_ms: int  # Time taken to load model


@dataclass
class InferRequest:
    """Standard inference request to worker."""

    payload: Dict[str, Any]  # Task-specific data
    request_id: str  # For tracking


@dataclass
class InferResponse:
    """Standard inference response from worker."""

    result: Dict[str, Any]  # Task-specific result
    request_id: str  # Matches request
    processing_time_ms: int  # Inference duration
    model: str  # Model that processed request


@dataclass
class WorkerHandle:
    """Coordinator's view of a running worker."""

    task: str  # Task type (embedding, ocr, etc.)
    model_id: str  # Model being served
    port: int  # HTTP port
    proc: Any  # subprocess.Popen
    state: WorkerState = WorkerState.STARTING
    spawned_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    python_env: Optional[str] = None  # Custom venv name if not main

    @property
    def key(self) -> str:
        """Unique key for this worker (task:model_id)."""
        return f"{self.task}:{self.model_id}"

    def is_alive(self) -> bool:
        """Check if worker process is still running."""
        return self.proc.poll() is None
