"""Worker subprocess architecture for AI inference.

This module provides process isolation for AI model inference, enabling:
- GPU memory protection via coordinator serialization
- Worker lingering (60s default) for better performance
- Multi-environment support for conflicting dependencies

Key components:
- coordinator: WorkerCoordinator for GPU serialization and worker lifecycle
- base: BaseWorker abstract class for implementing workers
- protocol: Shared request/response models
- utils: Helper functions for worker management
"""

from .coordinator import (
    WorkerCoordinator,
    get_coordinator,
    coordinator_infer,
)
from .protocol import WorkerHandle, WorkerState

__all__ = [
    "WorkerCoordinator",
    "get_coordinator",
    "coordinator_infer",
    "WorkerHandle",
    "WorkerState",
]
