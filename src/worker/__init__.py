"""Worker subprocess architecture for AI inference.

This module provides process isolation for AI model inference, enabling:
- GPU memory protection via coordinator serialization
- Worker lingering (60s default) for better performance
- Multi-environment support for conflicting dependencies
- On-demand environment installation

Key components:
- coordinator: WorkerCoordinator for GPU serialization and worker lifecycle
- env_manager: EnvironmentManager for on-demand env installation
- base: BaseWorker abstract class for implementing workers
- protocol: Shared request/response models
- utils: Helper functions for worker management
"""

from .coordinator import (
    WorkerCoordinator,
    get_coordinator,
    coordinator_infer,
)
from .env_manager import (
    EnvironmentManager,
    get_env_manager,
    EnvStatus,
    EnvInfo,
)
from .protocol import WorkerHandle, WorkerState

__all__ = [
    "WorkerCoordinator",
    "get_coordinator",
    "coordinator_infer",
    "EnvironmentManager",
    "get_env_manager",
    "EnvStatus",
    "EnvInfo",
    "WorkerHandle",
    "WorkerState",
]
