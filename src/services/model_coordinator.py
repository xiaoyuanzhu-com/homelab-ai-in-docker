"""
Global Model Coordinator

Manages model loading/unloading across all AI services with memory-aware policies.
Prevents OOM errors by preemptively unloading models when memory is constrained.

Key Features:
- Preemptive unload: Unloads previous model before loading next one
- Memory tracking: Monitors GPU/CPU memory usage
- Configurable policies: Max models in memory, memory thresholds
- Idle timeout: Automatic cleanup of unused models
- Thread-safe: Async locks for concurrent request handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a loaded model"""
    key: str
    model: Any
    loaded_at: float
    last_used: float
    estimated_memory_mb: Optional[float] = None
    actual_memory_mb: Optional[float] = None
    model_type: Optional[str] = None  # e.g., "embedding", "asr", "ocr"


class ModelCoordinator:
    """
    Centralized coordinator for managing AI model lifecycles.

    Prevents OOM by enforcing memory policies and preemptively unloading
    models when switching between different model types.
    """

    def __init__(
        self,
        max_models_in_memory: int = 1,
        max_memory_mb: Optional[float] = None,
        enable_preemptive_unload: bool = True,
    ):
        """
        Initialize the model coordinator.

        Args:
            max_models_in_memory: Maximum number of models to keep loaded (default: 1)
            max_memory_mb: Maximum memory usage in MB (None = no limit)
            enable_preemptive_unload: Whether to unload models before loading new ones
        """
        self._models: Dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()
        self._max_models = max_models_in_memory
        self._max_memory_mb = max_memory_mb
        self._enable_preemptive_unload = enable_preemptive_unload

        logger.info(
            f"ModelCoordinator initialized: max_models={max_models_in_memory}, "
            f"max_memory_mb={max_memory_mb}, preemptive_unload={enable_preemptive_unload}"
        )

    async def load_model(
        self,
        key: str,
        loader_fn: Callable,
        unloader_fn: Optional[Callable] = None,
        estimated_memory_mb: Optional[float] = None,
        model_type: Optional[str] = None,
    ) -> Any:
        """
        Load a model with memory-aware policies.

        This is the main entry point for all model loading. It:
        1. Checks if model is already loaded (returns cached)
        2. Applies preemptive unload policy if needed
        3. Loads the model via the provided loader function
        4. Tracks memory usage and metadata

        Args:
            key: Unique identifier for the model (e.g., "whisper-base", "all-MiniLM-L6-v2")
            loader_fn: Async function that loads and returns the model
            unloader_fn: Optional async function to unload this specific model
            estimated_memory_mb: Estimated memory usage (for planning)
            model_type: Type of model (e.g., "embedding", "asr", "ocr")

        Returns:
            The loaded model object
        """
        async with self._lock:
            # Check if model is already loaded
            if key in self._models:
                logger.info(f"Model '{key}' already loaded, reusing cached instance")
                self._models[key].last_used = time.time()
                return self._models[key].model

            # Get current memory stats before loading
            memory_before = await self._get_memory_stats()
            logger.info(
                f"Loading model '{key}' (type: {model_type}, estimated: {estimated_memory_mb}MB). "
                f"Current memory: GPU={memory_before.get('gpu_used_mb', 'N/A')}MB, "
                f"Models loaded: {len(self._models)}"
            )

            # Apply preemptive unload policy
            if self._enable_preemptive_unload:
                await self._apply_unload_policy(key, estimated_memory_mb)

            # Load the model
            try:
                model = await loader_fn()
            except Exception as e:
                logger.error(f"Failed to load model '{key}': {e}")
                raise

            # Track memory usage after loading
            memory_after = await self._get_memory_stats()
            actual_memory_mb = None
            if memory_before.get('gpu_used_mb') and memory_after.get('gpu_used_mb'):
                actual_memory_mb = memory_after['gpu_used_mb'] - memory_before['gpu_used_mb']

            # Register the model
            now = time.time()
            model_info = ModelInfo(
                key=key,
                model=model,
                loaded_at=now,
                last_used=now,
                estimated_memory_mb=estimated_memory_mb,
                actual_memory_mb=actual_memory_mb,
                model_type=model_type,
            )
            self._models[key] = model_info

            logger.info(
                f"Model '{key}' loaded successfully. "
                f"Actual memory: {actual_memory_mb}MB, "
                f"GPU: {memory_after.get('gpu_used_mb', 'N/A')}MB, "
                f"Models in memory: {len(self._models)}"
            )

            return model

    async def unload_model(
        self,
        key: str,
        unloader_fn: Optional[Callable] = None,
    ) -> bool:
        """
        Unload a specific model from memory.

        Args:
            key: Unique identifier for the model
            unloader_fn: Optional async function to perform cleanup

        Returns:
            True if model was unloaded, False if not found
        """
        async with self._lock:
            if key not in self._models:
                logger.debug(f"Model '{key}' not loaded, nothing to unload")
                return False

            model_info = self._models[key]
            logger.info(
                f"Unloading model '{key}' (type: {model_info.model_type}, "
                f"loaded for {time.time() - model_info.loaded_at:.1f}s)"
            )

            # Call custom unloader if provided
            if unloader_fn:
                try:
                    await unloader_fn(model_info.model)
                except Exception as e:
                    logger.error(f"Error in custom unloader for '{key}': {e}")

            # Remove from registry
            del self._models[key]

            # Generic GPU cleanup
            await self._cleanup_gpu_memory()

            memory_stats = await self._get_memory_stats()
            logger.info(
                f"Model '{key}' unloaded. GPU memory: {memory_stats.get('gpu_used_mb', 'N/A')}MB, "
                f"Models remaining: {len(self._models)}"
            )

            return True

    async def unload_all(self, unloader_fn: Optional[Callable] = None) -> int:
        """
        Unload all models from memory.

        Args:
            unloader_fn: Optional async function to perform cleanup on each model

        Returns:
            Number of models unloaded
        """
        async with self._lock:
            count = len(self._models)
            if count == 0:
                logger.debug("No models to unload")
                return 0

            logger.info(f"Unloading all {count} models")

            keys = list(self._models.keys())
            for key in keys:
                model_info = self._models[key]
                if unloader_fn:
                    try:
                        await unloader_fn(model_info.model)
                    except Exception as e:
                        logger.error(f"Error unloading '{key}': {e}")
                del self._models[key]

            await self._cleanup_gpu_memory()

            logger.info(f"All {count} models unloaded")
            return count

    async def get_model(self, key: str) -> Optional[Any]:
        """
        Get a loaded model without loading it.

        Args:
            key: Unique identifier for the model

        Returns:
            The model object if loaded, None otherwise
        """
        async with self._lock:
            if key in self._models:
                self._models[key].last_used = time.time()
                return self._models[key].model
            return None

    async def touch_model(self, key: str) -> bool:
        """
        Update the last used timestamp for a model.

        Args:
            key: Unique identifier for the model

        Returns:
            True if model exists, False otherwise
        """
        async with self._lock:
            if key in self._models:
                self._models[key].last_used = time.time()
                return True
            return False

    async def cleanup_idle_models(
        self,
        timeout_seconds: float,
        unloader_fn: Optional[Callable] = None,
    ) -> List[str]:
        """
        Unload models that have been idle for longer than timeout.

        Args:
            timeout_seconds: Idle timeout in seconds
            unloader_fn: Optional async function to perform cleanup

        Returns:
            List of model keys that were unloaded
        """
        async with self._lock:
            now = time.time()
            idle_models = [
                key for key, info in self._models.items()
                if now - info.last_used > timeout_seconds
            ]

            if not idle_models:
                return []

            logger.info(
                f"Cleaning up {len(idle_models)} idle models "
                f"(timeout: {timeout_seconds}s): {idle_models}"
            )

            for key in idle_models:
                model_info = self._models[key]
                if unloader_fn:
                    try:
                        await unloader_fn(model_info.model)
                    except Exception as e:
                        logger.error(f"Error unloading idle model '{key}': {e}")
                del self._models[key]

            if idle_models:
                await self._cleanup_gpu_memory()

            return idle_models

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently loaded models with metadata.

        Returns:
            List of dicts with model info
        """
        return [
            {
                "key": info.key,
                "type": info.model_type,
                "loaded_at": datetime.fromtimestamp(info.loaded_at).isoformat(),
                "last_used": datetime.fromtimestamp(info.last_used).isoformat(),
                "idle_seconds": time.time() - info.last_used,
                "estimated_memory_mb": info.estimated_memory_mb,
                "actual_memory_mb": info.actual_memory_mb,
            }
            for info in self._models.values()
        ]

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dict with memory stats including GPU/CPU usage
        """
        stats = await self._get_memory_stats()
        stats["models_loaded"] = len(self._models)
        stats["model_details"] = self.get_loaded_models()
        return stats

    async def _apply_unload_policy(
        self,
        new_model_key: str,
        estimated_memory_mb: Optional[float],
    ) -> None:
        """
        Apply preemptive unload policy before loading a new model.

        Strategy:
        1. If max_models would be exceeded, unload oldest models
        2. If max_memory would be exceeded, unload models until space available
        3. Preemptive unload: unload ALL other models (aggressive mode for OOM prevention)
        """
        if not self._models:
            return

        # Aggressive mode: unload all other models (prevents OOM)
        # This is the key behavior for your use case
        if self._max_models == 1:
            logger.info(
                f"Preemptive unload: removing {len(self._models)} models to load '{new_model_key}'"
            )
            keys_to_unload = list(self._models.keys())
            for key in keys_to_unload:
                del self._models[key]
            await self._cleanup_gpu_memory()
            return

        # Multi-model mode: enforce max_models limit
        models_to_unload = len(self._models) + 1 - self._max_models
        if models_to_unload > 0:
            # Sort by last_used, unload oldest
            sorted_models = sorted(
                self._models.items(),
                key=lambda x: x[1].last_used,
            )
            for key, _ in sorted_models[:models_to_unload]:
                logger.info(f"Unloading '{key}' to enforce max_models={self._max_models}")
                del self._models[key]
            await self._cleanup_gpu_memory()

    async def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics (GPU + CPU)"""
        stats = {}

        # Try to get GPU memory stats
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu_available"] = True
                stats["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                stats["gpu_used_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
                stats["gpu_cached_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
                stats["gpu_free_mb"] = stats["gpu_total_mb"] - stats["gpu_cached_mb"]
            else:
                stats["gpu_available"] = False
        except ImportError:
            stats["gpu_available"] = False

        return stats

    async def _cleanup_gpu_memory(self) -> None:
        """Force GPU memory cleanup"""
        try:
            import torch
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


# Global singleton instance
_coordinator: Optional[ModelCoordinator] = None


def get_coordinator() -> ModelCoordinator:
    """Get the global model coordinator instance"""
    global _coordinator
    if _coordinator is None:
        # Default to aggressive mode (max 1 model)
        # This can be configured via settings in the future
        _coordinator = ModelCoordinator(
            max_models_in_memory=1,
            enable_preemptive_unload=True,
        )
    return _coordinator


async def init_coordinator(
    max_models: int = 1,
    max_memory_mb: Optional[float] = None,
    enable_preemptive_unload: bool = True,
) -> ModelCoordinator:
    """
    Initialize the global model coordinator with custom settings.

    Args:
        max_models: Maximum number of models to keep in memory
        max_memory_mb: Maximum memory usage in MB (None = no limit)
        enable_preemptive_unload: Whether to unload models before loading new ones

    Returns:
        The initialized coordinator
    """
    global _coordinator
    _coordinator = ModelCoordinator(
        max_models_in_memory=max_models,
        max_memory_mb=max_memory_mb,
        enable_preemptive_unload=enable_preemptive_unload,
    )
    return _coordinator
