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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, AsyncIterator
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
    active_refs: int = 0  # Number of active inference operations using this model


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

    async def prepare_model(
        self,
        key: str,
        loader_fn: Callable,
        unloader_fn: Optional[Callable] = None,
        estimated_memory_mb: Optional[float] = None,
        model_type: Optional[str] = None,
    ) -> Any:
        """
        Prepare a model for use, waiting if necessary.

        This is the main entry point for all model loading. It:
        1. Checks if model is already loaded (increments ref count)
        2. If not loaded, waits for other models to release
        3. Loads the model via the provided loader function
        4. Returns with active_refs=1 (caller MUST call release_model)

        The returned model is protected from unloading until release_model() is called.

        Args:
            key: Unique identifier for the model (e.g., "whisper-base", "all-MiniLM-L6-v2")
            loader_fn: Async function that loads and returns the model
            unloader_fn: Optional async function to unload this specific model (stored for later)
            estimated_memory_mb: Estimated memory usage (for planning)
            model_type: Type of model (e.g., "embedding", "asr", "ocr")

        Returns:
            The loaded model object with active_refs=1

        IMPORTANT: Must be paired with release_model() call, preferably in a try/finally block
        """
        async with self._lock:
            # Check if model is already loaded
            if key in self._models:
                logger.info(f"Model '{key}' already loaded, reusing cached instance")
                self._models[key].active_refs += 1
                self._models[key].last_used = time.time()
                logger.debug(f"Acquired ref for '{key}', active_refs={self._models[key].active_refs}")
                return self._models[key].model

            # Get current memory stats before loading
            memory_before = await self._get_memory_stats()
            logger.info(
                f"Preparing model '{key}' (type: {model_type}, estimated: {estimated_memory_mb}MB). "
                f"Current memory: GPU={memory_before.get('gpu_used_mb', 'N/A')}MB, "
                f"Models loaded: {len(self._models)}"
            )

            # Wait for other models and unload them
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

            # Register the model with active_refs=1 (protected from the start)
            now = time.time()
            model_info = ModelInfo(
                key=key,
                model=model,
                loaded_at=now,
                last_used=now,
                estimated_memory_mb=estimated_memory_mb,
                actual_memory_mb=actual_memory_mb,
                model_type=model_type,
                active_refs=1,  # ← Starts with 1 ref!
            )
            self._models[key] = model_info

            logger.info(
                f"Model '{key}' prepared successfully. "
                f"Actual memory: {actual_memory_mb}MB, "
                f"GPU: {memory_after.get('gpu_used_mb', 'N/A')}MB, "
                f"Models in memory: {len(self._models)}, "
                f"active_refs=1"
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


    async def release_model(self, key: str) -> bool:
        """
        Release a model reference after use.

        Decrements active_refs by 1. When active_refs reaches 0, the model
        becomes eligible for idle cleanup after the timeout period.

        Must be called after every prepare_model() call, preferably in a finally block.

        Args:
            key: Unique identifier for the model

        Returns:
            True if model exists and ref was released, False otherwise
        """
        async with self._lock:
            if key in self._models:
                self._models[key].active_refs = max(0, self._models[key].active_refs - 1)
                self._models[key].last_used = time.time()
                logger.debug(f"Released '{key}', active_refs={self._models[key].active_refs}")
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
            # Only cleanup models with no active references
            idle_models = [
                key for key, info in self._models.items()
                if now - info.last_used > timeout_seconds and info.active_refs == 0
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
                "active_refs": info.active_refs,
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

        IMPORTANT: Never unload models with active_refs > 0 (active inference in progress)
        If models are blocked by active refs, this method will WAIT for them to be released.
        """
        if not self._models:
            return

        # Aggressive mode: unload all other models (prevents OOM)
        if self._max_models == 1:
            max_wait_time = 300  # 5 minutes max wait
            poll_interval = 2.0  # Check every 2 seconds
            waited_time = 0.0

            while self._models and waited_time < max_wait_time:
                # Filter models that can be unloaded (no active references)
                unloadable = {k: v for k, v in self._models.items() if v.active_refs == 0}
                active_models = {k: v for k, v in self._models.items() if v.active_refs > 0}

                if active_models:
                    logger.info(
                        f"Waiting for {len(active_models)} models with active inference to complete: "
                        f"{list(active_models.keys())} (waited {waited_time:.1f}s)"
                    )
                    # Release lock temporarily to allow other operations
                    # (like release_model_ref) to proceed
                    self._lock.release()
                    try:
                        await asyncio.sleep(poll_interval)
                    finally:
                        await self._lock.acquire()
                    waited_time += poll_interval
                    continue

                # All models are now unloadable
                if unloadable:
                    logger.info(
                        f"Preemptive unload: removing {len(unloadable)} idle models to load '{new_model_key}'"
                    )
                    keys_to_unload = list(unloadable.keys())
                    for key in keys_to_unload:
                        del self._models[key]
                    await self._cleanup_gpu_memory()
                break

            if waited_time >= max_wait_time and self._models:
                active_models = {k: v for k, v in self._models.items() if v.active_refs > 0}
                logger.warning(
                    f"Timeout waiting for models to release after {max_wait_time}s. "
                    f"Active models: {list(active_models.keys())}. "
                    f"Proceeding with load anyway (may cause OOM)."
                )
            return

        # Multi-model mode: enforce max_models limit
        models_to_unload = len(self._models) + 1 - self._max_models
        if models_to_unload > 0:
            unloadable = {k: v for k, v in self._models.items() if v.active_refs == 0}
            if unloadable:
                # Sort by last_used, unload oldest (but only unloadable ones)
                sorted_models = sorted(
                    unloadable.items(),
                    key=lambda x: x[1].last_used,
                )
                unload_count = min(models_to_unload, len(sorted_models))
                for key, _ in sorted_models[:unload_count]:
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


@asynccontextmanager
async def use_model(
    key: str,
    loader_fn: Callable,
    model_type: str,
    estimated_memory_mb: Optional[float] = None,
    unloader_fn: Optional[Callable] = None,
) -> AsyncIterator[Any]:
    """
    Context manager for safe model usage with automatic cleanup.

    This is the recommended way to use models. It guarantees that release_model()
    will be called even if an exception occurs during model usage.

    Example:
        ```python
        async with use_model(
            key="whisperx:large-v3",
            loader_fn=lambda: _load_asr_model_impl(...),
            model_type="whisperx",
        ) as model:
            result = model.transcribe(audio)
            aligned = whisperx.align(result, ...)
            # Model is automatically released when block exits
        ```

    Args:
        key: Unique identifier for the model
        loader_fn: Async function that loads and returns the model
        model_type: Type of model (e.g., "embedding", "asr", "ocr")
        estimated_memory_mb: Estimated memory usage (for planning)
        unloader_fn: Optional async function to unload this specific model

    Yields:
        The loaded model object (protected by active_refs)
    """
    coordinator = get_coordinator()
    model = await coordinator.prepare_model(
        key=key,
        loader_fn=loader_fn,
        unloader_fn=unloader_fn,
        estimated_memory_mb=estimated_memory_mb,
        model_type=model_type,
    )

    try:
        yield model
    finally:
        await coordinator.release_model(key)
