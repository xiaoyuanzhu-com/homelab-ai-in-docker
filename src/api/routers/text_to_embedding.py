"""Text to embedding API router for text vectorization."""

import logging
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
import asyncio
from sentence_transformers import SentenceTransformer
import torch

from ..models.text_to_embedding import EmbeddingRequest, EmbeddingResponse
from ...storage.history import history_storage
from ...config import get_data_dir, get_hf_endpoint, get_hf_model_cache_path
from ...db.catalog import get_model_dict
from ...services.model_coordinator import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-embedding"])

# Track current model name (coordinator manages the actual model cache)
_current_model_name: str = "all-MiniLM-L6-v2"


def get_total_gpu_memory_mb() -> float:
    """
    Get total GPU memory in MB.

    Returns:
        Total GPU memory in MB, or 0 if no GPU available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        # Get current device
        device = torch.cuda.current_device()
        # Get total memory in bytes
        total_memory_bytes = torch.cuda.mem_get_info(device)[1]
        # Convert to MB
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        return total_memory_mb
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return 0.0


def calculate_batch_size(total_memory_mb: float, memory_per_batch_mb: float = 2048, max_batch_size: int = 32) -> int:
    """
    Calculate optimal batch size based on total GPU memory.

    Uses the formula: batch_size = floor(total_memory / memory_per_batch)

    Args:
        total_memory_mb: Total GPU memory in MB
        memory_per_batch_mb: Memory consumption per batch (default: 2048 MB = 2 GB)
        max_batch_size: Maximum batch size cap (default: 32)

    Returns:
        Calculated batch size (minimum 1, maximum max_batch_size)
    """
    if total_memory_mb <= 0:
        # No GPU or no memory info, use conservative default
        logger.warning("No GPU memory available, using batch_size=1")
        return 1

    # Calculate batch size: total_memory / memory_per_batch
    calculated_batch = int(total_memory_mb / memory_per_batch_mb)

    # Clamp between 1 and max_batch_size
    batch_size = max(1, min(calculated_batch, max_batch_size))

    logger.info(
        f"Calculated batch_size={batch_size} "
        f"(total_memory={total_memory_mb:.0f}MB, "
        f"memory_per_batch={memory_per_batch_mb}MB)"
    )

    return batch_size


# Calculate batch size once at module initialization
from ...db.settings import get_setting_int
_memory_per_batch_mb = get_setting_int("embedding_memory_per_batch_mb", 2048)
_max_batch_size = get_setting_int("embedding_max_batch_size", 32)
_total_gpu_memory = get_total_gpu_memory_mb()
_default_batch_size = calculate_batch_size(
    total_memory_mb=_total_gpu_memory,
    memory_per_batch_mb=_memory_per_batch_mb,
    max_batch_size=_max_batch_size
)


def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    This function is called by periodic cleanup in main.py.
    Delegates to the global model coordinator for memory management.
    """
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    coordinator = get_coordinator()
    # Use asyncio.create_task to run cleanup asynchronously
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coordinator.cleanup_idle_models(idle_timeout, unloader_fn=_unload_model))
    except RuntimeError:
        # No event loop, skip cleanup
        pass


async def _load_model_impl(model_name: str) -> SentenceTransformer:
    """
    Internal async function to load the embedding model.

    Args:
        model_name: Model ID to load

    Returns:
        Loaded SentenceTransformer model
    """
    # Get model info from catalog
    model_info = get_model_dict(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in catalog")

    # Set HuggingFace endpoint for model loading
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()

    # Check for local download at HF standard cache path
    local_model_dir = get_hf_model_cache_path(model_name)
    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        # Load from local directory
        model_path = str(local_model_dir)
        logger.info(f"Using locally downloaded model from {model_path}")
    else:
        # Fall back to model ID (will download from HuggingFace to cache)
        model_path = model_name
        logger.info(f"Model not found locally, will download from HuggingFace: {model_name}")

    # Load model in thread pool to avoid blocking
    model = await asyncio.to_thread(SentenceTransformer, model_path)
    return model


async def _unload_model(model: SentenceTransformer) -> None:
    """
    Internal async function to unload and cleanup model.

    Args:
        model: Model to unload
    """
    # Move model to CPU first (helps with cleanup)
    try:
        if hasattr(model, 'cpu'):
            await asyncio.to_thread(model.cpu)
            logger.debug("Moved embedding model to CPU")
    except Exception as e:
        logger.warning(f"Error moving embedding model to CPU during cleanup: {e}")

    # Delete model reference
    del model


async def get_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Get or load the embedding model via the global coordinator.

    Models are managed by the coordinator to prevent OOM errors.
    The coordinator will preemptively unload other models if needed.

    Args:
        model_name: Skill ID of the model to load (optional)

    Returns:
        Loaded SentenceTransformer model

    Raises:
        HTTPException: If skill not found or not downloaded
    """
    global _current_model_name

    target_model = model_name or _current_model_name
    _current_model_name = target_model

    # Get model info for memory estimation
    model_info = get_model_dict(target_model)
    estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

    # Load through coordinator (handles preemptive unload)
    coordinator = get_coordinator()
    model = await coordinator.load_model(
        key=f"embedding:{target_model}",
        loader_fn=lambda: _load_model_impl(target_model),
        unloader_fn=_unload_model,
        estimated_memory_mb=estimated_memory_mb,
        model_type="embedding",
    )

    return model


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Delegates to the global model coordinator.
    """
    coordinator = get_coordinator()
    # Use asyncio to run cleanup synchronously
    try:
        loop = asyncio.get_running_loop()
        # Create task to unload all embedding models
        loop.create_task(coordinator.unload_model(f"embedding:{_current_model_name}", _unload_model))
    except RuntimeError:
        # No event loop, can't cleanup
        pass


@router.post("/text-to-embedding", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for text inputs.

    Converts text into dense vector representations for semantic search,
    similarity matching, and other NLP tasks.

    Args:
        request: Embedding request parameters

    Returns:
        Embedding vectors and metadata

    Raises:
        HTTPException: If embedding fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Load model via coordinator (handles preemptive unload)
        model = await get_model(request.model)

        # Generate embeddings off the event loop with pre-calculated batch size
        embeddings = await asyncio.to_thread(
            model.encode,
            request.texts,
            batch_size=_default_batch_size,
            convert_to_numpy=True
        )

        # Convert to list of lists
        embeddings_list = embeddings.tolist()

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = EmbeddingResponse(
            request_id=request_id,
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            model=_current_model_name,
            processing_time_ms=processing_time_ms,
        )

        # Save to history (exclude embeddings from storage to save space)
        history_storage.add_request(
            service="text-to-embedding",
            request_id=request_id,
            request_data=request.model_dump(),
            response_data={
                "request_id": response.request_id,
                "dimensions": response.dimensions,
                "model": response.model,
                "processing_time_ms": response.processing_time_ms,
                "num_embeddings": len(response.embeddings),
            },
            status="success",
        )

        # Note: Idle cleanup is handled by the global model coordinator
        return response

    except Exception as e:
        error_msg = f"Failed to generate embeddings: {str(e)}"
        logger.error(f"Embedding failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_FAILED",
                "message": error_msg,
                "request_id": request_id,
            },
        )
