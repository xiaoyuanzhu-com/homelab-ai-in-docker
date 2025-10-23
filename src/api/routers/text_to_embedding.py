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
from ...config import get_data_dir, get_hf_endpoint
from ...db.skills import get_skill_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-embedding"])

# Global model cache
_model_cache: Optional[SentenceTransformer] = None
_current_model_name: str = "all-MiniLM-L6-v2"
_last_access_time: Optional[float] = None
_idle_cleanup_task: Optional[asyncio.Task] = None


def check_and_cleanup_idle_model():
    """Check if model has been idle too long and cleanup if needed."""
    global _model_cache, _last_access_time, _current_model_name

    if _model_cache is None or _last_access_time is None:
        return

    # Get idle timeout from settings
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    # Check if model has been idle too long
    idle_duration = time.time() - _last_access_time
    if idle_duration >= idle_timeout:
        logger.info(
            f"Embedding model '{_current_model_name}' idle for {idle_duration:.1f}s "
            f"(timeout: {idle_timeout}s), unloading from GPU..."
        )
        cleanup()
        logger.info(f"Embedding model '{_current_model_name}' unloaded from GPU")


def schedule_idle_cleanup() -> None:
    """Schedule background cleanup after idle timeout."""
    global _idle_cleanup_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    if _idle_cleanup_task and not _idle_cleanup_task.done():
        _idle_cleanup_task.cancel()

    async def _watchdog(timeout_s: int):
        try:
            await asyncio.sleep(timeout_s)
            if _last_access_time is None:
                return
            idle_duration = time.time() - _last_access_time
            if idle_duration >= timeout_s and _model_cache is not None:
                logger.info(
                    f"Embedding model '{_current_model_name}' idle for {idle_duration:.1f}s "
                    f"(timeout: {timeout_s}s), unloading from GPU..."
                )
                cleanup()
        except asyncio.CancelledError:
            pass
        finally:
            # Clear task reference if it's the current task
            global _idle_cleanup_task
            try:
                current = asyncio.current_task()
            except Exception:
                current = None
            if current is not None and _idle_cleanup_task is current:
                _idle_cleanup_task = None

    _idle_cleanup_task = loop.create_task(_watchdog(idle_timeout))


def get_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Get or load the embedding model.

    Models downloaded via skills API are stored at: data/models/{org}/{model}
    Falls back to downloading from HuggingFace if not found locally.

    Args:
        model_name: Skill ID of the model to load (optional)

    Returns:
        Loaded SentenceTransformer model

    Raises:
        HTTPException: If skill not found or not downloaded
    """
    global _model_cache, _current_model_name, _last_access_time

    # Check if current model should be cleaned up due to idle timeout
    check_and_cleanup_idle_model()

    target_model = model_name or _current_model_name

    # Load model if not cached or if different model requested
    if _model_cache is None or target_model != _current_model_name:
        # Get skill info from database
        skill = get_skill_dict(target_model)
        if skill is None:
            raise ValueError(f"Skill '{target_model}' not found in database")

        # Set HuggingFace endpoint for model loading
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Check for local download at data/models/{org}/{model}
        local_model_dir = get_data_dir() / "models" / target_model
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            # Load from local directory
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
        else:
            # Fall back to model ID (will download from HuggingFace)
            model_path = target_model
            logger.info(f"Model not found locally, will download from HuggingFace: {target_model}")

        _model_cache = SentenceTransformer(model_path)
        _current_model_name = target_model

    # Update last access time
    _last_access_time = time.time()

    return _model_cache


def cleanup():
    """
    Release model resources immediately.
    Forces GPU memory cleanup to free resources for other services.
    """
    global _model_cache, _current_model_name, _last_access_time

    model_name = _current_model_name  # Save for logging

    if _model_cache is not None:
        # Move model to CPU first (helps with cleanup)
        try:
            if hasattr(_model_cache, 'cpu'):
                _model_cache.cpu()
                logger.debug(f"Moved embedding model '{model_name}' to CPU")
        except Exception as e:
            logger.warning(f"Error moving embedding model to CPU during cleanup: {e}")

        # Remove reference
        del _model_cache
        _model_cache = None

    _current_model_name = "all-MiniLM-L6-v2"
    _last_access_time = None

    # Force GPU memory release
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to finish
            logger.debug("GPU cache cleared and synchronized for embedding model")
    except Exception as e:
        logger.warning(f"Error releasing GPU memory: {e}")


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
        # Load model off the event loop
        model = await asyncio.to_thread(get_model, request.model)

        # Generate embeddings off the event loop
        embeddings = await asyncio.to_thread(model.encode, request.texts, convert_to_numpy=True)

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

        # Schedule background idle cleanup after request completes
        schedule_idle_cleanup()
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
