"""Text to embedding API router for text vectorization."""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
import torch

from ..models.text_to_embedding import EmbeddingRequest, EmbeddingResponse
from ...storage.history import history_storage
from ...config import get_data_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-embedding"])

# Global model cache
_model_cache: Optional[SentenceTransformer] = None
_current_model_name: str = "all-MiniLM-L6-v2"
_last_access_time: Optional[float] = None


def check_and_cleanup_idle_model():
    """Check if model has been idle too long and cleanup if needed."""
    global _model_cache, _last_access_time

    if _model_cache is None or _last_access_time is None:
        return

    # Get idle timeout from settings
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    # Check if model has been idle too long
    idle_duration = time.time() - _last_access_time
    if idle_duration >= idle_timeout:
        logger.info(f"Embedding model idle for {idle_duration:.1f}s (timeout: {idle_timeout}s), cleaning up...")
        cleanup()


def get_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Get or load the embedding model.

    Args:
        model_name: Name of the model to load (optional)

    Returns:
        Loaded SentenceTransformer model

    Raises:
        HTTPException: If model not found or not downloaded
    """
    global _model_cache, _current_model_name, _last_access_time

    # Check if current model should be cleaned up due to idle timeout
    check_and_cleanup_idle_model()

    target_model = model_name or _current_model_name

    # Load model if not cached or if different model requested
    if _model_cache is None or target_model != _current_model_name:
        # Use the downloaded models directory (preserves HuggingFace structure)
        # Path: data/models/sentence-transformers/all-MiniLM-L6-v2
        model_dir = get_data_dir() / "models" / target_model

        # Check if model exists
        if not model_dir.exists():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "MODEL_NOT_FOUND",
                    "message": f"Model '{target_model}' not found. Please download it first from the Models tab.",
                },
            )

        # Load model from local directory
        _model_cache = SentenceTransformer(str(model_dir))
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

    if _model_cache is not None:
        # Move model to CPU first (helps with cleanup)
        try:
            if hasattr(_model_cache, 'cpu'):
                _model_cache.cpu()
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
            logger.info("GPU memory released for embedding model")
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
        # Load model
        model = get_model(request.model)

        # Generate embeddings
        embeddings = model.encode(request.texts, convert_to_numpy=True)

        # Convert to list of lists
        embeddings_list = embeddings.tolist()

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = EmbeddingResponse(
            request_id=request_id,
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            model_used=_current_model_name,
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
                "model_used": response.model_used,
                "processing_time_ms": response.processing_time_ms,
                "num_embeddings": len(response.embeddings),
            },
            status="success",
        )

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
