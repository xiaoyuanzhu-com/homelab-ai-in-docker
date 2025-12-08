"""Text to embedding API router for text vectorization."""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from ..models.text_to_embedding import EmbeddingRequest, EmbeddingResponse
from ...storage.history import history_storage
from ...db.catalog import get_model_dict
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-embedding"])




def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    Workers handle their own idle timeouts, so this is a no-op.
    """
    pass


def cleanup():
    """
    Release model resources immediately.

    Workers handle their own cleanup, so this is a no-op.
    """
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
        # Validate model exists
        model_info = get_model_dict(request.model)
        if model_info is None:
            raise ValueError(f"Model '{request.model}' not found in catalog")

        # Call worker via coordinator
        result = await coordinator_infer(
            task="embedding",
            model_id=request.model,
            payload={"texts": request.texts},
            request_id=request_id,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        embeddings_list = result.get("embeddings", [])
        dimensions = result.get("dimensions", 0)

        response = EmbeddingResponse(
            request_id=request_id,
            embeddings=embeddings_list,
            dimensions=dimensions,
            model=request.model,
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

        return response

    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"Model error for request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": error_msg,
                "request_id": request_id,
            },
        )
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
