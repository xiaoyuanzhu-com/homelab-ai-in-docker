"""Embedding API router for text vectorization."""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer

from ..models.embedding import EmbeddingRequest, EmbeddingResponse
from ...storage.history import history_storage


router = APIRouter(prefix="/api", tags=["embedding"])

# Global model cache
_model_cache: Optional[SentenceTransformer] = None
_current_model_name: str = "all-MiniLM-L6-v2"


def get_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Get or load the embedding model.

    Args:
        model_name: Name of the model to load (optional)

    Returns:
        Loaded SentenceTransformer model
    """
    global _model_cache, _current_model_name

    target_model = model_name or _current_model_name

    # Load model if not cached or if different model requested
    if _model_cache is None or target_model != _current_model_name:
        _model_cache = SentenceTransformer(target_model)
        _current_model_name = target_model

    return _model_cache


@router.post("/embed", response_model=EmbeddingResponse)
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
            service="embed",
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
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_FAILED",
                "message": f"Failed to generate embeddings: {str(e)}",
                "request_id": request_id,
            },
        )
