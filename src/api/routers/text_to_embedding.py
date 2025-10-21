"""Text to embedding API router for text vectorization."""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer

from ..models.text_to_embedding import EmbeddingRequest, EmbeddingResponse
from ...storage.history import history_storage
from ...config import get_data_dir


router = APIRouter(prefix="/api", tags=["text-to-embedding"])

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

    Raises:
        HTTPException: If model not found or not downloaded
    """
    global _model_cache, _current_model_name

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

    return _model_cache


def cleanup():
    """Release model resources on shutdown."""
    global _model_cache, _current_model_name
    if _model_cache is not None:
        del _model_cache
        _model_cache = None
    _current_model_name = "all-MiniLM-L6-v2"


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
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_FAILED",
                "message": f"Failed to generate embeddings: {str(e)}",
                "request_id": request_id,
            },
        )
