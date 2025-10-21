"""Pydantic models for embedding API endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field

from .base import BaseResponse


class EmbeddingRequest(BaseModel):
    """Request model for text embedding."""

    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)
    model: Optional[str] = Field(
        default=None, description="Model to use for embedding (optional, uses default if not specified)"
    )


class EmbeddingResponse(BaseResponse):
    """Response model for embedding results."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimensions: int = Field(..., description="Dimensionality of embeddings")
    model: str = Field(..., description="Model that generated the embeddings")
