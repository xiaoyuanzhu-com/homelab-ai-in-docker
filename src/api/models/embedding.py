"""Pydantic models for embedding API endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for text embedding."""

    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)
    model: Optional[str] = Field(
        default=None, description="Model to use for embedding (optional)"
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding results."""

    request_id: str = Field(..., description="Unique request identifier")
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimensions: int = Field(..., description="Dimensionality of embeddings")
    model_used: str = Field(..., description="Model that generated the embeddings")
    processing_time_ms: int = Field(..., description="Time taken to process")
