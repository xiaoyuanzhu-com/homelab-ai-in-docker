"""Pydantic models for speaker embedding API."""

from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseResponse


class EmbeddingRequest(BaseModel):
    """Request model for speaker embedding extraction."""

    audio: str = Field(..., description="Base64-encoded audio file")
    model: str = Field(default="pyannote/embedding", description="Model to use for embedding extraction")
    mode: str = Field(
        default="whole",
        description="Extraction mode: 'whole' for entire file, 'segment' for a specific time range"
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start time in seconds (required for 'segment' mode)"
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End time in seconds (required for 'segment' mode)"
    )


class CompareEmbeddingsRequest(BaseModel):
    """Request model for comparing two speaker embeddings."""

    audio1: str = Field(..., description="Base64-encoded first audio file")
    audio2: str = Field(..., description="Base64-encoded second audio file")
    model: str = Field(default="pyannote/embedding", description="Model to use for embedding extraction")
    metric: str = Field(default="cosine", description="Distance metric: 'cosine', 'euclidean', or 'cityblock'")


class EmbeddingResponse(BaseResponse):
    """Response model for speaker embedding extraction."""

    embedding: list[float] = Field(..., description="Speaker embedding vector")
    dimension: int = Field(..., description="Dimensionality of the embedding vector")
    model: str = Field(..., description="Model used for extraction")
    duration: Optional[float] = Field(None, description="Duration of audio segment in seconds")


class CompareEmbeddingsResponse(BaseResponse):
    """Response model for speaker embedding comparison."""

    distance: float = Field(..., description="Distance between the two embeddings")
    similarity: float = Field(..., description="Similarity score (1 - distance for cosine)")
    metric: str = Field(..., description="Distance metric used")
    model: str = Field(..., description="Model used for extraction")
