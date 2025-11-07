"""Pydantic models for speaker embedding API."""

from typing import Optional, List, Dict
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


class TimeSegment(BaseModel):
    """A simple time segment."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class BatchEmbeddingRequest(BaseModel):
    """Request for extracting embeddings for multiple segments from one audio."""

    audio: str = Field(..., description="Base64-encoded audio file")
    model: str = Field(default="pyannote/embedding", description="Model to use")
    segments: List[TimeSegment] = Field(..., description="List of segments to crop and embed")


class BatchEmbeddingResponse(BaseResponse):
    """Response for batch embedding extraction."""

    embeddings: List[List[float]] = Field(..., description="Embeddings for each input segment")
    dimension: int = Field(..., description="Dimensionality of the embeddings")
    count: int = Field(..., description="Number of embeddings returned")
    model: str = Field(..., description="Model used for extraction")


class SpeakerRegistryEntry(BaseModel):
    name: str = Field(..., description="Application-level speaker name/ID")
    embeddings: List[List[float]] = Field(..., description="One or more embeddings for this speaker")


class MatchRequest(BaseModel):
    """Request to match query embeddings to a provided registry (stateless)."""

    query_embeddings: List[List[float]] = Field(..., description="Embeddings to classify (e.g., diarized clusters)")
    registry: List[SpeakerRegistryEntry] = Field(..., description="Known registry of speakers with embeddings")
    metric: str = Field(default="cosine", description="Distance metric: 'cosine', 'euclidean', 'cityblock'")
    threshold: Optional[float] = Field(default=0.75, description="Minimum similarity (cosine) to accept a match; set None to always return best")
    top_k: int = Field(default=1, description="Return top-k candidates per query for inspection")
    strategy: str = Field(default="centroid", description="'centroid' to compare to per-speaker centroid; 'best' to compare to best sample")


class MatchCandidate(BaseModel):
    name: str = Field(..., description="Matched speaker name")
    similarity: float = Field(..., description="Similarity score (1 - distance for cosine)")


class MatchResult(BaseModel):
    best: Optional[MatchCandidate] = Field(None, description="Best match above threshold (if any)")
    candidates: List[MatchCandidate] = Field(default_factory=list, description="Top-k candidates for inspection")


class MatchResponse(BaseResponse):
    """Match results for a batch of queries."""

    results: List[MatchResult] = Field(..., description="One result per query embedding")
