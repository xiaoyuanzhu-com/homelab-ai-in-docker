"""Models for embedding model management API."""

from typing import Optional
from pydantic import BaseModel


class EmbeddingModelInfo(BaseModel):
    """Information about an embedding model."""

    id: str
    name: str
    team: str
    license: str
    dimensions: int
    languages: str
    description: str
    size_mb: int
    link: str
    is_downloaded: bool
    downloaded_size_mb: Optional[int] = None


class ModelListResponse(BaseModel):
    """Response for listing available models."""

    models: list[EmbeddingModelInfo]


class ModelDownloadRequest(BaseModel):
    """Request to download a model."""

    model_id: str


class ModelDownloadResponse(BaseModel):
    """Response for model download status."""

    model_id: str
    status: str
    message: str
