"""Pydantic models for image caption API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field


class CaptionRequest(BaseModel):
    """Request model for image captioning."""

    image: str = Field(..., description="Base64-encoded image or image URL")
    model: Optional[str] = Field(
        default=None, description="Model to use for captioning (optional)"
    )


class CaptionResponse(BaseModel):
    """Response model for caption results."""

    request_id: str = Field(..., description="Unique request identifier")
    caption: str = Field(..., description="Generated caption")
    model_used: str = Field(..., description="Model that generated the caption")
    processing_time_ms: int = Field(..., description="Time taken to process")
