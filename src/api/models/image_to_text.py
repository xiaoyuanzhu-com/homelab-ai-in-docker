"""Pydantic models for image caption API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseResponse


class CaptionRequest(BaseModel):
    """Request model for image captioning."""

    image: str = Field(..., description="Base64-encoded image or image URL")
    model: str = Field(..., description="Model to use for captioning")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt/question for the model. If not provided, uses model's default prompt.",
    )


class CaptionResponse(BaseResponse):
    """Response model for caption results."""

    caption: str = Field(..., description="Generated caption")
    model: str = Field(..., description="Model that generated the caption")


class ModelInfo(BaseModel):
    """Information about a single model."""

    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name")
    team: str = Field(..., description="Model creator/team")
    task: str = Field(..., description="Task type")
    size_mb: int = Field(..., description="Model size in MB")
    parameters_m: int = Field(..., description="Number of parameters in millions")
    gpu_memory_mb: int = Field(..., description="Estimated GPU memory requirement")
    link: str = Field(..., description="HuggingFace model page URL")


class ModelsListResponse(BaseModel):
    """Response model for listing available models."""

    models: list[ModelInfo] = Field(..., description="Available models")
    default_model: str = Field(..., description="Default model ID")
