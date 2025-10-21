"""Pydantic models for text generation API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseResponse


class TextGenerationRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="Input prompt for text generation", min_length=1)
    model: str = Field(..., description="Model ID to use for generation")
    max_new_tokens: Optional[int] = Field(
        default=256, description="Maximum number of tokens to generate", ge=1, le=4096
    )
    temperature: Optional[float] = Field(
        default=0.7, description="Sampling temperature (0.0 = greedy, higher = more random)", ge=0.0, le=2.0
    )
    top_p: Optional[float] = Field(
        default=0.9, description="Nucleus sampling threshold", ge=0.0, le=1.0
    )
    do_sample: Optional[bool] = Field(
        default=True, description="Whether to use sampling (vs greedy decoding)"
    )


class TextGenerationResponse(BaseResponse):
    """Response model for text generation results."""

    generated_text: str = Field(..., description="Generated text output")
    model: str = Field(..., description="Model that generated the text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
