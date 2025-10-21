"""Pydantic models for image OCR API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseResponse


class OCRRequest(BaseModel):
    """Request model for image OCR."""

    image: str = Field(..., description="Base64-encoded image or image URL")
    model: str = Field(..., description="Model to use for OCR")
    language: Optional[str] = Field(
        default=None,
        description="Optional language hint for OCR (e.g., 'en', 'zh', 'auto')",
    )


class OCRResponse(BaseResponse):
    """Response model for OCR results."""

    text: str = Field(..., description="Extracted text from the image")
    model: str = Field(..., description="Model that performed the OCR")
