"""Pydantic models for image OCR API endpoints."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator

from .base import BaseResponse


class OCRRequest(BaseModel):
    """Request model for image OCR."""

    image: str = Field(..., description="Base64-encoded image or image URL")
    model: Optional[str] = Field(
        default=None,
        description="Model to use for OCR (mutually exclusive with 'lib')",
    )
    lib: Optional[str] = Field(
        default=None,
        description="Library/engine to use for OCR (mutually exclusive with 'model')",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language hint for OCR (e.g., 'en', 'zh', 'auto')",
    )
    output_format: Optional[Literal["text", "markdown"]] = Field(
        default="text",
        description="Output format: 'text' for plain text, 'markdown' for structured markdown (supported by PaddleOCR-VL, Granite Docling, MinerU, DeepSeek)",
    )

    @model_validator(mode="after")
    def _validate_choice(self):
        has_model = bool(self.model)
        has_lib = bool(self.lib)
        if has_model == has_lib:
            raise ValueError("Exactly one of 'model' or 'lib' must be provided")
        return self


class OCRResponse(BaseResponse):
    """Response model for OCR results."""

    text: str = Field(..., description="Extracted text from the image")
    model: str = Field(..., description="Model that performed the OCR")
    output_format: str = Field(..., description="Format of the output text")
