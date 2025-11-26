"""Pydantic models for Doc to Markdown API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator

from .base import BaseResponse


class DocToMarkdownRequest(BaseModel):
    """Request model for document-to-markdown conversion.

    The `file` field should contain a base64-encoded payload. Data URLs are accepted
    (e.g., "data:application/pdf;base64,<...>").
    """

    file: str = Field(..., description="Base64-encoded file data or data URL")
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename to infer file extension (e.g., report.pdf)",
    )
    lib: Optional[str] = Field(
        default="microsoft/markitdown",
        description="Library/engine to use; defaults to microsoft/markitdown",
    )

    @model_validator(mode="after")
    def _validate_lib(self):
        # Currently only supports MarkItDown, but keep the shape for parity with other endpoints
        if self.lib and self.lib != "microsoft/markitdown":
            raise ValueError("Only 'microsoft/markitdown' is supported for this endpoint")
        return self


class DocToMarkdownResponse(BaseResponse):
    """Response model for Doc to Markdown results."""

    markdown: str = Field(..., description="Converted Markdown content")
    model: str = Field(..., description="Library/engine used for conversion")

