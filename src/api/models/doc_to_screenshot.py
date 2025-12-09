"""Pydantic models for Doc to Screenshot API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator

from .base import BaseResponse


class DocToScreenshotRequest(BaseModel):
    """Request model for document-to-screenshot conversion.

    The `file` field should contain a base64-encoded payload. Data URLs are accepted
    (e.g., "data:application/pdf;base64,<...>").
    """

    file: str = Field(..., description="Base64-encoded file data or data URL")
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename to infer file extension (e.g., report.pdf)",
    )
    lib: Optional[str] = Field(
        default="screenitshot/screenitshot",
        description="Library/engine to use; defaults to screenitshot/screenitshot",
    )

    @model_validator(mode="after")
    def _validate_lib(self):
        # Currently only supports screenitshot, but keep the shape for parity with other endpoints
        if self.lib and self.lib != "screenitshot/screenitshot":
            raise ValueError("Only 'screenitshot/screenitshot' is supported for this endpoint")
        return self


class DocToScreenshotResponse(BaseResponse):
    """Response model for Doc to Screenshot results."""

    screenshot: str = Field(..., description="Base64-encoded PNG screenshot")
    model: str = Field(..., description="Library/engine used for conversion")
