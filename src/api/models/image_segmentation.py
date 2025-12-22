"""Pydantic models for image segmentation (Segment Anything) API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .base import BaseResponse


class SegmentationRequest(BaseModel):
    """Request model for promptable image segmentation."""

    image: str = Field(..., description="Base64-encoded image or data URL")
    prompt: str = Field(
        ...,
        min_length=1,
        description="Text prompt describing the target region",
    )
    lib: Optional[str] = Field(
        default="facebookresearch/sam3",
        description="Library/engine to use; defaults to facebookresearch/sam3",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to keep a mask (0-1)",
    )
    max_masks: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap on number of masks returned",
    )


class SegmentationMask(BaseModel):
    """One segmentation mask output."""

    mask: str = Field(..., description="Base64-encoded PNG mask")
    score: Optional[float] = Field(None, description="Mask confidence score")
    box: Optional[List[float]] = Field(None, description="Bounding box [x0, y0, x1, y1]")


class SegmentationResponse(BaseResponse):
    """Response model for image segmentation results."""

    model: str = Field(..., description="Library/engine used")
    prompt: str = Field(..., description="Text prompt used for segmentation")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
    masks: List[SegmentationMask] = Field(..., description="Segmentation masks")
