"""Pydantic models for image segmentation (Segment Anything) API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .base import BaseResponse


class SegmentationRequest(BaseModel):
    """Request model for promptable image segmentation."""

    image: str = Field(..., description="Base64-encoded image or data URL")
    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt describing the target region; omit or set to 'auto' to segment everything",
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
    points_per_side: Optional[int] = Field(
        default=None,
        ge=1,
        description="Auto mode: number of point prompts per side of the image grid",
    )
    points_per_batch: Optional[int] = Field(
        default=None,
        ge=1,
        description="Auto mode: number of point prompts per batch",
    )
    auto_iou_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Auto mode: IoU threshold to de-duplicate masks",
    )
    auto_min_area_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Auto mode: minimum mask area ratio to keep a mask",
    )


class SegmentationRLE(BaseModel):
    """COCO RLE mask encoding."""

    size: List[int] = Field(..., description="Mask size as [height, width]")
    counts: List[int] = Field(..., description="Run-length encoding counts in column-major order")


class SegmentationMask(BaseModel):
    """One segmentation mask output."""

    rle: SegmentationRLE = Field(..., description="COCO RLE-encoded mask")
    score: Optional[float] = Field(None, description="Mask confidence score")
    box: Optional[List[float]] = Field(None, description="Bounding box [x0, y0, x1, y1]")


class SegmentationResponse(BaseResponse):
    """Response model for image segmentation results."""

    model: str = Field(..., description="Library/engine used")
    prompt: str = Field(..., description="Text prompt used for segmentation")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
    masks: List[SegmentationMask] = Field(..., description="Segmentation masks")
