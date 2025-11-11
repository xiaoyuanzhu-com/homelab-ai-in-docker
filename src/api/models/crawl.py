"""Pydantic models for crawl API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field, HttpUrl

from .base import BaseResponse


class CrawlRequest(BaseModel):
    """Request model for crawling a URL."""

    url: HttpUrl = Field(..., description="URL to crawl")
    screenshot: bool = Field(default=True, description="Capture viewport screenshot (first screen)")
    screenshot_width: int = Field(
        default=1920,
        description="Screenshot viewport width in pixels",
        ge=320,
        le=7680
    )
    screenshot_height: int = Field(
        default=1080,
        description="Screenshot viewport height in pixels",
        ge=240,
        le=4320
    )
    page_timeout: int = Field(
        default=120000,
        description="Page navigation timeout in milliseconds (default: 120000ms / 2 minutes)",
        ge=10000,
        le=300000
    )
    chrome_cdp_url: Optional[str] = Field(
        default=None,
        description="Remote Chrome CDP URL (e.g., http://localhost:9222). If not provided, uses local browser.",
        examples=["http://chrome:9222", "ws://127.0.0.1:9222/devtools/browser/..."],
    )
    screenshot_fullpage: bool = Field(
        default=False,
        description="Also capture a full-page screenshot (stitched). Shares width/height with viewport",
    )
    include_html: bool = Field(
        default=False,
        description="Include raw HTML content in the response (off by default to reduce response size)",
    )


class CrawlResponse(BaseResponse):
    """Response model for crawl results."""

    url: str = Field(..., description="Crawled URL")
    title: Optional[str] = Field(None, description="Page title")
    markdown: str = Field(..., description="Extracted content in Markdown format")
    html: Optional[str] = Field(None, description="Raw HTML content (only included if include_html=true)")
    screenshot_base64: Optional[str] = Field(
        None, description="Base64-encoded screenshot (viewport screenshot if requested, otherwise full-page)"
    )
    screenshot_fullpage_base64: Optional[str] = Field(
        None, description="Base64-encoded full-page screenshot, if requested"
    )
    success: bool = Field(..., description="Whether the crawl was successful")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: dict = Field(
        ...,
        description="Error details",
        examples=[
            {
                "code": "CRAWL_FAILED",
                "message": "Failed to crawl the URL",
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "details": {"url": "https://example.com", "reason": "Timeout"},
            }
        ],
    )
