"""Pydantic models for crawl API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field, HttpUrl

from .base import BaseResponse


class CrawlRequest(BaseModel):
    """Request model for crawling a URL."""

    url: HttpUrl = Field(..., description="URL to crawl")
    screenshot: bool = Field(default=False, description="Capture screenshot")
    wait_for_js: bool = Field(
        default=True, description="Wait for JavaScript to execute"
    )
    chrome_cdp_url: Optional[str] = Field(
        default=None,
        description="Remote Chrome CDP URL (e.g., http://localhost:9222). If not provided, uses local browser.",
        examples=["http://chrome:9222", "ws://127.0.0.1:9222/devtools/browser/..."],
    )


class CrawlResponse(BaseResponse):
    """Response model for crawl results."""

    url: str = Field(..., description="Crawled URL")
    title: Optional[str] = Field(None, description="Page title")
    markdown: str = Field(..., description="Extracted content in Markdown format")
    html: Optional[str] = Field(None, description="Raw HTML content")
    screenshot_base64: Optional[str] = Field(
        None, description="Base64-encoded screenshot"
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
