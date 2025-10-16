"""Pydantic models for crawl API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class CrawlRequest(BaseModel):
    """Request model for crawling a URL."""

    url: HttpUrl = Field(..., description="URL to crawl")
    screenshot: bool = Field(default=False, description="Capture screenshot")
    wait_for_js: bool = Field(
        default=True, description="Wait for JavaScript to execute"
    )


class CrawlResponse(BaseModel):
    """Response model for crawl results."""

    request_id: str = Field(..., description="Unique request identifier")
    url: str = Field(..., description="Crawled URL")
    title: Optional[str] = Field(None, description="Page title")
    markdown: str = Field(..., description="Extracted content in Markdown format")
    html: Optional[str] = Field(None, description="Raw HTML content")
    screenshot_base64: Optional[str] = Field(
        None, description="Base64-encoded screenshot"
    )
    fetch_time_ms: int = Field(..., description="Time taken to fetch and process")
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
