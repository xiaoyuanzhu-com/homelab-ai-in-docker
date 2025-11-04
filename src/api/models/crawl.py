"""Pydantic models for crawl API endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

from .base import BaseResponse


class CrawlRequest(BaseModel):
    """Request model for crawling a URL."""

    url: HttpUrl = Field(..., description="URL to crawl")
    screenshot: bool = Field(default=False, description="Capture screenshot")
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
    wait_for_selector: Optional[str] = Field(
        default=None,
        description="Optional CSS selector to wait for after DOM content is loaded (helps with SPA rendering)",
        examples=["article[data-test-id='post-content']", "#main-content"],
    )
    wait_for_selector_timeout: Optional[int] = Field(
        default=10000,
        description="Timeout in milliseconds when waiting for selector (default: 10000ms). Ignored if wait_for_selector is not set.",
        ge=1000,
        le=60000,
    )
    content_selectors: Optional[List[str]] = Field(
        default=None,
        description="List of CSS selectors that should appear before considering the page fully rendered.",
        examples=[["article", "[data-testid='comment']"]],
    )
    min_content_selector_count: Optional[int] = Field(
        default=None,
        description="Minimum number of matching nodes across content_selectors before treating the page as stable.",
        ge=0,
        le=10000,
    )
    load_more_selectors: Optional[List[str]] = Field(
        default=None,
        description="Extra CSS selectors to click (e.g., 'Load more' buttons) while waiting for the page to stabilize.",
        examples=[["button.morecomments", "button[data-action='expand']"]],
    )
    max_scroll_rounds: int = Field(
        default=8,
        description="Maximum number of auto-scroll steps to trigger lazy loading.",
        ge=0,
        le=50,
    )
    scroll_delay_ms: int = Field(
        default=350,
        description="Pause between auto-scroll steps in milliseconds.",
        ge=50,
        le=2000,
    )
    load_more_clicks: int = Field(
        default=6,
        description="Maximum number of load-more click cycles.",
        ge=0,
        le=25,
    )
    stabilization_iterations: int = Field(
        default=2,
        description="Number of consecutive stabilization checks with no new content before concluding the page is complete.",
        ge=1,
        le=10,
    )
    stabilization_interval_ms: int = Field(
        default=700,
        description="Delay between stabilization checks in milliseconds.",
        ge=100,
        le=5000,
    )
    max_render_wait_ms: int = Field(
        default=20000,
        description="Maximum time in milliseconds to wait for dynamic content before returning.",
        ge=1000,
        le=60000,
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
