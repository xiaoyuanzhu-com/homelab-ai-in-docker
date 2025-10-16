"""Crawl API router for web scraping functionality."""

import asyncio
import base64
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from crawl4ai import AsyncWebCrawler

from ..models.crawl import CrawlRequest, CrawlResponse, ErrorResponse
from ...storage.history import history_storage


router = APIRouter(prefix="/api", tags=["crawl"])


async def crawl_url(
    url: str,
    screenshot: bool = False,
    wait_for_js: bool = True,
) -> dict:
    """
    Crawl a URL and extract content.

    Args:
        url: URL to crawl
        screenshot: Whether to capture a screenshot
        wait_for_js: Whether to wait for JavaScript execution

    Returns:
        Dictionary containing crawl results
    """
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                screenshot=screenshot,
                wait_for=2000 if wait_for_js else 0,  # Wait 2s for JS
            )

            return {
                "url": result.url,
                "title": result.metadata.get("title") if result.metadata else None,
                "markdown": result.markdown,
                "html": result.html,
                "screenshot": result.screenshot if screenshot else None,
                "success": result.success,
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CRAWL_FAILED",
                "message": f"Failed to crawl URL: {str(e)}",
                "url": url,
            },
        )


@router.post("/crawl", response_model=CrawlResponse, responses={500: {"model": ErrorResponse}})
async def crawl(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and extract clean text content.

    This endpoint uses crawl4ai to:
    - Fetch web pages with JavaScript rendering support
    - Extract clean text in Markdown format
    - Optionally capture screenshots

    Args:
        request: Crawl request parameters

    Returns:
        Crawl results including extracted content

    Raises:
        HTTPException: If crawling fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        result = await crawl_url(
            url=str(request.url),
            screenshot=request.screenshot,
            wait_for_js=request.wait_for_js,
        )

        fetch_time_ms = int((time.time() - start_time) * 1000)

        # Convert screenshot to base64 if present
        screenshot_base64 = None
        if request.screenshot and result.get("screenshot"):
            screenshot_base64 = base64.b64encode(result["screenshot"]).decode("utf-8")

        response = CrawlResponse(
            request_id=request_id,
            url=result["url"],
            title=result.get("title"),
            markdown=result["markdown"] or "",
            html=result.get("html"),
            screenshot_base64=screenshot_base64,
            fetch_time_ms=fetch_time_ms,
            success=result["success"],
        )

        # Save to history
        history_storage.add_request(
            service="crawl",
            request_id=request_id,
            request_data=request.model_dump(),
            response_data=response.model_dump(exclude={"html", "screenshot_base64"}),
            status="success",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CRAWL_FAILED",
                "message": f"Unexpected error during crawl: {str(e)}",
                "request_id": request_id,
            },
        )
