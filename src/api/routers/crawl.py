"""Crawl API router for web scraping functionality using worker."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from ..models.crawl import CrawlRequest, CrawlResponse, ErrorResponse
from ...storage.history import history_storage
from ...db.catalog import get_lib_dict
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["crawl"])


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

    # Get lib config for python_env
    lib_id = "crawl4ai/async-webcrawler"
    lib_config = get_lib_dict(lib_id)

    if not lib_config:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ENGINE_NOT_FOUND",
                "message": f"Library '{lib_id}' not found in catalog",
                "request_id": request_id,
            },
        )

    python_env = lib_config.get("python_env")

    try:
        # Delegate to worker
        result = await coordinator_infer(
            task="web-crawling",
            model_id=lib_id,
            payload={
                "url": str(request.url),
                "screenshot": request.screenshot,
                "screenshot_fullpage": request.screenshot_fullpage,
                "screenshot_width": request.screenshot_width,
                "screenshot_height": request.screenshot_height,
                "page_timeout": request.page_timeout,
                "chrome_cdp_url": request.chrome_cdp_url,
                "include_html": request.include_html,
            },
            request_id=request_id,
            python_env=python_env,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Screenshot is already base64-encoded
        screenshot_base64 = result.get("screenshot") if request.screenshot else None
        screenshot_fullpage_base64 = (
            result.get("screenshot_fullpage") if request.screenshot else None
        )

        response = CrawlResponse(
            request_id=request_id,
            url=result.get("url", str(request.url)),
            title=result.get("title"),
            markdown=result.get("markdown", ""),
            html=result.get("html") if request.include_html else None,
            screenshot_base64=screenshot_base64,
            screenshot_fullpage_base64=screenshot_fullpage_base64,
            processing_time_ms=processing_time_ms,
            success=result.get("success", True),
        )

        # Save to history
        history_storage.add_request(
            service="crawl",
            request_id=request_id,
            request_data=request.model_dump(mode="json"),
            response_data=response.model_dump(
                exclude={
                    "html",
                    "screenshot_base64",
                    "screenshot_fullpage_base64",
                }
            ),
            status="success",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during crawl: {str(e)}"
        logger.error(f"Crawl failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CRAWL_FAILED",
                "message": error_msg,
                "request_id": request_id,
            },
        )
