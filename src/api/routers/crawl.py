"""Crawl API router for web scraping functionality."""

import importlib
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from ..models.crawl import CrawlRequest, CrawlResponse, ErrorResponse
from ...storage.history import history_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["crawl"])

# Get default remote Chrome URL from environment
DEFAULT_CHROME_CDP_URL = os.environ.get("CHROME_CDP_URL")
DEFAULT_USER_AGENT = os.environ.get(
    "CRAWLER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)
DEFAULT_ACCEPT_LANGUAGE = os.environ.get("CRAWLER_ACCEPT_LANGUAGE", "en-US,en;q=0.9")
DEFAULT_REQUEST_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": DEFAULT_ACCEPT_LANGUAGE,
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DEFAULT_ENABLE_STEALTH = os.environ.get("CRAWLER_ENABLE_STEALTH", "true").lower() not in (
    "false",
    "0",
    "no",
)


_stealth_module = None
PLAYWRIGHT_STEALTH_SUPPORTED = False
if DEFAULT_ENABLE_STEALTH:
    try:
        _stealth_module = importlib.import_module('playwright_stealth')
    except ImportError:
        PLAYWRIGHT_STEALTH_SUPPORTED = False
    else:
        PLAYWRIGHT_STEALTH_SUPPORTED = hasattr(_stealth_module, 'Stealth')
        if not PLAYWRIGHT_STEALTH_SUPPORTED:
            logger.warning(
                'playwright_stealth does not provide the Stealth wrapper; stealth mode disabled.'
            )
if DEFAULT_ENABLE_STEALTH and not PLAYWRIGHT_STEALTH_SUPPORTED:
    logger.warning(
        'Stealth mode requested but playwright_stealth is unavailable or incompatible. ' +
        'Crawler will run without stealth fingerprinting.'
    )
ENABLE_STEALTH = DEFAULT_ENABLE_STEALTH and PLAYWRIGHT_STEALTH_SUPPORTED


async def crawl_url(
    url: str,
    screenshot: bool = False,
    screenshot_width: int = 1920,
    screenshot_height: int = 1080,
    wait_for_js: bool = True,
    page_timeout: int = 120000,
    chrome_cdp_url: Optional[str] = None,
    wait_for_selector: Optional[str] = None,
    wait_for_selector_timeout: Optional[int] = None,
) -> dict:
    """
    Crawl a URL and extract content.

    Args:
        url: URL to crawl
        screenshot: Whether to capture a screenshot
        screenshot_width: Screenshot viewport width in pixels
        screenshot_height: Screenshot viewport height in pixels
        wait_for_js: Whether to wait for JavaScript execution
        page_timeout: Page navigation timeout in milliseconds
        chrome_cdp_url: Remote Chrome CDP URL for browser connection
        wait_for_selector: Optional CSS selector to wait for once DOMContentLoaded fires
        wait_for_selector_timeout: Timeout for selector wait

    Returns:
        Dictionary containing crawl results
    """
    try:
        # Use remote Chrome if URL provided, otherwise use default from env or local
        cdp_url = chrome_cdp_url or DEFAULT_CHROME_CDP_URL

        # Configure browser with CDP support if URL is provided
        if cdp_url:
            # Connect to remote Chrome via CDP
            logger.info(f"Connecting to remote Chrome at: {cdp_url}")
            browser_config = BrowserConfig(
                browser_mode="cdp",
                cdp_url=cdp_url,
                headless=True,
                verbose=False,
                viewport_width=screenshot_width,
                viewport_height=screenshot_height,
                headers=dict(DEFAULT_REQUEST_HEADERS),
                user_agent=DEFAULT_USER_AGENT,
                enable_stealth=ENABLE_STEALTH,
            )
        else:
            # Use local browser (default mode)
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                viewport_width=screenshot_width,
                viewport_height=screenshot_height,
                headers=dict(DEFAULT_REQUEST_HEADERS),
                user_agent=DEFAULT_USER_AGENT,
                enable_stealth=ENABLE_STEALTH,
            )

        async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
            # Configure crawler run parameters
            run_config_params = {
                "screenshot": screenshot,
                "page_timeout": page_timeout,
            }

            if wait_for_js:
                # For JavaScript-heavy SPAs:
                # - wait_until="domcontentloaded" avoids networkidle hangs
                # - wait_for selector ensures target nodes render
                # - simulate_user/override_navigator reduce bot detection
                run_config_params["wait_until"] = "domcontentloaded"
                run_config_params["delay_before_return_html"] = 0.5
                run_config_params["simulate_user"] = True
                run_config_params["scan_full_page"] = True
                run_config_params["override_navigator"] = True

                if wait_for_selector:
                    run_config_params["wait_for"] = wait_for_selector
                    if wait_for_selector_timeout:
                        run_config_params["wait_for_timeout"] = wait_for_selector_timeout
            else:
                run_config_params["wait_until"] = "load"

            # Create crawler run configuration
            run_config = CrawlerRunConfig(**run_config_params)

            # Run the crawler with the configuration
            result = await crawler.arun(url=url, config=run_config)

            return {
                "url": result.url,
                "title": result.metadata.get("title") if result.metadata else None,
                "markdown": result.markdown,
                "html": result.html,
                "screenshot": result.screenshot if screenshot else None,
                "success": result.success,
            }
    except Exception as e:
        error_msg = f"Failed to crawl URL {url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
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
            screenshot_width=request.screenshot_width,
            screenshot_height=request.screenshot_height,
            wait_for_js=request.wait_for_js,
            page_timeout=request.page_timeout,
            chrome_cdp_url=request.chrome_cdp_url,
            wait_for_selector=request.wait_for_selector,
            wait_for_selector_timeout=request.wait_for_selector_timeout,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Screenshot is already base64-encoded by crawl4ai
        screenshot_base64 = result.get("screenshot") if request.screenshot else None

        response = CrawlResponse(
            request_id=request_id,
            url=result["url"],
            title=result.get("title"),
            markdown=result["markdown"] or "",
            html=result.get("html"),
            screenshot_base64=screenshot_base64,
            processing_time_ms=processing_time_ms,
            success=result["success"],
        )

        # Save to history (convert HttpUrl to string)
        history_storage.add_request(
            service="crawl",
            request_id=request_id,
            request_data=request.model_dump(mode="json"),
            response_data=response.model_dump(exclude={"html", "screenshot_base64"}),
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
