"""Crawl API router for web scraping functionality."""

import asyncio
import importlib
import logging
import os
import time
import types
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
# If not provided, let the browser supply its native User-Agent.
# No UA override: use browser's native User-Agent
# No global header overrides: let the browser send its native headers.

"""
Minimal, general crawler: no site-specific selectors, scrolling, or clicking.
"""

DEFAULT_ENABLE_STEALTH = os.environ.get("CRAWLER_ENABLE_STEALTH", "true").lower() not in (
    "false",
    "0",
    "no",
)

def _schedule_async_task(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        loop.create_task(coro)


async def _apply_stealth_to_context_async(context, stealth_async_fn, config):
    if getattr(context, "_haid_stealth_async_installed", False):
        return context

    setattr(context, "_haid_stealth_async_installed", True)

    original_new_page = context.new_page

    async def new_page_with_stealth(*args, **kwargs):
        page = await original_new_page(*args, **kwargs)
        await stealth_async_fn(page, config=config)
        return page

    context.new_page = types.MethodType(new_page_with_stealth, context)

    for page in list(context.pages):
        try:
            await stealth_async_fn(page, config=config)
        except Exception:
            continue

    context.on("page", lambda page: _schedule_async_task(stealth_async_fn(page, config=config)))
    return context


async def _apply_stealth_to_browser_async(browser, stealth_async_fn, config):
    if getattr(browser, "_haid_stealth_async_installed", False):
        return browser

    setattr(browser, "_haid_stealth_async_installed", True)

    original_new_page = getattr(browser, "new_page", None)
    if original_new_page is not None:

        async def browser_new_page(*args, **kwargs):
            page = await original_new_page(*args, **kwargs)
            await stealth_async_fn(page, config=config)
            return page

        browser.new_page = types.MethodType(browser_new_page, browser)

    original_new_context = getattr(browser, "new_context", None)
    if original_new_context is not None:

        async def browser_new_context(*args, **kwargs):
            context = await original_new_context(*args, **kwargs)
            await _apply_stealth_to_context_async(context, stealth_async_fn, config)
            return context

        browser.new_context = types.MethodType(browser_new_context, browser)

    for context in list(getattr(browser, "contexts", [])):
        await _apply_stealth_to_context_async(context, stealth_async_fn, config)

    if hasattr(browser, "on"):
        try:
            browser.on(
                "context",
                lambda ctx: _schedule_async_task(
                    _apply_stealth_to_context_async(ctx, stealth_async_fn, config)
                ),
            )
        except Exception:
            pass

    return browser


def _wrap_async_method(obj, attr_name, after_coroutine):
    original = getattr(obj, attr_name, None)
    if original is None or getattr(obj, f"_haid_wrapped_async_{attr_name}", False):
        return

    async def wrapped(*args, **kwargs):
        result = await original(*args, **kwargs)
        new_result = await after_coroutine(result)
        return new_result or result

    setattr(obj, attr_name, types.MethodType(wrapped, obj))
    setattr(obj, f"_haid_wrapped_async_{attr_name}", True)


def _wrap_sync_method(obj, attr_name, after_callable):
    original = getattr(obj, attr_name, None)
    if original is None or getattr(obj, f"_haid_wrapped_sync_{attr_name}", False):
        return

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        new_result = after_callable(result)
        return new_result or result

    setattr(obj, attr_name, types.MethodType(wrapped, obj))
    setattr(obj, f"_haid_wrapped_sync_{attr_name}", True)


async def _apply_async_stealth_to_playwright(playwright, stealth_async_fn, config):
    async def _after_browser(browser):
        return await _apply_stealth_to_browser_async(browser, stealth_async_fn, config)

    async def _after_context(context):
        return await _apply_stealth_to_context_async(context, stealth_async_fn, config)

    for browser_type_name in ("chromium", "firefox", "webkit"):
        browser_type = getattr(playwright, browser_type_name, None)
        if browser_type is None:
            continue
        _wrap_async_method(browser_type, "launch", _after_browser)
        _wrap_async_method(browser_type, "connect_over_cdp", _after_browser)
        _wrap_async_method(browser_type, "launch_persistent_context", _after_context)

    return playwright


def _apply_sync_stealth_to_context(context, stealth_sync_fn, config):
    if getattr(context, "_haid_stealth_sync_installed", False):
        return context

    setattr(context, "_haid_stealth_sync_installed", True)

    original_new_page = context.new_page

    def new_page_with_stealth(*args, **kwargs):
        page = original_new_page(*args, **kwargs)
        stealth_sync_fn(page, config=config)
        return page

    context.new_page = types.MethodType(new_page_with_stealth, context)

    for page in list(context.pages):
        try:
            stealth_sync_fn(page, config=config)
        except Exception:
            continue

    if hasattr(context, "on"):
        context.on("page", lambda page: stealth_sync_fn(page, config=config))
    return context


def _apply_sync_stealth_to_browser(browser, stealth_sync_fn, config):
    if getattr(browser, "_haid_stealth_sync_installed", False):
        return browser

    setattr(browser, "_haid_stealth_sync_installed", True)

    original_new_page = getattr(browser, "new_page", None)
    if original_new_page is not None:

        def browser_new_page(*args, **kwargs):
            page = original_new_page(*args, **kwargs)
            stealth_sync_fn(page, config=config)
            return page

        browser.new_page = types.MethodType(browser_new_page, browser)

    original_new_context = getattr(browser, "new_context", None)
    if original_new_context is not None:

        def browser_new_context(*args, **kwargs):
            context = original_new_context(*args, **kwargs)
            _apply_sync_stealth_to_context(context, stealth_sync_fn, config)
            return context

        browser.new_context = types.MethodType(browser_new_context, browser)

    for context in list(getattr(browser, "contexts", [])):
        _apply_sync_stealth_to_context(context, stealth_sync_fn, config)

    if hasattr(browser, "on"):
        try:
            browser.on(
                "context",
                lambda ctx: _apply_sync_stealth_to_context(ctx, stealth_sync_fn, config),
            )
        except Exception:
            pass

    return browser


def _apply_sync_stealth_to_playwright(playwright, stealth_sync_fn, config):

    def _after_browser(browser):
        return _apply_sync_stealth_to_browser(browser, stealth_sync_fn, config)

    def _after_context(context):
        return _apply_sync_stealth_to_context(context, stealth_sync_fn, config)

    for browser_type_name in ("chromium", "firefox", "webkit"):
        browser_type = getattr(playwright, browser_type_name, None)
        if browser_type is None:
            continue
        _wrap_sync_method(browser_type, "launch", _after_browser)
        _wrap_sync_method(browser_type, "connect_over_cdp", _after_browser)
        _wrap_sync_method(browser_type, "launch_persistent_context", _after_context)

    return playwright


def _install_playwright_stealth_compat(module) -> bool:
    stealth_async_fn = getattr(module, "stealth_async", None)
    stealth_sync_fn = getattr(module, "stealth_sync", None)
    StealthConfig = getattr(module, "StealthConfig", None)

    if stealth_async_fn is None or StealthConfig is None:
        return False

    class _CompatAsyncStealthContext:
        def __init__(self, inner_cm, stealth_fn, cfg):
            self._inner_cm = inner_cm
            self._stealth_fn = stealth_fn
            self._config = cfg

        async def __aenter__(self):
            self._cm = self._inner_cm
            self._playwright = await self._cm.__aenter__()
            await _apply_async_stealth_to_playwright(
                self._playwright, self._stealth_fn, self._config
            )
            return self._playwright

        async def __aexit__(self, exc_type, exc, tb):
            return await self._cm.__aexit__(exc_type, exc, tb)

    _CompatSyncStealthContext = None
    if stealth_sync_fn is not None:

        class _CompatSyncStealthContext:
            def __init__(self, inner_cm, stealth_fn, cfg):
                self._inner_cm = inner_cm
                self._stealth_fn = stealth_fn
                self._config = cfg

            def __enter__(self):
                self._cm = self._inner_cm
                self._playwright = self._cm.__enter__()
                _apply_sync_stealth_to_playwright(
                    self._playwright, self._stealth_fn, self._config
                )
                return self._playwright

            def __exit__(self, exc_type, exc, tb):
                return self._cm.__exit__(exc_type, exc, tb)

    class _CompatStealth:
        def __init__(self, config=None):
            self.config = config or StealthConfig()

        def use_async(self, playwright_cm):
            return _CompatAsyncStealthContext(playwright_cm, stealth_async_fn, self.config)

        def use(self, playwright_cm):
            if stealth_sync_fn is None or _CompatSyncStealthContext is None:
                raise RuntimeError(
                    "playwright-stealth does not expose synchronous helpers in this version."
                )
            return _CompatSyncStealthContext(playwright_cm, stealth_sync_fn, self.config)

    module.Stealth = _CompatStealth
    module.StealthAsync = _CompatAsyncStealthContext
    if stealth_sync_fn is not None:
        module.StealthSync = _CompatSyncStealthContext
    return True


## Removed site-specific and generic selector/interaction helpers to keep the
## crawler general. No auto-scroll or auto-click behavior.


_stealth_module = None
PLAYWRIGHT_STEALTH_SUPPORTED = False
if DEFAULT_ENABLE_STEALTH:
    try:
        _stealth_module = importlib.import_module('playwright_stealth')
    except ImportError:
        PLAYWRIGHT_STEALTH_SUPPORTED = False
    else:
        if any(
            hasattr(_stealth_module, attr_name)
            for attr_name in ("Stealth", "StealthAsync", "StealthSync")
        ):
            PLAYWRIGHT_STEALTH_SUPPORTED = True
        elif _install_playwright_stealth_compat(_stealth_module):
            PLAYWRIGHT_STEALTH_SUPPORTED = True
        else:
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
    page_timeout: int = 120000,
    chrome_cdp_url: Optional[str] = None,
) -> dict:
    """
    Crawl a URL and extract content.

    Args:
        url: URL to crawl
        screenshot: Whether to capture a screenshot
        screenshot_width: Screenshot viewport width in pixels
        screenshot_height: Screenshot viewport height in pixels
        page_timeout: Page navigation timeout in milliseconds
        chrome_cdp_url: Remote Chrome CDP URL for browser connection

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
            _cdp_kwargs = dict(
                browser_mode="cdp",
                cdp_url=cdp_url,
                headless=True,
                verbose=False,
                viewport_width=screenshot_width,
                viewport_height=screenshot_height,
                enable_stealth=ENABLE_STEALTH,
            )
            browser_config = BrowserConfig(**_cdp_kwargs)
        else:
            # Use local browser (default mode)
            _local_kwargs = dict(
                headless=True,
                verbose=False,
                viewport_width=screenshot_width,
                viewport_height=screenshot_height,
                enable_stealth=ENABLE_STEALTH,
            )
            browser_config = BrowserConfig(**_local_kwargs)

        async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
            # Configure crawler run parameters
            run_config_params = {
                "screenshot": screenshot,
                "page_timeout": page_timeout,
                # Use defaults for scrolling; we don't force interactions
            }

            # For JavaScript-heavy SPAs:
            # - wait_until="domcontentloaded" avoids networkidle hangs
            # - simulate_user/override_navigator reduce bot detection
            run_config_params["wait_until"] = "domcontentloaded"
            # Give SPAs a brief, generic settle window for dynamic content
            run_config_params["delay_before_return_html"] = 1.5
            # Do not simulate random user interactions to avoid accidental clicks
            run_config_params["simulate_user"] = False
            # Enable full-page scanning to capture content beyond initial viewport
            run_config_params["scan_full_page"] = True
            run_config_params["override_navigator"] = True

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
            page_timeout=request.page_timeout,
            chrome_cdp_url=request.chrome_cdp_url,
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
