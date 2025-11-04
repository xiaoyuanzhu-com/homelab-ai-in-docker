"""Crawl API router for web scraping functionality."""

import asyncio
import importlib
import json
import logging
import os
import time
import types
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
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

DEFAULT_CONTENT_SELECTORS: List[str] = [
    "[data-test-id='post-content']",
    "[data-testid='post-container']",
    "[data-test-id='comment']",
    "[data-testid='comment']",
    "div[data-test-id='comment']",
    "div[data-testid='comment']",
    "div[data-test-id='comment-list']",
    "shreddit-comment",
    "faceplate-tracker[data-tracker-name*='comment']",
    "[itemprop='articleBody']",
    "article.faceplate-comment",
    "article",
    "main article",
    "main",
]
DEFAULT_LOAD_MORE_SELECTORS: List[str] = [
    "button.morecomments",
    ".morecomments button",
    "button[aria-label*='more']",
    "button[aria-label*='More']",
    "button[aria-label*='load']",
    "button[aria-label*='Load']",
    "button[data-action='expand']",
    "button[data-click-id='comments']",
    "button[data-click-id='load_more']",
    "[data-testid='load-more']",
    "[data-testid='expand-button']",
    "[data-testid='caret']",
    "shreddit-comment-tree button",
    "faceplate-tracker button",
]
DEFAULT_LOAD_MORE_TEXTS: List[str] = [
    "Load more",
    "Show more",
    "View more",
    "More comments",
    "Load comments",
    "See more",
]
DEFAULT_MAX_RENDER_WAIT_MS = 20000

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


def _dedupe_selectors(selectors: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for selector in selectors:
        if not selector:
            continue
        key = selector.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


async def _wait_for_any_selector(page, selectors: List[str], timeout_ms: int) -> None:
    if not selectors:
        return
    for selector in selectors:
        try:
            await page.wait_for_selector(selector, timeout=timeout_ms)
            return
        except PlaywrightTimeoutError:
            continue


async def _count_matches(page, selector: str) -> int:
    try:
        return await page.evaluate(
            "sel => document.querySelectorAll(sel).length", selector
        )
    except Exception:
        return 0


async def _max_selector_count(page, selectors: List[str]) -> int:
    if not selectors:
        return 0
    counts = []
    for selector in selectors:
        counts.append(await _count_matches(page, selector))
    return max(counts) if counts else 0


def _track_network_activity(page):
    timestamps = {"last": time.monotonic()}

    def _update(*_args, **_kwargs):
        timestamps["last"] = time.monotonic()

    events = ["request", "requestfinished", "requestfailed", "response"]
    for event in events:
        page.on(event, _update)

    def _stop():
        for event in events:
            try:
                page.off(event, _update)
            except Exception:
                continue

    return timestamps, _stop


async def _wait_for_network_quiet(
    timestamps: dict,
    quiet_ms: int,
    timeout_ms: int,
) -> bool:
    start = time.monotonic()
    sleep_interval = max(quiet_ms / 2000.0, 0.1)
    while (time.monotonic() - start) * 1000 < timeout_ms:
        idle_ms = (time.monotonic() - timestamps["last"]) * 1000
        if idle_ms >= quiet_ms:
            return True
        await asyncio.sleep(sleep_interval)
    return False


async def _click_candidates(page, selectors: List[str], texts: List[str], delay_ms: int) -> bool:
    clicked = False

    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = await locator.count()
        except Exception:
            continue

        for idx in range(count):
            element = locator.nth(idx)
            try:
                if await element.is_visible() and await element.is_enabled():
                    await element.click(timeout=2000)
                    clicked = True
                    await page.wait_for_timeout(delay_ms)
            except Exception:
                continue

    for text in texts:
        text_selector = (
            ":is(button, a, div[role='button'], span[role='button'])"
            f":has-text({json.dumps(text)})"
        )
        locator = page.locator(text_selector)
        try:
            count = await locator.count()
        except Exception:
            continue

        for idx in range(count):
            element = locator.nth(idx)
            try:
                if await element.is_visible() and await element.is_enabled():
                    await element.click(timeout=2000)
                    clicked = True
                    await page.wait_for_timeout(delay_ms)
            except Exception:
                continue

    return clicked


async def _ensure_render_complete(page, url: str, logger: logging.Logger, settings: dict) -> None:
    selectors: List[str] = settings["content_selectors"]
    min_count: Optional[int] = settings["min_content_selector_count"]
    load_selectors: List[str] = settings["load_more_selectors"]
    load_texts: List[str] = settings["load_more_texts"]
    max_scroll_rounds: int = settings["max_scroll_rounds"]
    scroll_delay_ms: int = settings["scroll_delay_ms"]
    max_click_loops: int = settings["max_click_loops"]
    stabilization_iterations: int = settings["stabilization_iterations"]
    stabilization_interval_ms: int = settings["stabilization_interval_ms"]
    max_render_wait_ms: int = settings["max_render_wait_ms"]
    selector_wait_timeout: int = settings["selector_wait_timeout"]
    network_quiet_ms: int = settings["network_quiet_ms"]
    network_quiet_timeout_ms: int = settings["network_quiet_timeout_ms"]
    min_text_length: int = settings["min_text_length"]
    text_delta_threshold: int = settings["text_delta_threshold"]

    await _wait_for_any_selector(page, selectors, selector_wait_timeout)

    try:
        await page.wait_for_load_state("load")
    except PlaywrightTimeoutError:
        pass

    network_tracker, stop_tracking = _track_network_activity(page)

    start_time = time.monotonic()
    previous_count = -1
    stable_rounds = 0
    scroll_round = 0
    click_round = 0
    previous_height = 0
    stagnant_scrolls = 0
    last_text_length = -1

    try:
        while True:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if elapsed_ms >= max_render_wait_ms:
                logger.warning(
                    "Render stabilization timed out after %.0fms for %s",
                    elapsed_ms,
                    url,
                )
                break

            await _wait_for_network_quiet(network_tracker, network_quiet_ms, network_quiet_timeout_ms)

            if scroll_round < max_scroll_rounds:
                try:
                    current_height = await page.evaluate("() => document.body.scrollHeight")
                    await page.evaluate("() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
                    scroll_round += 1
                    if abs(current_height - previous_height) < 50:
                        stagnant_scrolls += 1
                    else:
                        stagnant_scrolls = 0
                    previous_height = current_height
                except Exception:
                    stagnant_scrolls += 1
                await page.wait_for_timeout(max(scroll_delay_ms, 50))

            clicked = False
            if click_round < max_click_loops:
                clicked = await _click_candidates(page, load_selectors, load_texts, scroll_delay_ms)
                if clicked:
                    click_round += 1

            current_count = await _max_selector_count(page, selectors)
            text_length = await page.evaluate("() => document.body.innerText.length")
            meets_minimum = min_count is None or current_count >= min_count
            meets_text = min_text_length is None or text_length >= min_text_length

            if current_count == previous_count and abs(text_length - last_text_length) <= text_delta_threshold:
                stable_rounds += 1
            else:
                stable_rounds = 0
            previous_count = current_count
            last_text_length = text_length

            if (meets_minimum or meets_text) and stable_rounds >= stabilization_iterations:
                break

            if (
                not clicked
                and scroll_round >= max_scroll_rounds
                and stable_rounds >= stabilization_iterations
                and (meets_minimum or meets_text)
                and stagnant_scrolls >= 2
            ):
                break

            await page.wait_for_timeout(stabilization_interval_ms)
    finally:
        stop_tracking()


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
    wait_for_selector: Optional[str] = None,
    wait_for_selector_timeout: Optional[int] = None,
    content_selectors: Optional[List[str]] = None,
    min_content_selector_count: Optional[int] = None,
    load_more_selectors: Optional[List[str]] = None,
    max_scroll_rounds: int = 8,
    scroll_delay_ms: int = 350,
    load_more_clicks: int = 6,
    stabilization_iterations: int = 2,
    stabilization_interval_ms: int = 700,
    max_render_wait_ms: Optional[int] = None,
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
        wait_for_selector: Optional CSS selector to wait for once DOMContentLoaded fires
        wait_for_selector_timeout: Timeout for selector wait
        content_selectors: Additional selectors that signal the important content has rendered
        min_content_selector_count: Minimum number of matched elements required before returning
        load_more_selectors: Extra selectors to click while waiting (e.g., “Load more” buttons)
        max_scroll_rounds: Maximum automatic scroll iterations to trigger lazy content
        scroll_delay_ms: Delay between scroll steps in milliseconds
        load_more_clicks: Maximum number of load-more click cycles
        stabilization_iterations: Consecutive iterations with no new content before finishing
        stabilization_interval_ms: Pause between stabilization checks in milliseconds
        max_render_wait_ms: Overall cap on dynamic render wait time

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

        merged_selectors: List[str] = []
        if wait_for_selector:
            merged_selectors.append(wait_for_selector)
        if content_selectors:
            merged_selectors.extend(content_selectors)
        merged_selectors.extend(DEFAULT_CONTENT_SELECTORS)
        merged_selectors = _dedupe_selectors(merged_selectors)

        merged_load_more = DEFAULT_LOAD_MORE_SELECTORS.copy()
        if load_more_selectors:
            merged_load_more = _dedupe_selectors(load_more_selectors + merged_load_more)

        selector_minimum = (
            min_content_selector_count
            if min_content_selector_count is not None
            else 1
        )

        render_settings = {
            "content_selectors": merged_selectors,
            "min_content_selector_count": selector_minimum,
            "load_more_selectors": merged_load_more,
            "load_more_texts": DEFAULT_LOAD_MORE_TEXTS,
            "max_scroll_rounds": max_scroll_rounds,
            "scroll_delay_ms": scroll_delay_ms,
            "max_click_loops": load_more_clicks,
            "stabilization_iterations": stabilization_iterations,
            "stabilization_interval_ms": stabilization_interval_ms,
            "max_render_wait_ms": max_render_wait_ms or DEFAULT_MAX_RENDER_WAIT_MS,
            "selector_wait_timeout": wait_for_selector_timeout or 10000,
            "network_quiet_ms": 1500,
            "network_quiet_timeout_ms": 15000,
            "min_text_length": 400,
            "text_delta_threshold": 80,
        }

        async def after_goto_hook(page, context=None, url: str = url, **_kwargs):
            try:
                await _ensure_render_complete(
                    page,
                    url,
                    logger,
                    render_settings,
                )
            except Exception as hook_error:
                logger.warning(
                    "Render completion hook failed for %s: %s",
                    url,
                    hook_error,
                    exc_info=True,
                )

        async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
            crawler.crawler_strategy.set_hook("after_goto", after_goto_hook)
            # Configure crawler run parameters
            run_config_params = {
                "screenshot": screenshot,
                "page_timeout": page_timeout,
                "max_scroll_steps": max_scroll_rounds,
                "scroll_delay": max(scroll_delay_ms / 1000.0, 0.05),
            }

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

            # Create crawler run configuration
            run_config = CrawlerRunConfig(**run_config_params)

            # Run the crawler with the configuration
            try:
                result = await crawler.arun(url=url, config=run_config)
            finally:
                crawler.crawler_strategy.set_hook("after_goto", None)

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
            wait_for_selector=request.wait_for_selector,
            wait_for_selector_timeout=request.wait_for_selector_timeout,
            content_selectors=request.content_selectors,
            min_content_selector_count=request.min_content_selector_count,
            load_more_selectors=request.load_more_selectors,
            max_scroll_rounds=request.max_scroll_rounds,
            scroll_delay_ms=request.scroll_delay_ms,
            load_more_clicks=request.load_more_clicks,
            stabilization_iterations=request.stabilization_iterations,
            stabilization_interval_ms=request.stabilization_interval_ms,
            max_render_wait_ms=request.max_render_wait_ms,
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
