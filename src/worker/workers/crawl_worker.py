"""Crawl4AI worker for web crawling and content extraction.

Crawls web pages with JavaScript rendering support and extracts
clean content in Markdown format.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import logging
import os
import sys
import types
from typing import Any, Dict, Optional

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger(__name__)

# Get default remote Chrome URL from environment
DEFAULT_CHROME_CDP_URL = os.environ.get("CHROME_CDP_URL")

# Stealth mode configuration
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


def _install_playwright_stealth_compat(module) -> bool:
    stealth_async_fn = getattr(module, "stealth_async", None)
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

    class _CompatStealth:
        def __init__(self, config=None):
            self.config = config or StealthConfig()

        def use_async(self, playwright_cm):
            return _CompatAsyncStealthContext(playwright_cm, stealth_async_fn, self.config)

    module.Stealth = _CompatStealth
    module.StealthAsync = _CompatAsyncStealthContext
    return True


class CrawlWorker(BaseWorker):
    """Worker for web crawling using Crawl4AI."""

    task_name = "web-crawling"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._stealth_module = None
        self._stealth_enabled = False

    def load_model(self) -> Any:
        """Initialize Crawl4AI and stealth libraries."""
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        # Initialize stealth support
        if DEFAULT_ENABLE_STEALTH:
            try:
                self._stealth_module = importlib.import_module("playwright_stealth")
                if any(
                    hasattr(self._stealth_module, attr)
                    for attr in ("Stealth", "StealthAsync", "StealthSync")
                ):
                    self._stealth_enabled = True
                elif _install_playwright_stealth_compat(self._stealth_module):
                    self._stealth_enabled = True
                else:
                    logger.warning("playwright_stealth incompatible, stealth disabled")
            except ImportError:
                logger.warning("playwright_stealth not available, stealth disabled")

        logger.info(f"Crawl4AI loaded, stealth_enabled={self._stealth_enabled}")

        # Return the classes we need
        return {
            "AsyncWebCrawler": AsyncWebCrawler,
            "BrowserConfig": BrowserConfig,
            "CrawlerRunConfig": CrawlerRunConfig,
        }

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Crawl a URL and extract content.

        Args:
            payload: {
                "url": URL to crawl,
                "screenshot": whether to capture viewport screenshot,
                "screenshot_fullpage": whether to capture full page screenshot,
                "screenshot_width": viewport width,
                "screenshot_height": viewport height,
                "page_timeout": navigation timeout in ms,
                "chrome_cdp_url": optional remote Chrome CDP URL,
                "include_html": whether to include raw HTML
            }

        Returns:
            {
                "url": final URL,
                "title": page title,
                "markdown": extracted markdown content,
                "html": raw HTML (if requested),
                "screenshot": base64 viewport screenshot,
                "screenshot_fullpage": base64 full page screenshot,
                "success": boolean
            }
        """
        # Run async crawl in event loop
        return asyncio.run(self._crawl_async(payload))

    async def _crawl_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Async implementation of crawling."""
        url = payload.get("url", "")
        screenshot = payload.get("screenshot", False)
        screenshot_fullpage = payload.get("screenshot_fullpage", False)
        screenshot_width = payload.get("screenshot_width", 1920)
        screenshot_height = payload.get("screenshot_height", 1080)
        page_timeout = payload.get("page_timeout", 120000)
        chrome_cdp_url = payload.get("chrome_cdp_url") or DEFAULT_CHROME_CDP_URL
        include_html = payload.get("include_html", False)

        # Normalize CDP URL: strip trailing slashes to avoid double-slash in path
        # e.g., "http://host:9223/" + "/json/version" -> "http://host:9223//json/version" (fails)
        if chrome_cdp_url:
            chrome_cdp_url = chrome_cdp_url.rstrip("/")

        AsyncWebCrawler = self._model["AsyncWebCrawler"]
        BrowserConfig = self._model["BrowserConfig"]
        CrawlerRunConfig = self._model["CrawlerRunConfig"]

        # Configure browser
        if chrome_cdp_url:
            logger.info(f"Connecting to remote Chrome at: {chrome_cdp_url}")
            browser_config = BrowserConfig(
                browser_mode="cdp",
                cdp_url=chrome_cdp_url,
                headless=True,
                verbose=False,
                viewport_width=screenshot_width,
                viewport_height=screenshot_height,
                enable_stealth=self._stealth_enabled,
            )
        else:
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                viewport_width=screenshot_width,
                viewport_height=screenshot_height,
                enable_stealth=self._stealth_enabled,
            )

        async with AsyncWebCrawler(config=browser_config, verbose=False) as crawler:
            run_config = CrawlerRunConfig(
                screenshot=bool(screenshot_fullpage),
                page_timeout=page_timeout,
                wait_until="domcontentloaded",
                delay_before_return_html=1.5,
                simulate_user=False,
                scan_full_page=True,
                override_navigator=True,
            )

            result = await crawler.arun(url=url, config=run_config)

            # Handle screenshots
            viewport_b64: Optional[str] = None
            fullpage_b64: Optional[str] = None

            if screenshot_fullpage:
                fullpage_b64 = result.screenshot

            if screenshot:
                viewport_b64 = await self._capture_viewport_screenshot(
                    url=url,
                    width=screenshot_width,
                    height=screenshot_height,
                    timeout_ms=page_timeout,
                    chrome_cdp_url=chrome_cdp_url,
                )

            # Note: result.markdown may be a StringCompatibleMarkdown object (crawl4ai 0.7+)
            # which inherits from str but has extra attributes that confuse Pydantic.
            # Convert to plain string explicitly.
            markdown_content = str(result.markdown) if result.markdown else ""

            return {
                "url": result.url,
                "title": result.metadata.get("title") if result.metadata else None,
                "markdown": markdown_content,
                "html": result.html if include_html else None,
                "screenshot": viewport_b64 or fullpage_b64,
                "screenshot_fullpage": fullpage_b64,
                "success": result.success,
            }

    async def _capture_viewport_screenshot(
        self,
        url: str,
        width: int,
        height: int,
        timeout_ms: int,
        chrome_cdp_url: Optional[str] = None,
    ) -> Optional[str]:
        """Capture a viewport screenshot using Playwright directly."""
        try:
            from playwright.async_api import async_playwright

            # Wrap with stealth if available
            if self._stealth_enabled and self._stealth_module is not None:
                stealth = self._stealth_module.Stealth()
                playwright_cm = stealth.use_async(async_playwright())
            else:
                playwright_cm = async_playwright()

            async with playwright_cm as p:
                browser = None
                context = None
                page = None
                try:
                    if chrome_cdp_url:
                        browser = await p.chromium.connect_over_cdp(chrome_cdp_url)
                        try:
                            context = await browser.new_context(
                                viewport={"width": width, "height": height}
                            )
                            page = await context.new_page()
                        except Exception:
                            contexts = getattr(browser, "contexts", []) or []
                            if contexts:
                                context = contexts[0]
                                page = await context.new_page()
                                try:
                                    await page.set_viewport_size({"width": width, "height": height})
                                except Exception:
                                    pass
                            else:
                                page = await browser.new_page()
                                try:
                                    await page.set_viewport_size({"width": width, "height": height})
                                except Exception:
                                    pass
                    else:
                        browser = await p.chromium.launch(headless=True)
                        context = await browser.new_context(
                            viewport={"width": width, "height": height}
                        )
                        page = await context.new_page()

                    try:
                        page.set_default_navigation_timeout(timeout_ms)
                        page.set_default_timeout(timeout_ms)
                    except Exception:
                        pass

                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    await asyncio.sleep(1.5)
                    img_bytes = await page.screenshot(full_page=False)
                    return base64.b64encode(img_bytes).decode("utf-8")
                finally:
                    try:
                        if page is not None:
                            await page.close()
                    except Exception:
                        pass
                    try:
                        if context is not None and not chrome_cdp_url:
                            await context.close()
                    except Exception:
                        pass
                    try:
                        if browser is not None and not chrome_cdp_url:
                            await browser.close()
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Viewport screenshot failed: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up resources."""
        self._stealth_module = None
        super().cleanup()


main = create_worker_main(CrawlWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
