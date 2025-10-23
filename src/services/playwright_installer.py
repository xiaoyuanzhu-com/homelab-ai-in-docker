"""Runtime installation and verification of Playwright browsers for crawl4ai."""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from ..db.settings import get_setting, set_setting

logger = logging.getLogger(__name__)

# Setting key for tracking installation status
PLAYWRIGHT_INSTALLED_KEY = "playwright_browsers_installed"


def _get_playwright_browsers_path() -> Path:
    """Get the Playwright browsers installation path."""
    # Check environment variable first
    if "PLAYWRIGHT_BROWSERS_PATH" in os.environ:
        return Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"])
    # Default to ~/.cache/ms-playwright
    return Path.home() / ".cache" / "ms-playwright"


def _cleanup_playwright_browsers() -> None:
    """
    Clean up Playwright browsers directory on installation failure.

    This ensures a clean state for the next installation attempt.
    """
    browsers_path = _get_playwright_browsers_path()

    if browsers_path.exists():
        try:
            logger.info(f"Cleaning up Playwright browsers directory: {browsers_path}")
            shutil.rmtree(browsers_path)
            logger.info("Playwright browsers directory cleaned up successfully")
        except Exception as e:
            logger.warning(f"Failed to clean up Playwright browsers directory: {e}")
    else:
        logger.debug(f"Playwright browsers directory does not exist: {browsers_path}")


async def check_playwright_installation() -> bool:
    """
    Check if Playwright browsers are properly installed.

    Returns:
        True if browsers are installed and ready, False otherwise
    """
    # First check database cache
    cached_status = get_setting(PLAYWRIGHT_INSTALLED_KEY)
    if cached_status == "true":
        # Verify it's still actually installed (could be deleted from data dir)
        if await _verify_playwright_browsers():
            return True
        else:
            # Cache was stale, clear it
            set_setting(PLAYWRIGHT_INSTALLED_KEY, "false")
            return False

    # Not cached or was false, check actual installation
    is_installed = await _verify_playwright_browsers()

    # Update cache
    set_setting(PLAYWRIGHT_INSTALLED_KEY, "true" if is_installed else "false")

    return is_installed


async def _verify_playwright_browsers() -> bool:
    """
    Verify that Playwright browsers are actually installed by checking the directory.

    Returns:
        True if browsers are available, False otherwise
    """
    try:
        browsers_path = _get_playwright_browsers_path()
        logger.debug(f"Checking for Playwright browsers at: {browsers_path}")

        # Check if the directory exists and has chromium
        if not browsers_path.exists():
            logger.debug(f"Browsers directory does not exist: {browsers_path}")
            return False

        # Look for chromium directory (pattern: chromium-*)
        chromium_dirs = list(browsers_path.glob("chromium-*"))
        if not chromium_dirs:
            logger.debug(f"No chromium directories found in: {browsers_path}")
            return False

        logger.debug(f"Found chromium installation: {chromium_dirs[0]}")
        return True

    except Exception as e:
        logger.debug(f"Playwright verification failed: {e}")
        return False


async def install_playwright_browsers() -> Tuple[bool, str]:
    """
    Install Playwright browsers for crawl4ai.

    This runs: playwright install chromium

    Note: We skip 'playwright install --with-deps' because it requires sudo for system dependencies,
    which won't work in non-interactive Docker environments. The Dockerfile pre-installs all
    required system dependencies (ffmpeg, etc.) at build time.

    We also skip 'crawl4ai-setup' because:
    1. It tries to run sudo for system packages (fails in Docker)
    2. It's not required - crawl4ai works fine with just playwright browsers installed
    3. The database initialization happens automatically on first use

    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info("Starting Playwright browsers installation...")

    try:
        # Install Playwright chromium browser (without --with-deps to avoid sudo requirement)
        browsers_path = _get_playwright_browsers_path()
        logger.info(f"Running: playwright install chromium (target: {browsers_path})")

        # Prepare environment with PLAYWRIGHT_BROWSERS_PATH
        env = os.environ.copy()
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "playwright", "install", "chromium",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        stdout, _ = await process.communicate()
        stdout_text = stdout.decode() if stdout else ""

        if process.returncode != 0:
            error_msg = f"Playwright installation failed with code {process.returncode}:\n{stdout_text}"
            logger.error(error_msg)
            return False, error_msg

        logger.info(f"Playwright browsers installed successfully to {browsers_path}")

        # Verify installation succeeded
        if await _verify_playwright_browsers():
            # Update database to mark as installed
            set_setting(
                PLAYWRIGHT_INSTALLED_KEY,
                "true",
                "Indicates whether Playwright browsers are installed and ready for crawl4ai"
            )
            logger.info("Playwright installation verification passed")
            return True, "Playwright browsers installed successfully"
        else:
            error_msg = "Installation completed but verification failed"
            logger.error(error_msg)
            # Cleanup on verification failure
            _cleanup_playwright_browsers()
            return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error during Playwright installation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Cleanup on exception
        _cleanup_playwright_browsers()
        return False, error_msg


async def ensure_playwright_installed() -> None:
    """
    Ensure Playwright browsers are installed, installing them if necessary.

    This function is idempotent and safe to call on every startup.
    It checks the database cache first to avoid unnecessary checks.

    On installation failure:
    - Cleans up the PLAYWRIGHT_BROWSERS_PATH directory
    - Logs a warning (does not raise exception)
    - Waits for next app restart or user intervention to retry
    """
    is_installed = await check_playwright_installation()

    if is_installed:
        logger.info("Playwright browsers already installed and verified")
        return

    logger.warning("Playwright browsers not detected, starting installation...")
    success, message = await install_playwright_browsers()

    if not success:
        logger.warning(
            f"Failed to install Playwright browsers: {message}. "
            f"The crawl service will not be available. "
            f"Installation will be retried on next app restart."
        )
        return

    logger.info(f"Playwright installation complete: {message}")
