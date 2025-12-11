#!/bin/bash
# Post-install script for crawl4ai environment
# Ensures Playwright browsers are installed for web crawling

set -e

# Get the expected browser path from playwright
# Playwright stores browsers in ~/.cache/ms-playwright/ by default
# or in PLAYWRIGHT_BROWSERS_PATH if set
BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"

# Check if chromium is already installed by looking for the executable
# Playwright creates directories like chromium-1234/ with the browser inside
if ls "$BROWSERS_PATH"/chromium*/chrome-linux/chrome 2>/dev/null || \
   ls "$BROWSERS_PATH"/chromium_headless_shell*/chrome-headless-shell-linux64/chrome-headless-shell 2>/dev/null; then
    echo "Playwright chromium already installed, skipping"
    exit 0
fi

echo "Installing Playwright chromium..."
playwright install chromium

echo "Crawl4AI post-install complete"
