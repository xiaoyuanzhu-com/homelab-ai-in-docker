#!/bin/bash
# Post-install script for crawl4ai environment
# Ensures Playwright browsers are installed for web crawling

set -e

# Use playwright's built-in check - it knows exactly which version it needs
# This is more reliable than manually checking browser paths
echo "Checking Playwright browser status..."

# playwright install is idempotent - it skips if the correct version is already installed
# Running it unconditionally is simpler and more reliable than version detection
playwright install chromium

echo "Crawl4AI post-install complete"
