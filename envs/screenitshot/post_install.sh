#!/bin/bash
# Post-install script for screenitshot environment
# Ensures Playwright browsers are installed for document rendering

set -e

# Use playwright's built-in check - it knows exactly which version it needs
# This is more reliable than manually checking browser paths
echo "Checking Playwright browser status..."

# playwright install is idempotent - it skips if the correct version is already installed
# Running it unconditionally is simpler and more reliable than version detection
playwright install chromium

echo "ScreenItShot post-install complete"
