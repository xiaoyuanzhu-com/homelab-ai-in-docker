#!/bin/bash

# Test Configuration
# Shared configuration for all integration tests

# API Base URL - override with TEST_API_URL environment variable
API_URL="${TEST_API_URL:-http://localhost:12310}"

# Test timeout in seconds
TEST_TIMEOUT=300

# Colors for output
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

# Test results tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
