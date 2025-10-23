#!/bin/bash

# Test: Web Crawl API
# Tests the POST /api/crawl endpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../test-utils.sh"

print_test_header "004-crawl" "POST /api/crawl - Web scraping with JS rendering"

# Test 1: Basic crawl (example.com)
echo "Test 1: Crawl example.com"
make_api_request "POST" "/api/crawl" '{
  "url": "https://example.com"
}'
assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "markdown" && \
assert_min_length "markdown" 50 && \
assert_field_not_empty "request_id"

# Test 2: Crawl with screenshot disabled
echo ""
echo "Test 2: Crawl with screenshot=false"
make_api_request "POST" "/api/crawl" '{
  "url": "https://example.com",
  "screenshot": false
}'
assert_status_200 && \
assert_field_exists "markdown" && \
assert_min_length "markdown" 50

# Test 3: Crawl with custom timeout
echo ""
echo "Test 3: Crawl with custom page_timeout"
make_api_request "POST" "/api/crawl" '{
  "url": "https://example.com",
  "page_timeout": 30000
}'
assert_status_200 && \
assert_field_exists "markdown" && \
assert_min_length "markdown" 50

print_summary
exit $?
