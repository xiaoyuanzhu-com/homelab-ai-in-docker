#!/bin/bash

# Test: History and Stats API
# Tests the /api/history endpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../test-utils.sh"

print_test_header "003-history-stats" "GET /api/history - Task history and statistics"

# Test 1: Get statistics
echo "Test 1: GET /api/history/stats"
make_api_request "GET" "/api/history/stats"
assert_status_200 && \
assert_field_exists "running" && \
assert_field_exists "today" && \
assert_field_exists "total"

# Test 2: Get recent history (limit 10)
echo ""
echo "Test 2: GET /api/history?limit=10"
make_api_request "GET" "/api/history?limit=10"
assert_status_200 && \
assert_field_exists "history"

# Test 3: Get recent history (limit 5)
echo ""
echo "Test 3: GET /api/history?limit=5"
make_api_request "GET" "/api/history?limit=5"
assert_status_200 && \
assert_field_exists "history"

print_summary
exit $?
