#!/bin/bash

# Test: Health and Readiness Checks
# Tests the /api/health and /api/ready endpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../test-utils.sh"

print_test_header "001-health" "Health and Readiness Checks"

# Test 1: Health Check
echo "Test 1: GET /api/health"
make_api_request "GET" "/api/health"
assert_status_200 && \
assert_field_exists "status" && \
assert_contains "status" "healthy"

# Test 2: Readiness Check
echo ""
echo "Test 2: GET /api/ready"
make_api_request "GET" "/api/ready"
assert_status_200 && \
assert_field_exists "status" && \
assert_field_exists "services"

# Test 3: Root API Info
echo ""
echo "Test 3: GET /api"
make_api_request "GET" "/api"
assert_status_200 && \
assert_field_exists "name" && \
assert_field_exists "version" && \
assert_field_exists "endpoints"

print_summary
exit $?
