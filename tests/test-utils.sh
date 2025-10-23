#!/bin/bash

# Test Utilities
# Shared helper functions for integration tests

# Source the test config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test-config.sh"

# Print test header
print_test_header() {
    local test_name="$1"
    local description="$2"

    echo ""
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}Test: ${test_name}${COLOR_RESET}"
    echo -e "${COLOR_BLUE}${description}${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo ""
}

# Print test result
print_result() {
    local status="$1"
    local message="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ "$status" = "PASS" ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo -e "${COLOR_GREEN}✓ PASS${COLOR_RESET}: $message"
    elif [ "$status" = "FAIL" ]; then
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo -e "${COLOR_RED}✗ FAIL${COLOR_RESET}: $message"
    elif [ "$status" = "SKIP" ]; then
        echo -e "${COLOR_YELLOW}⊘ SKIP${COLOR_RESET}: $message"
    fi
}

# Make API request and capture response
# Returns: status_code, response_body, request_id
make_api_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local content_type="${4:-application/json}"

    local url="${API_URL}${endpoint}"
    local response_file=$(mktemp)
    local headers_file=$(mktemp)

    # Make request
    if [ "$method" = "GET" ]; then
        http_code=$(curl -s -w "%{http_code}" \
            -o "$response_file" \
            -D "$headers_file" \
            --max-time "$TEST_TIMEOUT" \
            "$url")
    else
        http_code=$(curl -s -w "%{http_code}" \
            -X "$method" \
            -H "Content-Type: $content_type" \
            -d "$data" \
            -o "$response_file" \
            -D "$headers_file" \
            --max-time "$TEST_TIMEOUT" \
            "$url")
    fi

    response_body=$(cat "$response_file")

    # Extract request_id from response if present
    request_id=$(echo "$response_body" | grep -o '"request_id":"[^"]*"' | cut -d'"' -f4)

    # Clean up temp files
    rm -f "$response_file" "$headers_file"

    # Export for caller
    export HTTP_CODE="$http_code"
    export RESPONSE_BODY="$response_body"
    export REQUEST_ID="$request_id"
}

# Upload file via multipart form
upload_file() {
    local endpoint="$1"
    local file_path="$2"
    local field_name="${3:-file}"
    shift 3
    local extra_fields=("$@")

    local url="${API_URL}${endpoint}"
    local response_file=$(mktemp)
    local headers_file=$(mktemp)

    # Build curl command
    local curl_cmd="curl -s -w %{http_code} -o $response_file -D $headers_file --max-time $TEST_TIMEOUT"
    curl_cmd="$curl_cmd -F ${field_name}=@${file_path}"

    # Add extra fields
    for field in "${extra_fields[@]}"; do
        curl_cmd="$curl_cmd -F $field"
    done

    curl_cmd="$curl_cmd $url"

    http_code=$(eval "$curl_cmd")
    response_body=$(cat "$response_file")
    request_id=$(echo "$response_body" | grep -o '"request_id":"[^"]*"' | cut -d'"' -f4)

    rm -f "$response_file" "$headers_file"

    export HTTP_CODE="$http_code"
    export RESPONSE_BODY="$response_body"
    export REQUEST_ID="$request_id"
}

# Check if HTTP status code is 200
assert_status_200() {
    if [ "$HTTP_CODE" = "200" ]; then
        print_result "PASS" "HTTP status is 200"
        return 0
    else
        print_result "FAIL" "Expected HTTP 200, got $HTTP_CODE"
        echo "Response: $RESPONSE_BODY"
        return 1
    fi
}

# Check if response contains a field
assert_field_exists() {
    local field="$1"
    local field_name="${2:-$field}"

    if echo "$RESPONSE_BODY" | grep -q "\"$field\""; then
        print_result "PASS" "Response contains field '$field_name'"
        return 0
    else
        print_result "FAIL" "Response missing field '$field_name'"
        echo "Response: $RESPONSE_BODY"
        return 1
    fi
}

# Check if field value is not empty
assert_field_not_empty() {
    local field="$1"
    local field_name="${2:-$field}"

    local value=$(echo "$RESPONSE_BODY" | grep -o "\"$field\":\"[^\"]*\"" | cut -d'"' -f4)

    if [ -n "$value" ] && [ "$value" != "null" ]; then
        print_result "PASS" "Field '$field_name' is not empty: ${value:0:50}..."
        return 0
    else
        print_result "FAIL" "Field '$field_name' is empty or null"
        echo "Response: $RESPONSE_BODY"
        return 1
    fi
}

# Check if field is an array with items
assert_array_not_empty() {
    local field="$1"
    local field_name="${2:-$field}"

    # Extract array and check if it has elements
    if echo "$RESPONSE_BODY" | grep -q "\"$field\":\["; then
        # Check if array has at least one element (contains a '{' after the '[')
        if echo "$RESPONSE_BODY" | grep -A1 "\"$field\":\[" | grep -q "[{\"]"; then
            print_result "PASS" "Array '$field_name' is not empty"
            return 0
        else
            print_result "FAIL" "Array '$field_name' is empty"
            return 1
        fi
    else
        print_result "FAIL" "Field '$field_name' is not an array"
        echo "Response: $RESPONSE_BODY"
        return 1
    fi
}

# Check minimum text length
assert_min_length() {
    local field="$1"
    local min_length="$2"
    local field_name="${3:-$field}"

    local value=$(echo "$RESPONSE_BODY" | grep -o "\"$field\":\"[^\"]*\"" | cut -d'"' -f4)
    local actual_length=${#value}

    if [ "$actual_length" -ge "$min_length" ]; then
        print_result "PASS" "Field '$field_name' length ($actual_length) >= $min_length"
        return 0
    else
        print_result "FAIL" "Field '$field_name' length ($actual_length) < $min_length"
        echo "Value: $value"
        return 1
    fi
}

# Check if response contains substring
assert_contains() {
    local field="$1"
    local substring="$2"
    local field_name="${3:-$field}"

    if echo "$RESPONSE_BODY" | grep -q "$substring"; then
        print_result "PASS" "Response contains '$substring'"
        return 0
    else
        print_result "FAIL" "Response does not contain '$substring'"
        echo "Response: $RESPONSE_BODY"
        return 1
    fi
}

# Print summary
print_summary() {
    echo ""
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}Test Summary${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "Tests run:    $TESTS_RUN"
    echo -e "${COLOR_GREEN}Tests passed: $TESTS_PASSED${COLOR_RESET}"

    if [ "$TESTS_FAILED" -gt 0 ]; then
        echo -e "${COLOR_RED}Tests failed: $TESTS_FAILED${COLOR_RESET}"
        return 1
    else
        echo -e "${COLOR_GREEN}All tests passed!${COLOR_RESET}"
        return 0
    fi
}

# Wait for API to be ready
wait_for_api() {
    local max_attempts=30
    local attempt=0

    echo -n "Waiting for API to be ready..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s --max-time 2 "${API_URL}/api/health" > /dev/null 2>&1; then
            echo -e " ${COLOR_GREEN}✓${COLOR_RESET}"
            return 0
        fi

        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e " ${COLOR_RED}✗${COLOR_RESET}"
    echo -e "${COLOR_RED}API not ready after ${max_attempts} seconds${COLOR_RESET}"
    return 1
}
