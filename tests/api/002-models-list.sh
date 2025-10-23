#!/bin/bash

# Test: Models List API
# Tests the GET /api/models endpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../test-utils.sh"

print_test_header "002-models-list" "GET /api/models - List all available models"

# Test 1: Get all models
echo "Test 1: GET /api/models (all models)"
make_api_request "GET" "/api/models"
assert_status_200 && \
assert_field_exists "models" && \
assert_array_not_empty "models"

# Test 2: Get models by task (image-ocr)
echo ""
echo "Test 2: GET /api/models?task=image-ocr"
make_api_request "GET" "/api/models?task=image-ocr"
assert_status_200 && \
assert_field_exists "models" && \
assert_array_not_empty "models"

# Test 3: Get models by task (image-captioning)
echo ""
echo "Test 3: GET /api/models?task=image-captioning"
make_api_request "GET" "/api/models?task=image-captioning"
assert_status_200 && \
assert_field_exists "models" && \
assert_array_not_empty "models"

# Test 4: Get models by task (feature-extraction)
echo ""
echo "Test 4: GET /api/models?task=feature-extraction"
make_api_request "GET" "/api/models?task=feature-extraction"
assert_status_200 && \
assert_field_exists "models" && \
assert_array_not_empty "models"

# Test 5: Get models by task (automatic-speech-recognition)
echo ""
echo "Test 5: GET /api/models?task=automatic-speech-recognition"
make_api_request "GET" "/api/models?task=automatic-speech-recognition"
assert_status_200 && \
assert_field_exists "models" && \
assert_array_not_empty "models"

print_summary
exit $?
