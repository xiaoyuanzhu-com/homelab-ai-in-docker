#!/bin/bash

# Test: Image OCR - Paddle (Text Output)
# Tests PaddleOCR model with plain text output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "image-ocr/101-paddle-text" "PaddleOCR with text output"

# Check if test image exists
IMAGE_PATH="${SCRIPT_DIR}/../../fixtures/images/sample-ocr.jpg"
if [ ! -f "$IMAGE_PATH" ]; then
    print_result "SKIP" "Test image not found: $IMAGE_PATH"
    echo "Please add a sample image for OCR testing. See fixtures/README.md"
    exit 0
fi

# Note: Using a hypothetical PaddleOCR model ID - adjust based on actual manifest
# If you're using a different OCR model, update the model_id accordingly

echo "Test 1: OCR with PaddleOCR (text output)"
upload_file "/api/image-ocr" "$IMAGE_PATH" "image" \
    "model_id=opendatalab/MinerU2.5-2509-1.2B" \
    "output_format=text"

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "text" && \
assert_min_length "text" 5 "extracted text" && \
assert_field_not_empty "request_id"

# Display extracted text preview
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Extracted text preview:"
    echo "$RESPONSE_BODY" | grep -o '"text":"[^"]*"' | cut -d'"' -f4 | head -c 200
    echo "..."
fi

print_summary
exit $?
