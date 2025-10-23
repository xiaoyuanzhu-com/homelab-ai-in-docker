#!/bin/bash

# Test: Image OCR - Granite Docling (Markdown Output)
# Tests IBM Granite Docling model with markdown output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "image-ocr/105-granite-markdown" "Granite Docling with markdown output"

# Check if test image exists
IMAGE_PATH="${SCRIPT_DIR}/../../fixtures/images/sample-ocr.jpg"
if [ ! -f "$IMAGE_PATH" ]; then
    print_result "SKIP" "Test image not found: $IMAGE_PATH"
    echo "Please add a sample image for OCR testing. See fixtures/README.md"
    exit 0
fi

echo "Test 1: OCR with Granite Docling (markdown output)"
upload_file "/api/image-ocr" "$IMAGE_PATH" "image" \
    "model_id=ibm-granite/granite-docling-258M" \
    "output_format=markdown"

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "markdown" && \
assert_min_length "markdown" 5 "extracted markdown" && \
assert_field_not_empty "request_id"

# Display markdown preview
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Extracted markdown preview:"
    echo "$RESPONSE_BODY" | grep -o '"markdown":"[^"]*"' | cut -d'"' -f4 | head -c 200
    echo "..."
fi

print_summary
exit $?
