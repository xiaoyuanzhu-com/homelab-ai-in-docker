#!/bin/bash

# Test: Image Captioning - BLIP Base
# Tests BLIP base model for image captioning

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "image-captioning/201-blip-base" "BLIP base model captioning"

# Check if test image exists
IMAGE_PATH="${SCRIPT_DIR}/../../fixtures/images/sample-scene.jpg"
if [ ! -f "$IMAGE_PATH" ]; then
    # Fallback to OCR image if scene image not available
    IMAGE_PATH="${SCRIPT_DIR}/../../fixtures/images/sample-ocr.jpg"
    if [ ! -f "$IMAGE_PATH" ]; then
        print_result "SKIP" "Test image not found"
        echo "Please add a sample image for captioning testing. See fixtures/README.md"
        exit 0
    fi
fi

echo "Test 1: Caption image with BLIP base model"
upload_file "/api/image-captioning" "$IMAGE_PATH" "image" \
    "model_id=Salesforce/blip-image-captioning-base"

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "caption" && \
assert_min_length "caption" 5 "generated caption" && \
assert_field_not_empty "request_id"

# Display caption
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Generated caption:"
    echo "$RESPONSE_BODY" | grep -o '"caption":"[^"]*"' | cut -d'"' -f4
fi

print_summary
exit $?
