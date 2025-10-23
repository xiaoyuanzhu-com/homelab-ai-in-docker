#!/bin/bash

# Test: Image Captioning - LLaVA Detailed
# Tests LLaVA model with custom prompt for detailed captions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "image-captioning/203-llava-detailed" "LLaVA with detailed description prompt"

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

echo "Test 1: Detailed caption with LLaVA model and custom prompt"
upload_file "/api/image-captioning" "$IMAGE_PATH" "image" \
    "model_id=unsloth/llava-1.5-7b-hf-bnb-4bit" \
    "prompt=USER: <image>\nDescribe this image in detail, including colors, objects, and any text visible.\nASSISTANT:"

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "caption" && \
assert_min_length "caption" 10 "generated caption" && \
assert_field_not_empty "request_id"

# Display caption
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Generated detailed caption:"
    echo "$RESPONSE_BODY" | grep -o '"caption":"[^"]*"' | cut -d'"' -f4
fi

print_summary
exit $?
