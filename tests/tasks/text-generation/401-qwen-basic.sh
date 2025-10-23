#!/bin/bash

# Test: Text Generation - Qwen Basic
# Tests Qwen/Qwen3-0.6B model for text generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "text-generation/401-qwen-basic" "Qwen3-0.6B basic text generation"

# Test 1: Simple completion
echo "Test 1: Basic text generation with Qwen"
make_api_request "POST" "/api/text-generation" '{
  "model_id": "Qwen/Qwen3-0.6B",
  "prompt": "The capital of France is",
  "max_new_tokens": 50
}'

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "generated_text" && \
assert_min_length "generated_text" 5 "generated text" && \
assert_field_not_empty "request_id"

# Display generated text
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Generated text:"
    echo "$RESPONSE_BODY" | grep -o '"generated_text":"[^"]*"' | cut -d'"' -f4
fi

# Test 2: Story continuation
echo ""
echo "Test 2: Story continuation with Qwen"
make_api_request "POST" "/api/text-generation" '{
  "model_id": "Qwen/Qwen3-0.6B",
  "prompt": "Once upon a time, there was a",
  "max_new_tokens": 100
}'

assert_status_200 && \
assert_field_exists "generated_text" && \
assert_min_length "generated_text" 10 "generated text"

# Display generated text
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Generated text:"
    echo "$RESPONSE_BODY" | grep -o '"generated_text":"[^"]*"' | cut -d'"' -f4
fi

print_summary
exit $?
