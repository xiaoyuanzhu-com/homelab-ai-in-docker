#!/bin/bash

# Test: Text Generation - Gemma Chat
# Tests google/gemma-3-1b-it model for chat-style generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "text-generation/402-gemma-chat" "Gemma 3 1B chat-style generation"

# Test 1: Question answering
echo "Test 1: Question answering with Gemma"
make_api_request "POST" "/api/text-generation" '{
  "model_id": "google/gemma-3-1b-it",
  "prompt": "What are the three primary colors?",
  "max_new_tokens": 100
}'

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "generated_text" && \
assert_min_length "generated_text" 10 "generated text" && \
assert_field_not_empty "request_id"

# Display generated text
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Generated text:"
    echo "$RESPONSE_BODY" | grep -o '"generated_text":"[^"]*"' | cut -d'"' -f4
fi

# Test 2: Instructional prompt
echo ""
echo "Test 2: Instructional prompt with Gemma"
make_api_request "POST" "/api/text-generation" '{
  "model_id": "google/gemma-3-1b-it",
  "prompt": "Write a haiku about technology.",
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
