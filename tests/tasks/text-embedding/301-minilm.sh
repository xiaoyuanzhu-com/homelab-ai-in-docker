#!/bin/bash

# Test: Text Embedding - MiniLM
# Tests sentence-transformers/all-MiniLM-L6-v2 model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "text-embedding/301-minilm" "MiniLM embedding model"

# Test 1: Single text embedding
echo "Test 1: Embed single text with MiniLM"
make_api_request "POST" "/api/text-to-embedding" '{
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "text": "This is a test sentence for embedding."
}'

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "embedding" && \
assert_array_not_empty "embedding" && \
assert_field_not_empty "request_id"

# Test 2: Longer text embedding
echo ""
echo "Test 2: Embed longer text with MiniLM"
make_api_request "POST" "/api/text-to-embedding" '{
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "text": "The quick brown fox jumps over the lazy dog. This is a longer sentence that contains more semantic information for the embedding model to process."
}'

assert_status_200 && \
assert_field_exists "embedding" && \
assert_array_not_empty "embedding"

print_summary
exit $?
