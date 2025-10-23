#!/bin/bash

# Test: Text Embedding - BGE Large
# Tests BAAI/bge-large-en-v1.5 model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "text-embedding/302-bge-large" "BGE Large embedding model"

# Test 1: Single text embedding
echo "Test 1: Embed single text with BGE Large"
make_api_request "POST" "/api/text-to-embedding" '{
  "model_id": "BAAI/bge-large-en-v1.5",
  "text": "This is a test sentence for embedding with BGE large model."
}'

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "embedding" && \
assert_array_not_empty "embedding" && \
assert_field_not_empty "request_id"

# Test 2: Technical text embedding
echo ""
echo "Test 2: Embed technical text with BGE Large"
make_api_request "POST" "/api/text-to-embedding" '{
  "model_id": "BAAI/bge-large-en-v1.5",
  "text": "Machine learning models use neural networks to process data and generate embeddings that capture semantic meaning."
}'

assert_status_200 && \
assert_field_exists "embedding" && \
assert_array_not_empty "embedding"

print_summary
exit $?
