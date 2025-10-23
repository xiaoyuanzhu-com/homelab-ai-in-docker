# Integration Tests for Homelab AI Services

This directory contains curl-based integration tests designed to be easily used by AI agents and developers to verify API functionality.

## Overview

The test suite is organized into two main categories:

1. **API Tests** (`api/`) - Tests for general API endpoints (health, models, history, crawl)
2. **Task Tests** (`tasks/`) - Tests for model-based functionality (OCR, captioning, embeddings, etc.)

## Quick Start

### Prerequisites

1. **Running API Server**: Ensure your API is running on `http://localhost:12310` (default)
2. **Test Fixtures**: Add sample images and audio files (see [fixtures/README.md](fixtures/README.md))

### Running Tests

```bash
# Run all tests (API + Tasks)
./run-all.sh

# Run only API tests
./run-api-tests.sh

# Run only task tests
./run-task-tests.sh

# Run specific task tests
./run-task-tests.sh image-ocr
./run-task-tests.sh image-captioning
./run-task-tests.sh text-embedding
```

### Custom API URL

If your API runs on a different URL:

```bash
TEST_API_URL=http://localhost:8000 ./run-all.sh
```

## Directory Structure

```
tests/
├── README.md                   # This file
├── test-config.sh              # Shared configuration
├── test-utils.sh               # Helper functions
├── fixtures/                   # Test assets
│   ├── README.md              # Instructions for adding fixtures
│   ├── images/                # Sample images for OCR/captioning
│   └── audio/                 # Sample audio for ASR
├── api/                       # API endpoint tests (000-099)
│   ├── 001-health.sh          # Health & readiness checks
│   ├── 002-models-list.sh     # GET /api/models
│   ├── 003-history-stats.sh   # GET /api/history
│   └── 004-crawl.sh           # POST /api/crawl
├── tasks/                     # Model functionality tests
│   ├── image-ocr/             # OCR tests (100-199)
│   │   ├── 101-paddle-text.sh
│   │   ├── 102-paddle-markdown.sh
│   │   ├── 103-deepseek-text.sh
│   │   ├── 104-deepseek-markdown.sh
│   │   └── 105-granite-markdown.sh
│   ├── image-captioning/      # Captioning tests (200-299)
│   │   ├── 201-blip-base.sh
│   │   ├── 202-blip-large.sh
│   │   └── 203-llava-detailed.sh
│   ├── text-embedding/        # Embedding tests (300-399)
│   │   ├── 301-minilm.sh
│   │   └── 302-bge-large.sh
│   ├── text-generation/       # Generation tests (400-499)
│   │   ├── 401-qwen-basic.sh
│   │   └── 402-gemma-chat.sh
│   └── asr/                   # ASR tests (500-599)
│       ├── 501-whisper-turbo.sh
│       └── 502-speaker-diarization.sh
├── run-all.sh                 # Run all tests
├── run-api-tests.sh           # Run API tests only
└── run-task-tests.sh          # Run task tests only
```

## Test Naming Convention

Tests use a globally unique 3-digit numbering scheme:

- **3-digit prefix** (000-999): Globally unique test number
- **Descriptive name**: What the test does
- **Category ranges**:
  - **000-099**: API tests
  - **100-199**: Image OCR tests
  - **200-299**: Image captioning tests
  - **300-399**: Text embedding tests
  - **400-499**: Text generation tests
  - **500-599**: ASR tests

### Examples:

- `001-health.sh` - API health checks (test #001)
- `image-ocr/101-paddle-text.sh` - OCR with Paddle, text output (test #101)
- `image-ocr/102-paddle-markdown.sh` - OCR with Paddle, markdown (test #102)
- `text-embedding/301-minilm.sh` - MiniLM embedding (test #301)

This makes it easy to reference tests: "run test 101" or "run test image-ocr/101-paddle-text.sh"

## Test Philosophy

These tests are designed to be **"loose"** and flexible:

✅ **What we check:**
- HTTP 200 status codes
- Required fields exist in response
- Non-empty meaningful output
- Minimum content length

❌ **What we DON'T check:**
- Exact output matches
- Specific word counts
- Precise formatting
- Model accuracy

**Why?** Because we want to verify the API is working, not judge model quality.

## AI Agent Usage

These tests are designed to be easily referenced and executed by AI agents:

### Reference Format

```
Run test: tests/api/01-health.sh
Run test: tests/tasks/image-ocr/01-paddle-text.sh
Run all OCR tests: tests/run-task-tests.sh image-ocr
```

### Example AI Agent Instructions

**For testing a specific feature:**
```
Please run the image OCR tests to verify the API is working:
cd tests && ./run-task-tests.sh image-ocr
```

**For testing after changes:**
```
I've made changes to the API. Please run the full test suite:
cd tests && ./run-all.sh
```

**For debugging a specific model:**
```
Test the DeepSeek OCR with text output:
cd tests && bash tasks/image-ocr/03-deepseek-text.sh
```

## Test Output

Each test provides clear, colorized output:

```
========================================
Test: 01-health
Health and Readiness Checks
========================================

Test 1: GET /api/health
✓ PASS: HTTP status is 200
✓ PASS: Response contains field 'status'
✓ PASS: Response contains 'healthy'

========================================
Test Summary
========================================
Tests run:    3
Tests passed: 3
All tests passed!
```

## Adding New Tests

### 1. Create a New Test Script

```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "test-name" "Test Description"

# Your test logic here
make_api_request "POST" "/api/endpoint" '{
  "key": "value"
}'

assert_status_200 && \
assert_field_exists "field_name" && \
assert_field_not_empty "field_name"

print_summary
exit $?
```

### 2. Make it Executable

```bash
chmod +x your-test.sh
```

### 3. Run It

```bash
./your-test.sh
```

## Available Assertions

The `test-utils.sh` provides these helper functions:

- `make_api_request(method, endpoint, data)` - Make an API call
- `upload_file(endpoint, file_path, field_name, ...)` - Upload a file
- `assert_status_200()` - Check HTTP 200
- `assert_field_exists(field)` - Check field exists
- `assert_field_not_empty(field)` - Check field has value
- `assert_array_not_empty(field)` - Check array has items
- `assert_min_length(field, length)` - Check minimum text length
- `assert_contains(field, substring)` - Check for substring
- `print_result(status, message)` - Print test result
- `print_summary()` - Print test summary

## Test Fixtures

Tests rely on sample assets in the `fixtures/` directory:

### Required Fixtures:

- `fixtures/images/sample-ocr.jpg` - Image with text (for OCR tests)
- `fixtures/images/sample-scene.jpg` - Scene photo (for captioning tests)
- `fixtures/audio/sample-asr.wav` - Audio recording (for ASR tests)
- `fixtures/audio/sample-conversation.wav` - Multi-speaker audio (for diarization)

See [fixtures/README.md](fixtures/README.md) for instructions on adding fixtures.

## Troubleshooting

### Tests are skipped

**Issue:** Test shows "SKIP: Test image not found"

**Solution:** Add the required test fixtures to `fixtures/images/` or `fixtures/audio/`

### API not ready

**Issue:** "API not ready after 30 seconds"

**Solution:**
1. Check if the API is running: `curl http://localhost:12310/api/health`
2. Start the API: `cd /path/to/project && python main.py`
3. Check the API URL: `TEST_API_URL=http://localhost:8000 ./run-all.sh`

### Connection refused

**Issue:** "curl: (7) Failed to connect"

**Solution:** Ensure the API server is running and accessible on the specified port.

### Test timeout

**Issue:** Test hangs or times out

**Solution:**
1. Check if the model needs to be downloaded first
2. Increase timeout: Edit `test-config.sh` and increase `TEST_TIMEOUT`
3. Check API logs for errors

### Model not found

**Issue:** "Model '...' not found in database"

**Solution:**
1. Verify the model ID is correct in the test script
2. Check available models: `curl http://localhost:12310/api/models`
3. Update the test script with the correct model ID

## Configuration

Edit `test-config.sh` to customize:

- `API_URL` - Default API URL (override with `TEST_API_URL` env var)
- `TEST_TIMEOUT` - Request timeout in seconds (default: 300)
- Color settings for output

## CI/CD Integration

These tests are ideal for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    cd tests
    ./run-all.sh
  env:
    TEST_API_URL: http://localhost:12310
```

## Contributing

When adding new tests:

1. Follow the naming convention (numerical prefix + descriptive name)
2. Use the shared utility functions
3. Keep tests "loose" - check for API functionality, not model accuracy
4. Add clear comments and descriptions
5. Make tests self-contained and independent
6. Handle missing fixtures gracefully (skip with helpful message)

## License

These tests are part of the Homelab AI Services project.
