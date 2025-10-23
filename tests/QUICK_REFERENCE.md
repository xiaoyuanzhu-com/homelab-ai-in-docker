# Quick Reference - Integration Tests

## Test Commands

```bash
# Run all tests
./run-all.sh

# Run only API tests
./run-api-tests.sh

# Run only task tests
./run-task-tests.sh

# Run specific task type
./run-task-tests.sh image-ocr
./run-task-tests.sh image-captioning
./run-task-tests.sh text-embedding
./run-task-tests.sh text-generation
./run-task-tests.sh asr

# Run single test
bash api/001-health.sh
bash tasks/image-ocr/101-paddle-text.sh
```

## Test Numbering Scheme

Tests use globally unique 3-digit numbers:
- **000-099**: API tests
- **100-199**: Image OCR tests
- **200-299**: Image captioning tests
- **300-399**: Text embedding tests
- **400-499**: Text generation tests
- **500-599**: ASR tests

## Test Reference by Feature

### API Tests (000-099)

| # | Command | What it tests |
|---|---------|---------------|
| 001 | `bash api/001-health.sh` | /api/health, /api/ready, /api |
| 002 | `bash api/002-models-list.sh` | GET /api/models with filters |
| 003 | `bash api/003-history-stats.sh` | GET /api/history endpoints |
| 004 | `bash api/004-crawl.sh` | POST /api/crawl |

### Image OCR Tests (100-199)

| # | Command | Model | Output |
|---|---------|-------|--------|
| 101 | `bash tasks/image-ocr/101-paddle-text.sh` | MinerU2.5 | Text |
| 102 | `bash tasks/image-ocr/102-paddle-markdown.sh` | MinerU2.5 | Markdown |
| 103 | `bash tasks/image-ocr/103-deepseek-text.sh` | DeepSeek-OCR | Text |
| 104 | `bash tasks/image-ocr/104-deepseek-markdown.sh` | DeepSeek-OCR | Markdown |
| 105 | `bash tasks/image-ocr/105-granite-markdown.sh` | granite-docling | Markdown |

### Image Captioning Tests (200-299)

| # | Command | Model |
|---|---------|-------|
| 201 | `bash tasks/image-captioning/201-blip-base.sh` | blip-image-captioning-base |
| 202 | `bash tasks/image-captioning/202-blip-large.sh` | blip-image-captioning-large |
| 203 | `bash tasks/image-captioning/203-llava-detailed.sh` | llava-1.5-7b-hf-bnb-4bit |

### Text Embedding Tests (300-399)

| # | Command | Model |
|---|---------|-------|
| 301 | `bash tasks/text-embedding/301-minilm.sh` | all-MiniLM-L6-v2 |
| 302 | `bash tasks/text-embedding/302-bge-large.sh` | bge-large-en-v1.5 |

### Text Generation Tests (400-499)

| # | Command | Model |
|---|---------|-------|
| 401 | `bash tasks/text-generation/401-qwen-basic.sh` | Qwen3-0.6B |
| 402 | `bash tasks/text-generation/402-gemma-chat.sh` | gemma-3-1b-it |

### ASR Tests (500-599)

| # | Command | Model |
|---|---------|-------|
| 501 | `bash tasks/asr/501-whisper-turbo.sh` | whisper-large-v3-turbo |
| 502 | `bash tasks/asr/502-speaker-diarization.sh` | speaker-diarization-3.1 |

## AI Agent Usage Examples

### Example 1: Test a specific feature
```
Run the image OCR tests:
cd tests && ./run-task-tests.sh image-ocr
```

### Example 2: Test everything after changes
```
Run the full test suite:
cd tests && ./run-all.sh
```

### Example 3: Test a specific model
```
Test DeepSeek OCR with markdown output (test #104):
cd tests && bash tasks/image-ocr/104-deepseek-markdown.sh
```

### Example 4: Verify API is running
```
Quick health check (test #001):
cd tests && bash api/001-health.sh
```

## Environment Variables

```bash
# Custom API URL
TEST_API_URL=http://localhost:8000 ./run-all.sh

# Examples:
TEST_API_URL=http://192.168.1.100:12310 ./run-all.sh
TEST_API_URL=https://api.example.com ./run-all.sh
```

## Expected Output

### Success
```
========================================
Test: 001-health
Health and Readiness Checks
========================================

Test 1: GET /api/health
✓ PASS: HTTP status is 200
✓ PASS: Response contains field 'status'

========================================
Test Summary
========================================
Tests run:    3
Tests passed: 3
All tests passed!
```

### Skipped (missing fixture)
```
⊘ SKIP: Test image not found: fixtures/images/sample-ocr.jpg
Please add a sample image for OCR testing. See fixtures/README.md
```

### Failure
```
✗ FAIL: Expected HTTP 200, got 500
Response: {"error": "Internal server error"}
```

## Test Fixtures Required

Before running tests, add these files:

```bash
# Images (any format: jpg, png)
fixtures/images/sample-ocr.jpg      # Document/text image
fixtures/images/sample-scene.jpg    # Photo for captioning

# Audio (wav format preferred)
fixtures/audio/sample-asr.wav              # Speech recording
fixtures/audio/sample-conversation.wav     # Multi-speaker audio
```

See [fixtures/README.md](fixtures/README.md) for details.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API not ready | Check API is running: `curl http://localhost:12310/api/health` |
| Connection refused | Start API: `python main.py` |
| Tests skipped | Add test fixtures to `fixtures/` directory |
| Test timeout | Increase `TEST_TIMEOUT` in `test-config.sh` |
| Wrong model ID | Check available: `curl http://localhost:12310/api/models` |

## Quick Setup

```bash
# 1. Add test fixtures
cd tests/fixtures
# Add your images to images/
# Add your audio to audio/

# 2. Start API (in separate terminal)
cd /path/to/project
python main.py

# 3. Run tests
cd tests
./run-all.sh
```
