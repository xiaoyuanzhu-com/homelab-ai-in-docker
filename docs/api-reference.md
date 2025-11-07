# API Reference

Complete reference for all AI skill endpoints in Homelab AI Services API.

## Base URL

All endpoints are prefixed with `/api`:

- **Development**: `http://localhost:12310/api`
- **Production**: `http://your-domain:12310/api`

## Common Response Fields

All responses include these base fields:

```json
{
  "request_id": "uuid-string",
  "processing_time_ms": 123,
  ...
}
```

## Authentication

Currently no authentication required. For production deployments, use reverse proxy authentication or firewall restrictions.

---

## Text Generation

Generate text from prompts using language models.

### Endpoint

```
POST /api/text-generation
```

### Request

```json
{
  "prompt": "Write a haiku about coding",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Input prompt for text generation (min length: 1) |
| `model` | string | Yes | Model ID to use (see available models below) |
| `max_new_tokens` | integer | No | Maximum tokens to generate (default: 256, range: 1-4096) |
| `temperature` | float | No | Sampling temperature (default: 0.7, range: 0.0-2.0) |
| `top_p` | float | No | Nucleus sampling threshold (default: 0.9, range: 0.0-1.0) |
| `do_sample` | boolean | No | Use sampling vs greedy decoding (default: true) |

**Available Models:**
- `Qwen/Qwen2.5-0.5B-Instruct` - Lightweight model (0.5B parameters)
- `Qwen/Qwen2.5-1.5B-Instruct` - Mid-size model (1.5B parameters)
- `Qwen/Qwen2.5-3B-Instruct` - Larger model (3B parameters)

### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 1250,
  "generated_text": "Code flows like water,\nBugs hide in silent shadows,\nDebug brings the light.",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "tokens_generated": 24
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `generated_text` | string | Generated text output |
| `model` | string | Model that generated the text |
| `tokens_generated` | integer | Number of tokens generated |

---

## Text Embedding

Generate vector embeddings for semantic search and similarity.

### Endpoint

```
POST /api/text-to-embedding
```

### Request

```json
{
  "texts": ["Hello world", "Good morning"],
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | array[string] | Yes | List of texts to embed |
| `model` | string | No | Model ID (see available models below) |

**Available Models:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, 384 dimensions
- `BAAI/bge-large-en-v1.5` - High quality, 1024 dimensions
- `Alibaba-NLP/gte-large-en-v1.5` - High quality, 1024 dimensions

### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 45,
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimensions": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

## Image Captioning

Generate natural language descriptions of images.

### Endpoint

```
POST /api/image-captioning
```

### Request

```json
{
  "image": "base64-encoded-image-data",
  "model": "Salesforce/blip-image-captioning-base"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded image data |
| `model` | string | No | Model ID (default: blip-image-captioning-base) |

**Available Models:**
- `Salesforce/blip-image-captioning-base` - Fast and accurate
- `Salesforce/blip-image-captioning-large` - Higher quality

### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 320,
  "caption": "a cat sitting on a wooden table",
  "model": "Salesforce/blip-image-captioning-base"
}
```

---

## Image OCR

Extract text from images using optical character recognition.

### Endpoint

```
POST /api/image-ocr
```

### Request

```json
{
  "image": "base64-encoded-image-data",
  "model": "deepseek-ai/DeepSeek-OCR",
  "output_format": "markdown"
}
```

or using a library engine:

```json
{
  "image": "base64-encoded-image-data",
  "lib": "paddleocr/pp-ocrv5",
  "language": "en",
  "output_format": "text"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded image data |
| `model` | string | No | Model ID (use `/api/models?task=image-ocr`) |
| `lib` | string | No | Lib ID (use `/api/libs?task=image-ocr`) |
| `language` | string | No | Language hint ('en', 'zh', 'auto', etc.) |
| `output_format` | string | No | 'text' or 'markdown' (default: 'text') |

Exactly one of `model` or `lib` must be provided.
- `PaddlePaddle/PaddleOCR-VL` - Vision-language OCR with markdown support
- `deepseek-ai/DeepSeek-VL2-Tiny` - Advanced OCR with markdown support

### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 890,
  "text": "Extracted text from the image",
  "model": "PaddlePaddle/PaddleOCR",
  "output_format": "text"
}
```

---

## Automatic Speech Recognition

Transcribe audio files to text or perform speaker diarization.

### Endpoint

```
POST /api/automatic-speech-recognition
```

### Request (Transcription)

```json
{
  "audio": "base64-encoded-audio-data",
  "model": "openai/whisper-large-v3-turbo",
  "output_format": "transcription",
  "language": "en",
  "return_timestamps": false
}
```

### Request (Speaker Diarization)

```json
{
  "audio": "base64-encoded-audio-data",
  "model": "pyannote/speaker-diarization-3.1",
  "output_format": "diarization",
  "min_speakers": 2,
  "max_speakers": 4
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | Yes | Base64-encoded audio (mp3, mp4, mpeg, mpga, m4a, wav, webm) |
| `model` | string | No | Model ID (default: whisper-large-v3-turbo) |
| `output_format` | string | No | 'transcription' or 'diarization' (default: 'transcription') |
| `language` | string | No | Language code for transcription ('en', 'zh', 'es', etc.) |
| `return_timestamps` | boolean | No | Return word-level timestamps (transcription only) |
| `min_speakers` | integer | No | Minimum speakers (diarization only, ≥1) |
| `max_speakers` | integer | No | Maximum speakers (diarization only, ≥1) |
| `num_speakers` | integer | No | Exact speaker count if known (diarization only, ≥1) |

**Available Models:**
- **Transcription:**
  - `openai/whisper-large-v3-turbo` - Fast and accurate
  - `openai/whisper-large-v3` - Highest quality
  - `openai/whisper-medium` - Balanced speed/quality
  - `openai/whisper-small` - Lightweight
- **Diarization:**
  - `pyannote/speaker-diarization-3.1` - Speaker segmentation

### Response (Transcription)

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 3200,
  "text": "This is the transcribed text from the audio file.",
  "model": "openai/whisper-large-v3-turbo",
  "language": "en",
  "chunks": null
}
```

### Response (Diarization)

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 5400,
  "text": null,
  "model": "pyannote/speaker-diarization-3.1",
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "SPEAKER_00"
    },
    {
      "start": 3.5,
      "end": 7.8,
      "speaker": "SPEAKER_01"
    }
  ],
  "num_speakers": 2
}
```

---

## WhisperX (Aligned ASR + Diarization)

High-quality Whisper transcription with word-level alignment and optional diarization.

### Endpoint

```
POST /api/whisperx/transcribe
```

### Request

```json
{
  "audio": "base64-encoded-audio-data",
  "asr_model": "large-v3",
  "language": "en",
  "diarize": true,
  "batch_size": 16,
  "compute_type": "float16"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | Yes | Base64-encoded audio (mp3, mp4, mpeg, mpga, m4a, wav, webm) |
| `asr_model` | string | No | WhisperX model id (e.g., `large-v3`, `small.en`); default `large-v3` |
| `language` | string | No | Language code; if omitted, WhisperX will detect |
| `diarize` | boolean | No | If true, assigns speakers using pyannote integration |
| `batch_size` | integer | No | Inference batch size (default: 16) |
| `compute_type` | string | No | Compute type hint (`float16`, `float32`); auto-selected if omitted |

### Response

```json
{
  "request_id": "uuid",
  "processing_time_ms": 4120,
  "text": "Hello everyone, welcome...",
  "language": "en",
  "model": "large-v3",
  "segments": [
    {
      "start": 0.12,
      "end": 2.34,
      "text": "Hello everyone",
      "speaker": "SPEAKER_00",
      "words": [
        { "word": "Hello", "start": 0.12, "end": 0.52, "speaker": "SPEAKER_00" },
        { "word": "everyone", "start": 0.60, "end": 1.20, "speaker": "SPEAKER_00" }
      ]
    }
  ]
}
```

Notes
- Diarization uses your configured HuggingFace token (Settings → `hf_token`).
- This endpoint is independent of the legacy `/automatic-speech-recognition` route and can replace it.

---

## Speaker Embedding and Matching (Stateless)

Compute speaker embeddings and perform stateless matching against an app‑managed registry.

### Extract Single Embedding

```
POST /api/speaker-embedding/extract
```

```json
{
  "audio": "base64-audio",
  "model": "pyannote/embedding",
  "mode": "whole"
}
```

Response

```json
{
  "request_id": "uuid",
  "processing_time_ms": 210,
  "embedding": [0.12, -0.03, ...],
  "dimension": 192,
  "model": "pyannote/embedding"
}
```

### Extract Batch Embeddings (Segments)

```
POST /api/speaker-embedding/batch-extract
```

```json
{
  "audio": "base64-audio",
  "model": "pyannote/embedding",
  "segments": [
    { "start": 1.2, "end": 5.8 },
    { "start": 7.0, "end": 12.3 }
  ]
}
```

Response

```json
{
  "request_id": "uuid",
  "processing_time_ms": 380,
  "embeddings": [[...], [...]],
  "dimension": 192,
  "count": 2,
  "model": "pyannote/embedding"
}
```

### Compare Two Audio Files

```
POST /api/speaker-embedding/compare
```

```json
{
  "audio1": "base64-audio-A",
  "audio2": "base64-audio-B",
  "model": "pyannote/embedding",
  "metric": "cosine"
}
```

Response

```json
{
  "request_id": "uuid",
  "processing_time_ms": 95,
  "distance": 0.18,
  "similarity": 0.82,
  "metric": "cosine",
  "model": "pyannote/embedding"
}
```

### Match Against Registry (Stateless)

```
POST /api/speaker-embedding/match
```

```json
{
  "query_embeddings": [[...], [...]],
  "registry": [
    { "name": "Alice", "embeddings": [[...], [...]] },
    { "name": "Bob",   "embeddings": [[...]] }
  ],
  "metric": "cosine",
  "threshold": 0.78,
  "top_k": 3,
  "strategy": "centroid"
}
```

Response

```json
{
  "request_id": "uuid",
  "processing_time_ms": 35,
  "results": [
    {
      "best": { "name": "Alice", "similarity": 0.86 },
      "candidates": [
        { "name": "Alice", "similarity": 0.86 },
        { "name": "Bob",   "similarity": 0.73 },
        { "name": "Carol", "similarity": 0.55 }
      ]
    }
  ]
}
```

Notes
- The registry is provided by your app at call time; the API does not persist speaker data.
- Use ≥ 5–10s of clean speech per enrolled speaker for best results.
- `strategy: centroid` is robust; `best` compares against each sample and uses the maximum.

---

## Web Crawling

Scrape web pages with JavaScript rendering support.

### Endpoint

```
POST /api/crawl
```

### Request

```json
{
  "url": "https://example.com",
  "screenshot": true,
  "screenshot_fullpage": false
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | URL to crawl |
| `screenshot` | boolean | No | Capture viewport screenshot (default: true) |
| `screenshot_width` | integer | No | Screenshot viewport width (default: 1920, range: 320-7680) |
| `screenshot_height` | integer | No | Screenshot viewport height (default: 1080, range: 240-4320) |
| `page_timeout` | integer | No | Total navigation timeout in ms (default: 120000) |
| `chrome_cdp_url` | string | No | Remote Chrome CDP endpoint (attach to an existing browser) |
| `screenshot_fullpage` | boolean | No | Also capture a full-page screenshot (stitched). Shares width/height (default: false) |

Behavior
- General by default: no site-specific selectors, no auto scroll, no auto click.
- Uses headless browser rendering; returns after `domcontentloaded` + a brief settle window (~1.5s).
- Full-page scan is enabled to capture content beyond the initial viewport.
- Stealth is enabled; navigator is lightly overridden to reduce bot detection.
- No global header overrides; native browser User-Agent is used by default (no forced UA).
- If a site hides content behind interactions (e.g., “load more”), content may not appear by default; perform interactions downstream if needed.

Implementation notes
- `wait_until: "domcontentloaded"`
- `delay_before_return_html: 1.5`
- `scan_full_page: true`
- `simulate_user: false` (to avoid accidental clicks/navigation)
- `override_navigator: true`

### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time_ms": 2150,
  "url": "https://example.com",
  "title": "Example Domain",
  "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples...",
  "html": "<!DOCTYPE html>...",
  "screenshot_base64": null,
  "screenshot_viewport_base64": null,
  "screenshot_fullpage_base64": null,
  "success": true
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Final URL after redirects |
| `title` | string | Page title |
| `markdown` | string | Page content in Markdown format |
| `html` | string | Raw HTML content |
| `screenshot_base64` | string | Base64-encoded screenshot (if requested) |
| `screenshot_viewport_base64` | string | Base64-encoded viewport screenshot when `screenshot` is true |
| `screenshot_fullpage_base64` | string | Base64-encoded full-page screenshot when `screenshot_fullpage` is true |
| `success` | boolean | Whether crawl succeeded |

---

## Management Endpoints

### List Available Models

```
GET /api/models
```

Returns all downloadable AI models with their configurations, download status, and capabilities. Supports `?task=image-ocr` filter.

### List Available Libs

```
GET /api/libs
```

Returns built-in libraries/tools available by task. Supports `?task=image-ocr` filter.


### Task Options (Unified)

```
GET /api/task-options?task={task}
```

Returns a unified list of choices for a task, combining models and libs. Each option has a `type` field (`model` or `lib`).

Response

```json
{
  "task": "image-ocr",
  "options": [
    {
      "id": "deepseek-ai/DeepSeek-OCR",
      "label": "DeepSeek-OCR",
      "provider": "deepseek-ai",
      "type": "model",
      "supports_markdown": true,
      "requires_download": true,
      "status": "ready"
    },
    {
      "id": "paddleocr/pp-ocrv5",
      "label": "PP-OCRv5",
      "provider": "PaddlePaddle",
      "type": "lib",
      "supports_markdown": false,
      "requires_download": false,
      "status": "ready"
    }
  ]
}
```

### Hardware Information

```
GET /api/hardware
```

Returns GPU and system information including memory usage and availability.

### Request History

```
GET /api/history/{service}
```

Get request history for a specific service (e.g., `text-generation`, `image-ocr`).

### Health Checks

```
GET /api/health
```

Health check endpoint (returns `{"status": "healthy"}`).

```
GET /api/ready
```

Readiness check with service availability status.

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (invalid endpoint or model)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error

---

## Interactive Documentation

Visit `/api/docs` for interactive Swagger UI documentation where you can test all endpoints directly in your browser.
