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
  "model": "PaddlePaddle/PaddleOCR",
  "language": "en",
  "output_format": "text"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded image data |
| `model` | string | Yes | Model ID (see available models below) |
| `language` | string | No | Language hint ('en', 'zh', 'auto', etc.) |
| `output_format` | string | No | 'text' or 'markdown' (default: 'text') |

**Available Models:**
- `PaddlePaddle/PaddleOCR` - Multi-language OCR
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
  "screenshot": false
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | URL to crawl |
| `screenshot` | boolean | No | Capture screenshot (default: false) |
| `wait_for_js` | boolean | No | Wait for JavaScript to render (default: true) |

The crawler automatically scrolls, clicks common “load more” buttons, and waits for network idle/text stabilization to grab complete SPA content. If you need deeper control, see the advanced options below.

#### Advanced Options

| Field | Type | Description |
|-------|------|-------------|
| `screenshot_width` | integer | Screenshot viewport width (default: 1920, range: 320-7680) |
| `screenshot_height` | integer | Screenshot viewport height (default: 1080, range: 240-4320) |
| `wait_for_selector` | string | Guard selector to wait for before the render loop |
| `wait_for_selector_timeout` | integer | Timeout for the guard selector in ms (default: 15000) |
| `content_selectors` | array[string] | Additional selectors that must appear before returning (merged with defaults) |
| `min_content_selector_count` | integer | Minimum matches across `content_selectors` (default: 1) |
| `load_more_selectors` | array[string] | Extra “load more” buttons to click while content grows |
| `max_scroll_rounds` | integer | Auto-scroll passes to trigger lazy loading (default: 8) |
| `scroll_delay_ms` | integer | Delay between scrolls in ms (default: 350) |
| `load_more_clicks` | integer | Maximum load-more click cycles (default: 6) |
| `stabilization_iterations` | integer | Consecutive steady checks before finishing (default: 2) |
| `stabilization_interval_ms` | integer | Delay between stabilization checks in ms (default: 700) |
| `max_render_wait_ms` | integer | Hard cap for the dynamic render wait in ms (default: 20000) |
| `page_timeout` | integer | Total navigation timeout in ms (default: 120000) |
| `chrome_cdp_url` | string | Remote Chrome CDP endpoint (attach to existing browser) |

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
| `success` | boolean | Whether crawl succeeded |

---

## Management Endpoints

### List Available Skills

```
GET /api/skills
```

Returns all available AI models/skills with their configurations, download status, and capabilities.

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
