# API Reference

Complete reference for all endpoints in Homelab AI Services API.

## Base URL

All endpoints are prefixed with `/api`:

- **Development**: `http://localhost:12310/api`
- **Production**: `http://your-domain:12310/api`

## Common Response Fields

All responses include these base fields:

```json
{
  "request_id": "uuid-string",
  "processing_time_ms": 123
}
```

## Authentication

Currently no authentication required. For production deployments, use reverse proxy authentication or firewall restrictions.

---

# AI Skills

## Text Embedding

Generate vector embeddings for semantic search and similarity.

### `POST /api/text-to-embedding`

**Request:**

```json
{
  "texts": ["Hello world", "Good morning"],
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | array[string] | Yes | List of texts to embed |
| `model` | string | Yes | Model ID from catalog |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 45,
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimensions": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

## Text Generation

Generate text from prompts using language models.

### `POST /api/text-generation`

**Request:**

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

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Input prompt |
| `model` | string | Yes | Model ID from catalog |
| `max_new_tokens` | integer | No | Max tokens to generate (default: 256) |
| `temperature` | float | No | Sampling temperature (default: 0.7) |
| `top_p` | float | No | Nucleus sampling (default: 0.9) |
| `do_sample` | boolean | No | Use sampling vs greedy (default: true) |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 1250,
  "generated_text": "Code flows like water...",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "tokens_generated": 24
}
```

---

## Image Captioning

Generate natural language descriptions of images.

### `POST /api/image-captioning`

**Request:**

```json
{
  "image": "base64-encoded-image-data",
  "model": "Salesforce/blip-image-captioning-base",
  "prompt": "a photo of"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded image |
| `model` | string | Yes | Model ID from catalog |
| `prompt` | string | No | Optional prompt prefix |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 320,
  "caption": "a cat sitting on a wooden table",
  "model": "Salesforce/blip-image-captioning-base"
}
```

---

## Image OCR

Extract text from images using optical character recognition.

### `POST /api/image-ocr`

**Request:**

```json
{
  "image": "base64-encoded-image-data",
  "model": "deepseek-ai/DeepSeek-OCR",
  "output_format": "markdown",
  "language": "en"
}
```

Or using a library engine:

```json
{
  "image": "base64-encoded-image-data",
  "lib": "paddleocr/pp-ocrv5",
  "output_format": "text",
  "language": "en"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64-encoded image |
| `model` | string | No* | Model ID from catalog |
| `lib` | string | No* | Library ID from catalog |
| `output_format` | string | No | `text` or `markdown` (default: `text`) |
| `language` | string | No | Language hint (`en`, `zh`, etc.) |

*Exactly one of `model` or `lib` must be provided.

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 890,
  "text": "Extracted text from the image",
  "model": "deepseek-ai/DeepSeek-OCR",
  "output_format": "markdown"
}
```

---

## Automatic Speech Recognition

Unified endpoint for transcribing audio files. Supports multiple backends via the `lib` parameter.

### `POST /api/automatic-speech-recognition`

**Request:**

```json
{
  "audio": "base64-encoded-audio-data",
  "model": "openai/whisper-large-v3-turbo",
  "lib": "whisper",
  "language": "en",
  "diarization": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | Yes | Base64-encoded audio (mp3, mp4, wav, webm, etc.) |
| `model` | string | No | Model ID (default varies by lib) |
| `lib` | string | No | ASR library: `whisper`, `whisperx`, `funasr` (auto-detected if omitted) |
| `language` | string | No | Language code (e.g., `en`, `zh`). Auto-detect if omitted |
| `diarization` | boolean | No | Enable speaker diarization (default: false) |
| `min_speakers` | integer | No | Minimum speakers for diarization |
| `max_speakers` | integer | No | Maximum speakers for diarization |
| `num_speakers` | integer | No | Exact speaker count (whisper diarization only) |
| `return_timestamps` | boolean | No | Return word timestamps (whisper only) |
| `batch_size` | integer | No | Inference batch size (whisperx only, default: 4) |
| `compute_type` | string | No | Compute type (whisperx only, e.g., `float16`) |

**Library Options:**

| Library | Best For | Features |
|---------|----------|----------|
| `whisper` | Basic transcription | Fast, uses transformers pipeline |
| `whisperx` | High-quality with alignment | Word-level timestamps, speaker diarization with embeddings |
| `funasr` | Chinese/multilingual | Emotion detection, audio event detection |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 3200,
  "text": "Transcribed text from the audio.",
  "model": "openai/whisper-large-v3-turbo",
  "lib": "whisper",
  "language": "en",
  "segments": null,
  "speakers": null,
  "num_speakers": null,
  "chunks": null,
  "text_clean": null,
  "emotion": null,
  "event": null
}
```

**Response Fields by Library:**

| Field | whisper | whisperx | funasr |
|-------|---------|----------|--------|
| `text` | ✓ | ✓ | ✓ |
| `language` | ✓ | ✓ | ✓ |
| `segments` | diarization only | ✓ (with words) | - |
| `speakers` | - | ✓ (with embeddings) | - |
| `num_speakers` | ✓ | ✓ | - |
| `chunks` | with timestamps | - | - |
| `text_clean` | - | - | ✓ |
| `emotion` | - | - | ✓ |
| `event` | - | - | ✓ |

**Example: WhisperX with Diarization**

```json
{
  "audio": "base64-encoded-audio",
  "model": "large-v3",
  "lib": "whisperx",
  "diarization": true,
  "min_speakers": 2,
  "max_speakers": 4
}
```

Response includes word-aligned segments with speaker labels and speaker embeddings for voice fingerprinting:

```json
{
  "request_id": "uuid",
  "processing_time_ms": 4120,
  "text": "Hello everyone, welcome...",
  "language": "en",
  "lib": "whisperx",
  "model": "large-v3",
  "segments": [
    {
      "start": 0.12,
      "end": 2.34,
      "text": "Hello everyone",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.12, "end": 0.52, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "embedding": [0.1, 0.2, ...],
      "total_duration": 15.3,
      "segment_count": 5
    }
  ],
  "num_speakers": 2
}
```

**Example: FunASR with Emotion Detection**

```json
{
  "audio": "base64-encoded-audio",
  "model": "FunAudioLLM/SenseVoiceSmall",
  "lib": "funasr",
  "language": "zh"
}
```

Response includes emotion and audio event detection:

```json
{
  "request_id": "uuid",
  "processing_time_ms": 1500,
  "text": "<|zh|><|HAPPY|><|BGM|>你好世界",
  "text_clean": "你好世界",
  "language": "zh",
  "lib": "funasr",
  "model": "FunAudioLLM/SenseVoiceSmall",
  "emotion": "HAPPY",
  "event": "BGM"
}
```

---

### `WebSocket /api/automatic-speech-recognition/live`

Real-time live transcription via WebSocket.

**Query Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lib` | `whisperlivekit` | Streaming backend: `whisperlivekit` or `funasr` |
| `model` | varies | Model identifier |
| `language` | `en` / `zh` | Language code (or `auto`) |
| `diarization` | `false` | Enable speaker diarization (whisperlivekit only) |

**Protocol:**

1. Client connects with query params
2. Server sends config: `{"type": "config", "sampleRate": 16000, ...}`
3. Client streams raw PCM int16 audio bytes
4. Server sends partial results: `{"type": "partial", "lines": [...], "buffer_transcription": "..."}`
5. On disconnect, server sends: `{"type": "ready_to_stop"}`

**Example Connection URLs:**

```
# WhisperLiveKit (default)
ws://localhost:12310/api/automatic-speech-recognition/live?model=large-v3&language=en

# With diarization
ws://localhost:12310/api/automatic-speech-recognition/live?lib=whisperlivekit&model=large-v3&diarization=true

# FunASR for Chinese
ws://localhost:12310/api/automatic-speech-recognition/live?lib=funasr&language=zh
```

---

## Speaker Embedding

Extract and compare speaker voice embeddings.

### `POST /api/speaker-embedding/extract`

**Request:**

```json
{
  "audio": "base64-encoded-audio",
  "model": "pyannote/embedding",
  "mode": "whole"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | Yes | Base64-encoded audio |
| `model` | string | No | Model ID (default: `pyannote/embedding`) |
| `mode` | string | No | `whole` or `segment` |
| `start_time` | float | No | Start time (segment mode) |
| `end_time` | float | No | End time (segment mode) |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 210,
  "embedding": [0.12, -0.03, ...],
  "dimension": 192,
  "model": "pyannote/embedding"
}
```

### `POST /api/speaker-embedding/batch-extract`

Extract embeddings for multiple time segments from one audio file.

**Request:**

```json
{
  "audio": "base64-encoded-audio",
  "model": "pyannote/embedding",
  "segments": [
    {"start": 1.2, "end": 5.8},
    {"start": 7.0, "end": 12.3}
  ]
}
```

**Response:**

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

### `POST /api/speaker-embedding/compare`

Compare two audio files for speaker similarity.

**Request:**

```json
{
  "audio1": "base64-audio-A",
  "audio2": "base64-audio-B",
  "model": "pyannote/embedding",
  "metric": "cosine"
}
```

**Response:**

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

### `POST /api/speaker-embedding/match`

Match query embeddings against a speaker registry (stateless).

**Request:**

```json
{
  "query_embeddings": [[...], [...]],
  "registry": [
    {"name": "Alice", "embeddings": [[...], [...]]},
    {"name": "Bob", "embeddings": [[...]]}
  ],
  "metric": "cosine",
  "threshold": 0.78,
  "top_k": 3,
  "strategy": "centroid"
}
```

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 35,
  "results": [
    {
      "best": {"name": "Alice", "similarity": 0.86},
      "candidates": [
        {"name": "Alice", "similarity": 0.86},
        {"name": "Bob", "similarity": 0.73}
      ]
    }
  ]
}
```

---

# Document Processing

## Web Crawling

Scrape web pages with JavaScript rendering.

### `POST /api/crawl`

**Request:**

```json
{
  "url": "https://example.com",
  "screenshot": true,
  "screenshot_fullpage": false,
  "screenshot_width": 1920,
  "screenshot_height": 1080,
  "page_timeout": 120000,
  "include_html": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | URL to crawl |
| `screenshot` | boolean | No | Capture viewport screenshot (default: true) |
| `screenshot_fullpage` | boolean | No | Also capture full-page screenshot |
| `screenshot_width` | integer | No | Viewport width (default: 1920) |
| `screenshot_height` | integer | No | Viewport height (default: 1080) |
| `page_timeout` | integer | No | Timeout in ms (default: 120000) |
| `chrome_cdp_url` | string | No | Remote Chrome CDP endpoint |
| `include_html` | boolean | No | Include raw HTML in response |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 2150,
  "url": "https://example.com",
  "title": "Example Domain",
  "markdown": "# Example Domain\n\nThis domain is...",
  "html": null,
  "screenshot_base64": "base64-image...",
  "screenshot_fullpage_base64": null,
  "success": true
}
```

---

## Doc to Markdown

Convert documents (PDF, DOCX, PPTX, XLSX, HTML) to Markdown.

### `POST /api/doc-to-markdown`

**Request:**

```json
{
  "file": "base64-encoded-file",
  "filename": "report.pdf",
  "lib": "microsoft/markitdown"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes | Base64-encoded document (or data URL) |
| `filename` | string | No | Filename to detect format |
| `lib` | string | No | Engine (default: `microsoft/markitdown`) |

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 742,
  "markdown": "# Title\n...converted content...",
  "model": "microsoft/markitdown"
}
```

---

## Doc to Screenshot

Convert documents to PNG screenshots.

### `POST /api/doc-to-screenshot`

**Request:**

```json
{
  "file": "base64-encoded-file",
  "filename": "presentation.pptx",
  "lib": "screenitshot/screenitshot"
}
```

**Response:**

```json
{
  "request_id": "uuid",
  "processing_time_ms": 1200,
  "screenshot": "base64-png-image...",
  "model": "screenitshot/screenitshot"
}
```

---

# Catalog & Management

## Models

### `GET /api/models`

List all downloadable AI models.

**Query Parameters:**

| Parameter | Description |
|-----------|-------------|
| `task` | Filter by task (e.g., `image-ocr`, `text-generation`) |

**Response:**

```json
{
  "models": [
    {
      "id": "openai/whisper-large-v3-turbo",
      "label": "Whisper Large V3 Turbo",
      "provider": "openai",
      "tasks": ["automatic-speech-recognition"],
      "architecture": "whisper",
      "status": "ready",
      "size_mb": 3000,
      "gpu_memory_mb": 4000,
      "requires_download": true
    }
  ]
}
```

### `GET /api/models/download?model={model_id}`

Download a model. Returns SSE stream with progress events.

**SSE Events:**

```json
{"type": "progress", "current_mb": 500, "total_mb": 3000}
{"type": "complete", "percent": 100, "size_mb": 3000}
{"type": "error", "message": "Download failed"}
```

### `GET /api/models/{model_id}/logs`

Get download logs for a model.

### `DELETE /api/models/{model_id}`

Delete downloaded model assets.

---

## Libraries

### `GET /api/libs`

List all library/tool engines (non-HuggingFace).

**Query Parameters:**

| Parameter | Description |
|-----------|-------------|
| `task` | Filter by task |

**Response:**

```json
{
  "libs": [
    {
      "id": "whisperx/whisperx",
      "label": "WhisperX",
      "provider": "whisperx",
      "tasks": ["automatic-speech-recognition"],
      "architecture": "whisperlivekit",
      "status": "ready",
      "supports_live_streaming": true
    }
  ]
}
```

---

## Task Options (Unified)

### `GET /api/task-options`

Get combined model and library options for a task.

**Query Parameters:**

| Parameter | Description |
|-----------|-------------|
| `task` | Filter by task |

**Response:**

```json
{
  "task": "automatic-speech-recognition",
  "options": [
    {
      "id": "openai/whisper-large-v3-turbo",
      "label": "Whisper Large V3 Turbo",
      "provider": "openai",
      "type": "model",
      "architecture": "whisper",
      "supports_live_streaming": false,
      "status": "ready"
    },
    {
      "id": "whisperx/whisperx",
      "label": "WhisperX",
      "provider": "whisperx",
      "type": "lib",
      "architecture": "whisperlivekit",
      "supports_live_streaming": true,
      "status": "ready"
    }
  ]
}
```

---

## Environments

Worker Python environments that are installed on-demand.

### `GET /api/environments`

List all worker environments.

**Response:**

```json
{
  "environments": {
    "transformers": {
      "env_id": "transformers",
      "status": "ready",
      "size_mb": 2500.0,
      "python_version": "3.11"
    },
    "whisper": {
      "env_id": "whisper",
      "status": "not_installed",
      "size_mb": null
    }
  }
}
```

### `GET /api/environments/{env_id}`

Get status of a specific environment.

### `POST /api/environments/{env_id}/install`

Install an environment.

**Query Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wait` | `false` | Wait for installation to complete |

### `DELETE /api/environments/{env_id}`

Delete an installed environment to free disk space.

---

## Settings

### `GET /api/settings`

Get all application settings.

**Response:**

```json
{
  "settings": {
    "hf_token": "hf_xxx...",
    "hf_endpoint": "https://huggingface.co",
    "worker_idle_timeout_seconds": "60"
  }
}
```

### `GET /api/settings/{key}`

Get a specific setting value.

### `PUT /api/settings/{key}`

Update a setting.

**Request:**

```json
{
  "value": "new-value",
  "description": "Optional description"
}
```

---

## History

### `GET /api/history/stats`

Get task statistics.

**Response:**

```json
{
  "running": 0,
  "today": 15,
  "total": 1234
}
```

### `GET /api/history/all`

Get unified history across all services.

**Query Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 50 | Max entries (max 100) |
| `offset` | 0 | Entries to skip |

### `GET /api/history/{service}`

Get history for a specific service.

**Services:** `crawl`, `text-generation`, `text-to-embedding`, `image-captioning`, `image-ocr`, `automatic-speech-recognition`

### `GET /api/history/{service}/{request_id}`

Get a specific request by ID.

### `DELETE /api/history/{service}`

Clear all history for a service.

---

## Hardware

### `GET /api/hardware`

Get system hardware statistics.

**Response:**

```json
{
  "cpu": {
    "usage_percent": 15.2,
    "cores": 16,
    "frequency_mhz": 3600,
    "model": "AMD Ryzen 9 5950X",
    "temperature_c": 45.0
  },
  "memory": {
    "total_gb": 64.0,
    "available_gb": 48.5,
    "used_gb": 15.5,
    "usage_percent": 24.2
  },
  "gpu": {
    "available": true,
    "count": 1,
    "devices": [
      {
        "id": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "driver_version": "550.54.14",
        "cuda_version": "12.4",
        "total_memory_gb": 24.0,
        "used_memory_gb": 2.5,
        "free_memory_gb": 21.5,
        "temperature_c": 42
      }
    ]
  },
  "inference": {
    "device": "cuda",
    "description": "AI models will use CUDA for inference"
  }
}
```

### `GET /api/hardware/gpu/memory`

Get detailed GPU memory breakdown (PyTorch-specific).

---

# System Endpoints

### `GET /api`

Root endpoint with service information and available endpoints.

### `GET /api/health`

Health check with worker and environment status.

**Response:**

```json
{
  "status": "healthy",
  "workers": {},
  "gpu_lock_held": false,
  "environments": {
    "transformers": {"status": "ready", "size_mb": 2500}
  }
}
```

### `GET /api/ready`

Readiness check.

### `GET /api/docs`

Interactive Swagger UI documentation.

### `GET /api/redoc`

ReDoc API documentation.

---

# MCP Server

Model Context Protocol server for Claude Code integration.

**Endpoint:** `/mcp` (Streamable HTTP transport)

See [README.md](../README.md#mcp-integration) for setup instructions.

---

# Error Responses

All endpoints return standard error responses:

```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "request_id": "uuid"
  }
}
```

**Common HTTP Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (invalid endpoint or model) |
| 409 | Conflict (e.g., download already in progress) |
| 422 | Unprocessable Entity (validation error) |
| 500 | Internal Server Error |
| 504 | Gateway Timeout (e.g., OCR timeout) |
