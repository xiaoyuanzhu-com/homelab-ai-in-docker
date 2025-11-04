# Technical Design: Homelab AI Services API

## Architecture Overview

Python-based REST API service packaged as a Docker container, providing inference capabilities for pre-trained AI models.

## Technology Stack

- **API Framework**: FastAPI (OpenAPI/Swagger support, async, type hints)
- **ML Libraries**:
  - Transformers (Hugging Face) for text embeddings and image captioning
  - sentence-transformers for efficient embedding models
  - Playwright/Selenium for browser automation
- **Web UI**: Simple HTML/JS admin interface served by FastAPI
- **Containerization**: Docker with NVIDIA GPU support

### GPU Dependencies (Locked Versions)

**Python 3.13 + PyTorch 2.7.0 + CUDA 12.6 + flash-attn 2.8.3**

These versions are locked together for compatibility:
- PyTorch 2.7 removed CUDA 12.4 support (requires 12.6+)
- flash-attn 2.8.3 pre-built wheel from [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.4.12)
- Pre-built wheels avoid compilation (faster builds, no nvcc required)
- Configured via `[tool.uv.index]` and `[tool.uv.sources]` in pyproject.toml

**Note:** These are verified stable versions. Upgrading requires checking wheel availability for the Python/PyTorch/CUDA combination.

## API Design

### Core Endpoints

#### Image Captioning
```
POST /api/caption
Content-Type: multipart/form-data or application/json (base64)

Request:
- image: file or base64 string
- model: optional model identifier
- batch: optional array of images

Response:
{
  "request_id": "uuid",
  "caption": "A description of the image",
  "model_used": "blip-base",
  "processing_time_ms": 150
}
```

#### Text Embedding
```
POST /api/embed
Content-Type: application/json

Request:
{
  "texts": ["text1", "text2", ...],
  "model": "optional-model-name"
}

Response:
{
  "request_id": "uuid",
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dimensions": 384,
  "model_used": "all-MiniLM-L6-v2"
}
```

#### Browser Crawl
```
POST /api/crawl
Content-Type: application/json

Request:
{
  "url": "https://example.com",
  "screenshot": false,
  "wait_for_js": true,
  "wait_for_selector": "#main-content",
  "wait_for_selector_timeout": 15000
}

Response:
{
  "request_id": "uuid",
  "url": "https://example.com",
  "title": "Page Title",
  "text": "Extracted clean text...",
  "screenshot_base64": "optional",
  "fetch_time_ms": 2000
}
```

#### Browser Crawl Best Practices
- **Navigation waits**: Avoid Playwright's `wait_until="networkidle"` for modern SPAs like Reddit—long-lived connections keep the network busy forever. Prefer `wait_until="domcontentloaded"` plus an explicit `page.wait_for_selector(...)` for the first piece of real content (e.g., `article[data-test-id='post-content']` or the GDPR consent button).
- **Rendered-content checks**: After DOMContentLoaded, confirm that target text exists before scraping. `page.wait_for_selector` combined with `page.evaluate` sanity checks (count posts, ensure comment tree nodes render) provides deterministic completion signals.
- **Headers & fingerprint**: Mimic a current desktop browser. Set `User-Agent`, `Accept`, `Accept-Language`, `Sec-CH-UA*`, and `Viewport` in `BrowserConfig` to reduce bot detections. Inject `navigator.webdriver = false` and persist cookies between runs so consent/search preferences stick.
- **Rate limiting & etiquette**: Honor `robots.txt`, throttle to ~1 request/second per domain, and implement exponential backoff on 429/503 with respect for any `Retry-After`. Queue retries through the task scheduler instead of hammering from the crawler loop.
- **Consent & auth flows**: Expect interstitials (GDPR, login, age gates). Detect them via selectors and click/dismiss before scraping; persist resulting cookies so the modal doesn't reappear on subsequent runs.
- **Alt endpoints & fallbacks**: For Reddit, consider `old.reddit.com` or the `https://www.reddit.com/<path>/.json` API variant when HTML rendering proves fragile. Keep the primary path in place but fail over after repeated navigation timeouts.
- **Observability**: Capture console errors and navigation events. When navigation timeouts cluster, restart the remote Chromium session—stale websocket sessions are a common culprit. Log rendered HTML size, selector wait timing, and retry counts for easier post-mortems.
- **Dynamic render strategy**: The crawler auto-scrolls, clicks generic “load more” affordances, and waits for content counts to stabilize (or until `max_render_wait_ms` expires). Override the defaults with `content_selectors`, `load_more_selectors`, `max_scroll_rounds`, `load_more_clicks`, `stabilization_iterations`, etc., when a site needs custom tuning.
- **API support**: `wait_for_selector`, `content_selectors`, and companions like `min_content_selector_count` or `stabilization_interval_ms` let callers tailor the render-complete signal per site without changing server code.
- **Stealth support**: Stealth mode ships with a known-good `playwright-stealth==1.8.0` build so Crawl4AI keeps its fingerprint spoofing without surprises. Set `CRAWLER_ENABLE_STEALTH=false` if you need to disable it for troubleshooting.

**Render Strategy Parameters**

| Field | Purpose |
|-------|---------|
| `wait_for_selector` / `wait_for_selector_timeout` | Initial guard selector before dynamic waits kick in |
| `content_selectors` | Additional selectors that must appear before returning |
| `min_content_selector_count` | Minimum node count across `content_selectors` to consider the page populated |
| `max_scroll_rounds` / `scroll_delay_ms` | Auto-scroll depth and pacing used to trigger lazy loading |
| `load_more_selectors` / `load_more_clicks` | Generic “load more” controls to click repeatedly while content grows |
| `stabilization_iterations` / `stabilization_interval_ms` | Consecutive steady-state checks (no new nodes) before finishing |
| `max_render_wait_ms` | Hard cap on the whole render wait loop |

### Management Endpoints

```
GET  /api/models          # List available models
POST /api/models          # Add new model
GET  /api/models/{id}     # Model details
DELETE /api/models/{id}   # Remove model

GET  /api/status          # Service status and metrics
GET  /health              # K8s liveness probe
GET  /ready               # K8s readiness probe
GET  /metrics             # Prometheus metrics
```

## Model Management

### Model Storage

**Storage Location**: All models use HuggingFace's standard `HF_HOME` convention.

- **Root directory**:
  - In Docker images the app root (`/haid`) is present, so `HF_HOME` defaults to `/haid/data/models`
  - In local development `HF_HOME` resolves to `<project_root>/data/models` unless `HAID_DATA_DIR` is set
  - Any deployment can override the root by exporting `HAID_DATA_DIR` (the app then sets `HF_HOME=<HAID_DATA_DIR>/models`)
- **Cache layout**: `models/hub/models--{org}--{model}/`
- **Configuration**: `main.py` ensures `HF_HOME` is defined at startup based on the rules above
- **Benefits**:
  - Standard HuggingFace convention - all HF libraries automatically respect `HF_HOME`
  - Automatic organization with consistent structure across all model types
  - No custom cache management code needed
  - Easy to inspect and manage downloaded models
  - Future-proof for any HuggingFace model library

**Example structure**:
```
data/                # default root in local development
  models/            # HF_HOME
    hub/
      models--BAAI--bge-large-en-v1.5/
      models--Alibaba-NLP--gte-large-en-v1.5/
      models--Salesforce--blip-image-captioning-base/
      models--sentence-transformers--all-MiniLM-L6-v2/
```

**Detection**: To check if a model is downloaded, convert model ID to HF cache format:
- Model ID: `BAAI/bge-large-en-v1.5`
- Cache path: `data/models/hub/models--BAAI--bge-large-en-v1.5/`

### Unified Models Manifest

All available models are defined in `src/api/models/models_manifest.json`, grouped by task type (embedding, caption, ocr, etc.). Each model entry includes:
- `id`: HuggingFace model identifier (e.g., `BAAI/bge-large-en-v1.5`)
- `name`: Human-readable display name
- `team`: Model author/organization
- `size_mb`: Approximate download size
- Task-specific metadata (dimensions for embeddings, etc.)

### Model Management API

- `GET /api/models` - List all models across all task types with download status
- `GET /api/models/{type}` - Filter models by task type
- `POST /api/models/download` - Download any model via HuggingFace CLI
- `DELETE /api/models/{model_id}` - Remove model by deleting HF cache directory

### Lazy Loading Strategy
- Models NOT loaded at startup (saves memory)
- Load on first request to specific endpoint
- Configurable keep-alive timeout (default: 30 minutes)
- LRU eviction when memory threshold reached

### Model Configuration
```yaml
models:
  image_caption:
    - id: blip-base
      hf_model: Salesforce/blip-image-captioning-base
      enabled: true
      default: true
    - id: llava
      hf_model: llava-hf/llava-1.5-7b-hf
      enabled: false

  text_embedding:
    - id: minilm
      hf_model: sentence-transformers/all-MiniLM-L6-v2
      dimensions: 384
      enabled: true
      default: true
    - id: mpnet
      hf_model: sentence-transformers/all-mpnet-base-v2
      dimensions: 768
      enabled: false

resources:
  max_gpu_memory_gb: 8
  max_concurrent_requests: 4
  model_keep_alive_minutes: 30
```

## Error Handling

### Standardized Error Response
```json
{
  "error": {
    "code": "MODEL_NOT_LOADED",
    "message": "The requested model is not available",
    "request_id": "uuid",
    "details": {
      "model_id": "nonexistent-model",
      "available_models": ["blip-base", "llava"]
    }
  }
}
```

### Error Categories
- `INVALID_REQUEST` - Malformed input (400)
- `MODEL_NOT_FOUND` - Unknown model identifier (404)
- `MODEL_LOAD_FAILED` - Model initialization error (503)
- `INFERENCE_FAILED` - Model execution error (500)
- `RESOURCE_EXHAUSTED` - GPU/memory limits reached (503)
- `RATE_LIMIT_EXCEEDED` - Too many requests (429)

## Monitoring & Observability

### Metrics (Prometheus format)
- `api_requests_total{endpoint, status}` - Request counter
- `api_request_duration_seconds{endpoint}` - Response time histogram
- `model_load_duration_seconds{model}` - Model initialization time
- `model_inference_duration_seconds{model}` - Inference time
- `models_loaded{model}` - Currently loaded models gauge
- `gpu_memory_used_bytes` - GPU memory consumption
- `active_requests` - Current in-flight requests

### Access Logging
```
timestamp | request_id | client_ip | endpoint | status | duration_ms | model
```

## Security

### API Key Authentication
```
Authorization: Bearer <api-key>
```
- Keys stored in config file (hashed)
- Optional: allow unauthenticated localhost access
- Rate limiting per key

### IP Allowlist
```yaml
security:
  auth_enabled: true
  api_keys:
    - key_hash: "sha256_hash"
      name: "my-app"
  ip_allowlist:
    - 192.168.1.0/24
    - 127.0.0.1
```

## Docker Deployment

### Container Structure
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# Install Python 3.11, FastAPI, transformers, etc.
# Models downloaded on first use (not baked into image)
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Example
```yaml
services:
  ai-api:
    image: homelab-ai-api:latest
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./models:/app/models  # Model cache
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Performance Optimizations

### Batch Processing
- Automatically batch multiple requests if arriving within time window
- Single model inference call for better GPU utilization

### Caching Strategy (Future)
- Optional Redis-based result cache
- Cache key: hash(model_id + input)
- Particularly useful for embeddings (deterministic)

### Async Job Queue (Future)

## OCR Inference Isolation

Overview
- Each OCR model runs in its own worker process. A worker loads a single model and serves requests over a local HTTP endpoint.
- After a period of inactivity (idle timeout), the worker exits. Process exit fully releases the CUDA context and GPU memory.

Motivation
- Reliable GPU memory release: terminating a process frees CUDA contexts; in‑process unloads can leave allocator fragments.
- Fault isolation: a misbehaving model or library cannot crash the main API process.
- Predictable lifecycle: workers are spawned on demand and self‑terminate when idle.

Runtime Flow
- Router (manager): the FastAPI router spawns and tracks a per‑model worker using `src/worker/manager.py`.
  - On each request, the manager ensures a worker is running for the requested model and forwards the image to it.
  - After the request, it updates `last_active` and schedules a shutdown task that will terminate the worker if no further requests arrive before the timeout.
- Worker: `src/worker/image_ocr_worker.py` loads the model with `OCRInferenceEngine` and exposes endpoints:
  - `GET /healthz` – readiness/health check.
  - `POST /infer` – accepts base64 image and returns extracted text.
  - `POST /shutdown` – graceful shutdown endpoint (used by the manager).
  - The worker also has its own idle watchdog; if no requests arrive before the timeout, it cleans up and exits.

Idle Timeout / Short‑Lived Cache
- Default timeout is 5 seconds (DB setting `model_idle_timeout_seconds`).
- Effect: the most recently used model remains warm for 5s. Sequential requests (e.g., a batch of 100 images) reuse the same worker and avoid reload cost, provided inter‑request gaps are under the timeout.
- After the timeout, the worker exits and frees GPU memory.
- Note: workers receive the timeout at spawn; changing the setting applies to newly spawned workers. The manager also schedules shutdown using the current setting on each request.

Components
- Router changes: `src/api/routers/image_ocr.py` routes OCR requests via the worker manager exclusively.
- Manager: `src/worker/manager.py` (spawns, routes, and schedules shutdowns).
- Worker: `src/worker/image_ocr_worker.py` (serves a single model in a separate process).
- Shared engine: `src/inference/ocr.py` (unified inference; includes robust cleanup helpers).

Operational Notes
- Ports: manager assigns a free localhost port per worker. Workers bind to `127.0.0.1` only.
- Termination: manager first requests `/shutdown`, then sends SIGTERM, and finally SIGKILL if needed.
- GPU assignment: future enhancement could set `CUDA_VISIBLE_DEVICES` per worker if multiple GPUs are present.
- Logs: workers inherit parent environment; set `PYTHONUNBUFFERED=1` for timely logs.
- RQ or Celery for long-running tasks
- Return job ID immediately, poll for results
- Useful for large batch operations or browser crawls

## Frontend UI Architecture & Guidelines

The Next.js UI (see `ui/`) is structured around a small set of reusable primitives that keep every task page consistent and make history playback trivial.

### Core Building Blocks

- `ui/src/components/try/`
  - `try-layout.tsx`: two-column grid (side-by-side from `md` breakpoints up) that renders Input/Output panels.
  - `try-panel.tsx`: card shell used by both sides; accepts content, footer actions, and an optional raw payload toggle.
  - `raw-payload.tsx`: shared collapsible for “View Raw Request/Response”. Always include this for observability.
  - `model-selector.tsx`: dropdown wrapper that standardises loading/empty states for model selection.
- `ui/src/components/inputs/`
  - `image-upload.tsx`: drag-and-drop capable uploader with preview support (set `hideInput` to reuse preview in history mode).
  - `audio-upload.tsx`, `audio-recorder-controls.tsx`, `audio-player.tsx`: modular audio input primitives used by ASR and speaker embedding pages.
- Task-specific input/output components (e.g., `text-generation-input-output.tsx`, `image-caption-input-output.tsx`) compose the above primitives and expose a `mode` prop (`"try" | "history"`) so the same UI can render read-only history entries.

### Page Conventions

1. Every “Try” tab should render its task component inside `TryLayout`, passing Input/Output descriptors and wiring the raw payload props.
2. History views reuse the same components with `mode="history"`, populating `requestPayload`/`responsePayload` for the raw view.
3. Keep action buttons (generate, upload, compare, etc.) in the `footer` property of the input panel so they stay aligned across pages.
4. Prefer the shared upload/select controls over inline `<input>` elements to ensure accessibility and consistent styling.
- For image uploads, drag or click is supported automatically; supply `helperText` to describe accepted formats.
- For audio uploads, `AudioRecorderControls` handles recording availability and device messaging.
5. Task history is centralised via `ui/src/components/task-history.tsx` and displayed on the Status page. The component fetches the latest 10 entries, offers per-service filtering, and expands entries using the registered task renderers.

### Layout & Responsiveness

- `TryLayout` uses `grid-cols-1` on mobile and `md:grid-cols-2` for desktop screens to honour the desired left/right layout.
- Page containers typically use `max-w-6xl` with `px-4` padding; maintain this for new pages.
- Avoid fixed heights where possible; when scrollable sections are required (e.g. task history), wrap them in `ScrollArea` with an explicit height to prevent overlap with subsequent sections.

### Adding a New Task Page

1. Create a `<TaskName>InputOutput` component that accepts state props, implements both `try` and `history` modes, and exports the raw payload objects.
2. On the page, wire up data fetching/actions, then render `<TaskNameInputOutput>` inside the `Try` tab.
3. Register a renderer in `task-history.tsx` so history playback works with zero additional UI.
4. If uploads are needed, prefer the primitives in `inputs/`; extend them if new media types are introduced.

Following these guidelines keeps every task page visually consistent, reduces per-page duplication, and ensures history playback remains a first-class feature.

## Development Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Run locally
uvicorn main:app --reload

# Run tests
pytest

# Build Docker image
docker build -t homelab-ai-api .
```

## Testing Strategy

- Unit tests for API endpoints (mocked models)
- Integration tests with small test models
- Load testing with locust
- E2E tests for Docker deployment
