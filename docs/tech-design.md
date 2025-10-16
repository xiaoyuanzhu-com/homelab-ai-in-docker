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
- **Containerization**: Docker with optional NVIDIA GPU support

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
  "wait_for_js": true
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
- RQ or Celery for long-running tasks
- Return job ID immediately, poll for results
- Useful for large batch operations or browser crawls

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
