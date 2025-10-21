# Homelab AI Services API

REST API service wrapping common AI capabilities for homelab developers. Enables any programming language to access local AI features without managing Python ML dependencies.

## Features

- **Smart Web Scraping**: Crawl URLs with JavaScript rendering support, extract clean Markdown content
- **Text Embedding**: Convert text to vectors for semantic search and similarity matching
- **Image Captioning**: Generate natural language descriptions from images
- **Image OCR**: Extract text from images (runs in an isolated per‑model worker with a 5s idle shutdown to reliably free GPU memory)

## Quick Start

### Docker (Recommended for Production)

**Prerequisites for GPU Support:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Docker with GPU support enabled

**Option 1: Use pre-built image from GitHub Container Registry**

```bash
# Pull and run the latest image
docker pull ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest

# Run with docker-compose (includes GPU support)
docker-compose up -d

# Or run directly with GPU support
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/haid/data \
  --gpus all \
  --name homelab-ai \
  ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest

# Or run without GPU (CPU-only)
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/haid/data \
  --name homelab-ai \
  ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest
```

**Option 2: Build locally**

```bash
# Build and run with docker-compose (includes GPU support)
docker-compose up -d

# Or build manually
docker build -t homelab-ai .

# Run with GPU support
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/haid/data \
  --gpus all \
  --name homelab-ai \
  homelab-ai

# Or run without GPU (CPU-only)
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/haid/data \
  --name homelab-ai \
  homelab-ai
```

The API will be available at `http://localhost:8000`

> **Note:** The docker-compose.yml configuration includes GPU support by default. If you don't have a GPU or NVIDIA Container Toolkit installed, remove the `deploy` section from docker-compose.yml to run in CPU-only mode.

### Local Development

### Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Setup browser for crawling
crawl4ai-setup
```

### Running the Server

```bash
# With uv (no activation needed)
uv run python main.py

# Or activate and run
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python main.py
```

The API will be available at `http://localhost:8000`

- Interactive API docs: `http://localhost:8000/api/docs`
- Health check: `http://localhost:8000/api/health`
- API info: `http://localhost:8000/api`

### Example Usage

**Crawl a webpage:**

```bash
curl -X POST http://localhost:8000/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "screenshot": false,
    "wait_for_js": true
  }'
```

**Response:**

```json
{
  "request_id": "uuid",
  "url": "https://example.com/",
  "title": "Example Domain",
  "markdown": "# Example Domain\nThis domain is for use...",
  "html": "<!DOCTYPE html>...",
  "screenshot_base64": null,
  "fetch_time_ms": 1580,
  "success": true
}
```

## API Endpoints

All endpoints are under the `/api` prefix:

- `GET /api` - API status and information
- `POST /api/crawl` - Crawl and extract content from a URL
- `POST /api/embed` - Generate text embeddings
- `POST /api/caption` - Generate image captions
- `GET /api/history/{service}` - Get request history for a service
- `GET /api/health` - Health check
- `GET /api/ready` - Readiness check
- `GET /api/docs` - Interactive API documentation (Swagger UI)

## Directory Structure

```
/haid (or . in dev)
├── data/              # Model cache and history database (git-ignored)
│   ├── embedding/     # sentence-transformers models
│   ├── image-caption/ # BLIP models
│   └── history.db     # SQLite database for request history
├── src/               # Application source
└── main.py            # Application entry point
```

## Documentation

- [Product Design](docs/product-design.md) - User-focused overview
- [Technical Design](docs/tech-design.md) - Implementation details
- [Deployment Guide](docs/deployment.md) - Docker, GHCR, and production deployment
- [CLAUDE.md](CLAUDE.md) - AI assistant context

Architecture highlight: OCR inference runs in isolated worker processes with a short‑lived cache (5s linger) to balance batch throughput and fast GPU memory release. See Technical Design for details.

## Development

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run with auto-reload (using uv)
uv run uvicorn main:app --reload

# Or activate first
source .venv/bin/activate
uvicorn main:app --reload
```

## License

MIT
