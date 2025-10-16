# Homelab AI Services API

REST API service wrapping common AI capabilities for homelab developers. Enables any programming language to access local AI features without managing Python ML dependencies.

## Features

- **Smart Web Scraping**: Crawl URLs with JavaScript rendering support, extract clean Markdown content
- **Text Embedding**: (Coming soon) Convert text to vectors for semantic search
- **Image Captioning**: (Coming soon) Generate descriptions from images

## Quick Start

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
python main.py
```

The API will be available at `http://localhost:8000`

- Interactive API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

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

- `POST /api/crawl` - Crawl and extract content from a URL
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /docs` - Interactive API documentation

## Documentation

- [Product Design](docs/product-design.md) - User-focused overview
- [Technical Design](docs/tech-design.md) - Implementation details
- [CLAUDE.md](CLAUDE.md) - AI assistant context

## Development

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run with auto-reload
uvicorn main:app --reload
```

## License

MIT
