# Homelab AI Services Documentation

Welcome to the **Homelab AI Services** documentation! This is a REST API service that wraps common AI capabilities for homelab developers, allowing you to add AI features to your applications without dealing with Python dependencies or ML infrastructure.

## What is Homelab AI?

AI capabilities wrapped in API, specially crafted for homelab environments:

- **RESTful API in Docker** - Easy deployment with single container
- **Curated models** - Spanning low-end to high-end hardware
- **Smart pooling and queue** - Manages GPU contention automatically
- **Tech stack freedom** - Python and AI wrapped, use any language for your apps
- **Built-in observability** - Monitor usage and performance
- **Developer friendly** - Simple REST endpoints with interactive docs
- **LLM friendly** - Remote MCP server for Claude Code integration

## Supported Tasks

| Task | Status |
|------|--------|
| Text Generation (LLM) | ✅ |
| Feature Extraction (Text Embedding) | ✅ |
| Image Captioning | ✅ |
| Image OCR | ✅ |
| Automatic Speech Recognition | ✅ |
| Web Crawling | ✅ |
| Remote MCP Server | ✅ |

## Quick Start

### Docker Installation

```yaml
services:
  api:
    image: ghcr.io/xiaoyuanzhu-com/homelab-ai-in-docker:latest
    restart: unless-stopped
    ports:
      - "12310:12310"
    volumes:
      - ./data:/haid/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Start the service:

```bash
docker-compose up -d
```

The API will be available at `http://localhost:12310`

### Access the Services

- **Web UI**: `http://localhost:12310`
- **Interactive API docs**: `http://localhost:12310/api/docs`
- **API endpoint**: `http://localhost:12310/api`
- **Health check**: `http://localhost:12310/api/health`
- **MCP endpoint**: `http://localhost:12310/mcp`

## Example: Crawl a Webpage

```bash
curl -X POST http://localhost:12310/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "screenshot": false
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

## Documentation Sections

### User Guide

- **[Product Overview](product-design.md)** - Vision, target users, and AI capabilities
- **[Deployment](deployment.md)** - Docker, GHCR, and production deployment guide

### Developer Guide

- **[Technical Design](tech-design.md)** - Architecture, implementation details, and API reference

## Need Help?

- Check the [interactive API docs](http://localhost:12310/api/docs) for endpoint details
- Review the [technical design](tech-design.md) for implementation specifics
- MCP client integration tips are in the MCP section of the main README

## License

MIT
