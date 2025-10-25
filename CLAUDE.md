# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Notes

- **DO NOT commit without explicit user instruction** - Wait for the user to ask to commit changes before running git commit commands.
  - **CRITICAL**: This applies to EVERY set of changes, even within a single long session
  - Even if the user said "commit it" earlier, you MUST wait for explicit "commit" instruction for each subsequent change
  - Never assume permission to commit based on previous commits in the same conversation
  - Always wait for the user to review changes and explicitly say "commit" before running git commit

- **⚠️ ALWAYS SEARCH OFFICIAL DOCUMENTATION FOR OPTIMAL SOLUTIONS** - Do NOT implement workarounds or hacks to "just make it work"
  - **CRITICAL**: When encountering errors, warnings, or integration issues, ALWAYS search for official documentation first
  - Use WebSearch and WebFetch tools to find the authoritative, recommended approach from official sources
  - Present ALL available options to the user with pros/cons of each approach
  - Clearly identify which option is the optimal/recommended solution vs workarounds
  - Examples of what to search:
    - Official library documentation (PyPI, npm, GitHub repos)
    - Official installation guides and compatibility matrices
    - Version compatibility information
    - Best practices from the maintainers
  - **NEVER** implement a quick fix without first researching the proper solution
  - **ALWAYS** explain the tradeoffs between optimal solutions and workarounds
  - Let the user make informed decisions about which approach to take

## Project Overview

REST API service that wraps common AI capabilities for homelab developers. Built with FastAPI and Python, designed to run in Docker.

## Development Setup

- **Python Version**: 3.13 (specified in `.python-version`)
- **Package Manager**: uv (compatible with pip)
- **Framework**: FastAPI with uvicorn
- **Main Dependency**: crawl4ai for web scraping

### First Time Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
crawl4ai-setup  # Installs Playwright browsers
```

#### System Dependencies

**FFmpeg** (required for pyannote.audio speaker diarization):
- **Development (local)**: Install via `sudo apt-get install ffmpeg` on Debian/Ubuntu
- **Production (Docker)**: FFmpeg is automatically installed in the Docker image (see Dockerfile)
- **Why needed**: pyannote.audio uses torchcodec backend which requires FFmpeg for audio decoding
- **Supported versions**: FFmpeg 4.x through 7.x (version 8 also supported on Linux)
- **Without FFmpeg**: Speaker diarization will fail with "torchcodec is not installed correctly" warnings

**Notes**:
- Quantized models (4-bit/8-bit) require `bitsandbytes`, which installs automatically on Linux. On macOS/Windows, the package is skipped and quantized models will show a platform incompatibility message.

## Running the Application

### Docker (Recommended for Production)

The application is designed to run in Docker for production use. The Dockerfile includes all system dependencies (including FFmpeg) and builds both the UI and API into a single container.

**Build and run**:
```bash
docker build -t homelab-ai .
docker run -p 12310:12310 -v $(pwd)/data:/haid/data homelab-ai
```

**Key Docker features**:
- Multi-stage build (UI builder + Python runtime)
- FFmpeg pre-installed for speaker diarization
- Playwright browsers pre-installed for web crawling
- Volume mount for persistent model storage
- Health check endpoint configured

### Backend API (Development)
```bash
# Quick start with uv (no activation needed)
uv run python main.py


### Backend API
```bash
# IMPORTANT: Always use --reload-exclude to prevent model downloads from triggering reloads
uv run uvicorn main:app --reload --reload-exclude 'data/*'

# Or activate venv first
source .venv/bin/activate
uvicorn main:app --reload --reload-exclude 'data/*'

# Production (no reload)
python main.py
```

**⚠️ CRITICAL**: Always run with `--reload-exclude 'data/*'` in development. Model downloads will trigger server reloads and interrupt the download process if this flag is missing.

API runs on `http://localhost:8000`
- All endpoints under `/api` prefix
- Docs: `/api/docs`
- Health: `/api/health`
- Ready: `/api/ready`

### Frontend UI
```bash
cd ui
npm run dev
```

UI runs on `http://localhost:3000`
- **Note**: User manages the UI dev server separately. Assume it's running when working on UI features.
- **Development Proxy**: Next.js dev server proxies `/api/*` to `http://localhost:8000/api/*`
  - Single rewrite rule covers all API endpoints
  - Frontend code uses relative URLs (e.g., `fetch('/api/crawl')`, `fetch('/api/health')`)
  - In production, Python server will serve the built UI and handle API requests
  - This enables same-origin requests and simplifies deployment
- **UI Framework**: shadcn/ui (New York style) with Tailwind CSS v4
  - Add components: `npx shadcn@latest add <component>`
  - Components in `src/components/ui/`
  - Utilities in `src/lib/utils.ts`

## Project Structure

```
main.py                    - FastAPI application entry point
src/
  api/
    routers/
      crawl.py            - Crawl API endpoints
    models/
      crawl.py            - Pydantic models for crawl API
ui/
  src/app/
    page.tsx              - Homepage with API status and features
    layout.tsx            - Root layout and metadata
docs/
  product-design.md       - User-focused product overview
  tech-design.md          - Technical implementation details
```

## Architecture Notes

- **Async-first**: All endpoints use async/await
- **crawl4ai**: Uses Playwright under the hood for browser automation
- **No API versioning**: Keeping it simple for homelab use
- **Standard responses**: All responses include `request_id` for tracking

## Implemented Features

- GET `/api` - API status and info
- GET `/api/health` - Health check
- GET `/api/ready` - Readiness check
- POST `/api/crawl` - Web scraping with JS rendering
  - Returns Markdown-formatted content
  - Optional screenshot capture
  - Handles modern JS-heavy websites
- POST `/api/text-to-embedding` - Text embedding generation
- POST `/api/text-generation` - LLM text generation
- POST `/api/image-captioning` - Image caption generation
- POST `/api/image-ocr` - OCR text extraction
- POST `/api/automatic-speech-recognition` - Audio transcription
- GET `/api/hardware` - Hardware status and GPU info
- GET `/api/skills` - Model/skill management
- GET `/api/settings` - Settings management
- GET `/api/docs` - Interactive Swagger documentation
- **`/mcp`** - Model Context Protocol (MCP) server endpoint
  - Exposes all AI capabilities as remote MCP tools
  - Uses Streamable HTTP transport (modern MCP standard)
  - Compatible with Claude Code and other MCP clients
  - See [docs/mcp-setup.md](docs/mcp-setup.md) for details

## MCP (Model Context Protocol) Integration

The application now includes a **remote MCP server** that exposes all AI capabilities as MCP tools. This allows Claude Code and other MCP clients to use the homelab AI services programmatically.

**Implementation:**
- **Router:** `src/api/routers/mcp.py`
- **Framework:** FastMCP (high-level Python SDK)
- **Transport:** Streamable HTTP (modern standard, recommended for 2025+)
- **Endpoint:** `/mcp` (mounted in main.py)
- **Dependencies:** `mcp>=1.8.0` (added to pyproject.toml)

**Available MCP Tools:**
- `crawl_web` - Web scraping with JavaScript rendering
- `embed_text` - Text embedding generation
- `generate_text` - LLM text generation
- `caption_image` - Image captioning
- `ocr_image` - OCR text extraction
- `transcribe_audio` - Speech recognition
- `get_hardware_info` - Hardware/GPU information

**Authentication:** Currently open (no authentication). For production, use reverse proxy authentication or VPN/firewall restrictions.

**Testing:**
```bash
# Check MCP endpoint is mounted
curl http://localhost:12310/api | jq .endpoints.mcp

# Health check
curl http://localhost:12310/api/health
```

## Planned Features

- MCP authentication (bearer token or OAuth)
- Configuration UI for MCP settings
- Monitoring dashboard
