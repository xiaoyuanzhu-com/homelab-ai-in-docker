# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Notes

- **DO NOT commit without explicit user instruction** - Wait for the user to ask to commit changes before running git commit commands.

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

## Running the Application

### Backend API
```bash
# Quick start with uv (no activation needed)
uv run python main.py

# Or with auto-reload
uv run uvicorn main:app --reload

# Or activate venv first
source .venv/bin/activate
python main.py
```

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
- GET `/api/docs` - Interactive Swagger documentation

## Planned Features

- Text embedding (sentence-transformers)
- Image captioning (BLIP/LLaVA)
- Configuration UI
- Monitoring dashboard
