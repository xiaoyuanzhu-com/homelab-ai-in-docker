# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

```bash
# Development mode
python main.py

# Or with auto-reload
uvicorn main:app --reload
```

API runs on `http://localhost:8000`
- Docs: `/docs`
- Health: `/health`

## Project Structure

```
main.py                    - FastAPI application entry point
src/
  api/
    routers/
      crawl.py            - Crawl API endpoints
    models/
      crawl.py            - Pydantic models for crawl API
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

- POST `/api/crawl` - Web scraping with JS rendering
  - Returns Markdown-formatted content
  - Optional screenshot capture
  - Handles modern JS-heavy websites

## Planned Features

- Text embedding (sentence-transformers)
- Image captioning (BLIP/LLaVA)
- Configuration UI
- Monitoring dashboard
