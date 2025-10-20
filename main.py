"""Main FastAPI application for Homelab AI Services."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import crawl, embedding, caption, history, models_new as models, hardware

# Configure data directories for crawl4ai and playwright
# Use environment variables if set, otherwise default to /haid/data
HAID_DATA_DIR = Path(os.getenv("HAID_DATA_DIR", "/haid/data"))

# Set crawl4ai base directory

# Set HuggingFace cache directory for all models (transformers, sentence-transformers, etc.)
# This centralizes all model storage in data/models with HF's standard structure:
# data/models/hub/models--{org}--{model-name}/
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(HAID_DATA_DIR / "models")
if "CRAWL4AI_BASE_DIRECTORY" not in os.environ:
    os.environ["CRAWL4AI_BASE_DIRECTORY"] = str(HAID_DATA_DIR / "crawl4ai")

# Set playwright browsers path
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(HAID_DATA_DIR / "playwright")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title="Homelab AI Services API",
    description="REST API wrapping common AI capabilities for homelab developers",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Include routers
app.include_router(crawl.router)
app.include_router(embedding.router)
app.include_router(caption.router)
app.include_router(history.router)
app.include_router(models.router)
app.include_router(hardware.router)

# Serve static files from the UI build
UI_DIST_DIR = Path(__file__).parent / "ui" / "dist"
if UI_DIST_DIR.exists():
    # Mount static assets with caching
    app.mount("/_next", StaticFiles(directory=str(UI_DIST_DIR / "_next")), name="next-static")

    # Serve other static files (images, etc.)
    for static_dir in ["images", "icons", "fonts"]:
        static_path = UI_DIST_DIR / static_dir
        if static_path.exists():
            app.mount(f"/{static_dir}", StaticFiles(directory=str(static_path)), name=static_dir)


@app.get("/api")
async def root():
    """Root API endpoint with service information."""
    return {
        "name": "Homelab AI Services API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "crawl": "/api/crawl",
            "embed": "/api/embed",
            "caption": "/api/caption",
            "hardware": "/api/hardware",
            "docs": "/api/docs",
            "health": "/api/health",
            "ready": "/api/ready",
        },
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/ready")
async def ready():
    """Readiness check endpoint."""
    return {"status": "ready", "services": {"crawl": "available", "embedding": "available", "caption": "available"}}


# Catch-all route to serve the UI (must be last)
@app.get("/{full_path:path}")
async def serve_ui(full_path: str):
    """Serve the Next.js static UI for all non-API routes."""
    if UI_DIST_DIR.exists():
        # Check if the requested file exists
        file_path = UI_DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Check for .html extension
        html_path = UI_DIST_DIR / f"{full_path}.html"
        if html_path.is_file():
            return FileResponse(html_path)

        # Default to index.html for SPA routing
        index_path = UI_DIST_DIR / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)

    # If UI is not built, return a friendly message
    return JSONResponse(
        status_code=503,
        content={"error": "UI not available", "message": "The UI has not been built yet. Please build the UI first."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
