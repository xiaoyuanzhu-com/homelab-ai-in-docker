"""Main FastAPI application for Homelab AI Services."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.routers import crawl, embedding, caption, history, models, hardware

# Configure data directories for crawl4ai and playwright
# Use environment variables if set, otherwise default to /haid/data
HAID_DATA_DIR = Path(os.getenv("HAID_DATA_DIR", "/haid/data"))

# Set crawl4ai base directory
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
