"""Main FastAPI application for Homelab AI Services."""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import crawl, embedding, caption, history, models_new as models, hardware
from src.db.models import init_db, upsert_model

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


@app.on_event("startup")
async def startup_event():
    """Initialize database and load models manifest on startup."""
    logger = logging.getLogger(__name__)

    # Initialize database schema (models, history, and download_logs tables in haid.db)
    logger.info("Initializing database...")
    init_db()

    # Initialize download logs table
    from src.db.download_logs import init_download_logs_table
    init_download_logs_table()

    # Initialize history storage (creates request_history table)
    from src.storage.history import history_storage
    # history_storage.__init__() already called on import, table created

    # Load models manifest and upsert into database
    logger.info("Loading models manifest...")
    manifest_path = Path(__file__).parent / "src" / "api" / "models" / "models_manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Upsert all models from manifest
        model_count = 0
        for model_type, models_list in manifest.items():
            for model_info in models_list:
                upsert_model(
                    model_id=model_info["id"],
                    name=model_info["name"],
                    team=model_info["team"],
                    model_type=model_type,
                    task=model_info["task"],
                    size_mb=model_info["size_mb"],
                    parameters_m=model_info["parameters_m"],
                    gpu_memory_mb=model_info["gpu_memory_mb"],
                    link=model_info["link"],
                )
                model_count += 1

        logger.info(f"Loaded {model_count} models from manifest into database")
    else:
        logger.warning(f"Models manifest not found at {manifest_path}")


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
