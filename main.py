"""Main FastAPI application for Homelab AI Services."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import (
    crawl,
    text_to_embedding,
    image_to_text,
    history,
    models,
    hardware,
    settings,
)
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


async def periodic_model_cleanup():
    """
    Background task that periodically checks and cleans up idle models.

    Runs every second to check if models have exceeded their idle timeout.
    When a model is idle too long, it's automatically unloaded from GPU
    to free memory for other services.
    """
    logger = logging.getLogger(__name__)
    logger.info("Periodic model cleanup task started (checks every 1 second)")

    while True:
        try:
            # Check every second for idle models
            await asyncio.sleep(1)

            # Check image-to-text models
            try:
                image_to_text.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle image-to-text model: {e}")

            # Check text-to-embedding models
            try:
                text_to_embedding.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle text-to-embedding model: {e}")

        except asyncio.CancelledError:
            logger.info("Periodic model cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in periodic model cleanup: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events (startup and shutdown)."""
    logger = logging.getLogger(__name__)

    # Startup
    logger.info("Starting up application...")

    # Initialize database schema (models, history, download_logs, and settings tables in haid.db)
    logger.info("Initializing database...")
    init_db()

    # Initialize download logs table
    from src.db.download_logs import init_download_logs_table
    init_download_logs_table()

    # Initialize settings table
    from src.db.settings import init_settings_table
    init_settings_table()

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
                    architecture=model_info.get("architecture"),
                    default_prompt=model_info.get("default_prompt"),
                    platform_requirements=model_info.get("platform_requirements"),
                    requires_quantization=model_info.get("requires_quantization", False),
                )
                model_count += 1

        logger.info(f"Loaded {model_count} models from manifest into database")
    else:
        logger.warning(f"Models manifest not found at {manifest_path}")

    # Start background task for periodic model cleanup
    cleanup_task = asyncio.create_task(periodic_model_cleanup())

    logger.info("Startup complete")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")

    # Cancel background cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Clean up ML model resources
    from src.api.routers import text_to_embedding, image_to_text

    try:
        logger.info("Releasing text embedding model resources...")
        text_to_embedding.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up text embedding model: {e}")

    try:
        logger.info("Releasing image captioning model resources...")
        image_to_text.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up image captioning model: {e}")

    # Best-effort shutdown of any joblib/loky process executors to avoid
    # leaked semaphore warnings from Python's resource_tracker on exit.
    try:
        try:
            # Prefer the public import path if available
            from joblib.externals.loky import get_reusable_executor  # type: ignore
        except Exception:
            try:
                # Fallback for older/newer joblib structures
                from joblib.externals.loky.reusable_executor import get_reusable_executor  # type: ignore
            except Exception:
                get_reusable_executor = None  # type: ignore

        if get_reusable_executor is not None:  # type: ignore
            executor = get_reusable_executor()  # type: ignore
            if executor is not None:
                executor.shutdown(wait=True, kill_workers=True)
                logger.info("Shut down joblib/loky reusable executor")
    except Exception as e:
        # Only log at debug to keep shutdown clean in normal scenarios
        logger.debug(f"No joblib/loky executor to shutdown or cleanup failed: {e}")

    logger.info("Shutdown complete")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Homelab AI Services API",
    description="REST API wrapping common AI capabilities for homelab developers",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


# Include routers
app.include_router(crawl.router)
app.include_router(text_to_embedding.router)
app.include_router(image_to_text.router)
app.include_router(history.router)
app.include_router(models.router)
app.include_router(hardware.router)
app.include_router(settings.router)

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
        # Check if the requested file exists (for static assets like .js, .css, .svg, etc.)
        file_path = UI_DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Check for directory with index.html (Next.js static export with trailingSlash: true)
        if file_path.is_dir():
            index_in_dir = file_path / "index.html"
            if index_in_dir.is_file():
                return FileResponse(index_in_dir)

        # Check for .html extension (for paths without trailing slash)
        html_path = UI_DIST_DIR / f"{full_path}.html"
        if html_path.is_file():
            return FileResponse(html_path)

        # Check for path/index.html pattern (for paths without trailing slash)
        path_index = UI_DIST_DIR / full_path / "index.html"
        if path_index.is_file():
            return FileResponse(path_index)

        # Default to root index.html for SPA routing (fallback for unmatched routes)
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
