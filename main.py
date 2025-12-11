"""Main FastAPI application for Homelab AI Services.

This is the lean API server - all ML dependencies are in worker environments.
Workers are spawned on-demand with their own isolated Python environments.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import (
    crawl,
    text_to_embedding,
    text_generation,
    image_captioning,
    image_ocr,
    automatic_speech_recognition,
    speaker_embedding,
    whisperx,
    history,
    models,
    libs,
    task_options,
    hardware,
    settings,
    mcp,
    doc_to_markdown,
    doc_to_screenshot,
    environments,
)
from src.db.models import init_models_table, upsert_model, delete_model, get_all_models
from src.db.libs import init_libs_table, upsert_lib
from src.db.status import DownloadStatus

# Configure data directories for model caching
# Use environment variables if set, otherwise default to /haid/data
HAID_DATA_DIR = Path(os.getenv("HAID_DATA_DIR", "/haid/data"))

# Set HuggingFace home directory for model caching
# Workers inherit this via environment variables
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(HAID_DATA_DIR / "models")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def periodic_worker_status():
    """
    Background task that periodically logs worker status.

    Workers manage their own idle timeouts and self-terminate.
    This task just monitors and logs status for debugging.
    """
    logger = logging.getLogger(__name__)
    logger.info("Periodic worker status task started (checks every 30 seconds)")

    from src.worker import get_coordinator

    while True:
        try:
            await asyncio.sleep(30)

            try:
                coordinator = get_coordinator()
                status = await coordinator.get_worker_status()
                if status:
                    logger.debug(f"Active workers: {list(status.keys())}")
            except Exception as e:
                logger.debug(f"Error checking worker status: {e}")

        except asyncio.CancelledError:
            logger.info("Periodic worker status task cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in periodic worker status: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events (startup and shutdown)."""
    logger = logging.getLogger(__name__)

    cleanup_task: asyncio.Task | None = None
    mcp_lifespan_cm = None
    mcp_lifespan_started = False

    try:
        # Startup
        logger.info("Starting up application...")

        # Initialize database schema (models, libs, history, download_logs, settings)
        logger.info("Initializing database...")
        init_models_table()
        init_libs_table()

        # Initialize download logs table
        from src.db.download_logs import init_download_logs_table
        init_download_logs_table()

        # Initialize settings table
        from src.db.settings import init_settings_table, get_setting_int
        init_settings_table()

        # Initialize global worker coordinator
        from src.worker import get_coordinator
        coordinator = get_coordinator()
        idle_timeout = get_setting_int("worker_idle_timeout_seconds", 60)
        logger.info(
            f"Worker coordinator initialized: idle_timeout={idle_timeout}s"
        )

        # Initialize history storage (creates request_history table)
        from src.storage.history import history_storage
        # history_storage.__init__() already called on import, table created

        # Load catalog manifests (separate models and libs)
        models_manifest_path = Path(__file__).parent / "src" / "api" / "catalog" / "models.json"
        libs_manifest_path = Path(__file__).parent / "src" / "api" / "catalog" / "libs.json"

        # Models
        if models_manifest_path.exists():
            try:
                with open(models_manifest_path, "r", encoding="utf-8") as f:
                    models_manifest = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load models manifest: {e}")
            else:
                m_count = 0
                for item in models_manifest.get("models", []):
                    try:
                        upsert_model(
                            model_id=item["id"],
                            label=item["label"],
                            provider=item.get("provider", ""),
                            tasks=item.get("tasks", []),
                            architecture=item.get("architecture"),
                            default_prompt=item.get("default_prompt"),
                            platform_requirements=item.get("platform_requirements"),
                            supports_markdown=item.get("supports_markdown", False),
                            requires_quantization=item.get("requires_quantization", False),
                            requires_download=item.get("requires_download", True),
                            hf_model=item.get("hf_model"),
                            reference_url=item.get("reference_url"),
                            size_mb=item.get("size_mb"),
                            parameters_m=item.get("parameters_m"),
                            gpu_memory_mb=item.get("gpu_memory_mb"),
                            dimensions=item.get("dimensions"),
                            python_env=item.get("python_env"),
                            initial_status=DownloadStatus.INIT if item.get("requires_download", True) else DownloadStatus.READY,
                        )
                        m_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to upsert model '{item.get('id')}': {e}")
                logger.info(f"Loaded {m_count} models from manifest into database")

                # Clean up models not in manifest
                manifest_ids = {item["id"] for item in models_manifest.get("models", [])}
                db_models = get_all_models()
                removed_count = 0
                for row in db_models:
                    if row["id"] not in manifest_ids:
                        delete_model(row["id"])
                        removed_count += 1
                        logger.info(f"Removed model '{row['id']}' (not in manifest)")
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} models not in manifest")
        else:
            logger.warning(f"Models manifest not found at {models_manifest_path}")

        # Libs
        if libs_manifest_path.exists():
            try:
                with open(libs_manifest_path, "r", encoding="utf-8") as f:
                    libs_manifest = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load libs manifest: {e}")
            else:
                l_count = 0
                for item in libs_manifest.get("libs", []):
                    try:
                        upsert_lib(
                            lib_id=item["id"],
                            label=item["label"],
                            provider=item.get("provider", ""),
                            tasks=item.get("tasks", []),
                            architecture=item.get("architecture"),
                            default_prompt=item.get("default_prompt"),
                            python_env=item.get("python_env"),
                            platform_requirements=item.get("platform_requirements"),
                            supports_markdown=item.get("supports_markdown", False),
                            requires_quantization=item.get("requires_quantization", False),
                            requires_download=item.get("requires_download", False),
                            reference_url=item.get("reference_url"),
                            size_mb=item.get("size_mb"),
                            parameters_m=item.get("parameters_m"),
                            gpu_memory_mb=item.get("gpu_memory_mb"),
                            dimensions=item.get("dimensions"),
                            initial_status="ready",
                        )
                        l_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to upsert lib '{item.get('id')}': {e}")
                logger.info(f"Loaded {l_count} libs from manifest into database")
        else:
            logger.warning(f"Libs manifest not found at {libs_manifest_path}")

        # Start background task for periodic worker status monitoring
        cleanup_task = asyncio.create_task(periodic_worker_status())

        # Ensure the MCP sub-application lifecycle is running so its session manager is ready
        mcp_app_instance = globals().get("mcp_app")
        if mcp_app_instance is not None:
            try:
                mcp_lifespan_cm = mcp_app_instance.router.lifespan_context(mcp_app_instance)
                await mcp_lifespan_cm.__aenter__()
                mcp_lifespan_started = True
                logger.info("MCP server lifespan started")
            except Exception as e:
                logger.error(f"Failed to start MCP server lifespan: {e}", exc_info=True)
                mcp_lifespan_cm = None
        else:
            logger.info("MCP server not configured; skipping lifespan startup")

        logger.info("Startup complete")

        yield  # Application runs here

    finally:
        # Shutdown
        logger.info("Shutting down application...")

        # Cancel background cleanup task
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop MCP session manager before tearing down shared resources
        if mcp_lifespan_cm is not None and mcp_lifespan_started:
            try:
                await mcp_lifespan_cm.__aexit__(None, None, None)
                logger.info("MCP server lifespan stopped")
            except Exception as e:
                logger.warning(f"Error shutting down MCP server lifespan: {e}", exc_info=True)

        # Shutdown all workers
        try:
            from src.worker import get_coordinator
            logger.info("Shutting down all workers...")
            coordinator = get_coordinator()
            await coordinator.shutdown_all()
            logger.info("All workers shut down")
        except Exception as e:
            logger.warning(f"Error shutting down workers: {e}")

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
app.include_router(text_generation.router)
app.include_router(image_captioning.router)
app.include_router(image_ocr.router)
app.include_router(automatic_speech_recognition.router)
app.include_router(speaker_embedding.router)
app.include_router(whisperx.router)
app.include_router(history.router)
app.include_router(models.router)
app.include_router(libs.router)
app.include_router(task_options.router)
app.include_router(hardware.router)
app.include_router(settings.router)
app.include_router(doc_to_markdown.router)
app.include_router(doc_to_screenshot.router)
app.include_router(environments.router)

# Initialize logger for startup messages
logger = logging.getLogger(__name__)

# Mount MCP (Model Context Protocol) server at /mcp
# Note: FastMCP creates a Starlette app with routes, but we need to ensure
# the catch-all route doesn't interfere. The catch-all explicitly skips "/mcp"
mcp_app = None
try:
    mcp_app = mcp.get_mcp_app()
    app.mount("/mcp", mcp_app)
    logger.info("MCP server mounted at /mcp")
except Exception as e:
    logger.warning(f"Failed to mount MCP server (mcp package may not be installed): {e}")


@app.middleware("http")
async def trailing_slash_redirect(request: Request, call_next):
    """Redirect bare /mcp and /doc requests to their trailing-slash versions."""
    from fastapi.responses import RedirectResponse

    # Redirect /mcp to /mcp/ for the mounted Streamable HTTP app
    if request.method in {"GET", "POST"} and request.url.path == "/mcp":
        if globals().get("mcp_app") is None:
            raise HTTPException(status_code=404, detail="MCP server not available")
        return RedirectResponse(url="/mcp/", status_code=307)

    # Redirect /doc to /doc/ for the mounted MkDocs static files
    if request.method == "GET" and request.url.path == "/doc":
        return RedirectResponse(url="/doc/", status_code=307)

    return await call_next(request)

# Mount MkDocs documentation at /doc
DOCS_SITE_DIR = Path(__file__).parent / "site"
if DOCS_SITE_DIR.exists():
    app.mount("/doc", StaticFiles(directory=str(DOCS_SITE_DIR), html=True), name="mkdocs")
    logger.info("MkDocs documentation mounted at /doc")

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
            "embed": "/api/text-to-embedding",
            "text_generation": "/api/text-generation",
            "image_captioning": "/api/image-captioning",
            "image_ocr": "/api/image-ocr",
            "automatic_speech_recognition": "/api/automatic-speech-recognition",
            "whisperx": "/api/whisperx",
            "hardware": "/api/hardware",
            "docs": "/api/docs",
            "health": "/api/health",
            "ready": "/api/ready",
            "mcp": "/mcp",
            "documentation": "/doc",
        },
    }


@app.get("/api/health")
async def health():
    """Health check endpoint with worker and environment status."""
    from src.worker import get_coordinator, get_env_manager

    try:
        coordinator = get_coordinator()
        workers = await coordinator.get_worker_status()
        gpu_lock_held = coordinator.is_gpu_locked()
    except Exception:
        workers = {}
        gpu_lock_held = False

    try:
        env_manager = get_env_manager()
        envs = env_manager.list_environments()
        environments = {
            env_id: {"status": info.status.value, "size_mb": info.size_mb}
            for env_id, info in envs.items()
        }
    except Exception:
        environments = {}

    return {
        "status": "healthy",
        "workers": workers,
        "gpu_lock_held": gpu_lock_held,
        "environments": environments,
    }


@app.get("/api/ready")
async def ready():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "services": {
            "crawl": "available",
            "embedding": "available",
            "text_generation": "available",
            "image_captioning": "available",
            "image_ocr": "available",
            "automatic_speech_recognition": "available",
        },
    }


# Serve UI at root path (mount at "/" as catch-all, must be LAST)
# Note: Mounts are evaluated in reverse order, so specific mounts like /api, /mcp, /doc
# will be matched first before falling back to the root mount
if UI_DIST_DIR.exists():
    # For Next.js static export with SPA routing, we need a custom StaticFiles class
    # that serves index.html for all unmatched routes
    class SPAStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except Exception:
                # Don't intercept paths that should be handled by other mounts
                # (FastAPI/Starlette should have already routed those, but be defensive)
                request_path = scope.get("path", "")
                if request_path.startswith("/doc"):
                    # Let the /doc mount handle this (don't fall back to UI)
                    raise

                # If file not found, serve index.html for SPA routing
                index_path = Path(self.directory) / "index.html"
                if index_path.is_file():
                    return FileResponse(index_path)
                raise

    app.mount("/", SPAStaticFiles(directory=str(UI_DIST_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12310)
