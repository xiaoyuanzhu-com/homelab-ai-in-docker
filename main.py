"""Main FastAPI application for Homelab AI Services."""

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
    history,
    skills,
    hardware,
    settings,
    mcp,
)
from src.db.skills import init_skills_table, upsert_skill, SkillStatus

# Configure data directories for model caching
# Use environment variables if set, otherwise default to /haid/data
HAID_DATA_DIR = Path(os.getenv("HAID_DATA_DIR", "/haid/data"))

# Set HuggingFace home directory for model caching (replaces deprecated TRANSFORMERS_CACHE)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(HAID_DATA_DIR / "models")

# Configure PyTorch TF32 for better performance on Ampere+ GPUs
# This suppresses pyannote.audio reproducibility warnings
try:
    import torch

    if torch.cuda.is_available():
        # Prefer new TF32 configuration API to avoid deprecation warnings (PyTorch 2.9+)
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        except AttributeError:
            torch.backends.cuda.matmul.allow_tf32 = True

        try:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except AttributeError:
            torch.backends.cudnn.allow_tf32 = True
except ImportError:
    pass  # torch not installed, skip

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

            # Check image captioning models
            try:
                image_captioning.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle image captioning model: {e}")

            # Check image OCR models
            try:
                image_ocr.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle image OCR model: {e}")

            # Check text-to-embedding models
            try:
                text_to_embedding.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle text-to-embedding model: {e}")

            # Check text generation models
            try:
                text_generation.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle text generation model: {e}")

            # Check ASR models
            try:
                automatic_speech_recognition.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle ASR model: {e}")

            # Check speaker embedding models
            try:
                speaker_embedding.check_and_cleanup_idle_model()
            except Exception as e:
                logger.debug(f"Error checking idle speaker embedding model: {e}")

        except asyncio.CancelledError:
            logger.info("Periodic model cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in periodic model cleanup: {e}", exc_info=True)


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

        # Initialize database schema (skills, history, download_logs, and settings tables in haid.db)
        logger.info("Initializing database...")
        init_skills_table()

        # Initialize download logs table
        from src.db.download_logs import init_download_logs_table
        init_download_logs_table()

        # Initialize settings table
        from src.db.settings import init_settings_table
        init_settings_table()

        # Initialize history storage (creates request_history table)
        from src.storage.history import history_storage
        # history_storage.__init__() already called on import, table created

        # Load skills manifest
        skills_manifest_path = Path(__file__).parent / "src" / "api" / "skills" / "skills_manifest.json"
        if skills_manifest_path.exists():
            with open(skills_manifest_path, "r") as f:
                skills_manifest = json.load(f)

            skill_count = 0
            for skill in skills_manifest.get("skills", []):
                upsert_skill(
                    skill_id=skill["id"],
                    label=skill["label"],
                    provider=skill.get("provider", ""),
                    tasks=skill.get("tasks", []),
                    architecture=skill.get("architecture"),
                    default_prompt=skill.get("default_prompt"),
                    platform_requirements=skill.get("platform_requirements"),
                    supports_markdown=skill.get("supports_markdown", False),
                    requires_quantization=skill.get("requires_quantization", False),
                    requires_download=skill.get("requires_download", True),
                    hf_model=skill.get("hf_model"),
                    reference_url=skill.get("reference_url"),
                    size_mb=skill.get("size_mb"),
                    parameters_m=skill.get("parameters_m"),
                    gpu_memory_mb=skill.get("gpu_memory_mb"),
                    dimensions=skill.get("dimensions"),
                    initial_status=(
                        SkillStatus.READY if not skill.get("requires_download", True) else SkillStatus.INIT
                    ),
                )
                skill_count += 1

            logger.info(f"Loaded {skill_count} skills from manifest into database")
        else:
            logger.warning(f"Skills manifest not found at {skills_manifest_path}")

        # Start background task for periodic model cleanup
        cleanup_task = asyncio.create_task(periodic_model_cleanup())

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

        # Clean up ML model resources
        from src.api.routers import text_to_embedding, text_generation, image_captioning, image_ocr, automatic_speech_recognition

        try:
            logger.info("Releasing text embedding model resources...")
            text_to_embedding.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up text embedding model: {e}")

        try:
            logger.info("Releasing text generation model resources...")
            text_generation.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up text generation model: {e}")

        try:
            logger.info("Releasing image captioning model resources...")
            image_captioning.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up image captioning model: {e}")

        try:
            logger.info("Releasing image OCR model resources...")
            image_ocr.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up image OCR model: {e}")

        try:
            logger.info("Releasing ASR model resources...")
            automatic_speech_recognition.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up ASR model: {e}")

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
app.include_router(history.router)
app.include_router(skills.router)
app.include_router(hardware.router)
app.include_router(settings.router)

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
    """Health check endpoint."""
    return {"status": "healthy"}


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
