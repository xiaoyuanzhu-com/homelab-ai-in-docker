"""Models API router for managing embedding models."""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..models.models import (
    DownloadProgressEvent,
    EmbeddingModelInfo,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelListResponse,
)

router = APIRouter(prefix="/api", tags=["models"])

# Active downloads tracking: {model_id: subprocess.Popen}
active_downloads: Dict[str, subprocess.Popen] = {}


def load_model_catalog() -> list[dict]:
    """Load model catalog from JSON file."""
    manifest_path = Path(__file__).parent.parent / "models" / "embedding_models.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return data["models"]


def check_model_downloaded(model_id: str) -> tuple[bool, Optional[int]]:
    """
    Check if a model is already downloaded.

    Args:
        model_id: The model identifier

    Returns:
        Tuple of (is_downloaded, size_in_mb)
    """
    # Don't use get_model_cache_dir() as it creates the directory
    # Instead, construct path without creating it
    from ...config import get_data_dir

    # Preserve the original HuggingFace path structure
    cache_dir = get_data_dir() / "embedding" / "models" / model_id

    # Check if the model directory exists and has content
    if cache_dir.exists():
        # Calculate directory size
        files = list(cache_dir.rglob("*"))
        # Only consider downloaded if there are actual files
        if files:
            total_size = sum(
                f.stat().st_size for f in files if f.is_file()
            )
            size_mb = total_size // (1024 * 1024)
            # Only mark as downloaded if size > 0
            if size_mb > 0:
                return True, size_mb

    return False, None


@router.get("/models/embedding", response_model=ModelListResponse)
async def list_embedding_models() -> ModelListResponse:
    """
    List all available embedding models.

    Returns:
        List of embedding models with download status
    """
    catalog = load_model_catalog()
    models = []

    for model_info in catalog:
        is_downloaded, downloaded_size_mb = check_model_downloaded(model_info["id"])

        models.append(
            EmbeddingModelInfo(
                id=model_info["id"],
                name=model_info["name"],
                team=model_info["team"],
                license=model_info["license"],
                dimensions=model_info["dimensions"],
                languages=model_info["languages"],
                description=model_info["description"],
                size_mb=model_info["size_mb"],
                link=model_info["link"],
                is_downloaded=is_downloaded,
                downloaded_size_mb=downloaded_size_mb,
            )
        )

    return ModelListResponse(models=models)


async def download_model_with_progress(
    model_id: str, expected_size_mb: int, request: Request
) -> AsyncGenerator[str, None]:
    """
    Download a model using huggingface-cli and stream progress via SSE.

    Args:
        model_id: The model identifier
        expected_size_mb: Expected model size from manifest
        request: FastAPI request object for disconnect detection

    Yields:
        SSE events with download progress
    """
    from ...config import get_data_dir

    # Preserve the original HuggingFace path structure (e.g., BAAI/bge-large-en-v1.5)
    cache_dir = get_data_dir() / "embedding" / "models" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Start huggingface-cli download subprocess
        # Pass environment variables including HF_ENDPOINT if set
        import os
        import logging

        logger = logging.getLogger(__name__)
        env = os.environ.copy()

        # Log all HF-related environment variables for transparency
        logger.info(f"=== Download Starting for {model_id} ===")
        logger.info(f"Target directory: {cache_dir}")

        hf_vars = {k: v for k, v in env.items() if k.startswith('HF_')}
        logger.info(f"Current HF environment variables: {hf_vars}")

        # Check for HF_ENDPOINT and ensure both env vars are set
        hf_endpoint = env.get("HF_ENDPOINT")
        if hf_endpoint:
            # Set both HF_ENDPOINT and HF_HUB_ENDPOINT for compatibility
            env["HF_ENDPOINT"] = hf_endpoint
            env["HF_HUB_ENDPOINT"] = hf_endpoint
            logger.info(f"✓ Mirror configured: {hf_endpoint}")
            logger.info(f"✓ Set HF_ENDPOINT={hf_endpoint}")
            logger.info(f"✓ Set HF_HUB_ENDPOINT={hf_endpoint}")
        else:
            logger.warning("⚠ No HF_ENDPOINT set - using default huggingface.co")

        cmd = ["huggingface-cli", "download", model_id, "--local-dir", str(cache_dir)]
        logger.info(f"Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout for unified logging
            universal_newlines=True,
            bufsize=1,
            env=env,  # Pass all environment variables including HF_ENDPOINT
        )

        # Track active download
        active_downloads[model_id] = process

        # Log subprocess details
        logger.info(f"✓ Download process started (PID: {process.pid})")
        logger.info(f"=== Monitoring download progress ===")

        last_progress_time = time.time()
        last_size = 0
        last_log_time = time.time()

        # Monitor download progress
        while process.poll() is None:
            # Read and log subprocess output (non-blocking)
            if process.stdout:
                try:
                    import select
                    # Check if there's output available (with short timeout)
                    if select.select([process.stdout], [], [], 0.1)[0]:
                        line = process.stdout.readline()
                        if line:
                            line_stripped = line.strip()
                            # Always log lines containing URLs or important info
                            if any(keyword in line_stripped for keyword in ['http', 'Downloading', 'Fetching', 'hf.co', 'hf-mirror']):
                                logger.info(f"CLI: {line_stripped}")
                            # Log other output every 5 seconds to avoid spam
                            elif time.time() - last_log_time >= 5.0:
                                logger.info(f"CLI: {line_stripped}")
                                last_log_time = time.time()
                except Exception as e:
                    pass  # Ignore read errors
            # Check if client disconnected
            if await request.is_disconnected():
                process.terminate()
                process.wait(timeout=5)
                del active_downloads[model_id]
                return

            # Get current directory size
            current_size = 0
            if cache_dir.exists():
                current_size = sum(
                    f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
                )

            current_size_mb = current_size // (1024 * 1024)

            # Send progress update every second or if significant size change
            # Don't calculate percent since manifest sizes are unreliable
            current_time = time.time()
            if current_time - last_progress_time >= 1.0 or current_size_mb > last_size + 10:
                event = DownloadProgressEvent(
                    type="progress",
                    percent=None,  # Don't show percentage - manifest sizes are unreliable
                    current_mb=current_size_mb,
                    total_mb=expected_size_mb,  # Show as reference only
                )
                yield event.model_dump_json()
                last_progress_time = current_time
                last_size = current_size_mb

            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.5)

        # Download completed - check exit code
        returncode = process.returncode
        del active_downloads[model_id]

        logger.info(f"=== Download Process Completed ===")
        logger.info(f"Exit code: {returncode}")

        if returncode != 0:
            # Download failed - read remaining output
            stdout_output = process.stdout.read() if process.stdout else "Unknown error"
            logger.error(f"✗ Download FAILED for {model_id}")
            logger.error(f"Error output: {stdout_output[:500]}")
            event = DownloadProgressEvent(
                type="error", message=f"Download failed: {stdout_output[:200]}"
            )
            yield event.model_dump_json()
            # Clean up partial download
            import shutil

            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            return

        # Calculate final size for reporting
        final_size = 0
        if cache_dir.exists():
            final_size = sum(
                f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
            )
        final_size_mb = final_size // (1024 * 1024)

        logger.info(f"✓ Download SUCCESS for {model_id}")
        logger.info(f"Downloaded size: {final_size_mb} MB")
        logger.info(f"Location: {cache_dir}")
        logger.info(f"=== Download Complete ===")

        # Success! (no size verification - sizes vary too much)
        event = DownloadProgressEvent(
            type="complete", percent=100, size_mb=final_size_mb
        )
        yield event.model_dump_json()

    except Exception as e:
        # Unexpected error
        if model_id in active_downloads:
            del active_downloads[model_id]
        event = DownloadProgressEvent(type="error", message=f"Unexpected error: {str(e)}")
        yield event.model_dump_json()


@router.get("/models/embedding/{model_id:path}/download")
async def download_embedding_model_stream(model_id: str, request: Request):
    """
    Download an embedding model with SSE progress streaming.

    Args:
        model_id: The model identifier (can contain slashes, e.g., BAAI/bge-large-en-v1.5)
        request: FastAPI request

    Returns:
        EventSourceResponse with progress events

    Raises:
        HTTPException: If model not found or already downloaded
    """
    # Validate model exists in catalog
    catalog = load_model_catalog()
    model_info = next((m for m in catalog if m["id"] == model_id), None)

    if not model_info:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": f"Model '{model_id}' not found in catalog",
            },
        )

    # Check if already downloaded
    is_downloaded, _ = check_model_downloaded(model_id)
    if is_downloaded:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ALREADY_DOWNLOADED",
                "message": f"Model '{model_id}' is already downloaded",
            },
        )

    # Check if already downloading
    if model_id in active_downloads:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DOWNLOAD_IN_PROGRESS",
                "message": f"Model '{model_id}' is already being downloaded",
            },
        )

    return EventSourceResponse(
        download_model_with_progress(model_id, model_info["size_mb"], request)
    )


@router.post("/models/embedding/download", response_model=ModelDownloadResponse)
async def download_embedding_model(
    request: ModelDownloadRequest,
) -> ModelDownloadResponse:
    """
    Download an embedding model.

    Args:
        request: Model download request

    Returns:
        Download status

    Raises:
        HTTPException: If download fails
    """
    # Validate model exists in catalog
    catalog = load_model_catalog()
    valid_model_ids = [m["id"] for m in catalog]
    if request.model_id not in valid_model_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": f"Model '{request.model_id}' not found in catalog",
            },
        )

    try:
        # Check if already downloaded
        is_downloaded, size_mb = check_model_downloaded(request.model_id)
        if is_downloaded:
            return ModelDownloadResponse(
                model_id=request.model_id,
                status="already_downloaded",
                message=f"Model already downloaded ({size_mb} MB)",
            )

        # Download the model by loading it
        # Create cache directory only when downloading
        from ...config import get_data_dir

        safe_model_name = request.model_id.replace("/", "--")
        cache_dir = get_data_dir() / "embedding" / safe_model_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        model = SentenceTransformer(request.model_id, cache_folder=str(cache_dir))

        # Get final size
        _, size_mb = check_model_downloaded(request.model_id)

        return ModelDownloadResponse(
            model_id=request.model_id,
            status="downloaded",
            message=f"Model downloaded successfully ({size_mb} MB)",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DOWNLOAD_FAILED",
                "message": f"Failed to download model: {str(e)}",
            },
        )


@router.delete("/models/embedding/{model_id:path}/download")
async def cancel_download(model_id: str):
    """
    Cancel an in-progress model download.

    Args:
        model_id: The model identifier (can contain slashes)

    Returns:
        Cancellation status
    """
    if model_id not in active_downloads:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NOT_DOWNLOADING",
                "message": f"Model '{model_id}' is not currently being downloaded",
            },
        )

    try:
        process = active_downloads[model_id]
        process.terminate()
        process.wait(timeout=5)
        del active_downloads[model_id]

        # Clean up partial download
        from ...config import get_data_dir
        import shutil

        cache_dir = get_data_dir() / "embedding" / "models" / model_id

        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        return {
            "model_id": model_id,
            "status": "cancelled",
            "message": "Download cancelled and partial files removed",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CANCEL_FAILED",
                "message": f"Failed to cancel download: {str(e)}",
            },
        )


@router.delete("/models/embedding/{model_id:path}")
async def delete_embedding_model(model_id: str):
    """
    Delete a downloaded embedding model.

    Args:
        model_id: The model identifier (can contain slashes)

    Returns:
        Deletion status

    Raises:
        HTTPException: If deletion fails
    """
    # Validate model exists in catalog
    catalog = load_model_catalog()
    valid_model_ids = [m["id"] for m in catalog]
    if model_id not in valid_model_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": f"Model '{model_id}' not found in catalog",
            },
        )

    try:
        from ...config import get_data_dir

        cache_dir = get_data_dir() / "embedding" / "models" / model_id

        if not cache_dir.exists():
            return {
                "model_id": model_id,
                "status": "not_found",
                "message": "Model not downloaded",
            }

        # Delete the model directory
        import shutil
        shutil.rmtree(cache_dir)

        return {
            "model_id": model_id,
            "status": "deleted",
            "message": "Model deleted successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DELETE_FAILED",
                "message": f"Failed to delete model: {str(e)}",
            },
        )
