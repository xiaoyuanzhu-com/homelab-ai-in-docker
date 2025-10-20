"""Unified models API router for managing all AI models."""

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/api", tags=["models"])

# Active downloads tracking: {model_id: subprocess.Popen}
active_downloads: Dict[str, subprocess.Popen] = {}


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    team: str
    type: str
    task: str
    size_mb: int
    parameters_m: int
    gpu_memory_mb: int
    link: str
    is_downloaded: bool
    downloaded_size_mb: Optional[int] = None


class ModelsResponse(BaseModel):
    """Response containing list of models."""
    models: list[ModelInfo]


class DownloadProgressEvent(BaseModel):
    """Download progress event for SSE."""
    type: str  # "progress", "complete", "error"
    percent: Optional[int] = None
    current_mb: Optional[int] = None
    total_mb: Optional[int] = None
    size_mb: Optional[int] = None
    message: Optional[str] = None


def load_models_manifest() -> dict:
    """Load unified models manifest from JSON file."""
    manifest_path = Path(__file__).parent.parent / "models" / "models_manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def check_model_downloaded_hf(model_id: str) -> tuple[bool, Optional[int]]:
    """
    Check if a model is downloaded in custom directory structure.

    Models are stored in: data/models/{org}/{model}/

    Args:
        model_id: The model identifier (e.g., "BAAI/bge-large-en-v1.5")

    Returns:
        Tuple of (is_downloaded, size_in_mb)
    """
    from ...config import get_data_dir

    # Use custom directory structure: data/models/{org}/{model}
    # "BAAI/bge-large-en-v1.5" -> "data/models/BAAI/bge-large-en-v1.5"
    cache_path = get_data_dir() / "models" / model_id

    if cache_path.exists():
        # Calculate directory size
        files = list(cache_path.rglob("*"))
        if files:
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size // (1024 * 1024)
            if size_mb > 0:
                return True, size_mb

    return False, None


@router.get("/models", response_model=ModelsResponse)
async def list_all_models() -> ModelsResponse:
    """
    List all available AI models across all types.

    Returns:
        List of all models with download status
    """
    manifest = load_models_manifest()
    models = []

    # Process each model type
    for model_type, type_models in manifest.items():
        for model_info in type_models:
            is_downloaded, downloaded_size_mb = check_model_downloaded_hf(model_info["id"])

            models.append(
                ModelInfo(
                    id=model_info["id"],
                    name=model_info["name"],
                    team=model_info["team"],
                    type=model_type,
                    task=model_info["task"],
                    size_mb=model_info["size_mb"],
                    parameters_m=model_info["parameters_m"],
                    gpu_memory_mb=model_info["gpu_memory_mb"],
                    link=model_info["link"],
                    is_downloaded=is_downloaded,
                    downloaded_size_mb=downloaded_size_mb,
                )
            )

    return ModelsResponse(models=models)


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
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Use custom directory structure: data/models/{org}/{model}
    # model_id like "sentence-transformers/all-MiniLM-L6-v2" becomes
    # "data/models/sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = get_data_dir() / "models" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = os.environ.copy()
        logger.info(f"=== Download Starting for {model_id} ===")
        logger.info(f"Target directory: {cache_dir}")

        cmd = ["huggingface-cli", "download", model_id, "--local-dir", str(cache_dir)]
        logger.info(f"Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        active_downloads[model_id] = process
        logger.info(f"✓ Download process started (PID: {process.pid})")

        last_progress_time = time.time()
        last_size = 0
        output_lines = []

        # Monitor download progress
        while process.poll() is None:
            # Check if client disconnected
            if await request.is_disconnected():
                process.terminate()
                process.wait(timeout=5)
                del active_downloads[model_id]
                return

            # Read any available output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    logger.info(f"Download output: {line.strip()}")

            # Get current directory size
            current_size = 0
            if cache_dir.exists():
                current_size = sum(
                    f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
                )

            current_size_mb = current_size // (1024 * 1024)

            # Send progress update every second
            current_time = time.time()
            if current_time - last_progress_time >= 1.0 or current_size_mb > last_size + 10:
                event = DownloadProgressEvent(
                    type="progress",
                    current_mb=current_size_mb,
                    total_mb=expected_size_mb,
                )
                yield event.model_dump_json()
                last_progress_time = current_time
                last_size = current_size_mb

            await asyncio.sleep(0.5)

        # Read any remaining output
        if process.stdout:
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.splitlines():
                    if line.strip():
                        output_lines.append(line.strip())
                        logger.info(f"Download output: {line.strip()}")

        # Download completed - check exit code
        returncode = process.returncode
        del active_downloads[model_id]

        logger.info(f"=== Download Process Completed ===")
        logger.info(f"Exit code: {returncode}")

        if returncode != 0:
            logger.error(f"✗ Download FAILED for {model_id}")
            # Log last few lines of output for debugging
            if output_lines:
                logger.error(f"Last output lines: {output_lines[-10:]}")

            error_msg = "Download failed"
            if output_lines:
                # Try to extract meaningful error from output
                error_lines = [l for l in output_lines[-5:] if "error" in l.lower() or "failed" in l.lower()]
                if error_lines:
                    error_msg = error_lines[-1][:200]  # Limit message length

            event = DownloadProgressEvent(
                type="error", message=error_msg
            )
            yield event.model_dump_json()
            # Clean up partial download
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            return

        # Calculate final size
        final_size = 0
        if cache_dir.exists():
            final_size = sum(
                f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
            )
        final_size_mb = final_size // (1024 * 1024)

        logger.info(f"✓ Download SUCCESS for {model_id}")
        logger.info(f"Downloaded size: {final_size_mb} MB")

        event = DownloadProgressEvent(
            type="complete", percent=100, size_mb=final_size_mb
        )
        yield event.model_dump_json()

    except Exception as e:
        if model_id in active_downloads:
            del active_downloads[model_id]
        event = DownloadProgressEvent(type="error", message=f"Error: {str(e)}")
        yield event.model_dump_json()


@router.get("/models/download")
async def download_model_stream(model: str, request: Request):
    """
    Download a model with SSE progress streaming.

    Args:
        model: The model identifier (e.g., "BAAI/bge-large-en-v1.5")
        request: FastAPI request

    Returns:
        EventSourceResponse with progress events
    """
    # Validate model exists in manifest
    manifest = load_models_manifest()
    all_models = {}
    for type_models in manifest.values():
        for m in type_models:
            all_models[m["id"]] = m

    if model not in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found in catalog",
        )

    # Check if already downloaded
    is_downloaded, _ = check_model_downloaded_hf(model)
    if is_downloaded:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is already downloaded",
        )

    # Check if already downloading
    if model in active_downloads:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model}' is already being downloaded",
        )

    model_info = all_models[model]
    return EventSourceResponse(
        download_model_with_progress(model, model_info["size_mb"], request)
    )


@router.delete("/models/{model_id:path}")
async def delete_model(model_id: str):
    """
    Delete a downloaded model from HuggingFace cache.

    Args:
        model_id: The model identifier (can contain slashes)

    Returns:
        Deletion status
    """
    # Validate model exists in manifest
    manifest = load_models_manifest()
    all_models = []
    for type_models in manifest.values():
        all_models.extend([m["id"] for m in type_models])

    if model_id not in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not found in catalog",
        )

    try:
        from ...config import get_data_dir

        # Use custom directory structure: data/models/{org}/{model}
        cache_path = get_data_dir() / "models" / model_id

        if not cache_path.exists():
            return {
                "model_id": model_id,
                "status": "not_found",
                "message": "Model not downloaded",
            }

        # Delete the model directory
        shutil.rmtree(cache_path)

        return {
            "model_id": model_id,
            "status": "deleted",
            "message": "Model deleted successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}",
        )
