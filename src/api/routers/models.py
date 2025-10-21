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
    status: str  # init, downloading, failed, downloaded
    downloaded_size_mb: Optional[int] = None
    error_message: Optional[str] = None


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


@router.get("/models", response_model=ModelsResponse)
async def list_all_models(task: Optional[str] = None) -> ModelsResponse:
    """
    List all available AI models across all types from database.

    Args:
        task: Optional task ID filter (e.g., "image-captioning", "image-ocr", "embedding")

    Returns:
        List of all models with download status, filtered by task if specified
    """
    from ...db.models import get_all_models

    db_models = get_all_models()
    models = []

    for db_model in db_models:
        # Filter by task ID if specified (case-insensitive)
        if task and db_model["task"].lower() != task.lower():
            continue

        models.append(
            ModelInfo(
                id=db_model["id"],
                name=db_model["name"],
                team=db_model["team"],
                type=db_model["type"],
                task=db_model["task"],
                size_mb=db_model["size_mb"],
                parameters_m=db_model["parameters_m"],
                gpu_memory_mb=db_model["gpu_memory_mb"],
                link=db_model["link"],
                status=db_model["status"],
                downloaded_size_mb=db_model["downloaded_size_mb"],
                error_message=db_model["error_message"],
            )
        )

    return ModelsResponse(models=models)


@router.get("/models/{model_id:path}/logs")
async def get_model_download_logs(model_id: str):
    """
    Get download logs for a specific model.

    Args:
        model_id: The model identifier (can contain slashes)

    Returns:
        List of log entries with timestamps
    """
    from ...db.download_logs import get_logs

    logs = get_logs(model_id)
    return {"model_id": model_id, "logs": logs}


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
    from ...db.models import update_model_status, ModelStatus
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Use custom directory structure: data/models/{org}/{model}
    # model_id like "sentence-transformers/all-MiniLM-L6-v2" becomes
    # "data/models/sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = get_data_dir() / "models" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Update status to downloading
    update_model_status(model_id, ModelStatus.DOWNLOADING)

    # Clear old logs for this model
    from ...db.download_logs import clear_logs, add_log_line
    clear_logs(model_id)

    process = None
    try:
        from ...config import get_hf_endpoint

        env = os.environ.copy()
        hf_endpoint = get_hf_endpoint()

        logger.info(f"=== Download Starting for {model_id} ===")
        logger.info(f"HuggingFace Endpoint: {hf_endpoint}")
        logger.info(f"Target directory: {cache_dir}")

        # Use hfd (huggingface downloader with aria2) for better mirror support and resume capability
        # hfd properly respects HF_ENDPOINT and uses aria2c for faster multi-threaded downloads
        cmd_str = f"HF_ENDPOINT={hf_endpoint} hfd {model_id} --local-dir {cache_dir}"
        logger.info(f"Command: {cmd_str}")

        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=True,
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
                    stripped_line = line.strip()
                    output_lines.append(stripped_line)
                    logger.info(f"Download output: {stripped_line}")
                    # Persist log to database
                    try:
                        add_log_line(model_id, stripped_line)
                    except Exception as log_err:
                        logger.warning(f"Failed to persist log: {log_err}")

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
                    stripped_line = line.strip()
                    if stripped_line:
                        output_lines.append(stripped_line)
                        logger.info(f"Download output: {stripped_line}")
                        # Persist log to database
                        try:
                            add_log_line(model_id, stripped_line)
                        except Exception as log_err:
                            logger.warning(f"Failed to persist log: {log_err}")

        # Download completed - check exit code
        returncode = process.returncode
        if model_id in active_downloads:
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

            # Update status to failed in database
            update_model_status(model_id, ModelStatus.FAILED, error_message=error_msg)

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

        # Update status to downloaded in database
        update_model_status(model_id, ModelStatus.DOWNLOADED, downloaded_size_mb=final_size_mb)

        event = DownloadProgressEvent(
            type="complete", percent=100, size_mb=final_size_mb
        )
        yield event.model_dump_json()

    except Exception as e:
        if model_id in active_downloads:
            del active_downloads[model_id]

        error_msg = f"Error: {str(e)}"
        # Update status to failed in database
        update_model_status(model_id, ModelStatus.FAILED, error_message=error_msg)

        event = DownloadProgressEvent(type="error", message=error_msg)
        yield event.model_dump_json()
    finally:
        # Ensure process is cleaned up and not left in downloading state
        if process and model_id in active_downloads:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except:
                pass
            del active_downloads[model_id]


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
    from ...db.models import get_model, ModelStatus

    # Validate model exists in database
    db_model = get_model(model)
    if not db_model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found in catalog",
        )

    # Check if already downloaded
    if db_model["status"] == ModelStatus.DOWNLOADED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is already downloaded",
        )

    # Check if there's an actual active download process running
    # Only block if there's a real process, not just a stale "downloading" status in DB
    if model in active_downloads:
        # Verify the process is actually still running
        process = active_downloads[model]
        if process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model}' is already being downloaded",
            )
        else:
            # Process died but wasn't cleaned up - remove it
            del active_downloads[model]
            # Reset status to failed if it was stuck in downloading
            if db_model["status"] == ModelStatus.DOWNLOADING.value:
                from ...db.models import update_model_status
                update_model_status(model, ModelStatus.FAILED, error_message="Download process died unexpectedly")

    return EventSourceResponse(
        download_model_with_progress(model, db_model["size_mb"], request)
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
    from ...config import get_data_dir
    from ...db.models import get_model, update_model_status, ModelStatus

    # Validate model exists in database
    db_model = get_model(model_id)
    if not db_model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not found in catalog",
        )

    try:
        # Use custom directory structure: data/models/{org}/{model}
        cache_path = get_data_dir() / "models" / model_id

        if not cache_path.exists():
            # Reset status to init if directory doesn't exist
            update_model_status(model_id, ModelStatus.INIT, downloaded_size_mb=None, error_message=None)
            return {
                "model_id": model_id,
                "status": "not_found",
                "message": "Model not downloaded",
            }

        # Delete the model directory
        shutil.rmtree(cache_path)

        # Reset status to init after deletion
        update_model_status(model_id, ModelStatus.INIT, downloaded_size_mb=None, error_message=None)

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
