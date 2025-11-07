"""Models API router for listing and downloading models.

Replaces the legacy skills endpoints for models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...db.catalog import get_model_dict, list_models
from ...db.models import update_model_status
from ...db.status import DownloadStatus
from ...db.download_logs import add_log_line, clear_logs, get_logs
from ...config import get_data_dir, get_hf_endpoint, get_hf_token, get_hf_username
import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path


router = APIRouter(prefix="/api", tags=["models"])

# Active downloads tracking: {model_id: subprocess.Popen}
active_downloads: Dict[str, subprocess.Popen] = {}


class ModelInfo(BaseModel):
    id: str
    label: str
    provider: str
    tasks: list[str]
    architecture: Optional[str] = None
    default_prompt: Optional[str] = None
    platform_requirements: Optional[str] = None
    supports_markdown: bool = False
    requires_quantization: bool = False
    requires_download: bool = True
    hf_model: Optional[str] = None
    reference_url: Optional[str] = None
    size_mb: Optional[int] = None
    parameters_m: Optional[int] = None
    gpu_memory_mb: Optional[int] = None
    status: DownloadStatus
    downloaded_size_mb: Optional[int] = None
    error_message: Optional[str] = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


def _serialize_model(d: Dict[str, Any]) -> ModelInfo:
    return ModelInfo(
        id=d["id"],
        label=d["label"],
        provider=d["provider"],
        tasks=d.get("tasks", []),
        architecture=d.get("architecture"),
        default_prompt=d.get("default_prompt"),
        platform_requirements=d.get("platform_requirements"),
        supports_markdown=bool(d.get("supports_markdown", False)),
        requires_quantization=bool(d.get("requires_quantization", False)),
        requires_download=bool(d.get("requires_download", True)),
        hf_model=d.get("hf_model"),
        reference_url=d.get("reference_url"),
        size_mb=d.get("size_mb"),
        parameters_m=d.get("parameters_m"),
        gpu_memory_mb=d.get("gpu_memory_mb"),
        status=DownloadStatus(d.get("status", DownloadStatus.INIT.value)),
        downloaded_size_mb=d.get("downloaded_size_mb"),
        error_message=d.get("error_message"),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_available_models(task: Optional[str] = None) -> ModelsResponse:
    models = [_serialize_model(m) for m in list_models(task=task)]
    return ModelsResponse(models=models)


@router.get("/models/{model_id:path}/logs")
async def get_model_download_logs(model_id: str):
    return {"model_id": model_id, "logs": get_logs(model_id)}


def _model_cache_dir(hf_model: str) -> Path:
    cache_root = get_data_dir() / "models"
    parts = hf_model.split("/")
    return cache_root.joinpath(*parts)


async def _download_model_with_progress(
    model_id: str,
    hf_model: str,
    expected_size_mb: Optional[int],
    request: Request,
):
    import select
    logger = __import__(__name__).logging.getLogger(__name__)  # reuse module logger

    cache_dir = _model_cache_dir(hf_model)
    cache_dir.mkdir(parents=True, exist_ok=True)

    update_model_status(model_id, DownloadStatus.DOWNLOADING)
    clear_logs(model_id)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    hf_endpoint = get_hf_endpoint()
    hf_username = get_hf_username()
    hf_token = get_hf_token()

    logger.info(f"=== Download starting for model '{model_id}' ({hf_model}) ===")
    logger.info(f"HF endpoint: {hf_endpoint}")
    logger.info(f"HF username: {'[configured]' if hf_username else '[not set]'}")
    logger.info(f"HF token: {'[configured]' if hf_token else '[not set]'}")
    logger.info(f"Download directory: {cache_dir}")

    import shlex
    cmd_parts = [
        f"HF_ENDPOINT={hf_endpoint}",
        "stdbuf",
        "-oL",
        "hfd",
        hf_model,
        "--local-dir",
        str(cache_dir),
    ]
    if hf_username and hf_token:
        cmd_parts.extend(["--hf_username", hf_username, "--hf_token", hf_token])
    cmd_str = " ".join(shlex.quote(part) if i > 0 else part for i, part in enumerate(cmd_parts))
    if hf_token:
        logger.info("Command: %s", cmd_str.replace(hf_token, "[REDACTED]"))
    else:
        logger.info("Command: %s", cmd_str)

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
    output_lines: list[str] = []
    last_progress_time = time.time()
    last_size = 0

    async def try_readline_nonblock() -> Optional[str]:
        if process.stdout is None:
            return None
        fd = process.stdout.fileno()
        readable, _, _ = await asyncio.to_thread(select.select, [fd], [], [], 0)
        if readable:
            return await asyncio.to_thread(process.stdout.readline)
        return None

    async def compute_dir_size_mb() -> int:
        def _calc() -> int:
            total = 0
            if cache_dir.exists():
                for path in cache_dir.rglob("*"):
                    if path.is_file():
                        try:
                            total += path.stat().st_size
                        except OSError:
                            pass
            return total // (1024 * 1024)

        return await asyncio.to_thread(_calc)

    try:
        while process.poll() is None:
            if await request.is_disconnected():
                process.terminate()
                process.wait(timeout=5)
                active_downloads.pop(model_id, None)
                update_model_status(model_id, DownloadStatus.FAILED, error_message="Client disconnected")
                return

            try:
                line = await try_readline_nonblock()
            except Exception as err:
                line = None
                logger.debug("Failed to read download output: %s", err)

            if line:
                stripped = line.strip()
                if stripped:
                    output_lines.append(stripped)
                    logger.info("Download output: %s", stripped)
                    try:
                        add_log_line(model_id, stripped)
                    except Exception as log_err:
                        logger.warning("Failed to persist log: %s", log_err)

            current_time = time.time()
            if current_time - last_progress_time >= 1.0:
                try:
                    current_size_mb = await compute_dir_size_mb()
                except Exception as size_err:
                    logger.debug("Failed to compute download size: %s", size_err)
                    current_size_mb = last_size

                yield {"type": "progress", "current_mb": current_size_mb, "total_mb": expected_size_mb}
                last_progress_time = current_time
                last_size = current_size_mb

            await asyncio.sleep(0.2)

        # Drain remaining output
        if process.stdout:
            remaining = await asyncio.to_thread(process.stdout.read)
            if remaining:
                for line in remaining.splitlines():
                    stripped = line.strip()
                    if stripped:
                        output_lines.append(stripped)
                        logger.info("Download output: %s", stripped)
                        try:
                            add_log_line(model_id, stripped)
                        except Exception as log_err:
                            logger.warning("Failed to persist log: %s", log_err)

        returncode = process.returncode
        active_downloads.pop(model_id, None)
        logger.info("=== Download finished for model '%s' (exit %s) ===", model_id, returncode)

        if returncode != 0:
            error_msg = "Download failed"
            for line in reversed(output_lines[-10:]):
                if "error" in line.lower() or "failed" in line.lower():
                    error_msg = line[:200]
                    break

            update_model_status(model_id, DownloadStatus.FAILED, error_message=error_msg)
            if cache_dir.exists():
                await asyncio.to_thread(shutil.rmtree, cache_dir)
            yield {"type": "error", "message": error_msg}
            return

        final_size_mb = await compute_dir_size_mb()
        update_model_status(model_id, DownloadStatus.READY, downloaded_size_mb=final_size_mb)
        yield {"type": "complete", "percent": 100, "size_mb": final_size_mb}

    finally:
        proc = active_downloads.pop(model_id, None)
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
            except Exception:
                pass


@router.get("/models/download")
async def download_model(model: str, request: Request):
    model_info = get_model_dict(model)
    if model_info is None:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found")

    if not model_info.get("requires_download", True):
        raise HTTPException(status_code=400, detail=f"Model '{model}' does not require downloading")

    hf_model = model_info.get("hf_model")
    if not hf_model:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is missing hf_model metadata")

    if model_info.get("status") == DownloadStatus.READY.value:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is already ready")

    process = active_downloads.get(model)
    if process and process.poll() is None:
        raise HTTPException(status_code=409, detail=f"Model '{model}' is already being downloaded")

    async def _stream():
        async for event in _download_model_with_progress(
            model_id=model,
            hf_model=hf_model,
            expected_size_mb=model_info.get("size_mb"),
            request=request,
        ):
            # Ensure proper SSE formatting
            from json import dumps
            yield dumps(event)

    return EventSourceResponse(_stream())


@router.delete("/models/{model_id:path}")
async def delete_model_assets(model_id: str):
    model_info = get_model_dict(model_id)
    if model_info is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not found")

    hf_model = model_info.get("hf_model")
    if not hf_model:
        update_model_status(model_id, DownloadStatus.READY)
        return {
            "model_id": model_id,
            "status": "skip",
            "message": "Model uses built-in assets; nothing to delete.",
        }

    from shutil import rmtree
    cache_dir = _model_cache_dir(hf_model)
    if not cache_dir.exists():
        update_model_status(model_id, DownloadStatus.INIT, downloaded_size_mb=None, error_message=None)
        return {
            "model_id": model_id,
            "status": "not_found",
            "message": "Model assets not found on disk.",
        }

    try:
        rmtree(cache_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete model assets: {exc}") from exc

    update_model_status(model_id, DownloadStatus.INIT, downloaded_size_mb=None, error_message=None)
    return {
        "model_id": model_id,
        "status": "deleted",
        "message": "Model assets deleted successfully.",
    }
