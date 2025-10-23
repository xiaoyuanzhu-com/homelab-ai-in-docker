"""Skills API router for listing and downloading capability definitions."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...config import get_data_dir, get_hf_endpoint, get_hf_token, get_hf_username
from ...db.download_logs import add_log_line, clear_logs, get_logs
from ...db.skills import (
    SkillStatus,
    get_skill_dict,
    list_skills,
    update_skill_status,
)

router = APIRouter(prefix="/api", tags=["skills"])

# Active downloads tracking: {skill_id: subprocess.Popen}
active_downloads: Dict[str, subprocess.Popen] = {}


class SkillInfo(BaseModel):
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
    status: SkillStatus
    downloaded_size_mb: Optional[int] = None
    error_message: Optional[str] = None


class SkillsResponse(BaseModel):
    skills: list[SkillInfo]


class DownloadProgressEvent(BaseModel):
    type: str  # "progress", "complete", "error"
    percent: Optional[int] = None
    current_mb: Optional[int] = None
    total_mb: Optional[int] = None
    size_mb: Optional[int] = None
    message: Optional[str] = None


def _serialize_skill(skill: Dict[str, Any]) -> SkillInfo:
    return SkillInfo(
        id=skill["id"],
        label=skill["label"],
        provider=skill["provider"],
        tasks=skill["tasks"],
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
        status=SkillStatus(skill.get("status", SkillStatus.INIT.value)),
        downloaded_size_mb=skill.get("downloaded_size_mb"),
        error_message=skill.get("error_message"),
    )


def _skill_cache_dir(hf_model: str) -> Path:
    cache_root = get_data_dir() / "models"
    parts = hf_model.split("/")
    return cache_root.joinpath(*parts)


@router.get("/skills", response_model=SkillsResponse)
async def list_available_skills(task: Optional[str] = None) -> SkillsResponse:
    skills = [_serialize_skill(skill) for skill in list_skills(task=task)]
    return SkillsResponse(skills=skills)


@router.get("/skills/{skill_id:path}/logs")
async def get_skill_download_logs(skill_id: str):
    return {"skill_id": skill_id, "logs": get_logs(skill_id)}


async def _download_skill_with_progress(
    skill_id: str,
    hf_model: str,
    expected_size_mb: Optional[int],
    request: Request,
) -> AsyncGenerator[str, None]:
    import logging
    import select

    logger = logging.getLogger(__name__)

    cache_dir = _skill_cache_dir(hf_model)
    cache_dir.mkdir(parents=True, exist_ok=True)

    update_skill_status(skill_id, SkillStatus.DOWNLOADING)
    clear_logs(skill_id)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    hf_endpoint = get_hf_endpoint()
    hf_username = get_hf_username()
    hf_token = get_hf_token()

    logger.info(f"=== Download starting for skill '{skill_id}' ({hf_model}) ===")
    logger.info(f"HF endpoint: {hf_endpoint}")
    logger.info(f"HF username: {'[configured]' if hf_username else '[not set]'}")
    logger.info(f"HF token: {'[configured]' if hf_token else '[not set]'}")
    logger.info(f"Target directory: {cache_dir}")

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

    active_downloads[skill_id] = process
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
                active_downloads.pop(skill_id, None)
                update_skill_status(skill_id, SkillStatus.FAILED, error_message="Client disconnected")
                return

            try:
                line = await try_readline_nonblock()
            except Exception as err:  # pragma: no cover - defensive logging
                line = None
                logger.debug("Failed to read download output: %s", err)

            if line:
                stripped = line.strip()
                if stripped:
                    output_lines.append(stripped)
                    logger.info("Download output: %s", stripped)
                    try:
                        add_log_line(skill_id, stripped)
                    except Exception as log_err:
                        logger.warning("Failed to persist log: %s", log_err)

            current_time = time.time()
            if current_time - last_progress_time >= 1.0:
                try:
                    current_size_mb = await compute_dir_size_mb()
                except Exception as size_err:
                    logger.debug("Failed to compute download size: %s", size_err)
                    current_size_mb = last_size

                yield DownloadProgressEvent(
                    type="progress",
                    current_mb=current_size_mb,
                    total_mb=expected_size_mb,
                ).model_dump_json()
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
                            add_log_line(skill_id, stripped)
                        except Exception as log_err:
                            logger.warning("Failed to persist log: %s", log_err)

        returncode = process.returncode
        active_downloads.pop(skill_id, None)
        logger.info("=== Download finished for skill '%s' (exit %s) ===", skill_id, returncode)

        if returncode != 0:
            error_msg = "Download failed"
            for line in reversed(output_lines[-10:]):
                if "error" in line.lower() or "failed" in line.lower():
                    error_msg = line[:200]
                    break

            update_skill_status(skill_id, SkillStatus.FAILED, error_message=error_msg)

            # Cleanup partial directory
            if cache_dir.exists():
                await asyncio.to_thread(shutil.rmtree, cache_dir)

            yield DownloadProgressEvent(type="error", message=error_msg).model_dump_json()
            return

        final_size_mb = await compute_dir_size_mb()
        update_skill_status(skill_id, SkillStatus.DOWNLOADED, downloaded_size_mb=final_size_mb)
        yield DownloadProgressEvent(
            type="complete",
            percent=100,
            size_mb=final_size_mb,
        ).model_dump_json()

    finally:
        proc = active_downloads.pop(skill_id, None)
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
            except Exception:
                pass


@router.get("/skills/download")
async def download_skill(skill: str, request: Request):
    skill_info = get_skill_dict(skill)
    if skill_info is None:
        raise HTTPException(status_code=400, detail=f"Skill '{skill}' not found")

    if not skill_info.get("requires_download", True):
        raise HTTPException(status_code=400, detail=f"Skill '{skill}' does not require downloading")

    hf_model = skill_info.get("hf_model")
    if not hf_model:
        raise HTTPException(status_code=400, detail=f"Skill '{skill}' is missing hf_model metadata")

    if skill_info.get("status") == SkillStatus.DOWNLOADED.value:
        raise HTTPException(status_code=400, detail=f"Skill '{skill}' is already downloaded")

    process = active_downloads.get(skill)
    if process and process.poll() is None:
        raise HTTPException(status_code=409, detail=f"Skill '{skill}' is already being downloaded")

    return EventSourceResponse(
        _download_skill_with_progress(
            skill_id=skill,
            hf_model=hf_model,
            expected_size_mb=skill_info.get("size_mb"),
            request=request,
        )
    )


@router.delete("/skills/{skill_id:path}")
async def delete_skill_assets(skill_id: str):
    skill_info = get_skill_dict(skill_id)
    if skill_info is None:
        raise HTTPException(status_code=400, detail=f"Skill '{skill_id}' not found")

    hf_model = skill_info.get("hf_model")
    if not hf_model:
        update_skill_status(skill_id, SkillStatus.DOWNLOADED)
        return {
            "skill_id": skill_id,
            "status": "skip",
            "message": "Skill uses built-in assets; nothing to delete.",
        }

    cache_dir = _skill_cache_dir(hf_model)
    if not cache_dir.exists():
        update_skill_status(skill_id, SkillStatus.INIT, downloaded_size_mb=None, error_message=None)
        return {
            "skill_id": skill_id,
            "status": "not_found",
            "message": "Skill assets not found on disk.",
        }

    try:
        shutil.rmtree(cache_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete skill assets: {exc}") from exc

    update_skill_status(skill_id, SkillStatus.INIT, downloaded_size_mb=None, error_message=None)
    return {
        "skill_id": skill_id,
        "status": "deleted",
        "message": "Skill assets deleted successfully.",
    }
