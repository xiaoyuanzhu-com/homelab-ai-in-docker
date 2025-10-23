"""Skills API router providing read-only access to skill metadata."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ...db.skills import list_skills, SkillStatus

router = APIRouter(prefix="/api", tags=["skills"])


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


@router.get("/skills", response_model=SkillsResponse)
async def list_available_skills(task: Optional[str] = None) -> SkillsResponse:
    rows = list_skills(task=task)
    skills = [
        SkillInfo(
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
        for skill in rows
    ]
    return SkillsResponse(skills=skills)
