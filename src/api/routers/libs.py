"""Libs API router.

Lists library/tool entries (non-HuggingFace) separately from models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ...db.catalog import list_libs


router = APIRouter(prefix="/api", tags=["libs"])


class LibInfo(BaseModel):
    id: str
    label: str
    provider: str
    tasks: list[str]
    architecture: Optional[str] = None
    default_prompt: Optional[str] = None
    platform_requirements: Optional[str] = None
    supports_markdown: bool = False
    requires_quantization: bool = False
    requires_download: bool = False
    reference_url: Optional[str] = None
    status: str
    downloaded_size_mb: Optional[int] = None
    error_message: Optional[str] = None


class LibsResponse(BaseModel):
    libs: list[LibInfo]


def _serialize_lib(d: Dict[str, Any]) -> LibInfo:
    return LibInfo(
        id=d["id"],
        label=d["label"],
        provider=d["provider"],
        tasks=d.get("tasks", []),
        architecture=d.get("architecture"),
        default_prompt=d.get("default_prompt"),
        platform_requirements=d.get("platform_requirements"),
        supports_markdown=bool(d.get("supports_markdown", False)),
        requires_quantization=bool(d.get("requires_quantization", False)),
        requires_download=bool(d.get("requires_download", False)),
        reference_url=d.get("reference_url"),
        status=str(d.get("status", "init")),
        downloaded_size_mb=d.get("downloaded_size_mb"),
        error_message=d.get("error_message"),
    )


@router.get("/libs", response_model=LibsResponse)
async def list_available_libs(task: Optional[str] = None) -> LibsResponse:
    libs = [_serialize_lib(x) for x in list_libs(task=task)]
    return LibsResponse(libs=libs)
