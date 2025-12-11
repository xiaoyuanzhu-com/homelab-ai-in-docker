"""Unified task options endpoint.

Returns combined model and lib choices for a given task.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal, Optional

from ...db.catalog import list_models, list_libs


router = APIRouter(prefix="/api", tags=["task-options"])


class TaskOption(BaseModel):
    id: str
    label: str
    provider: str
    type: Literal["model", "lib"]
    architecture: Optional[str] = None
    supports_markdown: bool = False
    supports_live_streaming: bool = False
    requires_download: bool = False
    status: str


class TaskOptionsResponse(BaseModel):
    task: Optional[str] = None
    options: list[TaskOption]


@router.get("/task-options", response_model=TaskOptionsResponse)
async def get_task_options(task: Optional[str] = None) -> TaskOptionsResponse:
    models = list_models(task=task)
    libs = list_libs(task=task)
    options: list[TaskOption] = []
    for m in models:
        options.append(
            TaskOption(
                id=m["id"],
                label=m.get("label", m["id"]),
                provider=m.get("provider", ""),
                type="model",
                architecture=m.get("architecture"),
                supports_markdown=bool(m.get("supports_markdown", False)),
                supports_live_streaming=bool(m.get("supports_live_streaming", False)),
                requires_download=bool(m.get("requires_download", True)),
                status=str(m.get("status", "init")),
            )
        )
    for lib in libs:
        options.append(
            TaskOption(
                id=lib["id"],
                label=lib.get("label", lib["id"]),
                provider=lib.get("provider", ""),
                type="lib",
                architecture=lib.get("architecture"),
                supports_markdown=bool(lib.get("supports_markdown", False)),
                supports_live_streaming=bool(lib.get("supports_live_streaming", False)),
                requires_download=bool(lib.get("requires_download", False)),
                status=str(lib.get("status", "ready")),
            )
        )
    # Sort options by provider+label for consistent UI order
    options.sort(key=lambda o: (o.provider.lower(), o.label.lower()))
    return TaskOptionsResponse(task=task, options=options)

