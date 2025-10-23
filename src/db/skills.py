"""Database schema and helpers for managing skills manifest."""

from __future__ import annotations

import json
import sqlite3
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Iterable

from .db_config import get_db


class SkillStatus(str, Enum):
    INIT = "init"
    DOWNLOADING = "downloading"
    FAILED = "failed"
    READY = "ready"


def init_skills_table() -> None:
    """Create the skills table (idempotent)."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                provider TEXT NOT NULL,
                tasks TEXT NOT NULL,
                architecture TEXT,
                default_prompt TEXT,
                platform_requirements TEXT,
                supports_markdown INTEGER DEFAULT 0,
                requires_quantization INTEGER DEFAULT 0,
                requires_download INTEGER DEFAULT 1,
                hf_model TEXT,
                reference_url TEXT,
                size_mb INTEGER,
                parameters_m INTEGER,
                gpu_memory_mb INTEGER,
                status TEXT NOT NULL DEFAULT 'init',
                downloaded_size_mb INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_skills_status
            ON skills(status)
            """
        )


def _tasks_to_json(tasks: Sequence[str]) -> str:
    return json.dumps(sorted(set(tasks)))


def upsert_skill(
    *,
    skill_id: str,
    label: str,
    provider: str,
    tasks: Sequence[str],
    architecture: Optional[str] = None,
    default_prompt: Optional[str] = None,
    platform_requirements: Optional[str] = None,
    supports_markdown: bool = False,
    requires_quantization: bool = False,
    requires_download: bool = True,
    hf_model: Optional[str] = None,
    reference_url: Optional[str] = None,
    size_mb: Optional[int] = None,
    parameters_m: Optional[int] = None,
    gpu_memory_mb: Optional[int] = None,
    initial_status: Optional[SkillStatus] = None,
) -> None:
    """
    Insert or update a skill definition.

    The status remains untouched unless `initial_status` is supplied or the record
    does not yet exist.
    """
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO skills (
                id, label, provider, tasks, architecture, default_prompt,
                platform_requirements, supports_markdown, requires_quantization,
                requires_download, hf_model, reference_url, size_mb, parameters_m,
                gpu_memory_mb, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label = excluded.label,
                provider = excluded.provider,
                tasks = excluded.tasks,
                architecture = excluded.architecture,
                default_prompt = excluded.default_prompt,
                platform_requirements = excluded.platform_requirements,
                supports_markdown = excluded.supports_markdown,
                requires_quantization = excluded.requires_quantization,
                requires_download = excluded.requires_download,
                hf_model = excluded.hf_model,
                reference_url = excluded.reference_url,
                size_mb = excluded.size_mb,
                parameters_m = excluded.parameters_m,
                gpu_memory_mb = excluded.gpu_memory_mb,
                status = CASE
                    WHEN skills.status IN ('downloading') THEN skills.status
                    ELSE excluded.status
                END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                skill_id,
                label,
                provider,
                _tasks_to_json(tasks),
                architecture,
                default_prompt,
                platform_requirements,
                1 if supports_markdown else 0,
                1 if requires_quantization else 0,
                1 if requires_download else 0,
                hf_model,
                reference_url,
                size_mb,
                parameters_m,
                gpu_memory_mb,
                (initial_status or (SkillStatus.INIT if requires_download else SkillStatus.READY)).value,
            ),
        )


def get_skill(skill_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,))
        return cursor.fetchone()


def get_all_skills() -> Iterable[sqlite3.Row]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM skills ORDER BY provider, label")
        return cursor.fetchall()


def delete_skill(skill_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))


def update_skill_status(
    skill_id: str,
    status: SkillStatus,
    downloaded_size_mb: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE skills
            SET status = ?,
                downloaded_size_mb = COALESCE(?, downloaded_size_mb),
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status.value, downloaded_size_mb, error_message, skill_id),
        )


def _deserialize_tasks(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def _row_to_skill(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "label": row["label"],
        "provider": row["provider"],
        "tasks": _deserialize_tasks(row["tasks"]),
        "architecture": row["architecture"],
        "default_prompt": row["default_prompt"],
        "platform_requirements": row["platform_requirements"],
        "supports_markdown": bool(row["supports_markdown"]),
        "requires_quantization": bool(row["requires_quantization"]),
        "requires_download": bool(row["requires_download"]),
        "hf_model": row["hf_model"],
        "reference_url": row["reference_url"],
        "size_mb": row["size_mb"],
        "parameters_m": row["parameters_m"],
        "gpu_memory_mb": row["gpu_memory_mb"],
        "status": row["status"],
        "downloaded_size_mb": row["downloaded_size_mb"],
        "error_message": row["error_message"],
    }


def list_skills(task: Optional[str] = None) -> list[Dict[str, Any]]:
    rows = get_all_skills()
    skills = [_row_to_skill(row) for row in rows]
    if task:
        task_lower = task.lower()
        skills = [s for s in skills if any(t.lower() == task_lower for t in s["tasks"])]
    return skills


def get_skill_dict(skill_id: str) -> Optional[Dict[str, Any]]:
    row = get_skill(skill_id)
    if row is None:
        return None
    return _row_to_skill(row)
