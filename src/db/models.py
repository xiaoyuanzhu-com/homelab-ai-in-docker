"""Database schema and helpers for managing downloadable models."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, Optional, Sequence

from .db_config import get_db
from .status import DownloadStatus


def _create_models_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
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
            dimensions INTEGER,
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
        CREATE INDEX IF NOT EXISTS idx_models_status
        ON models(status)
        """
    )


def init_models_table() -> None:
    with get_db() as conn:
        # Create if not exists
        _create_models_schema(conn)
        # Verify schema; if mismatched (e.g., missing 'label'), drop and recreate (fresh impl)
        try:
            cur = conn.execute("PRAGMA table_info(models)")
            cols = {row[1] for row in cur.fetchall()}
        except Exception:
            cols = set()
        required = {
            "id",
            "label",
            "provider",
            "tasks",
            "status",
        }
        if not required.issubset(cols):
            conn.execute("DROP TABLE IF EXISTS models")
            _create_models_schema(conn)


def _tasks_to_json(tasks: Sequence[str]) -> str:
    return json.dumps(sorted(set(tasks)))


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


def upsert_model(
    *,
    model_id: str,
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
    dimensions: Optional[int] = None,
    initial_status: Optional[DownloadStatus] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO models (
                id, label, provider, tasks, architecture, default_prompt,
                platform_requirements, supports_markdown, requires_quantization,
                requires_download, hf_model, reference_url, size_mb, parameters_m,
                gpu_memory_mb, dimensions, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                dimensions = excluded.dimensions,
                status = CASE
                    WHEN models.status IN ('downloading', 'ready') THEN models.status
                    ELSE excluded.status
                END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                model_id,
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
                dimensions,
                (initial_status or (DownloadStatus.INIT if requires_download else DownloadStatus.READY)).value,
            ),
        )


def get_model(model_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        return cur.fetchone()


def get_all_models() -> Iterable[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM models ORDER BY provider, label")
        return cur.fetchall()


def delete_model(model_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))


def update_model_status(
    model_id: str,
    status: DownloadStatus,
    downloaded_size_mb: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE models
            SET status = ?,
                downloaded_size_mb = COALESCE(?, downloaded_size_mb),
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status.value, downloaded_size_mb, error_message, model_id),
        )


def _row_to_model(row: sqlite3.Row) -> Dict[str, Any]:
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
        "dimensions": row["dimensions"],
        "status": row["status"],
        "downloaded_size_mb": row["downloaded_size_mb"],
        "error_message": row["error_message"],
    }


def list_models(task: Optional[str] = None) -> list[Dict[str, Any]]:
    rows = get_all_models()
    items = [_row_to_model(r) for r in rows]
    if task:
        tl = task.lower()
        items = [s for s in items if any(t.lower() == tl for t in s["tasks"])]
    return items


def get_model_dict2(model_id: str) -> Optional[Dict[str, Any]]:
    row = get_model(model_id)
    return _row_to_model(row) if row is not None else None
