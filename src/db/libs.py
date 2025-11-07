"""Database schema and helpers for managing built-in libraries/tools."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, Optional, Sequence

from .db_config import get_db


def _create_libs_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS libs (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            provider TEXT NOT NULL,
            tasks TEXT NOT NULL,
            architecture TEXT,
            default_prompt TEXT,
            platform_requirements TEXT,
            supports_markdown INTEGER DEFAULT 0,
            requires_quantization INTEGER DEFAULT 0,
            requires_download INTEGER DEFAULT 0,
            reference_url TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            downloaded_size_mb INTEGER,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_libs_status
        ON libs(status)
        """
    )


def init_libs_table() -> None:
    with get_db() as conn:
        _create_libs_schema(conn)
        # Verify schema; if mismatched, drop and recreate
        try:
            cur = conn.execute("PRAGMA table_info(libs)")
            cols = {row[1] for row in cur.fetchall()}
        except Exception:
            cols = set()
        required = {"id", "label", "provider", "tasks", "status"}
        if not required.issubset(cols):
            conn.execute("DROP TABLE IF EXISTS libs")
            _create_libs_schema(conn)


def _tasks_to_json(tasks: Sequence[str]) -> str:
    import json as _json
    return _json.dumps(sorted(set(tasks)))


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


def upsert_lib(
    *,
    lib_id: str,
    label: str,
    provider: str,
    tasks: Sequence[str],
    architecture: Optional[str] = None,
    default_prompt: Optional[str] = None,
    platform_requirements: Optional[str] = None,
    supports_markdown: bool = False,
    requires_quantization: bool = False,
    requires_download: bool = False,
    reference_url: Optional[str] = None,
    size_mb: Optional[int] = None,  # accepted but not stored
    parameters_m: Optional[int] = None,  # accepted but not stored
    gpu_memory_mb: Optional[int] = None,  # accepted but not stored
    dimensions: Optional[int] = None,  # accepted but not stored
    initial_status: Optional[str] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO libs (
                id, label, provider, tasks, architecture, default_prompt,
                platform_requirements, supports_markdown, requires_quantization,
                requires_download, reference_url, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                reference_url = excluded.reference_url,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                lib_id,
                label,
                provider,
                _tasks_to_json(tasks),
                architecture,
                default_prompt,
                platform_requirements,
                1 if supports_markdown else 0,
                1 if requires_quantization else 0,
                1 if requires_download else 0,
                reference_url,
                (initial_status or "ready"),
            ),
        )


def get_lib(lib_id: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM libs WHERE id = ?", (lib_id,))
        return cur.fetchone()


def get_all_libs() -> Iterable[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM libs ORDER BY provider, label")
        return cur.fetchall()


def delete_lib(lib_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM libs WHERE id = ?", (lib_id,))


def _row_to_lib(row: sqlite3.Row) -> Dict[str, Any]:
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
        "reference_url": row["reference_url"],
        "status": row["status"],
        "downloaded_size_mb": row["downloaded_size_mb"] if "downloaded_size_mb" in row.keys() else None,
        "error_message": row["error_message"] if "error_message" in row.keys() else None,
    }


def list_libs(task: Optional[str] = None) -> list[Dict[str, Any]]:
    rows = get_all_libs()
    libs = [_row_to_lib(r) for r in rows]
    if task:
        tl = task.lower()
        libs = [s for s in libs if any(t.lower() == tl for t in s["tasks"])]
    return libs


def get_lib_dict2(lib_id: str) -> Optional[Dict[str, Any]]:
    row = get_lib(lib_id)
    return _row_to_lib(row) if row is not None else None
