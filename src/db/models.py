"""Database models and schema for AI model tracking."""

import sqlite3
from typing import Optional
from enum import Enum

from .db_config import get_db


class ModelStatus(str, Enum):
    """Model download status."""
    INIT = "init"
    DOWNLOADING = "downloading"
    FAILED = "failed"
    DOWNLOADED = "downloaded"


def init_db():
    """Initialize database schema."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                team TEXT NOT NULL,
                type TEXT NOT NULL,
                task TEXT NOT NULL,
                size_mb INTEGER NOT NULL,
                parameters_m INTEGER NOT NULL,
                gpu_memory_mb INTEGER NOT NULL,
                link TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'init',
                downloaded_size_mb INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on status for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_models_status
            ON models(status)
        """)


def upsert_model(
    model_id: str,
    name: str,
    team: str,
    model_type: str,
    task: str,
    size_mb: int,
    parameters_m: int,
    gpu_memory_mb: int,
    link: str,
) -> None:
    """
    Insert or update model metadata from manifest.
    Only updates metadata fields, preserves status and download info.
    """
    with get_db() as conn:
        conn.execute("""
            INSERT INTO models (
                id, name, team, type, task, size_mb,
                parameters_m, gpu_memory_mb, link
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                team = excluded.team,
                type = excluded.type,
                task = excluded.task,
                size_mb = excluded.size_mb,
                parameters_m = excluded.parameters_m,
                gpu_memory_mb = excluded.gpu_memory_mb,
                link = excluded.link,
                updated_at = CURRENT_TIMESTAMP
        """, (model_id, name, team, model_type, task, size_mb, parameters_m, gpu_memory_mb, link))


def get_all_models():
    """Get all models from database."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT * FROM models ORDER BY team, name
        """)
        return cursor.fetchall()


def get_model(model_id: str) -> Optional[sqlite3.Row]:
    """Get a specific model by ID."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT * FROM models WHERE id = ?
        """, (model_id,))
        return cursor.fetchone()


def update_model_status(
    model_id: str,
    status: ModelStatus,
    downloaded_size_mb: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update model download status."""
    with get_db() as conn:
        conn.execute("""
            UPDATE models
            SET status = ?,
                downloaded_size_mb = COALESCE(?, downloaded_size_mb),
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status.value, downloaded_size_mb, error_message, model_id))


def delete_model_record(model_id: str) -> None:
    """Delete model record from database."""
    with get_db() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
