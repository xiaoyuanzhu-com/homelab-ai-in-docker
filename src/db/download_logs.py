"""Database schema and operations for download logs."""

from typing import List, Dict, Any
from datetime import datetime

from .db_config import get_db


def init_download_logs_table():
    """Initialize download_logs table schema."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS download_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                log_line TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)

        # Create index on model_id for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_download_logs_model_id
            ON download_logs(model_id, timestamp)
        """)


def add_log_line(model_id: str, log_line: str) -> None:
    """Add a log line for a model download."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO download_logs (model_id, log_line)
            VALUES (?, ?)
        """, (model_id, log_line))


def get_logs(model_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Get download logs for a model.

    Args:
        model_id: The model identifier
        limit: Maximum number of log lines to return (most recent first)

    Returns:
        List of log entries with timestamp and log_line
    """
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT log_line, timestamp
            FROM download_logs
            WHERE model_id = ?
            ORDER BY id ASC
            LIMIT ?
        """, (model_id, limit))

        rows = cursor.fetchall()
        return [
            {
                "log_line": row["log_line"],
                "timestamp": row["timestamp"]
            }
            for row in rows
        ]


def clear_logs(model_id: str) -> None:
    """Clear all logs for a model."""
    with get_db() as conn:
        conn.execute("""
            DELETE FROM download_logs
            WHERE model_id = ?
        """, (model_id,))
