"""Worker execution logs with GPU memory tracking.

Tracks task execution metrics for observability and debugging.
"""

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .db_config import get_db


def ensure_table() -> None:
    """Create worker_logs table if it doesn't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Task identification
                task TEXT NOT NULL,
                model_id TEXT NOT NULL,
                request_id TEXT,

                -- Timing
                started_at DATETIME NOT NULL,
                completed_at DATETIME,
                duration_ms INTEGER,

                -- GPU memory (MB) - from nvidia-smi process memory
                gpu_memory_before_mb REAL,
                gpu_memory_peak_mb REAL,
                gpu_memory_after_mb REAL,

                -- Worker metadata
                worker_port INTEGER,
                worker_pid INTEGER,

                -- Request/Response metadata
                input_size_bytes INTEGER,
                output_size_bytes INTEGER,

                -- Status
                status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
                error_message TEXT,

                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for querying by task and time
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_worker_logs_task_time
            ON worker_logs (task, started_at DESC)
        """)

        # Index for querying by model
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_worker_logs_model
            ON worker_logs (model_id, started_at DESC)
        """)

        # Index for querying by status
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_worker_logs_status
            ON worker_logs (status)
        """)


def create_log(
    task: str,
    model_id: str,
    request_id: Optional[str] = None,
    worker_port: Optional[int] = None,
    worker_pid: Optional[int] = None,
    gpu_memory_before_mb: Optional[float] = None,
    input_size_bytes: Optional[int] = None,
) -> int:
    """
    Create a new worker log entry when inference starts.

    Returns:
        The log ID for updating later
    """
    ensure_table()

    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO worker_logs (
                task, model_id, request_id, started_at,
                worker_port, worker_pid, gpu_memory_before_mb,
                input_size_bytes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (
                task, model_id, request_id,
                datetime.utcnow().isoformat(),
                worker_port, worker_pid, gpu_memory_before_mb,
                input_size_bytes
            )
        )
        return cursor.lastrowid


def complete_log(
    log_id: int,
    duration_ms: int,
    gpu_memory_peak_mb: Optional[float] = None,
    gpu_memory_after_mb: Optional[float] = None,
    output_size_bytes: Optional[int] = None,
    status: str = "completed",
    error_message: Optional[str] = None,
) -> None:
    """Update a log entry when inference completes."""
    with get_db() as conn:
        conn.execute(
            """
            UPDATE worker_logs SET
                completed_at = ?,
                duration_ms = ?,
                gpu_memory_peak_mb = ?,
                gpu_memory_after_mb = ?,
                output_size_bytes = ?,
                status = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                datetime.utcnow().isoformat(),
                duration_ms,
                gpu_memory_peak_mb,
                gpu_memory_after_mb,
                output_size_bytes,
                status,
                error_message,
                log_id,
            )
        )


def get_recent_logs(
    limit: int = 100,
    task: Optional[str] = None,
    model_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent worker logs with optional filtering.

    Returns list of log entries as dictionaries.
    """
    ensure_table()

    conditions = []
    params = []

    if task:
        conditions.append("task = ?")
        params.append(task)
    if model_id:
        conditions.append("model_id = ?")
        params.append(model_id)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    params.append(limit)

    with get_db() as conn:
        cursor = conn.execute(
            f"""
            SELECT * FROM worker_logs
            {where_clause}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            params
        )
        return [dict(row) for row in cursor.fetchall()]


def get_log_stats(
    task: Optional[str] = None,
    model_id: Optional[str] = None,
    hours: int = 24,
) -> Dict[str, Any]:
    """
    Get aggregated statistics for worker logs.

    Returns dict with:
    - total_requests: Total number of requests
    - completed: Number completed successfully
    - failed: Number that failed
    - avg_duration_ms: Average duration
    - avg_peak_gpu_mb: Average peak GPU memory
    - max_peak_gpu_mb: Maximum peak GPU memory
    """
    ensure_table()

    conditions = ["started_at >= datetime('now', ?)", "status != 'running'"]
    params = [f"-{hours} hours"]

    if task:
        conditions.append("task = ?")
        params.append(task)
    if model_id:
        conditions.append("model_id = ?")
        params.append(model_id)

    where_clause = "WHERE " + " AND ".join(conditions)

    with get_db() as conn:
        cursor = conn.execute(
            f"""
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(duration_ms) as avg_duration_ms,
                AVG(gpu_memory_peak_mb) as avg_peak_gpu_mb,
                MAX(gpu_memory_peak_mb) as max_peak_gpu_mb
            FROM worker_logs
            {where_clause}
            """,
            params
        )
        row = cursor.fetchone()
        return dict(row) if row else {}


def cleanup_old_logs(days: int = 30) -> int:
    """
    Delete logs older than specified days.

    Returns number of deleted rows.
    """
    with get_db() as conn:
        cursor = conn.execute(
            """
            DELETE FROM worker_logs
            WHERE started_at < datetime('now', ?)
            """,
            (f"-{days} days",)
        )
        return cursor.rowcount
