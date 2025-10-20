"""Shared database configuration for HAID (Homelab AI Docker)."""

import sqlite3
from pathlib import Path
from contextlib import contextmanager


def get_db_path() -> Path:
    """Get the shared database file path."""
    from ..config import get_data_dir
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "haid.db"


@contextmanager
def get_db(timeout: float = 5.0):
    """
    Get database connection context manager.

    Args:
        timeout: Database lock timeout in seconds (default: 5.0)
    """
    conn = sqlite3.connect(
        get_db_path(),
        timeout=timeout,
        check_same_thread=False,  # Allow connections across threads
    )
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for better concurrency (allows reads during writes)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
