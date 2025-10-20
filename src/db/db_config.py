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
def get_db():
    """Get database connection context manager."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
