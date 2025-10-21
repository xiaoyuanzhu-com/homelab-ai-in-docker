"""Database schema and functions for application settings."""

import sqlite3
from typing import Optional, Any
import json

from .db_config import get_db


def init_settings_table():
    """Initialize settings table schema."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert default settings if not exists
        defaults = [
            ("model_idle_timeout_seconds", "5", "Seconds of inactivity before unloading models from GPU memory"),
            ("hf_endpoint", "https://huggingface.co", "HuggingFace endpoint URL for model downloads and loading"),
        ]
        
        for key, value, description in defaults:
            conn.execute("""
                INSERT OR IGNORE INTO settings (key, value, description)
                VALUES (?, ?, ?)
            """, (key, value, description))


def get_setting(key: str, default: Any = None) -> Optional[str]:
    """
    Get a setting value by key.
    
    Args:
        key: Setting key
        default: Default value if setting not found
        
    Returns:
        Setting value as string, or default if not found
    """
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT value FROM settings WHERE key = ?
        """, (key,))
        row = cursor.fetchone()
        return row["value"] if row else default


def get_setting_int(key: str, default: int = 0) -> int:
    """
    Get a setting value as integer.
    
    Args:
        key: Setting key
        default: Default value if setting not found or invalid
        
    Returns:
        Setting value as integer
    """
    value = get_setting(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def set_setting(key: str, value: str, description: Optional[str] = None) -> None:
    """
    Set or update a setting value.
    
    Args:
        key: Setting key
        value: Setting value (will be stored as string)
        description: Optional description of the setting
    """
    with get_db() as conn:
        if description is not None:
            conn.execute("""
                INSERT INTO settings (key, value, description, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    description = excluded.description,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, str(value), description))
        else:
            conn.execute("""
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, str(value)))


def get_all_settings() -> dict:
    """
    Get all settings as a dictionary.
    
    Returns:
        Dictionary of all settings {key: value}
    """
    with get_db() as conn:
        cursor = conn.execute("SELECT key, value FROM settings")
        return {row["key"]: row["value"] for row in cursor.fetchall()}
