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
            ("max_models_in_memory", "1", "Maximum number of models to keep loaded in memory simultaneously (1=aggressive, 2+=allow multiple if memory permits)"),
            ("enable_preemptive_unload", "true", "Unload previous model before loading next one to prevent OOM (true/false)"),
            ("max_memory_mb", "", "Maximum GPU memory usage in MB (empty=no limit, for multi-model mode)"),
            ("hf_endpoint", "https://huggingface.co", "HuggingFace endpoint URL for model downloads and loading"),
            ("hf_username", "", "HuggingFace username for accessing gated models (optional, not email)"),
            ("hf_token", "", "HuggingFace API token for accessing private models (optional)"),
            ("embedding_memory_per_batch_mb", "2048", "GPU memory used per embedding batch in MB (2048 = 2GB per batch)"),
            ("embedding_max_batch_size", "32", "Maximum batch size for embedding generation (prevents excessive batching)"),
            ("whisperx_align_device", "cpu", "Device for WhisperX alignment model (cpu/cuda). CPU reduces GPU memory pressure."),
            ("whisperx_diar_device", "cuda", "Device for WhisperX diarization pipeline (cpu/cuda). GPU is 3-5x faster."),
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


def get_setting_bool(key: str, default: bool = False) -> bool:
    """
    Get a setting value as boolean.

    Args:
        key: Setting key
        default: Default value if setting not found or invalid

    Returns:
        Setting value as boolean
    """
    value = get_setting(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_setting_float(key: str, default: Optional[float] = None) -> Optional[float]:
    """
    Get a setting value as float.

    Args:
        key: Setting key
        default: Default value if setting not found or invalid

    Returns:
        Setting value as float, or None if empty/invalid
    """
    value = get_setting(key)
    if not value or value.strip() == "":
        return default
    try:
        return float(value)
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
