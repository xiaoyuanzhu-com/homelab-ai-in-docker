"""Application configuration for model cache paths and settings."""

import os
from pathlib import Path


def get_app_root() -> Path:
    """
    Get the application root directory.

    Returns /haid in production, or project root in development.
    """
    # Check if we're in production (Docker environment)
    if os.path.exists("/haid"):
        return Path("/haid")

    # Development: return project root
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory for model storage."""
    return get_app_root() / "data"


def get_model_cache_dir(service: str, model_name: str) -> Path:
    """
    Get the cache directory for a specific model.

    Args:
        service: Service name (e.g., 'embedding', 'image-caption')
        model_name: Model identifier (e.g., 'all-MiniLM-L6-v2')

    Returns:
        Path to model cache directory
    """
    # Sanitize model name for filesystem (replace / with --)
    safe_model_name = model_name.replace("/", "--")

    cache_dir = get_data_dir() / service / safe_model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


# Environment variable names for HuggingFace cache
HF_HOME_ENV = "HF_HOME"
TRANSFORMERS_CACHE_ENV = "TRANSFORMERS_CACHE"
SENTENCE_TRANSFORMERS_HOME_ENV = "SENTENCE_TRANSFORMERS_HOME"
