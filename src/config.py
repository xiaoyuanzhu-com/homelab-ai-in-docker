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
    """
    Get the data directory for model storage.

    Checks HAID_DATA_DIR environment variable first, then falls back to:
    - /haid/data in production (Docker)
    - {project_root}/data in development
    """
    # Check environment variable first
    if env_data_dir := os.getenv("HAID_DATA_DIR"):
        return Path(env_data_dir)

    # Fall back to default behavior
    return get_app_root() / "data"


def get_hf_model_cache_path(model_id: str) -> Path:
    """
    Get the HuggingFace cache path for a model following HF conventions.

    Converts model_id to HF cache path: $HF_HOME/hub/models--{model_id with '/' replaced by '--'}

    Args:
        model_id: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')

    Returns:
        Path to model cache directory in HF format

    Examples:
        >>> get_hf_model_cache_path('bert-base-uncased')
        Path('/haid/data/models/hub/models--bert-base-uncased')
        >>> get_hf_model_cache_path('sentence-transformers/all-MiniLM-L6-v2')
        Path('/haid/data/models/hub/models--sentence-transformers--all-MiniLM-L6-v2')
    """
    hf_home = Path(os.getenv("HF_HOME", get_data_dir() / "models"))
    # Convert model_id to HF cache format: replace '/' with '--'
    safe_model_id = model_id.replace("/", "--")
    return hf_home / "hub" / f"models--{safe_model_id}"


def get_model_cache_dir(service: str, model_name: str) -> Path:
    """
    Get the cache directory for a specific model.

    DEPRECATED: Use get_hf_model_cache_path() instead.
    This function is kept for backward compatibility.

    Uses HuggingFace standard storage structure: data/models/hub/models--{org}--{model}

    Args:
        service: Service name (deprecated, kept for compatibility)
        model_name: Model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')

    Returns:
        Path to model cache directory
    """
    return get_hf_model_cache_path(model_name)


# Environment variable names for HuggingFace cache
HF_HOME_ENV = "HF_HOME"
TRANSFORMERS_CACHE_ENV = "TRANSFORMERS_CACHE"  # Deprecated: Use HF_HOME instead
SENTENCE_TRANSFORMERS_HOME_ENV = "SENTENCE_TRANSFORMERS_HOME"


def get_hf_endpoint() -> str:
    """
    Get the HuggingFace endpoint URL with priority order:
    1. Database settings
    2. Environment variable (HF_ENDPOINT)
    3. Default (https://huggingface.co)

    Returns:
        HuggingFace endpoint URL
    """
    # Avoid circular import by importing here
    from src.db.settings import get_setting

    # Priority 1: Database settings
    db_endpoint = get_setting("hf_endpoint")
    if db_endpoint:
        return db_endpoint

    # Priority 2: Environment variable
    env_endpoint = os.getenv("HF_ENDPOINT")
    if env_endpoint:
        return env_endpoint

    # Priority 3: Default
    return "https://huggingface.co"


def get_hf_username() -> str:
    """
    Get the HuggingFace username with priority order:
    1. Database settings
    2. Environment variable (HF_USERNAME)
    3. Default (empty string)

    Returns:
        HuggingFace username (empty string if not configured)
    """
    # Avoid circular import by importing here
    from src.db.settings import get_setting

    # Priority 1: Database settings
    db_username = get_setting("hf_username")
    if db_username:
        return db_username

    # Priority 2: Environment variable
    env_username = os.getenv("HF_USERNAME")
    if env_username:
        return env_username

    # Priority 3: Default (empty)
    return ""


def get_hf_token() -> str:
    """
    Get the HuggingFace API token with priority order:
    1. Database settings
    2. Environment variable (HF_TOKEN)
    3. Default (empty string)

    Returns:
        HuggingFace API token (empty string if not configured)
    """
    # Avoid circular import by importing here
    from src.db.settings import get_setting

    # Priority 1: Database settings
    db_token = get_setting("hf_token")
    if db_token:
        return db_token

    # Priority 2: Environment variable
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    # Priority 3: Default (empty)
    return ""
