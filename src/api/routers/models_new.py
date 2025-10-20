"""Unified models API router for managing all AI models."""

import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["models"])


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    team: str
    type: str
    license: Optional[str] = None
    dimensions: Optional[int] = None
    languages: Optional[list[str]] = None
    description: str
    size_mb: int
    link: str
    is_downloaded: bool
    downloaded_size_mb: Optional[int] = None


class ModelsResponse(BaseModel):
    """Response containing list of models."""
    models: list[ModelInfo]


def load_models_manifest() -> dict:
    """Load unified models manifest from JSON file."""
    manifest_path = Path(__file__).parent.parent / "models" / "models_manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def check_model_downloaded_hf(model_id: str) -> tuple[bool, Optional[int]]:
    """
    Check if a model is downloaded in HuggingFace cache structure.

    HF stores models in: HF_HOME/hub/models--{org}--{model}/

    Args:
        model_id: The model identifier (e.g., "BAAI/bge-large-en-v1.5")

    Returns:
        Tuple of (is_downloaded, size_in_mb)
    """
    from ...config import get_data_dir

    # Convert model_id to HF cache directory name
    # "BAAI/bge-large-en-v1.5" -> "models--BAAI--bge-large-en-v1.5"
    hf_model_dir = f"models--{model_id.replace('/', '--')}"
    cache_path = get_data_dir() / "models" / "hub" / hf_model_dir

    if cache_path.exists():
        # Calculate directory size
        files = list(cache_path.rglob("*"))
        if files:
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size // (1024 * 1024)
            if size_mb > 0:
                return True, size_mb

    return False, None


@router.get("/models", response_model=ModelsResponse)
async def list_all_models() -> ModelsResponse:
    """
    List all available AI models across all types.

    Returns:
        List of all models with download status
    """
    manifest = load_models_manifest()
    models = []

    # Process each model type
    for model_type, type_models in manifest.items():
        for model_info in type_models:
            is_downloaded, downloaded_size_mb = check_model_downloaded_hf(model_info["id"])

            models.append(
                ModelInfo(
                    id=model_info["id"],
                    name=model_info["name"],
                    team=model_info["team"],
                    type=model_type,
                    license=model_info.get("license"),
                    dimensions=model_info.get("dimensions"),
                    languages=model_info.get("languages"),
                    description=model_info["description"],
                    size_mb=model_info["size_mb"],
                    link=model_info["link"],
                    is_downloaded=is_downloaded,
                    downloaded_size_mb=downloaded_size_mb,
                )
            )

    return ModelsResponse(models=models)


@router.post("/models/download")
async def download_model(model_id: str):
    """
    Download a model using huggingface-cli.

    Args:
        model_id: The model identifier (e.g., "BAAI/bge-large-en-v1.5")

    Returns:
        Download status

    Note: This is a simple implementation. For production, use SSE streaming
    from the existing /models/embedding/{model_id}/download endpoint.
    """
    # Validate model exists in manifest
    manifest = load_models_manifest()
    all_models = []
    for type_models in manifest.values():
        all_models.extend([m["id"] for m in type_models])

    if model_id not in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not found in catalog",
        )

    # Check if already downloaded
    is_downloaded, size_mb = check_model_downloaded_hf(model_id)
    if is_downloaded:
        return {
            "model_id": model_id,
            "status": "already_downloaded",
            "message": f"Model already downloaded ({size_mb} MB)",
        }

    # For now, return a message to use the streaming endpoint
    # In production, you'd implement download here or redirect to SSE endpoint
    return {
        "model_id": model_id,
        "status": "use_streaming_endpoint",
        "message": "Please use /api/models/embedding/{model_id}/download for streaming progress",
    }


@router.delete("/models/{model_id:path}")
async def delete_model(model_id: str):
    """
    Delete a downloaded model from HuggingFace cache.

    Args:
        model_id: The model identifier (can contain slashes)

    Returns:
        Deletion status
    """
    # Validate model exists in manifest
    manifest = load_models_manifest()
    all_models = []
    for type_models in manifest.values():
        all_models.extend([m["id"] for m in type_models])

    if model_id not in all_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not found in catalog",
        )

    try:
        from ...config import get_data_dir

        # Get HF cache path
        hf_model_dir = f"models--{model_id.replace('/', '--')}"
        cache_path = get_data_dir() / "models" / "hub" / hf_model_dir

        if not cache_path.exists():
            return {
                "model_id": model_id,
                "status": "not_found",
                "message": "Model not downloaded",
            }

        # Delete the model directory
        shutil.rmtree(cache_path)

        return {
            "model_id": model_id,
            "status": "deleted",
            "message": "Model deleted successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}",
        )
