"""Models API router for managing embedding models."""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer

from ..models.models import (
    EmbeddingModelInfo,
    ModelListResponse,
    ModelDownloadRequest,
    ModelDownloadResponse,
)

router = APIRouter(prefix="/api", tags=["models"])


def load_model_catalog() -> list[dict]:
    """Load model catalog from JSON file."""
    manifest_path = Path(__file__).parent.parent / "models" / "embedding_models.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return data["models"]


def check_model_downloaded(model_id: str) -> tuple[bool, Optional[int]]:
    """
    Check if a model is already downloaded.

    Args:
        model_id: The model identifier

    Returns:
        Tuple of (is_downloaded, size_in_mb)
    """
    # Don't use get_model_cache_dir() as it creates the directory
    # Instead, construct path without creating it
    from ...config import get_data_dir

    safe_model_name = model_id.replace("/", "--")
    cache_dir = get_data_dir() / "embedding" / safe_model_name

    # Check if the model directory exists and has content
    if cache_dir.exists():
        # Calculate directory size
        files = list(cache_dir.rglob("*"))
        # Only consider downloaded if there are actual files
        if files:
            total_size = sum(
                f.stat().st_size for f in files if f.is_file()
            )
            size_mb = total_size // (1024 * 1024)
            # Only mark as downloaded if size > 0
            if size_mb > 0:
                return True, size_mb

    return False, None


@router.get("/models/embedding", response_model=ModelListResponse)
async def list_embedding_models() -> ModelListResponse:
    """
    List all available embedding models.

    Returns:
        List of embedding models with download status
    """
    catalog = load_model_catalog()
    models = []

    for model_info in catalog:
        is_downloaded, downloaded_size_mb = check_model_downloaded(model_info["id"])

        models.append(
            EmbeddingModelInfo(
                id=model_info["id"],
                name=model_info["name"],
                team=model_info["team"],
                license=model_info["license"],
                dimensions=model_info["dimensions"],
                languages=model_info["languages"],
                description=model_info["description"],
                size_mb=model_info["size_mb"],
                link=model_info["link"],
                is_downloaded=is_downloaded,
                downloaded_size_mb=downloaded_size_mb,
            )
        )

    return ModelListResponse(models=models)


@router.post("/models/embedding/download", response_model=ModelDownloadResponse)
async def download_embedding_model(
    request: ModelDownloadRequest,
) -> ModelDownloadResponse:
    """
    Download an embedding model.

    Args:
        request: Model download request

    Returns:
        Download status

    Raises:
        HTTPException: If download fails
    """
    # Validate model exists in catalog
    catalog = load_model_catalog()
    valid_model_ids = [m["id"] for m in catalog]
    if request.model_id not in valid_model_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": f"Model '{request.model_id}' not found in catalog",
            },
        )

    try:
        # Check if already downloaded
        is_downloaded, size_mb = check_model_downloaded(request.model_id)
        if is_downloaded:
            return ModelDownloadResponse(
                model_id=request.model_id,
                status="already_downloaded",
                message=f"Model already downloaded ({size_mb} MB)",
            )

        # Download the model by loading it
        # Create cache directory only when downloading
        from ...config import get_data_dir

        safe_model_name = request.model_id.replace("/", "--")
        cache_dir = get_data_dir() / "embedding" / safe_model_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        model = SentenceTransformer(request.model_id, cache_folder=str(cache_dir))

        # Get final size
        _, size_mb = check_model_downloaded(request.model_id)

        return ModelDownloadResponse(
            model_id=request.model_id,
            status="downloaded",
            message=f"Model downloaded successfully ({size_mb} MB)",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DOWNLOAD_FAILED",
                "message": f"Failed to download model: {str(e)}",
            },
        )


@router.delete("/models/embedding/{model_id}")
async def delete_embedding_model(model_id: str):
    """
    Delete a downloaded embedding model.

    Args:
        model_id: The model identifier

    Returns:
        Deletion status

    Raises:
        HTTPException: If deletion fails
    """
    # Validate model exists in catalog
    catalog = load_model_catalog()
    valid_model_ids = [m["id"] for m in catalog]
    if model_id not in valid_model_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": f"Model '{model_id}' not found in catalog",
            },
        )

    try:
        from ...config import get_data_dir

        safe_model_name = model_id.replace("/", "--")
        cache_dir = get_data_dir() / "embedding" / safe_model_name

        if not cache_dir.exists():
            return {
                "model_id": model_id,
                "status": "not_found",
                "message": "Model not downloaded",
            }

        # Delete the model directory
        import shutil
        shutil.rmtree(cache_dir)

        return {
            "model_id": model_id,
            "status": "deleted",
            "message": "Model deleted successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DELETE_FAILED",
                "message": f"Failed to delete model: {str(e)}",
            },
        )
