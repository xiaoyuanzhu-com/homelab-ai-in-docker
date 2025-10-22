"""History API router for request history management."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...storage.history import history_storage


router = APIRouter(prefix="/api/history", tags=["history"])


SERVICE_ALIASES = {
    "crawl": "crawl",
    "text-generation": "text-generation",
    "text-to-embedding": "text-to-embedding",
    "embed": "text-to-embedding",
    "image-captioning": "image-captioning",
    "caption": "image-captioning",
    "image-ocr": "image-ocr",
    "ocr": "image-ocr",
    "automatic-speech-recognition": "automatic-speech-recognition",
    "asr": "automatic-speech-recognition",
}


class HistoryEntry(BaseModel):
    """History entry model."""

    service: str
    timestamp: str
    request_id: str
    status: str
    request: dict
    response: dict


@router.get("/stats")
async def get_stats():
    """
    Get overall task statistics.

    Returns:
        Dictionary with running, today, and total task counts
    """
    return history_storage.get_stats()


def _resolve_service(service: str) -> str:
    resolved = SERVICE_ALIASES.get(service)
    if not resolved:
        raise HTTPException(status_code=400, detail="Invalid service name")
    return resolved


@router.get("/all", response_model=List[HistoryEntry])
async def get_all_history(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get unified request history across all services.

    Args:
        limit: Maximum number of entries (max 100)
        offset: Number of entries to skip

    Returns:
        List of history entries sorted by timestamp (most recent first)
    """
    history = history_storage.get_history(limit=limit, offset=offset)
    return history


@router.get("/{service}", response_model=List[HistoryEntry])
async def get_history(
    service: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get request history for a service.

    Args:
        service: Service name (crawl, embed, caption, text-generation, ocr)
        limit: Maximum number of entries (max 100)
        offset: Number of entries to skip

    Returns:
        List of history entries
    """
    resolved_service = _resolve_service(service)
    history = history_storage.get_history(resolved_service, limit=limit, offset=offset)
    return history


@router.get("/{service}/{request_id}", response_model=HistoryEntry)
async def get_request(service: str, request_id: str):
    """
    Get a specific request by ID.

    Args:
        service: Service name
        request_id: Request ID

    Returns:
        History entry

    Raises:
        HTTPException: If request not found
    """
    resolved_service = _resolve_service(service)

    entry = history_storage.get_request(resolved_service, request_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Request not found")

    return entry


@router.delete("/{service}")
async def clear_history(service: str):
    """
    Clear all history for a service.

    Args:
        service: Service name

    Returns:
        Success message
    """
    resolved_service = _resolve_service(service)

    history_storage.clear_history(resolved_service)
    return {"message": f"History cleared for {resolved_service}"}
