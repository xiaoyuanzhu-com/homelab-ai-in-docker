"""History API router for request history management."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...storage.history import history_storage


router = APIRouter(prefix="/api/history", tags=["history"])


class HistoryEntry(BaseModel):
    """History entry model."""

    timestamp: str
    request_id: str
    status: str
    request: dict
    response: dict


@router.get("/{service}", response_model=List[HistoryEntry])
async def get_history(
    service: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get request history for a service.

    Args:
        service: Service name (crawl, embed, caption)
        limit: Maximum number of entries (max 100)
        offset: Number of entries to skip

    Returns:
        List of history entries
    """
    if service not in ["crawl", "embed", "caption"]:
        raise HTTPException(status_code=400, detail="Invalid service name")

    history = history_storage.get_history(service, limit=limit, offset=offset)
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
    if service not in ["crawl", "embed", "caption"]:
        raise HTTPException(status_code=400, detail="Invalid service name")

    entry = history_storage.get_request(service, request_id)
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
    if service not in ["crawl", "embed", "caption"]:
        raise HTTPException(status_code=400, detail="Invalid service name")

    history_storage.clear_history(service)
    return {"message": f"History cleared for {service}"}
