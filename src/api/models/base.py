"""Base Pydantic models with common fields for all API endpoints."""

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model with common fields for all API responses."""

    request_id: str = Field(..., description="Unique request identifier")
    processing_time_ms: int = Field(..., description="Time taken to process the request in milliseconds")
