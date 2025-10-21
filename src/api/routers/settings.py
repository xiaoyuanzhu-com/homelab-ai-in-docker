"""Settings API router for application configuration."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...db.settings import get_all_settings, get_setting, set_setting

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])


class SettingValue(BaseModel):
    """Request model for updating a setting value."""
    value: str = Field(..., description="Setting value")
    description: str | None = Field(None, description="Optional description")


class SettingsResponse(BaseModel):
    """Response model for settings."""
    settings: Dict[str, str] = Field(..., description="All settings as key-value pairs")


@router.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """
    Get all application settings.
    
    Returns:
        All settings as key-value pairs
    """
    try:
        settings = get_all_settings()
        return SettingsResponse(settings=settings)
    except Exception as e:
        logger.error(f"Error getting settings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SETTINGS_ERROR",
                "message": f"Failed to get settings: {str(e)}"
            }
        )


@router.put("/settings/{key}")
async def update_setting(key: str, setting: SettingValue) -> Dict[str, Any]:
    """
    Update a setting value.
    
    Args:
        key: Setting key to update
        setting: New value and optional description
        
    Returns:
        Success message with updated value
    """
    try:
        set_setting(key, setting.value, setting.description)
        logger.info(f"Updated setting '{key}' to '{setting.value}'")
        return {
            "success": True,
            "key": key,
            "value": setting.value,
            "message": f"Setting '{key}' updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating setting '{key}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "UPDATE_ERROR",
                "message": f"Failed to update setting: {str(e)}"
            }
        )


@router.get("/settings/{key}")
async def get_setting_value(key: str) -> Dict[str, Any]:
    """
    Get a specific setting value.
    
    Args:
        key: Setting key
        
    Returns:
        Setting value
    """
    try:
        value = get_setting(key)
        if value is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SETTING_NOT_FOUND",
                    "message": f"Setting '{key}' not found"
                }
            )
        return {
            "key": key,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting setting '{key}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SETTINGS_ERROR",
                "message": f"Failed to get setting: {str(e)}"
            }
        )
