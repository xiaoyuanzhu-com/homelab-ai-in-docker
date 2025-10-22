"""Settings API router for application configuration."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...db.settings import get_all_settings, get_setting, set_setting

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])

# Sensitive keys that should be redacted in logs
SENSITIVE_KEYS = {"hf_token", "token", "password", "secret", "api_key", "apikey"}


def should_redact(key: str) -> bool:
    """Check if a setting key contains sensitive data that should be redacted."""
    key_lower = key.lower()
    return any(sensitive in key_lower for sensitive in SENSITIVE_KEYS)


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

    Note: Sensitive values (tokens, passwords, etc.) are returned as-is since
    the frontend needs them to display in password fields. The frontend should
    use type="password" for these fields.

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

        # Redact sensitive values in logs
        if should_redact(key):
            logger.info(f"Updated setting '{key}' to '[REDACTED]'")
        else:
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
