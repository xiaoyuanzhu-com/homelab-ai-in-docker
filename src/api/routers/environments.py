"""Environment management API router.

Provides endpoints for managing worker environments:
- List all environments and their status
- Get specific environment status
- Pre-install an environment
- Delete an environment to free disk space
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ...worker.env_manager import (
    EnvInfo,
    EnvStatus,
    get_env_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/environments", tags=["environments"])


class EnvStatusResponse(BaseModel):
    """Response model for environment status."""

    env_id: str = Field(..., description="Environment identifier")
    status: str = Field(..., description="Current status (not_installed, installing, ready, failed)")
    size_mb: Optional[float] = Field(None, description="Installed size in MB")
    python_version: Optional[str] = Field(None, description="Python version")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    install_time_seconds: Optional[float] = Field(None, description="Installation duration or time elapsed")


class EnvListResponse(BaseModel):
    """Response model for listing environments."""

    environments: Dict[str, EnvStatusResponse] = Field(
        ..., description="Map of env_id to status"
    )


class InstallResponse(BaseModel):
    """Response model for install endpoint."""

    env_id: str
    status: str
    message: str
    background: bool = Field(False, description="Whether installation is running in background")


def _env_info_to_response(info: EnvInfo) -> EnvStatusResponse:
    """Convert EnvInfo to API response model."""
    return EnvStatusResponse(
        env_id=info.env_id,
        status=info.status.value,
        size_mb=info.size_mb,
        python_version=info.python_version,
        error_message=info.error_message,
        install_time_seconds=info.install_time_seconds,
    )


@router.get("", response_model=EnvListResponse)
async def list_environments() -> EnvListResponse:
    """
    List all available worker environments and their status.

    Returns:
        Map of environment IDs to their current status
    """
    try:
        env_manager = get_env_manager()
        envs = env_manager.list_environments()

        return EnvListResponse(
            environments={
                env_id: _env_info_to_response(info)
                for env_id, info in envs.items()
            }
        )
    except Exception as e:
        logger.error(f"Error listing environments: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ENV_LIST_ERROR",
                "message": f"Failed to list environments: {str(e)}",
            },
        )


@router.get("/{env_id}", response_model=EnvStatusResponse)
async def get_environment_status(env_id: str) -> EnvStatusResponse:
    """
    Get status of a specific worker environment.

    Args:
        env_id: Environment identifier (e.g., "transformers", "whisper")

    Returns:
        Current environment status
    """
    try:
        env_manager = get_env_manager()
        info = env_manager.get_env_status(env_id)

        if info.status == EnvStatus.NOT_FOUND:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ENV_NOT_FOUND",
                    "message": f"Environment '{env_id}' not found",
                },
            )

        return _env_info_to_response(info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environment status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ENV_STATUS_ERROR",
                "message": f"Failed to get environment status: {str(e)}",
            },
        )


@router.post("/{env_id}/install", response_model=InstallResponse)
async def install_environment(
    env_id: str,
    background_tasks: BackgroundTasks,
    wait: bool = False,
) -> InstallResponse:
    """
    Install a worker environment.

    By default, runs in background and returns immediately.
    Set wait=true to wait for installation to complete (may take 1-2 minutes).

    Args:
        env_id: Environment identifier
        wait: If true, wait for installation to complete

    Returns:
        Installation status
    """
    env_manager = get_env_manager()

    # Check current status
    info = env_manager.get_env_status(env_id)

    if info.status == EnvStatus.NOT_FOUND:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ENV_NOT_FOUND",
                "message": f"Environment template '{env_id}' not found",
            },
        )

    if info.status == EnvStatus.READY:
        return InstallResponse(
            env_id=env_id,
            status="ready",
            message=f"Environment '{env_id}' is already installed",
        )

    if info.status == EnvStatus.INSTALLING:
        return InstallResponse(
            env_id=env_id,
            status="installing",
            message=f"Environment '{env_id}' is already being installed",
            background=True,
        )

    if wait:
        # Wait for installation to complete
        try:
            await env_manager.ensure_installed(env_id)
            return InstallResponse(
                env_id=env_id,
                status="ready",
                message=f"Environment '{env_id}' installed successfully",
            )
        except Exception as e:
            logger.error(f"Environment installation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INSTALL_FAILED",
                    "message": f"Failed to install environment '{env_id}': {str(e)}",
                },
            )
    else:
        # Start installation in background
        async def install_task():
            try:
                await env_manager.ensure_installed(env_id)
                logger.info(f"Background installation of '{env_id}' completed")
            except Exception as e:
                logger.error(f"Background installation of '{env_id}' failed: {e}")

        background_tasks.add_task(install_task)

        return InstallResponse(
            env_id=env_id,
            status="installing",
            message=f"Environment '{env_id}' installation started",
            background=True,
        )


@router.delete("/{env_id}")
async def delete_environment(env_id: str) -> Dict[str, Any]:
    """
    Delete an installed environment to free disk space.

    The environment template is preserved; only the installed .venv is removed.
    The environment can be re-installed later.

    Args:
        env_id: Environment identifier

    Returns:
        Deletion result
    """
    try:
        env_manager = get_env_manager()
        info = env_manager.get_env_status(env_id)

        if info.status == EnvStatus.NOT_FOUND:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ENV_NOT_FOUND",
                    "message": f"Environment '{env_id}' not found",
                },
            )

        if info.status == EnvStatus.INSTALLING:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "ENV_INSTALLING",
                    "message": f"Cannot delete environment '{env_id}' while installing",
                },
            )

        if info.status == EnvStatus.NOT_INSTALLED:
            return {
                "success": True,
                "env_id": env_id,
                "message": f"Environment '{env_id}' is not installed",
                "freed_mb": 0,
            }

        size_mb = info.size_mb or 0
        deleted = await env_manager.delete_env(env_id)

        return {
            "success": deleted,
            "env_id": env_id,
            "message": f"Environment '{env_id}' deleted" if deleted else "Nothing to delete",
            "freed_mb": size_mb if deleted else 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting environment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DELETE_ERROR",
                "message": f"Failed to delete environment: {str(e)}",
            },
        )
