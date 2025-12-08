"""Image captioning API router for generating image descriptions."""

import logging
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ..models.image_captioning import CaptionRequest, CaptionResponse
from ...storage.history import history_storage
from ...db.catalog import get_model_dict, list_models
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["image-captioning"])


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get model configuration from catalog.

    Args:
        model_id: Model identifier

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model not found in catalog
    """
    model = get_model_dict(model_id)

    if model is None:
        raise ValueError(f"Model '{model_id}' not found in catalog")

    return model


def get_available_models() -> list[str]:
    """
    Load available caption models from the catalog.

    Returns:
        List of model IDs that can be used
    """
    caption_models = list_models(task="image-captioning")
    return [m["id"] for m in caption_models]


def validate_model(model_name: str) -> None:
    """
    Validate that the model is supported.

    Args:
        model_name: Model identifier to validate

    Raises:
        ValueError: If model is not supported
    """
    available = get_available_models()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {', '.join(available)}"
        )


def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    Workers handle their own idle timeouts, so this is a no-op.
    """
    pass


def cleanup():
    """
    Release model resources immediately.

    Workers handle their own cleanup, so this is a no-op.
    """
    pass


@router.post("/image-captioning", response_model=CaptionResponse)
async def caption_image(request: CaptionRequest) -> CaptionResponse:
    """
    Generate a caption for an image.

    Creates descriptive text for images using vision-language models.

    Args:
        request: Caption request parameters

    Returns:
        Generated caption and metadata

    Raises:
        HTTPException: If captioning fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Validate model
        validate_model(request.model)

        # Call worker via coordinator
        result = await coordinator_infer(
            task="captioning",
            model_id=request.model,
            payload={
                "image": request.image,
                "prompt": request.prompt,
            },
            request_id=request_id,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        caption = result.get("caption", "")

        response = CaptionResponse(
            request_id=request_id,
            caption=caption,
            model=request.model,
            processing_time_ms=processing_time_ms,
        )

        # Save to history (exclude image data to save space)
        history_storage.add_request(
            service="image-captioning",
            request_id=request_id,
            request_data={"model": request.model, "prompt": request.prompt},
            response_data=response.model_dump(),
            status="success",
        )

        return response

    except ValueError as e:
        error_msg = str(e)
        if "Model" in error_msg or "model" in error_msg:
            code = "INVALID_MODEL"
            logger.warning(f"Model error for request {request_id}: {error_msg}")
        else:
            code = "INVALID_IMAGE"
            logger.warning(f"Image decode error for request {request_id}: {error_msg}")

        raise HTTPException(
            status_code=400,
            detail={
                "code": code,
                "message": error_msg,
                "request_id": request_id,
            },
        )
    except Exception as e:
        error_msg = f"Failed to generate caption: {str(e)}"
        logger.error(f"Caption failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CAPTION_FAILED",
                "message": error_msg,
                "request_id": request_id,
            },
        )
