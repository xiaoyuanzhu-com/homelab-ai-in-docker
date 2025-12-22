"""Image segmentation API router using Segment Anything (SAM3) worker."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from ..models.image_segmentation import SegmentationRequest, SegmentationResponse
from ...storage.history import history_storage
from ...db.catalog import list_libs, get_lib_dict
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["image-segmentation"])


def _available_segmentation_libs() -> list[str]:
    try:
        return [lib["id"] for lib in list_libs(task="image-segmentation")]
    except Exception:
        return []


@router.post("/image-segmentation", response_model=SegmentationResponse)
async def segment_image(request: SegmentationRequest) -> SegmentationResponse:
    """Run promptable image segmentation using Segment Anything (SAM3)."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    lib_id = request.lib or "facebookresearch/sam3"
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_PROMPT",
                "message": "Prompt must not be empty",
                "request_id": request_id,
            },
        )
    supported = _available_segmentation_libs()

    if lib_id not in supported:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ENGINE",
                "message": (
                    f"Engine '{lib_id}' is not available. "
                    f"Available: {', '.join(supported) if supported else 'none'}"
                ),
                "request_id": request_id,
            },
        )

    lib_config = get_lib_dict(lib_id)
    if not lib_config:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ENGINE",
                "message": f"Library '{lib_id}' not found in catalog",
                "request_id": request_id,
            },
        )

    python_env = lib_config.get("python_env")

    try:
        result = await coordinator_infer(
            task="image-segmentation",
            model_id=lib_id,
            payload={
                "image": request.image,
                "prompt": prompt,
                "confidence_threshold": request.confidence_threshold,
                "max_masks": request.max_masks,
            },
            request_id=request_id,
            python_env=python_env,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = SegmentationResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            model=lib_id,
            prompt=result.get("prompt", prompt),
            image_width=result.get("image_width", 0),
            image_height=result.get("image_height", 0),
            masks=result.get("masks", []),
        )

        history_storage.add_request(
            service="image-segmentation",
            request_id=request_id,
            request_data={"lib": lib_id, "prompt": prompt},
            response_data={
                "request_id": request_id,
                "processing_time_ms": processing_time_ms,
                "model": response.model,
                "mask_count": len(response.masks),
            },
            status="success",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image segmentation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SEGMENTATION_FAILED",
                "message": f"Failed to segment image: {e}",
                "request_id": request_id,
            },
        )
