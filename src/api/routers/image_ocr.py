"""Image OCR API router for extracting text from images."""

import logging
import os
import signal
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ..models.image_ocr import OCRRequest, OCRResponse
from ...storage.history import history_storage
from ...db.catalog import list_models, list_libs, get_model_dict, get_lib_dict
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["image-ocr"])


# Ensure Paddle's SIGTERM handler (which logs a FatalError message) doesn't
# pollute shutdown logs when the dev reloader sends SIGTERM. We forward
# SIGTERM to SIGINT so Uvicorn performs its normal graceful shutdown.
_sigterm_forwarder_installed = False


def _install_sigterm_forwarder():
    global _sigterm_forwarder_installed
    if _sigterm_forwarder_installed:
        return

    def _on_sigterm(signum, frame):
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except Exception:
            pass
        try:
            logger.info("Received SIGTERM; forwarding as SIGINT for clean shutdown")
        except Exception:
            pass
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
        _sigterm_forwarder_installed = True
    except Exception:
        _sigterm_forwarder_installed = False


def get_available_choices() -> list[str]:
    """Return all OCR-capable engines (models + libs)."""
    ids = [m["id"] for m in list_models(task="image-ocr")]
    ids += [lib["id"] for lib in list_libs(task="image-ocr")]
    return sorted(set(ids))


def validate_engine(engine_id: str) -> None:
    """Validate that the selected engine (model or lib) is supported."""
    available = get_available_choices()
    if engine_id not in available:
        raise ValueError(
            f"Engine '{engine_id}' is not supported. "
            f"Available: {', '.join(available)}"
        )


def get_model_config(model_id: str) -> Dict[str, Any]:
    """Return configuration for either a model or a lib by id."""
    d = get_model_dict(model_id) or get_lib_dict(model_id)
    if d is None:
        raise ValueError(f"Engine '{model_id}' not found in catalog")
    return d


# Backward-compat alias for legacy code paths
def validate_model(model_name: str) -> None:
    validate_engine(model_name)


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


@router.post("/image-ocr", response_model=OCRResponse)
async def ocr_image(request: OCRRequest) -> OCRResponse:
    """
    Extract text from an image using OCR.

    Performs optical character recognition to extract text content from images.

    Args:
        request: OCR request parameters

    Returns:
        Extracted text and metadata

    Raises:
        HTTPException: If OCR fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Determine selected engine (model or lib) and validate
        selected = request.model or request.lib
        assert selected is not None
        validate_engine(selected)

        # Get model config for python_env
        model_config = get_model_config(selected)
        python_env = model_config.get("python_env")

        # Build extra args for language
        extra_args = {}
        if request.language:
            extra_args["language"] = request.language

        # Call worker via coordinator
        result = await coordinator_infer(
            task="ocr",
            model_id=selected,
            payload={
                "image": request.image,
                "output_format": request.output_format or "text",
            },
            request_id=request_id,
            python_env=python_env,
            extra_args=extra_args,
        )

        final_text = result.get("text", "")
        output_format = result.get("output_format", request.output_format or "text")

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = OCRResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            text=final_text,
            model=selected,
            output_format=output_format,
        )

        # Log to history
        history_storage.add_request(
            service="image-ocr",
            request_id=request_id,
            request_data={"model": selected},  # Exclude image
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
    except TimeoutError as e:
        error_msg = (
            "OCR request timed out. The model may be loading for the first time "
            "or the image is too complex. Please try again."
        )
        logger.error(f"OCR timeout for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail={
                "code": "TIMEOUT",
                "message": error_msg,
                "request_id": request_id,
            },
        )
    except Exception as e:
        error_msg = f"Failed to perform OCR: {str(e)}"
        logger.error(f"OCR failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "OCR_FAILED",
                "message": error_msg,
                "request_id": request_id,
            },
        )
