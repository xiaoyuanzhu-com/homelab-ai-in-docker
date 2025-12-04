"""Image OCR API router for extracting text from images."""

import base64
import io
import logging
import os
import signal
import time
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
import asyncio
from PIL import Image

from ..models.image_ocr import OCRRequest, OCRResponse
from ...storage.history import history_storage
from ...db.catalog import list_models, list_libs, get_model_dict, get_lib_dict
from ...worker.manager import manager as ocr_manager

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

    def _on_sigterm(signum, frame):  # type: ignore[override]
        try:
            # Restore default for SIGTERM to avoid recursion
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except Exception:
            pass
        try:
            logger.info("Received SIGTERM; forwarding as SIGINT for clean shutdown")
        except Exception:
            pass
        # Send SIGINT to current process so Uvicorn handles graceful shutdown
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            # Fallback: exit immediately
            raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
        _sigterm_forwarder_installed = True
    except Exception:
        # Best-effort; continue without failing OCR if we cannot set handler
        _sigterm_forwarder_installed = False


def get_available_choices() -> list[str]:
    """Return all OCR-capable engines (models + libs)."""
    ids = [m["id"] for m in list_models(task="image-ocr")]
    ids += [l["id"] for l in list_libs(task="image-ocr")]
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
    """Return configuration for either a model or a lib by id.

    Kept for compatibility with in-process loading paths; prefers model entries
    and falls back to lib entries.
    """
    d = get_model_dict(model_id) or get_lib_dict(model_id)
    if d is None:
        raise ValueError(f"Engine '{model_id}' not found in catalog")
    return d


# Backward-compat alias for legacy code paths
def validate_model(model_name: str) -> None:  # pragma: no cover - compatibility
    validate_engine(model_name)


def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    This function is called by periodic cleanup in main.py.
    Note: OCR uses worker manager for isolation - cleanup handled by worker manager.
    """
    # Worker manager handles its own cleanup
    pass


async def _load_model_impl(model_name: str, language: Optional[str] = None) -> tuple[Any, Dict[str, Any]]:
    """
    Internal async function to load the OCR model.

    Note: This is legacy code kept for compatibility. The current implementation
    uses worker manager for isolation.

    Args:
        model_name: Model ID to load
        language: Language code for OCR (e.g., 'ch', 'en', 'fr'). Defaults to 'ch' for multilingual.

    Returns:
        Tuple of (model, model_config)
    """
    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Determine language: request param > model config > default 'ch' (multilingual)
    # 'ch' supports Chinese (Simplified/Traditional) + English + Japanese + Pinyin
    lang = language or model_config.get("language", "ch")

    # Load model in thread pool to avoid blocking
    def _load():
        # Install SIGTERM forwarder for PaddleOCR models
        arch = model_config.get("architecture", "paddleocr-legacy")
        if arch in ("paddleocr", "paddleocr-legacy"):
            _install_sigterm_forwarder()

        logger.info(f"Loading OCR model '{model_name}' with language='{lang}'...")

        # Note: OCRInferenceEngine would be imported here if this code was active
        # Since this is legacy code, we'll raise NotImplementedError
        raise NotImplementedError(
            "Direct model loading is deprecated. OCR now uses worker manager for isolation."
        )

    try:
        model = await asyncio.to_thread(_load)
        return model, model_config
    except Exception as e:
        logger.error(f"Failed to load OCR model '{model_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load OCR model: {str(e)}")


async def _unload_model(model_tuple: tuple[Any, Dict[str, Any]]) -> None:
    """
    Internal async function to unload and cleanup model.

    Note: This is legacy code kept for compatibility. The current implementation
    uses worker manager for isolation.

    Args:
        model_tuple: Tuple of (model, config) to unload
    """
    model, config = model_tuple

    # Cleanup model if it has a cleanup method
    try:
        if hasattr(model, 'cleanup'):
            await asyncio.to_thread(model.cleanup)
            logger.debug("Cleaned up OCR model")
    except Exception as e:
        logger.warning(f"Error cleaning up OCR model during unload: {e}")

    # Delete references
    del model
    del config


async def get_model(model_name: str, language: Optional[str] = None) -> tuple[Any, Dict[str, Any]]:
    """
    Get or load the OCR model.

    Note: This is legacy code - NOT USED. The current implementation
    uses worker manager for isolation (see ocr_image endpoint).

    Args:
        model_name: Model identifier to load
        language: Language code for OCR

    Returns:
        Tuple of (model, model_config)

    Raises:
        NotImplementedError: This function is deprecated
    """
    raise NotImplementedError(
        "Direct model loading is deprecated. OCR now uses worker manager for isolation."
    )


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Note: OCR uses worker manager - cleanup handled by worker manager.
    """
    # Worker manager handles its own cleanup
    pass


def decode_image(image_data: str) -> Image.Image:
    """
    Decode base64 image or load from URL.

    Args:
        image_data: Base64-encoded image string

    Returns:
        PIL Image object
    """
    try:
        # Assume base64 for now
        if image_data.startswith('data:image'):
            # Remove data URL prefix if present
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


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
        selected = request.model or request.lib  # model_validator ensures exactly one
        assert selected is not None
        validate_engine(selected)
        # Use isolated worker manager; pass raw image string (base64 or data URL)
        infer_data = await ocr_manager.infer(
            selected,
            request.language,
            request.image,
            request.output_format or "text"
        )
        final_text = infer_data.get("text", "")
        output_format = infer_data.get("output_format", request.output_format or "text")

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create response
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

        # In isolated mode workers self-terminate; no in-process cleanup needed

        return response

    except ValueError as e:
        error_msg = str(e)
        # Distinguish between image and model errors
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
    except NotImplementedError as e:
        error_msg = str(e)
        logger.info(f"OCR not implemented for request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=501,
            detail={
                "code": "NOT_IMPLEMENTED",
                "message": error_msg,
                "request_id": request_id,
            },
        )
    except TimeoutError as e:
        error_msg = "OCR request timed out. The model may be loading for the first time or the image is too complex. Please try again."
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
