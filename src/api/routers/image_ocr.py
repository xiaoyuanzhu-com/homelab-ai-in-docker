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
from ...db.skills import get_skill_dict, list_skills
from ...worker.manager import manager as ocr_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["image-ocr"])

# Global model cache (unused in isolated mode; kept for compatibility)
_model_cache: Optional[Any] = None
_current_model_name: str = ""
_current_language: str = ""
_current_model_config: Optional[Dict[str, Any]] = None
_last_access_time: Optional[float] = None
_idle_cleanup_task: Optional[asyncio.Task] = None


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


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get skill configuration from database.

    Args:
        model_id: Skill identifier

    Returns:
        Skill configuration dictionary

    Raises:
        ValueError: If skill not found in database
    """
    skill = get_skill_dict(model_id)

    if skill is None:
        raise ValueError(f"Skill '{model_id}' not found in database")

    return skill


def get_available_models() -> list[str]:
    """
    Load available OCR skills from the database.

    Returns:
        List of skill IDs that can be used
    """
    # Filter for image-ocr task
    ocr_skills = list_skills(task="image-ocr")
    return [skill["id"] for skill in ocr_skills]


def validate_model(model_name: str) -> None:
    """
    Validate that the skill is supported.

    Args:
        model_name: Skill identifier to validate

    Raises:
        ValueError: If skill is not supported
    """
    available = get_available_models()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {', '.join(available)}"
        )


def check_and_cleanup_idle_model():
    """Check if model has been idle too long and cleanup if needed."""
    global _model_cache, _last_access_time, _current_model_name

    if _model_cache is None or _last_access_time is None:
        return

    # Get idle timeout from settings
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    # Check if model has been idle too long
    idle_duration = time.time() - _last_access_time
    if idle_duration >= idle_timeout:
        logger.info(
            f"OCR model '{_current_model_name}' idle for {idle_duration:.1f}s "
            f"(timeout: {idle_timeout}s), unloading from GPU..."
        )
        cleanup()
        logger.info(f"OCR model '{_current_model_name}' unloaded from GPU")


def schedule_idle_cleanup() -> None:
    """Schedule a background task that will cleanup the model after idle timeout.

    This ensures cleanup happens even if no further requests arrive.
    """
    global _idle_cleanup_task

    # Attempt to get the running loop; if not available, skip scheduling
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    # Fetch timeout each time so changes to settings apply dynamically
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    # Cancel any pending watchdog
    if _idle_cleanup_task and not _idle_cleanup_task.done():
        _idle_cleanup_task.cancel()

    async def _watchdog(start_time: float, timeout_s: int):
        try:
            # Sleep for the configured timeout, then verify we are still idle
            await asyncio.sleep(timeout_s)
            # Double-check idle state to avoid races with new activity
            if _last_access_time is None:
                return
            idle_duration = time.time() - _last_access_time
            if idle_duration >= timeout_s and _model_cache is not None:
                logger.info(
                    f"OCR model '{_current_model_name}' idle for {idle_duration:.1f}s "
                    f"(timeout: {timeout_s}s), unloading from GPU..."
                )
                cleanup()
        except asyncio.CancelledError:
            # Expected during reschedule; ignore
            pass
        finally:
            # Clear the task reference when finished if still current
            global _idle_cleanup_task
            try:
                current = asyncio.current_task()
            except Exception:
                current = None
            if current is not None and _idle_cleanup_task is current:
                _idle_cleanup_task = None

    _idle_cleanup_task = loop.create_task(_watchdog(time.time(), idle_timeout))



def get_model(model_name: str, language: Optional[str] = None):
    """
    Get or load the OCR model.

    Args:
        model_name: Model identifier to load
        language: Language code for OCR (e.g., 'ch', 'en', 'fr'). Defaults to 'ch' for multilingual.

    Returns:
        Loaded model and configuration

    Raises:
        ValueError: If model is not supported
        RuntimeError: If model loading fails
    """
    global _model_cache, _current_model_name, _current_language, _current_model_config, _last_access_time

    # Check if current model should be cleaned up due to idle timeout
    check_and_cleanup_idle_model()

    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Determine language: request param > model config > default 'ch' (multilingual)
    # 'ch' supports Chinese (Simplified/Traditional) + English + Japanese + Pinyin
    lang = language or model_config.get("language", "ch")

    # Check if we need to reload the model (model name or language changed)
    if model_name != _current_model_name or lang != _current_language:
        # Clear existing cache
        if _model_cache is not None:
            _model_cache.cleanup()
            _model_cache = None
        _current_model_name = model_name
        _current_language = lang
        _current_model_config = model_config

    if _model_cache is None:
        try:
            # Install SIGTERM forwarder for PaddleOCR models
            arch = model_config.get("architecture", "paddleocr-legacy")
            if arch in ("paddleocr", "paddleocr-legacy"):
                _install_sigterm_forwarder()

            logger.info(f"Loading OCR model '{model_name}' with language='{lang}'...")

            # Create inference engine
            # Note: output_format will be set per request in get_model
            _model_cache = OCRInferenceEngine(
                model_id=model_name,
                architecture=arch,
                model_config=model_config,
                language=lang,
                output_format="text",  # Default, will be updated per request
            )

            # Load the model
            _model_cache.load()

            logger.info(f"OCR model '{model_name}' loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load OCR model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load OCR model: {str(e)}")

    # Update last access time
    _last_access_time = time.time()

    return _model_cache, _current_model_config


def cleanup():
    """
    Release model resources immediately.
    Forces GPU memory cleanup to free resources for other services.
    """
    global _model_cache, _current_model_name, _current_language, _current_model_config, _last_access_time

    model_name = _current_model_name  # Save for logging

    if _model_cache is not None:
        try:
            logger.debug(f"Cleaning up OCR model '{model_name}'")
            _model_cache.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up OCR model: {e}")

        _model_cache = None

    _current_model_name = ""
    _current_language = ""
    _current_model_config = None
    _last_access_time = None


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
        # Validate model exists
        validate_model(request.model)
        # Use isolated worker manager; pass raw image string (base64 or data URL)
        infer_data = await ocr_manager.infer(
            request.model,
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
            model=request.model,
            output_format=output_format,
        )

        # Log to history
        history_storage.add_request(
            service="image-ocr",
            request_id=request_id,
            request_data={"model": request.model},  # Exclude image
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
