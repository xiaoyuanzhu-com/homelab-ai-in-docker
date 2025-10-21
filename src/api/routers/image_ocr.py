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
from PIL import Image

from ..models.image_ocr import OCRRequest, OCRResponse
from ...storage.history import history_storage
from ...config import get_model_cache_dir
from ...db.models import get_model as get_model_from_db, get_all_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["image-ocr"])

# Global model cache
_model_cache: Optional[Any] = None
_current_model_name: str = ""
_current_language: str = ""
_current_model_config: Optional[Dict[str, Any]] = None
_last_access_time: Optional[float] = None


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
    Get model configuration from database.

    Args:
        model_id: Model identifier

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model not found in database
    """
    db_model = get_model_from_db(model_id)

    if db_model is None:
        raise ValueError(f"Model '{model_id}' not found in database")

    # Convert sqlite3.Row to dict
    return {
        "id": db_model["id"],
        "name": db_model["name"],
        "team": db_model["team"],
        "task": db_model["task"],
        "architecture": db_model["architecture"],
        "default_prompt": db_model["default_prompt"],
        "platform_requirements": db_model["platform_requirements"],
        "requires_quantization": bool(db_model["requires_quantization"]),
        "size_mb": db_model["size_mb"],
        "parameters_m": db_model["parameters_m"],
        "gpu_memory_mb": db_model["gpu_memory_mb"],
        "link": db_model["link"],
    }


def get_available_models() -> list[str]:
    """
    Load available OCR models from the database.

    Returns:
        List of model IDs that can be used
    """
    all_models = get_all_models()
    # Filter for image OCR models only
    return [model["id"] for model in all_models if model["task"] == "image-ocr"]


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
        NotImplementedError: PaddleOCR support coming soon
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
            del _model_cache
            _model_cache = None
        _current_model_name = model_name
        _current_language = lang
        _current_model_config = model_config

    if _model_cache is None:
        # Load PaddleOCR model
        try:
            from paddleocr import PaddleOCR

            # Forward SIGTERM to SIGINT after Paddle is imported to avoid
            # Paddle's FatalError log on normal shutdown under reloaders.
            _install_sigterm_forwarder()

            logger.info(f"Loading OCR model '{model_name}' with language='{lang}'...")

            # Initialize PaddleOCR
            # PaddleOCR 3.x uses 'device' parameter instead of 'use_gpu'
            # Check if PaddlePaddle has GPU support
            try:
                import paddle
                has_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
                device = "gpu:0" if has_gpu else "cpu"
            except Exception:
                # Fallback to CPU if paddle check fails
                device = "cpu"

            _model_cache = PaddleOCR(
                lang=lang,
                device=device
            )

            logger.info(f"PaddleOCR initialized with lang={lang}, device={device}")

            logger.info(f"OCR model '{model_name}' loaded successfully")

        except ImportError:
            raise RuntimeError(
                "PaddleOCR is not installed. Please install it with: "
                "pip install paddlepaddle-gpu paddleocr"
            )
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
            # TODO: Add proper cleanup for PaddleOCR models
            logger.debug(f"Cleaning up OCR model '{model_name}'")
        except Exception as e:
            logger.warning(f"Error cleaning up OCR model: {e}")

        del _model_cache
        _model_cache = None

    _current_model_name = ""
    _current_language = ""
    _current_model_config = None
    _last_access_time = None

    # Force GPU memory release
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared and synchronized for OCR model")
    except Exception as e:
        logger.warning(f"Error releasing GPU memory: {e}")


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
        # Decode image
        image = decode_image(request.image)

        # Load model and get config (with optional language parameter)
        model, model_config = get_model(request.model, language=request.language)

        # Save image temporarily for PaddleOCR (it expects file path)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Perform OCR
            # PaddleOCR 3.x: ocr() is an alias for predict(), no cls parameter needed
            result = model.predict(tmp_path)

            # Parse results - PaddleOCR 3.x returns list of dicts with 'rec_texts' field
            # Format: [{'rec_texts': ['text1', 'text2', ...], 'rec_scores': [0.95, 0.89, ...], ...}]
            extracted_text = []
            if result and len(result) > 0:
                # Get the first result (single image)
                page_result = result[0]
                if 'rec_texts' in page_result:
                    extracted_text = page_result['rec_texts']

            # Join all detected text with newlines
            final_text = "\n".join(extracted_text) if extracted_text else ""

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create response
        response = OCRResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            text=final_text,
            model=request.model,
        )

        # Log to history
        history_storage.add_request(
            service="image-ocr",
            request_id=request_id,
            request_data={"model": request.model},  # Exclude image
            response_data=response.model_dump(),
            status="success",
        )

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
