"""Image captioning API router for generating image descriptions."""

import base64
import io
import logging
import os
import platform
import time
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
import asyncio
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    Blip2ForConditionalGeneration,
)
import torch

from ..models.image_captioning import CaptionRequest, CaptionResponse
from ...storage.history import history_storage
from ...config import get_model_cache_dir
from ...db.models import get_model as get_model_from_db, get_all_models

logger = logging.getLogger(__name__)

# Check if bitsandbytes is available (Linux only)
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


router = APIRouter(prefix="/api", tags=["image-captioning"])

# Global model cache
_model_cache: Optional[Any] = None
_processor_cache: Optional[Any] = None
_current_model_name: str = ""
_current_model_config: Optional[Dict[str, Any]] = None
_last_access_time: Optional[float] = None


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
    Load available caption models from the database.

    Returns:
        List of model IDs that can be used
    """
    all_models = get_all_models()
    # Filter for image captioning models only
    return [model["id"] for model in all_models if model["task"] == "image-captioning"]


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
            f"Image captioning model '{_current_model_name}' idle for {idle_duration:.1f}s "
            f"(timeout: {idle_timeout}s), unloading from GPU..."
        )
        # Preserve name for accurate post-cleanup logging
        unloaded_name = _current_model_name
        cleanup()
        logger.info(f"Image captioning model '{unloaded_name}' unloaded from GPU")


def get_model(model_name: str):
    """
    Get or load the image captioning model using Auto classes.

    Args:
        model_name: Model identifier to load. If different from currently
                   loaded model, will reload with the new model.

    Returns:
        Tuple of (processor, model, model_config)

    Raises:
        ValueError: If model is not supported
    """
    global _model_cache, _processor_cache, _current_model_name, _current_model_config, _last_access_time

    # Check if current model should be cleaned up due to idle timeout
    check_and_cleanup_idle_model()

    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Check if model requires quantization support
    if model_config.get("requires_quantization") and not HAS_BITSANDBYTES:
        platform_req = model_config.get("platform_requirements", "Linux")
        raise ValueError(
            f"Model '{model_name}' requires bitsandbytes which is not available. "
            f"Platform requirements: {platform_req}. "
            f"Current platform: {platform.system()}. "
            f"On Linux, install with: pip install bitsandbytes"
        )

    # Check if we need to reload the model
    if model_name != _current_model_name:
        # Clear existing cache
        if _model_cache is not None:
            del _model_cache
            _model_cache = None
        if _processor_cache is not None:
            del _processor_cache
            _processor_cache = None
        _current_model_name = model_name
        _current_model_config = model_config

    if _model_cache is None or _processor_cache is None:
        # Get custom cache directory
        cache_dir = get_model_cache_dir("image-caption", _current_model_name)

        # Set HuggingFace cache environment variable
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

        # Set HuggingFace endpoint for model loading
        from ...config import get_hf_endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        try:
            # Load processor using Auto class (works for all architectures)
            _processor_cache = AutoProcessor.from_pretrained(
                _current_model_name, cache_dir=str(cache_dir), use_fast=True
            )

            # Load model based on architecture
            architecture = model_config.get("architecture", "").lower()

            # Common loading kwargs
            load_kwargs = {
                "cache_dir": str(cache_dir),
                "low_cpu_mem_usage": True,     # Reduce CPU memory usage
            }

            # Check if this model requires quantization (from database)
            if model_config.get("requires_quantization"):
                # Pre-quantized models already have quantization config baked in
                # Only need device_map - the model handles its own quantization
                load_kwargs["device_map"] = "auto"
            else:
                # Non-quantized models - use fp16 for efficiency
                load_kwargs["dtype"] = torch.float16

            if architecture == "llava":
                # LLaVA requires specific class
                _model_cache = LlavaForConditionalGeneration.from_pretrained(
                    _current_model_name, **load_kwargs
                )
            elif architecture == "llava_next":
                # LLaVA-NeXT (v1.6) uses different class
                _model_cache = LlavaNextForConditionalGeneration.from_pretrained(
                    _current_model_name, **load_kwargs
                )
            elif architecture == "blip2":
                # BLIP-2 uses specific class
                _model_cache = Blip2ForConditionalGeneration.from_pretrained(
                    _current_model_name, **load_kwargs
                )
            else:
                # BLIP and others use AutoModelForVision2Seq
                _model_cache = AutoModelForVision2Seq.from_pretrained(
                    _current_model_name, **load_kwargs
                )

        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    # Update last access time
    _last_access_time = time.time()

    return _processor_cache, _model_cache, _current_model_config


def cleanup():
    """
    Release model and processor resources immediately.
    Forces GPU memory cleanup to free resources for other services.
    """
    global _model_cache, _processor_cache, _current_model_name, _current_model_config, _last_access_time

    model_name = _current_model_name  # Save for logging

    if _model_cache is not None:
        # Move model to CPU first (helps with cleanup)
        try:
            if hasattr(_model_cache, 'cpu'):
                _model_cache.cpu()
                logger.debug(f"Moved image captioning model '{model_name}' to CPU")
        except Exception as e:
            logger.warning(f"Error moving model to CPU during cleanup: {e}")

        # Remove reference
        del _model_cache
        _model_cache = None

    if _processor_cache is not None:
        del _processor_cache
        _processor_cache = None

    _current_model_name = ""
    _current_model_config = None
    _last_access_time = None

    # Force GPU memory release
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to finish
            logger.debug("GPU cache cleared and synchronized for image captioning model")
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


@router.post("/image-captioning", response_model=CaptionResponse)
async def caption_image(request: CaptionRequest) -> CaptionResponse:
    """
    Generate a caption for an image.

    Creates descriptive text for images using BLIP model.

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
        # Decode image off the event loop
        image = await asyncio.to_thread(decode_image, request.image)

        # Load model and get config off the event loop
        processor, model, model_config = await asyncio.to_thread(get_model, request.model)

        # Run the full preprocessing + generation pipeline in a worker thread
        def _run_inference():
            prompt = request.prompt
            if prompt is None and model_config.get("default_prompt"):
                prompt = model_config["default_prompt"]

            # Process image and generate caption with unified interface
            if prompt:
                inputs = processor(text=prompt, images=image, return_tensors="pt")
            else:
                inputs = processor(images=image, return_tensors="pt")

            # When a model is dispatched with Accelerate (device_map="auto"),
            # different submodules can live on different devices (CPU/GPU).
            # In that case, inputs should generally remain on CPU and the
            # dispatch hooks will move them to the correct device internally.
            # Only move inputs to a single device when the model is on one.
            hf_device_map = getattr(model, "hf_device_map", None) or getattr(model, "device_map", None)
            if not hf_device_map:
                model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=150)

            # Decode output
            _caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            # Clean up caption formatting if necessary
            if "ASSISTANT:" in _caption:
                _caption = _caption.split("ASSISTANT:")[-1].strip()
            elif prompt and _caption.startswith(prompt):
                _caption = _caption[len(prompt):].strip()

            return _caption

        caption = await asyncio.to_thread(_run_inference)

        processing_time_ms = int((time.time() - start_time) * 1000)

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
            request_data={"model": request.model, "prompt": request.prompt},  # Exclude image
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
