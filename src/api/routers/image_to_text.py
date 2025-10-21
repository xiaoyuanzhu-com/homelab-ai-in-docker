"""Image to text API router for generating image descriptions."""

import base64
import io
import json
import logging
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
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

from ..models.image_to_text import CaptionRequest, CaptionResponse
from ...storage.history import history_storage
from ...config import get_model_cache_dir

logger = logging.getLogger(__name__)

# Check if bitsandbytes is available (Linux only)
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


router = APIRouter(prefix="/api", tags=["image-to-text"])

# Global model cache
_model_cache: Optional[Any] = None
_processor_cache: Optional[Any] = None
_current_model_name: str = ""
_current_model_config: Optional[Dict[str, Any]] = None

# Load available models from manifest
_available_models: Optional[list[str]] = None
_models_manifest: Optional[Dict[str, list[Dict[str, Any]]]] = None


def load_models_manifest() -> Dict[str, list[Dict[str, Any]]]:
    """
    Load models manifest from JSON file.

    Returns:
        Dictionary with model type as key and list of model configs as value
    """
    global _models_manifest

    if _models_manifest is None:
        manifest_path = Path(__file__).parent.parent / "models" / "models_manifest.json"
        try:
            with open(manifest_path) as f:
                _models_manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load models manifest: {e}")

    return _models_manifest


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get model configuration from manifest.

    Args:
        model_id: Model identifier

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model not found in manifest
    """
    manifest = load_models_manifest()

    for model_config in manifest.get("caption", []):
        if model_config["id"] == model_id:
            return model_config

    raise ValueError(f"Model '{model_id}' not found in manifest")


def get_available_models() -> list[str]:
    """
    Load available caption models from the manifest.

    Returns:
        List of model IDs that can be used
    """
    global _available_models

    if _available_models is None:
        manifest = load_models_manifest()
        _available_models = [model["id"] for model in manifest.get("caption", [])]

    return _available_models


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
    global _model_cache, _processor_cache, _current_model_name, _current_model_config

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

            # Check if this is a quantized model
            is_quantized = "bnb" in _current_model_name.lower() or "4bit" in _current_model_name.lower()

            if is_quantized:
                # Pre-quantized models - load with device_map="auto"
                # Don't set dtype - quantized models have their own precision (4-bit)
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

    return _processor_cache, _model_cache, _current_model_config


def cleanup():
    """Release model and processor resources on shutdown."""
    global _model_cache, _processor_cache, _current_model_name, _current_model_config
    if _model_cache is not None:
        del _model_cache
        _model_cache = None
    if _processor_cache is not None:
        del _processor_cache
        _processor_cache = None
    _current_model_name = ""
    _current_model_config = None


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


@router.post("/image-to-text", response_model=CaptionResponse)
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
        # Decode image
        image = decode_image(request.image)

        # Load model and get config
        processor, model, model_config = get_model(request.model)

        # Determine prompt to use (user-provided or default from config)
        prompt = request.prompt
        if prompt is None and model_config.get("default_prompt"):
            prompt = model_config["default_prompt"]

        # Process image and generate caption with unified interface
        if prompt:
            # Models that support text prompts (BLIP-2, LLaVA, etc.)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        else:
            # Models that only need image (BLIP base/large)
            inputs = processor(images=image, return_tensors="pt")

        # Move inputs to the same device as the model
        # Works for both quantized models (with device_map) and regular models
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=150)

        # Decode output
        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Clean up caption (remove prompt if it was echoed back)
        if prompt and caption.startswith(prompt):
            caption = caption[len(prompt):].strip()

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = CaptionResponse(
            request_id=request_id,
            caption=caption,
            model_used=request.model,
            processing_time_ms=processing_time_ms,
        )

        # Save to history (exclude image data to save space)
        history_storage.add_request(
            service="image-to-text",
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
