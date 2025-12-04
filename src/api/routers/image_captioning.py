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
from ...db.catalog import get_model_dict, list_models
from ...services.model_coordinator import use_model

logger = logging.getLogger(__name__)

# Check if bitsandbytes is available (Linux only)
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


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

    This function is called by periodic cleanup in main.py.
    Delegates to the global model coordinator for memory management.
    """
    # Cleanup now handled entirely by the global model coordinator
    pass


async def _load_model_impl(model_name: str) -> tuple[Any, Any, Dict[str, Any]]:
    """
    Internal async function to load the image captioning model.

    Args:
        model_name: Model ID to load

    Returns:
        Tuple of (processor, model, model_config)
    """
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

    # Check for local download at HF standard cache path
    from ...config import get_hf_endpoint, get_hf_model_cache_path
    local_model_dir = get_hf_model_cache_path(model_name)

    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        model_path = str(local_model_dir)
        logger.info(f"Using locally downloaded model from {model_path}")
        extra_kwargs = {"local_files_only": True}
    else:
        model_path = model_name
        logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
        extra_kwargs = {}

    # Set HuggingFace endpoint for model loading
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()

    # Load processor and model in thread pool to avoid blocking
    def _load():
        # Load processor using Auto class (works for all architectures)
        processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, **extra_kwargs
        )

        # Load model based on architecture
        architecture = model_config.get("architecture", "").lower()

        # Common loading kwargs
        load_kwargs = {
            "low_cpu_mem_usage": True,     # Reduce CPU memory usage
            **extra_kwargs,
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
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
        elif architecture == "llava_next":
            # LLaVA-NeXT (v1.6) uses different class
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
        elif architecture == "blip2":
            # BLIP-2 uses specific class
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
        else:
            # BLIP and others use AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_path, **load_kwargs
            )

        return processor, model

    try:
        processor, model = await asyncio.to_thread(_load)
        return processor, model, model_config
    except Exception as e:
        error_msg = f"Failed to load model '{model_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


async def _unload_model(model_tuple: tuple[Any, Any, Dict[str, Any]]) -> None:
    """
    Internal async function to unload and cleanup model.

    Args:
        model_tuple: Tuple of (processor, model, config) to unload
    """
    processor, model, config = model_tuple

    # Move model to CPU first (helps with cleanup)
    try:
        if hasattr(model, 'cpu'):
            await asyncio.to_thread(model.cpu)
            logger.debug("Moved image captioning model to CPU")
    except Exception as e:
        logger.warning(f"Error moving image captioning model to CPU during cleanup: {e}")

    # Delete references
    del processor
    del model
    del config


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Cleanup now handled entirely by the global model coordinator.
    """
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

        # Get model info for memory estimation
        model_config = get_model_config(request.model)
        estimated_memory_mb = model_config.get("gpu_memory_mb") if model_config else None

        # Use context manager for automatic cleanup
        async with use_model(
            key=f"image-caption:{request.model}",
            loader_fn=lambda: _load_model_impl(request.model),
            model_type="image-captioning",
            estimated_memory_mb=estimated_memory_mb,
            unloader_fn=_unload_model,
        ) as model_tuple:
            processor, model, model_config = model_tuple

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
                # We need to move inputs to the device of the first layer.
                # Reference: https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    # Get first layer's device from the device map
                    first_device = model.hf_device_map[next(iter(model.hf_device_map))]
                    inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                else:
                    # Model on single device - use model's device
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

            # Model automatically released when context exits

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

        # Note: Idle cleanup is handled by the global model coordinator
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
