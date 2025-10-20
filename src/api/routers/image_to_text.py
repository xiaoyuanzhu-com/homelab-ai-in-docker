"""Image to text API router for generating image descriptions."""

import base64
import io
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from ..models.caption import CaptionRequest, CaptionResponse
from ...storage.history import history_storage
from ...config import get_model_cache_dir


router = APIRouter(prefix="/api", tags=["image-to-text"])

# Global model cache
_model_cache: Optional[BlipForConditionalGeneration] = None
_processor_cache: Optional[BlipProcessor] = None
_current_model_name: str = "Salesforce/blip-image-captioning-base"


def get_model():
    """
    Get or load the image captioning model.

    Returns:
        Tuple of (processor, model)
    """
    global _model_cache, _processor_cache, _current_model_name

    if _model_cache is None or _processor_cache is None:
        # Get custom cache directory
        cache_dir = get_model_cache_dir("image-caption", _current_model_name)

        # Set HuggingFace cache environment variable
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

        # Load model and processor with custom cache location
        _processor_cache = BlipProcessor.from_pretrained(
            _current_model_name, cache_dir=str(cache_dir)
        )
        _model_cache = BlipForConditionalGeneration.from_pretrained(
            _current_model_name, cache_dir=str(cache_dir)
        )

    return _processor_cache, _model_cache


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

        # Load model
        processor, model = get_model()

        # Process image and generate caption
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)

        caption = processor.decode(out[0], skip_special_tokens=True)

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = CaptionResponse(
            request_id=request_id,
            caption=caption,
            model_used=_current_model_name,
            processing_time_ms=processing_time_ms,
        )

        # Save to history (exclude image data to save space)
        history_storage.add_request(
            service="image-to-text",
            request_id=request_id,
            request_data={"model": request.model},  # Exclude image
            response_data=response.model_dump(),
            status="success",
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE",
                "message": str(e),
                "request_id": request_id,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CAPTION_FAILED",
                "message": f"Failed to generate caption: {str(e)}",
                "request_id": request_id,
            },
        )
