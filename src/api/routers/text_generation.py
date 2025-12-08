"""Text generation API router for generating text from prompts."""

import logging
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ..models.text_generation import TextGenerationRequest, TextGenerationResponse
from ...storage.history import history_storage
from ...db.catalog import get_model_dict, list_models
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-generation"])


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
    Load available text generation models from the catalog.

    Returns:
        List of model IDs that can be used
    """
    text_gen_models = list_models(task="text-generation")
    return [m["id"] for m in text_gen_models]


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


@router.post("/text-generation", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest) -> TextGenerationResponse:
    """
    Generate text from a prompt using a language model.

    Creates text completions using causal language models like Qwen3 and Gemma.

    Args:
        request: Text generation request parameters

    Returns:
        Generated text and metadata

    Raises:
        HTTPException: If generation fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Validate model
        validate_model(request.model)

        # Call worker via coordinator
        result = await coordinator_infer(
            task="text-generation",
            model_id=request.model,
            payload={
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.do_sample,
            },
            request_id=request_id,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        generated_text = result.get("generated_text", "")
        tokens_generated = result.get("tokens_generated", 0)

        response = TextGenerationResponse(
            request_id=request_id,
            generated_text=generated_text,
            model=request.model,
            tokens_generated=tokens_generated,
            processing_time_ms=processing_time_ms,
        )

        # Save to history
        history_storage.add_request(
            service="text-generation",
            request_id=request_id,
            request_data={
                "model": request.model,
                "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
            },
            response_data={
                "request_id": response.request_id,
                "model": response.model,
                "tokens_generated": response.tokens_generated,
                "processing_time_ms": response.processing_time_ms,
            },
            status="success",
        )

        return response

    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"Model error for request {request_id}: {error_msg}")

        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_MODEL",
                "message": error_msg,
                "request_id": request_id,
            },
        )
    except Exception as e:
        error_msg = f"Failed to generate text: {str(e)}"
        logger.error(f"Text generation failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "GENERATION_FAILED",
                "message": error_msg,
                "request_id": request_id,
            },
        )
