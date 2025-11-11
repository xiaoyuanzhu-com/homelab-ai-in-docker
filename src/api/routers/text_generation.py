"""Text generation API router for generating text from prompts."""

import logging
import os
import time
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from ..models.text_generation import TextGenerationRequest, TextGenerationResponse
from ...storage.history import history_storage
from ...config import get_hf_endpoint, get_hf_model_cache_path
from ...db.catalog import get_model_dict, list_models
from ...services.model_coordinator import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-generation"])

# Track current model name (coordinator manages the actual model cache)
_current_model_name: str = ""


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

    This function is called by periodic cleanup in main.py.
    Delegates to the global model coordinator for memory management.
    """
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 60)

    coordinator = get_coordinator()
    # Use asyncio.create_task to run cleanup asynchronously
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coordinator.cleanup_idle_models(idle_timeout, unloader_fn=_unload_model))
    except RuntimeError:
        # No event loop, skip cleanup
        pass


async def _load_model_impl(model_name: str) -> tuple[Any, Any, Dict[str, Any]]:
    """
    Internal async function to load the text generation model.

    Args:
        model_name: Model ID to load

    Returns:
        Tuple of (tokenizer, model, model_config)

    Raises:
        ValueError: If model is not supported
    """
    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Check for local download at HF standard cache path
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

    try:
        # Load tokenizer and model in thread pool to avoid blocking
        def _load():
            tokenizer = AutoTokenizer.from_pretrained(model_path, **extra_kwargs)

            # Common loading kwargs
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "dtype": torch.float16,  # Use fp16 for efficiency
                **extra_kwargs,
            }

            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to("cuda")

            return tokenizer, model

        tokenizer, model = await asyncio.to_thread(_load)
        return tokenizer, model, model_config

    except Exception as e:
        error_msg = f"Failed to load model '{model_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


async def _unload_model(model_tuple: tuple[Any, Any, Dict[str, Any]]) -> None:
    """
    Internal async function to unload and cleanup model.

    Args:
        model_tuple: Tuple of (tokenizer, model, config) to unload
    """
    tokenizer, model, config = model_tuple

    # Move model to CPU first (helps with cleanup)
    try:
        if hasattr(model, 'cpu'):
            await asyncio.to_thread(model.cpu)
            logger.debug("Moved text generation model to CPU")
    except Exception as e:
        logger.warning(f"Error moving text generation model to CPU during cleanup: {e}")

    # Delete references
    del tokenizer
    del model
    del config


async def get_model(model_name: str) -> tuple[Any, Any, Dict[str, Any]]:
    """
    Get or load the text generation model via the global coordinator.

    Models are managed by the coordinator to prevent OOM errors.
    The coordinator will preemptively unload other models if needed.

    Args:
        model_name: Model identifier to load

    Returns:
        Tuple of (tokenizer, model, model_config)

    Raises:
        ValueError: If model is not supported
    """
    global _current_model_name

    _current_model_name = model_name

    # Get model info for memory estimation
    model_info = get_model_dict(model_name)
    estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

    # Load through coordinator (handles preemptive unload)
    coordinator = get_coordinator()
    model_tuple = await coordinator.load_model(
        key=f"text-gen:{model_name}",
        loader_fn=lambda: _load_model_impl(model_name),
        unloader_fn=_unload_model,
        estimated_memory_mb=estimated_memory_mb,
        model_type="text-generation",
    )

    return model_tuple


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Delegates to the global model coordinator.
    """
    coordinator = get_coordinator()
    # Use asyncio to run cleanup synchronously
    try:
        loop = asyncio.get_running_loop()
        # Create task to unload all text generation models
        loop.create_task(coordinator.unload_model(f"text-gen:{_current_model_name}", _unload_model))
    except RuntimeError:
        # No event loop, can't cleanup
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
        # Load model via coordinator (handles preemptive unload)
        tokenizer, model, model_config = await get_model(request.model)

        # Run the full tokenization + generation pipeline in a worker thread
        def _run_inference():
            # Tokenize input
            inputs = tokenizer(request.prompt, return_tensors="pt")

            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    pad_token_id=tokenizer.eos_token_id,  # Avoid warnings
                )

            # Decode output (skip the input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output_ids[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_text, len(generated_tokens)

        generated_text, tokens_generated = await asyncio.to_thread(_run_inference)

        processing_time_ms = int((time.time() - start_time) * 1000)

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

        # Note: Idle cleanup is handled by the global model coordinator
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
