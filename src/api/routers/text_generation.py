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
from ...config import get_hf_endpoint
from ...db.skills import get_skill_dict, list_skills

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-generation"])

# Global model cache
_model_cache: Optional[Any] = None
_tokenizer_cache: Optional[Any] = None
_current_model_name: str = ""
_current_model_config: Optional[Dict[str, Any]] = None
_last_access_time: Optional[float] = None
_idle_cleanup_task: Optional[asyncio.Task] = None
_model_in_use: bool = False  # Flag to indicate model is actively being used


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
    Load available text generation skills from the database.

    Returns:
        List of skill IDs that can be used
    """
    text_gen_skills = list_skills(task="text-generation")
    return [skill["id"] for skill in text_gen_skills]


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
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 60)

    # Check if model has been idle too long
    idle_duration = time.time() - _last_access_time
    if idle_duration >= idle_timeout:
        logger.info(
            f"Text generation model '{_current_model_name}' idle for {idle_duration:.1f}s "
            f"(timeout: {idle_timeout}s), unloading from GPU..."
        )
        # Preserve name for accurate post-cleanup logging
        unloaded_name = _current_model_name
        cleanup()
        logger.info(f"Text generation model '{unloaded_name}' unloaded from GPU")


def schedule_idle_cleanup() -> None:
    """Schedule background cleanup after idle timeout."""
    global _idle_cleanup_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 60)

    if _idle_cleanup_task and not _idle_cleanup_task.done():
        _idle_cleanup_task.cancel()

    async def _watchdog(timeout_s: int):
        try:
            await asyncio.sleep(timeout_s)
            if _last_access_time is None:
                return
            # Don't cleanup if model is actively in use
            if _model_in_use:
                return
            idle_duration = time.time() - _last_access_time
            if idle_duration >= timeout_s and _model_cache is not None:
                logger.info(
                    f"Text generation model '{_current_model_name}' idle for {idle_duration:.1f}s "
                    f"(timeout: {timeout_s}s), unloading from GPU..."
                )
                cleanup()
        except asyncio.CancelledError:
            pass
        finally:
            global _idle_cleanup_task
            try:
                current = asyncio.current_task()
            except Exception:
                current = None
            if current is not None and _idle_cleanup_task is current:
                _idle_cleanup_task = None

    _idle_cleanup_task = loop.create_task(_watchdog(idle_timeout))


def get_model(model_name: str):
    """
    Get or load the text generation model.

    Args:
        model_name: Model identifier to load. If different from currently
                   loaded model, will reload with the new model.

    Returns:
        Tuple of (tokenizer, model, model_config)

    Raises:
        ValueError: If model is not supported
    """
    global _model_cache, _tokenizer_cache, _current_model_name, _current_model_config, _last_access_time

    # Check if current model should be cleaned up due to idle timeout
    check_and_cleanup_idle_model()

    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Check if we need to reload the model
    if model_name != _current_model_name:
        # Clear existing cache
        if _model_cache is not None:
            del _model_cache
            _model_cache = None
        if _tokenizer_cache is not None:
            del _tokenizer_cache
            _tokenizer_cache = None
        _current_model_name = model_name
        _current_model_config = model_config

    if _model_cache is None or _tokenizer_cache is None:
        # Load model directly by ID - HuggingFace automatically checks HF_HOME cache (data/models)
        model_path = _current_model_name
        logger.info(f"Loading model '{_current_model_name}' (HF_HOME cache lookup)")
        extra_kwargs = {}

        # Set HuggingFace endpoint for model loading
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        try:
            # Load tokenizer
            _tokenizer_cache = AutoTokenizer.from_pretrained(
                model_path, **extra_kwargs
            )

            # Common loading kwargs
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "dtype": torch.float16,  # Use fp16 for efficiency
                **extra_kwargs,
            }

            # Load model
            _model_cache = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                _model_cache = _model_cache.to("cuda")

        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    # Update last access time
    _last_access_time = time.time()

    return _tokenizer_cache, _model_cache, _current_model_config


def cleanup():
    """
    Release model and tokenizer resources immediately.
    Forces GPU memory cleanup to free resources for other services.
    """
    global _model_cache, _tokenizer_cache, _current_model_name, _current_model_config, _last_access_time

    model_name = _current_model_name  # Save for logging

    if _model_cache is not None:
        # Move model to CPU first (helps with cleanup)
        try:
            if hasattr(_model_cache, 'cpu'):
                _model_cache.cpu()
                logger.debug(f"Moved text generation model '{model_name}' to CPU")
        except Exception as e:
            logger.warning(f"Error moving model to CPU during cleanup: {e}")

        # Remove reference
        del _model_cache
        _model_cache = None

    if _tokenizer_cache is not None:
        del _tokenizer_cache
        _tokenizer_cache = None

    _current_model_name = ""
    _current_model_config = None
    _last_access_time = None

    # Force GPU memory release
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to finish
            logger.debug("GPU cache cleared and synchronized for text generation model")
    except Exception as e:
        logger.warning(f"Error releasing GPU memory: {e}")


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
        # Load model and get config off the event loop
        tokenizer, model, model_config = await asyncio.to_thread(get_model, request.model)

        # Mark model as in use BEFORE starting inference to prevent cleanup
        global _model_in_use, _last_access_time
        _model_in_use = True

        try:
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

            # Update access time after successful inference to prevent immediate cleanup
            _last_access_time = time.time()
        finally:
            # Always clear the in-use flag when done (success or error)
            _model_in_use = False

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

        # Schedule background idle cleanup after request completes
        schedule_idle_cleanup()
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
