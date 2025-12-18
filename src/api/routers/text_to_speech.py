"""Text-to-speech API router using CosyVoice."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..models.text_to_speech import (
    TextToSpeechRequest,
    TextToSpeechResponse,
)
from ...storage.history import history_storage
from ...db.catalog import get_model_dict, list_models
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-speech"])


# =============================================================================
# Helper Functions
# =============================================================================


def get_model_config(model_id: str) -> Dict[str, Any]:
    """Get model configuration from catalog."""
    model = get_model_dict(model_id)
    if model is None:
        raise ValueError(f"Model '{model_id}' not found in catalog")
    return model


def get_available_models() -> list[str]:
    """Load available TTS models from the catalog."""
    tts_models = list_models(task="text-to-speech")
    return [m["id"] for m in tts_models]


def validate_model(model_name: str) -> None:
    """Validate that the model is supported in the catalog."""
    available = get_available_models()
    if model_name not in available:
        # Allow short names and non-catalog models for flexibility
        logger.warning(
            f"Model '{model_name}' not in catalog. "
            f"Available models: {', '.join(available)}"
        )


# =============================================================================
# Main API Endpoint
# =============================================================================


@router.post("/text-to-speech", response_model=TextToSpeechResponse)
async def synthesize_speech(request: TextToSpeechRequest) -> TextToSpeechResponse:
    """
    Synthesize speech from text using CosyVoice.

    Supports multiple synthesis modes:
    - **zero_shot**: Voice cloning from a reference audio sample
    - **cross_lingual**: Cross-lingual synthesis (speak text in one language with voice from another)
    - **instruct**: Natural language instruction for speech style control
    - **sft**: Use pre-trained speaker voices

    ## Example Usage

    ### Zero-shot voice cloning
    ```json
    {
        "text": "Hello, this is a test.",
        "mode": "zero_shot",
        "prompt_text": "This is my voice sample.",
        "prompt_audio": "data:audio/wav;base64,..."
    }
    ```

    ### Instruction-based synthesis
    ```json
    {
        "text": "Today is a wonderful day!",
        "mode": "instruct",
        "instruction": "Speak happily with enthusiasm"
    }
    ```
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"TTS request: model={request.model}, mode={request.mode}, "
        f"text_length={len(request.text)}"
    )

    try:
        # Validate mode requirements
        if request.mode == "zero_shot" and not request.prompt_audio:
            raise ValueError("prompt_audio is required for zero_shot mode")
        if request.mode == "cross_lingual" and not request.prompt_audio:
            raise ValueError("prompt_audio is required for cross_lingual mode")
        if request.mode == "instruct" and not request.instruction:
            raise ValueError("instruction is required for instruct mode")

        # Get model config for python_env
        try:
            model_config = get_model_config(request.model)
            python_env = model_config.get("python_env", "cosyvoice")
        except ValueError:
            # Model not in catalog, use default environment
            python_env = "cosyvoice"
            logger.info(f"Using default python_env=cosyvoice for model {request.model}")

        # Call worker via coordinator
        result = await coordinator_infer(
            task="tts",
            model_id=request.model,
            payload={
                "text": request.text,
                "mode": request.mode,
                "prompt_text": request.prompt_text,
                "prompt_audio": request.prompt_audio,
                "instruction": request.instruction,
                "speaker_id": request.speaker_id,
                "speed": request.speed,
            },
            request_id=request_id,
            python_env=python_env,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = TextToSpeechResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            audio=result.get("audio", ""),
            sample_rate=result.get("sample_rate", 22050),
            duration_seconds=result.get("duration_seconds", 0.0),
            model=request.model,
            mode=result.get("mode", request.mode),
        )

        # Log to history (truncate audio for storage)
        truncated_audio = (
            request.prompt_audio[:100] + "..."
            if request.prompt_audio and len(request.prompt_audio) > 100
            else request.prompt_audio
        )
        history_storage.add_request(
            service="text-to-speech",
            request_id=request_id,
            request_data={
                "model": request.model,
                "mode": request.mode,
                "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
                "prompt_audio": truncated_audio,
            },
            response_data={
                "sample_rate": response.sample_rate,
                "duration_seconds": response.duration_seconds,
                "audio_length": len(response.audio),
            },
            status="success",
        )

        return response

    except ValueError as e:
        error_msg = str(e)
        code = "INVALID_REQUEST"
        logger.warning(f"{code} for request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail={"code": code, "message": error_msg, "request_id": request_id},
        )
    except Exception as e:
        error_msg = f"Failed to synthesize speech: {str(e)}"
        logger.error(f"TTS failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "SYNTHESIS_FAILED", "message": error_msg, "request_id": request_id},
        )
