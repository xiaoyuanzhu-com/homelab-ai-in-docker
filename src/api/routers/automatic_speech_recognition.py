"""Automatic speech recognition API router for transcribing audio files."""

import asyncio
import logging
import os
import time
import uuid
from typing import Dict, Any

import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.automatic_speech_recognition import (
    TranscriptionRequest,
    TranscriptionResponse,
    SpeakerSegment,
)
from ...storage.history import history_storage
from ...db.catalog import get_model_dict, list_models
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["automatic-speech-recognition"])

# Global WhisperLiveKit engine (shared across connections)
_live_transcription_engine = None
_live_engine_config = None  # Track current engine config (model, language, diarization)
_live_engine_lock = asyncio.Lock()


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
    Load available ASR models from the catalog.

    Returns:
        List of model IDs that can be used (both Whisper and pyannote)
    """
    asr_models = list_models(task="automatic-speech-recognition")
    return [m["id"] for m in asr_models]


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


@router.post("/automatic-speech-recognition", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest) -> TranscriptionResponse:
    """
    Perform automatic speech recognition on audio.

    Supports both transcription (Whisper models) and speaker diarization (pyannote models).

    Args:
        request: ASR request parameters

    Returns:
        Transcription or diarization results based on model type

    Raises:
        HTTPException: If processing fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Route to appropriate processing based on output_format
        if request.output_format == "diarization":
            return await _process_diarization(request, request_id, start_time)
        else:
            return await _process_transcription(request, request_id, start_time)

    except ValueError as e:
        error_msg = str(e)
        code = "INVALID_MODEL" if "Model" in error_msg or "model" in error_msg else "INVALID_AUDIO"
        logger.warning(f"{code} for request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail={"code": code, "message": error_msg, "request_id": request_id},
        )
    except Exception as e:
        error_msg = f"Failed to process audio: {str(e)}"
        logger.error(f"Processing failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "PROCESSING_FAILED", "message": error_msg, "request_id": request_id},
        )


async def _process_transcription(
    request: TranscriptionRequest, request_id: str, start_time: float
) -> TranscriptionResponse:
    """Process Whisper transcription via worker."""
    # Validate model
    validate_model(request.model)

    # Call worker via coordinator
    result = await coordinator_infer(
        task="asr",
        model_id=request.model,
        payload={
            "audio": request.audio,
            "language": request.language,
            "return_timestamps": request.return_timestamps,
        },
        request_id=request_id,
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    response = TranscriptionResponse(
        request_id=request_id,
        text=result.get("text", ""),
        model=request.model,
        language=result.get("language"),
        chunks=result.get("chunks"),
        processing_time_ms=processing_time_ms,
    )

    # Save to history
    truncated_audio = request.audio[:100] + "..." if len(request.audio) > 100 else request.audio
    history_storage.add_request(
        service="automatic-speech-recognition",
        request_id=request_id,
        request_data={
            "model": request.model,
            "language": request.language,
            "return_timestamps": request.return_timestamps,
            "audio": truncated_audio,
        },
        response_data=response.model_dump(),
        status="success",
    )

    return response


async def _process_diarization(
    request: TranscriptionRequest, request_id: str, start_time: float
) -> TranscriptionResponse:
    """Process pyannote speaker diarization via worker."""
    # Validate model
    validate_model(request.model)

    # Call worker via coordinator
    result = await coordinator_infer(
        task="speaker-diarization",
        model_id=request.model,
        payload={
            "audio": request.audio,
            "num_speakers": request.num_speakers,
            "min_speakers": request.min_speakers,
            "max_speakers": request.max_speakers,
        },
        request_id=request_id,
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Convert segments to SpeakerSegment objects
    segments = [
        SpeakerSegment(
            start=seg["start"],
            end=seg["end"],
            speaker=seg["speaker"],
        )
        for seg in result.get("segments", [])
    ]

    response = TranscriptionResponse(
        request_id=request_id,
        model=request.model,
        segments=segments,
        num_speakers=result.get("num_speakers", 0),
        processing_time_ms=processing_time_ms,
    )

    # Save to history
    truncated_audio = request.audio[:100] + "..." if len(request.audio) > 100 else request.audio
    history_storage.add_request(
        service="automatic-speech-recognition",
        request_id=request_id,
        request_data={
            "model": request.model,
            "min_speakers": request.min_speakers,
            "max_speakers": request.max_speakers,
            "num_speakers": request.num_speakers,
            "audio": truncated_audio,
        },
        response_data=response.model_dump(),
        status="success",
    )

    return response


# =============================================================================
# Live Transcription WebSocket Endpoint (WhisperLiveKit)
# Note: This remains in-process as WebSocket requires persistent connection
# =============================================================================


async def _get_live_engine(model: str = "large-v3", language: str = "en", diarization: bool = False):
    """
    Get or create the shared WhisperLiveKit transcription engine.

    The engine is shared across all WebSocket connections for efficiency.
    If parameters change, the engine is recreated with the new configuration.
    """
    global _live_transcription_engine, _live_engine_config

    requested_config = (model, language, diarization)

    async with _live_engine_lock:
        # Check if we need to recreate the engine due to config change
        if _live_transcription_engine is not None and _live_engine_config != requested_config:
            logger.info(
                f"Engine config changed from {_live_engine_config} to {requested_config}, recreating engine..."
            )
            # Clean up existing engine if possible
            try:
                if hasattr(_live_transcription_engine, "cleanup"):
                    await asyncio.to_thread(_live_transcription_engine.cleanup)
            except Exception as e:
                logger.warning(f"Error cleaning up old engine: {e}")
            _live_transcription_engine = None
            _live_engine_config = None

            # Force garbage collection to free GPU memory
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if _live_transcription_engine is None:
            try:
                from whisperlivekit import TranscriptionEngine

                # WhisperLiveKit expects "auto" string for auto-detection, not None
                effective_language = language if language else "auto"

                logger.info(
                    f"Initializing WhisperLiveKit TranscriptionEngine "
                    f"(model={model}, language={effective_language}, diarization={diarization})"
                )

                # Initialize engine in thread pool to avoid blocking
                def _create_engine():
                    return TranscriptionEngine(
                        model=model,
                        lan=effective_language,
                        transcription=True,
                        diarization=diarization,
                        buffer_trimming="sentence",
                    )

                _live_transcription_engine = await asyncio.to_thread(_create_engine)
                _live_engine_config = requested_config
                logger.info("WhisperLiveKit TranscriptionEngine initialized successfully")

            except ImportError as e:
                logger.error(f"WhisperLiveKit not installed: {e}")
                raise RuntimeError(
                    "WhisperLiveKit is not installed. Install with: pip install whisperlivekit"
                )
            except Exception as e:
                logger.error(f"Failed to initialize WhisperLiveKit engine: {e}", exc_info=True)
                raise

        return _live_transcription_engine


async def _handle_websocket_results(websocket: WebSocket, results_generator):
    """
    Consume results from the audio processor and send them via WebSocket.
    """
    try:
        async for response in results_generator:
            try:
                # FrontData from WhisperLiveKit has a to_dict() method
                if hasattr(response, "to_dict") and callable(response.to_dict):
                    data = response.to_dict()
                elif hasattr(response, "model_dump") and callable(response.model_dump):
                    data = response.model_dump()
                elif isinstance(response, dict):
                    data = response
                else:
                    # Try to convert dataclass or similar to dict
                    import dataclasses

                    if dataclasses.is_dataclass(response):
                        data = dataclasses.asdict(response)
                    else:
                        # Last resort: use vars or repr
                        data = {"raw": str(response)}

                await websocket.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to serialize response: {e}, type={type(response)}")
                await websocket.send_json({"error": f"Serialization error: {str(e)}"})

        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@router.websocket("/automatic-speech-recognition/live")
async def live_transcription_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live transcription using WhisperLiveKit.

    Protocol:
    1. Client connects to ws://host/api/automatic-speech-recognition/live
    2. Server sends config: {"type": "config", "useAudioWorklet": false}
    3. Client streams audio bytes (WebM from MediaRecorder or PCM from AudioWorklet)
    4. Server sends transcription results as JSON with "lines" and "buffer" fields
    5. When client stops, server sends {"type": "ready_to_stop"}

    Query parameters:
    - model: Whisper model size (default: "large-v3")
    - language: Source language code (default: "en", use "auto" for detection)
    - diarization: Enable speaker diarization (default: "false")
    """
    from whisperlivekit import AudioProcessor

    # Extract query parameters
    model = websocket.query_params.get("model", "large-v3")
    language = websocket.query_params.get("language", "en")
    diarization = websocket.query_params.get("diarization", "false").lower() == "true"

    logger.info(
        f"WebSocket connection request: model={model}, language={language}, diarization={diarization}"
    )

    # Get shared transcription engine
    try:
        transcription_engine = await _get_live_engine(
            model=model, language=language, diarization=diarization
        )
        logger.info(f"Using engine with config: {_live_engine_config}")
    except Exception as e:
        logger.error(f"Failed to get transcription engine: {e}")
        await websocket.close(code=1011, reason=str(e))
        return

    # Create per-connection audio processor
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)

    await websocket.accept()
    logger.info(f"Live transcription WebSocket connection opened (model={model}, language={language})")

    # Send config to client (we use MediaRecorder, not AudioWorklet by default)
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": False})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    # Start result generator
    results_generator = await audio_processor.create_tasks()

    # Start task to handle results
    websocket_task = asyncio.create_task(_handle_websocket_results(websocket, results_generator))

    try:
        while True:
            # Receive audio bytes from client
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)

    except WebSocketDisconnect:
        logger.info("Live transcription WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"Unexpected error in live transcription: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up live transcription WebSocket...")

        # Cancel the results task if still running
        if not websocket_task.done():
            websocket_task.cancel()
            try:
                await websocket_task
            except asyncio.CancelledError:
                pass

        # Cleanup audio processor
        try:
            await audio_processor.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up audio processor: {e}")

        logger.info("Live transcription WebSocket cleaned up")
