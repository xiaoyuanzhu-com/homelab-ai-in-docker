"""Automatic speech recognition API router for transcribing audio files."""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.automatic_speech_recognition import (
    TranscriptionRequest,
    TranscriptionResponse,
    SpeakerSegment,
)
from ...storage.history import history_storage
from ...db.catalog import get_model_dict, list_models
from ...worker import coordinator_infer
from ...worker.coordinator import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["automatic-speech-recognition"])


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

    # Get python_env for worker isolation
    model_config = get_model_config(request.model)
    python_env = model_config.get("python_env")

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
        python_env=python_env,
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

    # Get python_env for worker isolation
    model_config = get_model_config(request.model)
    python_env = model_config.get("python_env")

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
        python_env=python_env,
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
# Live Transcription WebSocket Endpoint
# Proxies to ASR streaming worker (keeps main process lean)
# =============================================================================


@router.websocket("/automatic-speech-recognition/live")
async def live_transcription_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live transcription.

    This endpoint proxies to an ASR streaming worker subprocess.

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
    import websockets

    # Extract query parameters
    model = websocket.query_params.get("model", "large-v3")
    language = websocket.query_params.get("language", "en")
    diarization = websocket.query_params.get("diarization", "false").lower() == "true"

    logger.info(
        f"Live transcription request: model={model}, language={language}, diarization={diarization}"
    )

    # Get or spawn the streaming worker
    coordinator = get_coordinator()
    try:
        worker_url = await coordinator.get_or_spawn_worker(
            task="asr-streaming",
            model_id=model,
            python_env="whisper",
            extra_args={
                "language": language,
                "diarization": str(diarization).lower(),
            },
        )
        logger.info(f"Using streaming worker at {worker_url}")
    except Exception as e:
        logger.error(f"Failed to spawn streaming worker: {e}")
        await websocket.close(code=1011, reason=str(e))
        return

    # Accept client connection
    await websocket.accept()

    # Connect to worker's WebSocket
    worker_ws_url = worker_url.replace("http://", "ws://") + "/stream"
    logger.info(f"Connecting to worker WebSocket at {worker_ws_url}")

    try:
        async with websockets.connect(worker_ws_url) as worker_ws:
            # Proxy messages bidirectionally
            async def forward_client_to_worker():
                """Forward messages from client to worker."""
                try:
                    while True:
                        # Try to receive either bytes or text
                        message = await websocket.receive()
                        if "bytes" in message:
                            await worker_ws.send(message["bytes"])
                        elif "text" in message:
                            await worker_ws.send(message["text"])
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.debug(f"Client->worker forward ended: {e}")

            async def forward_worker_to_client():
                """Forward messages from worker to client."""
                try:
                    async for message in worker_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception as e:
                    logger.debug(f"Worker->client forward ended: {e}")

            # Run both forwarding tasks
            await asyncio.gather(
                forward_client_to_worker(),
                forward_worker_to_client(),
                return_exceptions=True,
            )

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Worker WebSocket closed: {e}")
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}", exc_info=True)
    finally:
        logger.info("Live transcription WebSocket proxy closed")
