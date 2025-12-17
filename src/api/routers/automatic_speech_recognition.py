"""Unified automatic speech recognition API router.

Supports multiple ASR backends via the `lib` parameter:
- whisper: Basic Whisper transcription via transformers pipeline
- whisperx: WhisperX with word-level alignment and speaker diarization
- funasr: Alibaba FunASR with emotion/event detection

Live streaming is available via WebSocket with `lib` parameter:
- whisperlivekit: Real-time streaming (whisper/whisperx backend)
- funasr: Real-time streaming (FunASR backend)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.automatic_speech_recognition import (
    TranscriptionRequest,
    TranscriptionResponse,
    Segment,
    Word,
    Speaker,
)
from ...storage.history import history_storage
from ...db.catalog import get_model_dict, list_models
from ...worker import coordinator_infer
from ...worker.coordinator import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["automatic-speech-recognition"])


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
    """Load available ASR models from the catalog."""
    asr_models = list_models(task="automatic-speech-recognition")
    return [m["id"] for m in asr_models]


def validate_model(model_name: str) -> None:
    """Validate that the model is supported in the catalog."""
    available = get_available_models()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {', '.join(available)}"
        )


def infer_lib_from_model(model_id: str) -> str:
    """Infer the library to use based on model ID patterns."""
    model_lower = model_id.lower()

    # FunASR models
    if "funasr" in model_lower or "sensevoice" in model_lower or "paraformer" in model_lower:
        return "funasr"

    # WhisperX short model names
    if model_id in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]:
        return "whisperx"
    if model_id.endswith(".en"):  # e.g., "small.en"
        return "whisperx"

    # Default to whisper for catalog models
    return "whisper"


# =============================================================================
# FunASR Helpers
# =============================================================================


def _parse_sensevoice_output(text: str) -> tuple[str, str | None, str | None, str | None]:
    """Parse SenseVoice output which may contain tags.

    SenseVoice outputs: <|LANG|><|EMOTION|><|EVENT|>text
    Example: <|zh|><|HAPPY|><|BGM|>你好世界

    Returns:
        tuple: (clean_text, language, emotion, event)
    """
    pattern = r"<\|([^|]+)\|>"
    matches = re.findall(pattern, text)
    clean_text = re.sub(pattern, "", text).strip()

    language = None
    emotion = None
    event = None

    lang_codes = {"zh", "en", "ja", "ko", "yue", "auto"}
    emotions = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
    events = {"BGM", "Applause", "Laughter", "Speech", "Silence"}

    for tag in matches:
        tag_upper = tag.upper()
        tag_lower = tag.lower()
        if tag_lower in lang_codes:
            language = tag_lower
        elif tag_upper in emotions:
            emotion = tag_upper
        elif tag in events or tag_upper in {e.upper() for e in events}:
            event = tag

    return clean_text, language, emotion, event


# =============================================================================
# Main API Endpoint
# =============================================================================


@router.post("/automatic-speech-recognition", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest) -> TranscriptionResponse:
    """
    Unified automatic speech recognition endpoint.

    Supports multiple backends via the `lib` parameter:
    - whisper: Basic Whisper transcription via transformers pipeline
    - whisperx: WhisperX with word-level alignment and speaker diarization
    - funasr: Alibaba FunASR with emotion/event detection

    The library is auto-detected from the model if not specified.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Determine which library to use
    lib = request.lib or infer_lib_from_model(request.model)
    logger.info(f"ASR request: model={request.model}, lib={lib}, diarization={request.diarization}")

    try:
        if lib == "whisperx":
            return await _process_whisperx(request, request_id, start_time)
        elif lib == "funasr":
            return await _process_funasr(request, request_id, start_time)
        else:
            # Default whisper processing
            if request.diarization:
                return await _process_whisper_diarization(request, request_id, start_time)
            else:
                return await _process_whisper_transcription(request, request_id, start_time)

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


# =============================================================================
# Whisper Processing (Basic)
# =============================================================================


async def _process_whisper_transcription(
    request: TranscriptionRequest, request_id: str, start_time: float
) -> TranscriptionResponse:
    """Process basic Whisper transcription via worker."""
    validate_model(request.model)
    model_config = get_model_config(request.model)
    python_env = model_config.get("python_env")

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
        lib="whisper",
        language=result.get("language"),
        chunks=result.get("chunks"),
        processing_time_ms=processing_time_ms,
    )

    truncated_audio = request.audio[:100] + "..." if len(request.audio) > 100 else request.audio
    history_storage.add_request(
        service="automatic-speech-recognition",
        request_id=request_id,
        request_data={
            "model": request.model,
            "lib": "whisper",
            "language": request.language,
            "audio": truncated_audio,
        },
        response_data=response.model_dump(),
        status="success",
    )

    return response


async def _process_whisper_diarization(
    request: TranscriptionRequest, request_id: str, start_time: float
) -> TranscriptionResponse:
    """Process pyannote speaker diarization via worker."""
    # For diarization, we need a pyannote model
    diar_model = "pyannote/speaker-diarization-3.1"

    try:
        model_config = get_model_config(diar_model)
    except ValueError:
        raise ValueError(
            f"Diarization model '{diar_model}' not found. "
            "For speaker diarization with whisper, use lib='whisperx' instead."
        )

    python_env = model_config.get("python_env")

    result = await coordinator_infer(
        task="speaker-diarization",
        model_id=diar_model,
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

    segments = [
        Segment(
            start=seg["start"],
            end=seg["end"],
            text="",  # Diarization-only doesn't have text
            speaker=seg["speaker"],
        )
        for seg in result.get("segments", [])
    ]

    response = TranscriptionResponse(
        request_id=request_id,
        model=diar_model,
        lib="whisper",
        segments=segments,
        num_speakers=result.get("num_speakers", 0),
        processing_time_ms=processing_time_ms,
    )

    truncated_audio = request.audio[:100] + "..." if len(request.audio) > 100 else request.audio
    history_storage.add_request(
        service="automatic-speech-recognition",
        request_id=request_id,
        request_data={
            "model": diar_model,
            "lib": "whisper",
            "diarization": True,
            "audio": truncated_audio,
        },
        response_data=response.model_dump(),
        status="success",
    )

    return response


# =============================================================================
# WhisperX Processing
# =============================================================================


async def _process_whisperx(
    request: TranscriptionRequest, request_id: str, start_time: float
) -> TranscriptionResponse:
    """Process transcription using WhisperX worker with alignment and optional diarization."""
    logger.info(
        f"WhisperX transcription: model={request.model}, batch_size={request.batch_size}, "
        f"language={request.language}, diarization={request.diarization}"
    )

    # Call the whisperx worker
    result = await coordinator_infer(
        task="whisperx",
        model_id=request.model,
        payload={
            "audio": request.audio,
            "language": request.language,
            "diarization": request.diarization,
            "batch_size": request.batch_size,
            "min_speakers": request.min_speakers,
            "max_speakers": request.max_speakers,
        },
        request_id=request_id,
        python_env="whisper",
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Convert worker result to response models
    segments_out = None
    if result.get("segments"):
        segments_out = [
            Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker"),
                words=[
                    Word(
                        word=w["word"],
                        start=w.get("start"),
                        end=w.get("end"),
                        speaker=w.get("speaker"),
                    )
                    for w in seg.get("words", [])
                ] if seg.get("words") else None,
            )
            for seg in result["segments"]
        ]

    speakers_out = None
    if result.get("speakers"):
        speakers_out = [
            Speaker(
                speaker_id=spk["speaker_id"],
                embedding=spk.get("embedding"),
                total_duration=spk.get("total_duration"),
                segment_count=spk.get("segment_count"),
            )
            for spk in result["speakers"]
        ]

    response = TranscriptionResponse(
        request_id=request_id,
        processing_time_ms=processing_time_ms,
        text=result.get("text", ""),
        language=result.get("language"),
        model=request.model,
        lib="whisperx",
        segments=segments_out,
        speakers=speakers_out,
        num_speakers=result.get("num_speakers"),
    )

    truncated_audio = request.audio[:100] + "..." if len(request.audio) > 100 else request.audio
    history_storage.add_request(
        service="automatic-speech-recognition",
        request_id=request_id,
        request_data={
            "model": request.model,
            "lib": "whisperx",
            "language": request.language,
            "diarization": request.diarization,
            "audio": truncated_audio,
        },
        response_data=response.model_dump(),
        status="success",
    )

    return response


# =============================================================================
# FunASR Processing
# =============================================================================


async def _process_funasr(
    request: TranscriptionRequest, request_id: str, start_time: float
) -> TranscriptionResponse:
    """Process transcription using FunASR."""
    logger.info(f"FunASR transcription: model={request.model}, language={request.language}")

    result = await coordinator_infer(
        task="funasr",
        model_id=request.model,
        payload={
            "audio": request.audio,
            "language": request.language,
        },
        request_id=request_id,
        python_env="funasr",
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    raw_text = result.get("text", "")
    clean_text, detected_lang, emotion, event = _parse_sensevoice_output(raw_text)
    language = detected_lang or request.language

    response = TranscriptionResponse(
        request_id=request_id,
        processing_time_ms=processing_time_ms,
        text=raw_text,
        text_clean=clean_text if clean_text != raw_text else None,
        language=language,
        model=request.model,
        lib="funasr",
        emotion=emotion,
        event=event,
    )

    history_storage.add_request(
        service="automatic-speech-recognition",
        request_id=request_id,
        request_data={
            "model": request.model,
            "lib": "funasr",
            "language": request.language,
        },
        response_data=response.model_dump(),
        status="success",
    )

    return response


# =============================================================================
# Live Transcription WebSocket Endpoint
# =============================================================================


@router.websocket("/automatic-speech-recognition/live")
async def live_transcription_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live transcription.

    Supports multiple backends via the `lib` query parameter:
    - whisperlivekit (default): Real-time streaming with Whisper
    - funasr: Real-time streaming with FunASR

    Query parameters:
    - lib: Backend library (whisperlivekit, funasr)
    - model: Model identifier
    - language: Source language code (default: "en" for whisper, "zh" for funasr)
    - diarization: Enable speaker diarization (whisperlivekit only, "true"/"false")
    """
    import websockets

    # Extract query parameters
    lib = websocket.query_params.get("lib", "whisperlivekit")
    model = websocket.query_params.get("model")
    language = websocket.query_params.get("language")
    diarization = websocket.query_params.get("diarization", "false").lower() == "true"

    # Set defaults based on lib
    if lib == "funasr":
        model = model or "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        language = language or "zh"

        # Fun-ASR-Nano is an LLM-based model that doesn't support streaming
        # It requires complete audio input, not chunk-based processing
        if "Fun-ASR-Nano" in (model or ""):
            logger.warning(f"Fun-ASR-Nano does not support live streaming: {model}")
            await websocket.close(
                code=1008,
                reason="Fun-ASR-Nano does not support live streaming. Use paraformer-zh-streaming or use the batch API instead."
            )
            return

        task = "funasr-streaming"
        python_env = "funasr"
        extra_args = {"language": language}
    else:
        # whisperlivekit
        model = model or "large-v3"
        language = language or "en"
        task = "asr-streaming"
        python_env = "whisper"
        extra_args = {"language": language, "diarization": str(diarization).lower()}

    logger.info(f"Live transcription: lib={lib}, model={model}, language={language}, diarization={diarization}")

    # Get or spawn the streaming worker
    coordinator = get_coordinator()
    try:
        worker_url = await coordinator.get_or_spawn_worker(
            task=task,
            model_id=model,
            python_env=python_env,
            extra_args=extra_args,
        )
        logger.info(f"Using streaming worker at {worker_url}")
    except Exception as e:
        logger.error(f"Failed to spawn streaming worker: {e}")
        await websocket.close(code=1011, reason=str(e))
        return

    await websocket.accept()

    worker_ws_url = worker_url.replace("http://", "ws://") + "/stream"
    logger.info(f"Connecting to worker WebSocket at {worker_ws_url}")

    try:
        async with websockets.connect(worker_ws_url) as worker_ws:
            async def forward_client_to_worker():
                try:
                    while True:
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
                try:
                    async for message in worker_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception as e:
                    logger.debug(f"Worker->client forward ended: {e}")

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
