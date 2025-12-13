"""FunASR-based transcription API router."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.funasr import (
    FunASRTranscriptionRequest,
    FunASRTranscriptionResponse,
)
from ...storage.history import history_storage
from ...worker import coordinator_infer
from ...worker.coordinator import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/funasr", tags=["funasr"])

# Default streaming model
DEFAULT_STREAMING_MODEL = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


def _parse_sensevoice_output(text: str) -> tuple[str, str | None, str | None, str | None]:
    """
    Parse SenseVoice output which may contain tags.

    SenseVoice outputs: <|LANG|><|EMOTION|><|EVENT|>text
    Example: <|zh|><|HAPPY|><|BGM|>你好世界

    Returns:
        tuple: (clean_text, language, emotion, event)
    """
    # Pattern to match SenseVoice tags
    pattern = r"<\|([^|]+)\|>"
    matches = re.findall(pattern, text)

    # Remove all tags to get clean text
    clean_text = re.sub(pattern, "", text).strip()

    # Parse tags - SenseVoice uses specific tag positions
    # First tag is usually language, second is emotion, third is event
    language = None
    emotion = None
    event = None

    # Known language codes
    lang_codes = {"zh", "en", "ja", "ko", "yue", "auto"}
    # Known emotions
    emotions = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
    # Known events
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


@router.post("/transcribe", response_model=FunASRTranscriptionResponse)
async def transcribe(request: FunASRTranscriptionRequest) -> FunASRTranscriptionResponse:
    """
    Transcribe audio using FunASR models.

    Supports:
    - SenseVoiceSmall: Multi-language ASR with emotion/event detection
    - Paraformer: High-accuracy Mandarin ASR with VAD and punctuation
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        logger.info(
            f"FunASR transcription request: model={request.model}, "
            f"language={request.language}"
        )

        # Call worker via coordinator
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

        # Get raw text from worker
        raw_text = result.get("text", "")

        # Parse SenseVoice output for emotion/event tags
        clean_text, detected_lang, emotion, event = _parse_sensevoice_output(raw_text)

        # Use detected language or requested language
        language = detected_lang or request.language

        response = FunASRTranscriptionResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            text=raw_text,
            text_clean=clean_text if clean_text != raw_text else None,
            language=language,
            model=request.model,
            emotion=emotion,
            event=event,
        )

        # Save to history (exclude audio)
        history_storage.add_request(
            service="funasr",
            request_id=request_id,
            request_data={
                "model": request.model,
                "language": request.language,
            },
            response_data=response.model_dump(),
            status="success",
        )

        return response

    except Exception as e:
        logger.error(f"FunASR processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe: {e}")


# =============================================================================
# Live Transcription WebSocket Endpoint
# Proxies to FunASR streaming worker
# =============================================================================


@router.websocket("/live")
async def live_transcription_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live transcription using FunASR.

    This endpoint proxies to a FunASR streaming worker subprocess.

    Protocol:
    1. Client connects to ws://host/api/funasr/live
    2. Server sends config: {"type": "config", "sampleRate": 16000, ...}
    3. Client streams audio bytes (WebM from MediaRecorder or PCM)
    4. Server sends transcription results as JSON
    5. When client stops, server sends {"type": "ready_to_stop"}

    Query parameters:
    - model: FunASR model (default: paraformer-zh-streaming)
    - language: Source language code (default: "zh")
    """
    import websockets

    # Extract query parameters
    model = websocket.query_params.get("model", DEFAULT_STREAMING_MODEL)
    language = websocket.query_params.get("language", "zh")

    logger.info(f"FunASR live transcription request: model={model}, language={language}")

    # Get or spawn the streaming worker
    coordinator = get_coordinator()
    try:
        worker_url = await coordinator.get_or_spawn_worker(
            task="funasr-streaming",
            model_id=model,
            python_env="funasr",
            extra_args={"language": language},
        )
        logger.info(f"Using FunASR streaming worker at {worker_url}")
    except Exception as e:
        logger.error(f"Failed to spawn FunASR streaming worker: {e}")
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
        logger.info("FunASR live transcription WebSocket proxy closed")
