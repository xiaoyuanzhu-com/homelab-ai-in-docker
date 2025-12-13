"""FunASR streaming worker with WebSocket support for live transcription."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger("funasr_streaming_worker")


class WorkerHealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="'ok' or 'error'")
    model: str = Field(..., description="Model ID being served")
    ready: bool = Field(..., description="True when ready for inference")
    task: str = Field(..., description="Task type")
    streaming: bool = Field(default=True, description="Supports streaming")


class WorkerInfoResponse(BaseModel):
    """Model and memory info response."""

    model: str
    task: str
    memory_mb: float
    load_time_ms: int
    active_streams: int


class FunASRStreamingWorker:
    """
    FunASR worker with WebSocket streaming support.

    Uses FunASR's streaming models (paraformer-zh-streaming) for real-time transcription.
    """

    task_name = "funasr-streaming"

    # Streaming configuration
    # chunk_size[1] * 60ms = display latency
    # chunk_size[2] * 60ms = lookahead
    CHUNK_SIZE = [0, 10, 5]  # 600ms latency, 300ms lookahead
    ENCODER_CHUNK_LOOK_BACK = 4
    DECODER_CHUNK_LOOK_BACK = 1

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        language: str = "zh",
    ):
        self.model_id = model_id
        self.port = port
        self.idle_timeout = idle_timeout
        self.language = language

        self._model = None
        self._load_time_ms: int = 0
        self._ready = False
        self._last_active = time.time()
        self._idle_task: Optional[asyncio.Task] = None
        self._active_streams: int = 0

        # Audio parameters
        self._sample_rate = 16000
        self._chunk_stride = self.CHUNK_SIZE[1] * 960  # samples per chunk

        # Create FastAPI app
        self.app = FastAPI(title=f"{self.task_name} worker")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register HTTP and WebSocket endpoints."""

        @self.app.get("/healthz", response_model=WorkerHealthResponse)
        async def healthz() -> WorkerHealthResponse:
            return WorkerHealthResponse(
                status="ok" if self._ready else "starting",
                model=self.model_id,
                ready=self._ready,
                task=self.task_name,
                streaming=True,
            )

        @self.app.get("/info", response_model=WorkerInfoResponse)
        async def info() -> WorkerInfoResponse:
            from ..utils import get_gpu_memory_mb

            return WorkerInfoResponse(
                model=self.model_id,
                task=self.task_name,
                memory_mb=get_gpu_memory_mb(),
                load_time_ms=self._load_time_ms,
                active_streams=self._active_streams,
            )

        @self.app.post("/shutdown")
        async def shutdown_endpoint():
            asyncio.create_task(self._shutdown())
            return {"status": "shutting_down"}

        @self.app.websocket("/stream")
        async def stream_endpoint(websocket: WebSocket):
            await self._handle_stream(websocket)

    async def _handle_stream(self, websocket: WebSocket) -> None:
        """Handle a streaming WebSocket connection."""
        import numpy as np

        await websocket.accept()
        self._active_streams += 1
        self._last_active = time.time()
        self._cancel_idle_shutdown()

        logger.info(
            f"Stream started (model={self.model_id}, language={self.language}, "
            f"active_streams={self._active_streams})"
        )

        # Send config to client
        try:
            await websocket.send_json({
                "type": "config",
                "sampleRate": self._sample_rate,
                "chunkMs": self.CHUNK_SIZE[1] * 60,
            })
        except Exception as e:
            logger.warning(f"Failed to send config: {e}")

        # Per-connection state for streaming
        cache = {}
        audio_buffer = np.array([], dtype=np.float32)
        total_text = ""

        try:
            while True:
                # Receive audio data (WebM or raw PCM)
                message = await websocket.receive_bytes()

                # Convert to float32 samples
                # WebM from browser needs decoding, raw PCM can be used directly
                try:
                    audio_chunk = await self._decode_audio_chunk(message)
                except Exception as e:
                    logger.warning(f"Failed to decode audio: {e}")
                    continue

                # Append to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                # Process complete chunks
                while len(audio_buffer) >= self._chunk_stride:
                    chunk = audio_buffer[: self._chunk_stride]
                    audio_buffer = audio_buffer[self._chunk_stride :]

                    # Run inference
                    result = await asyncio.to_thread(
                        self._model.generate,
                        input=chunk,
                        cache=cache,
                        is_final=False,
                        chunk_size=self.CHUNK_SIZE,
                        encoder_chunk_look_back=self.ENCODER_CHUNK_LOOK_BACK,
                        decoder_chunk_look_back=self.DECODER_CHUNK_LOOK_BACK,
                    )

                    # Extract text from result
                    if result and len(result) > 0:
                        text = self._extract_text(result[0])
                        if text:
                            total_text += text
                            await websocket.send_json({
                                "type": "partial",
                                "text": text,
                                "buffer_transcription": total_text,
                            })

        except WebSocketDisconnect:
            logger.info("Stream disconnected by client")
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
        finally:
            # Process final chunk with is_final=True
            try:
                if len(audio_buffer) > 0:
                    result = await asyncio.to_thread(
                        self._model.generate,
                        input=audio_buffer,
                        cache=cache,
                        is_final=True,
                        chunk_size=self.CHUNK_SIZE,
                        encoder_chunk_look_back=self.ENCODER_CHUNK_LOOK_BACK,
                        decoder_chunk_look_back=self.DECODER_CHUNK_LOOK_BACK,
                    )
                    if result and len(result) > 0:
                        text = self._extract_text(result[0])
                        if text:
                            total_text += text

                # Send final result
                await websocket.send_json({
                    "type": "final",
                    "text": total_text,
                    "lines": [{"text": total_text, "speaker": 0, "start": "0", "end": ""}],
                })
                await websocket.send_json({"type": "ready_to_stop"})
            except Exception as e:
                logger.warning(f"Error sending final result: {e}")

            self._active_streams -= 1
            self._last_active = time.time()

            # Schedule idle shutdown if no active streams
            if self._active_streams == 0:
                self._schedule_idle_shutdown()

            logger.info(f"Stream ended (active_streams={self._active_streams})")

    async def _decode_audio_chunk(self, data: bytes) -> "np.ndarray":
        """Decode audio chunk from raw PCM int16 to float32 samples.

        The UI sends raw 16-bit PCM at 16kHz via AudioWorklet.
        Each chunk is 1600 samples (100ms) = 3200 bytes.
        """
        import numpy as np

        # Validate buffer size (must be multiple of 2 for int16)
        if len(data) % 2 != 0:
            raise ValueError(f"Buffer size {len(data)} is not a multiple of 2 (int16)")

        # Convert int16 PCM to float32 [-1, 1]
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        return samples

    def _extract_text(self, result: Any) -> str:
        """Extract text from FunASR result."""
        if isinstance(result, dict):
            return result.get("text", "")
        elif isinstance(result, str):
            return result
        elif hasattr(result, "text"):
            return result.text
        return str(result) if result else ""

    def _cancel_idle_shutdown(self) -> None:
        """Cancel pending idle shutdown."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None

    def _schedule_idle_shutdown(self) -> None:
        """Schedule shutdown after idle timeout."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._cancel_idle_shutdown()

        async def _watchdog():
            try:
                await asyncio.sleep(self.idle_timeout)
                if self._active_streams == 0:
                    idle = time.time() - self._last_active
                    if idle >= self.idle_timeout:
                        logger.info(f"Worker idle for {idle:.1f}s, shutting down")
                        await self._shutdown()
            except asyncio.CancelledError:
                pass

        self._idle_task = loop.create_task(_watchdog())

    async def _shutdown(self) -> None:
        """Cleanup and exit."""
        try:
            self.cleanup()
        finally:
            os._exit(0)

    def load_model(self) -> Any:
        """Load FunASR streaming model."""
        from funasr import AutoModel

        from src.config import get_hf_endpoint

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        logger.info(f"Loading FunASR streaming model: {self.model_id}")

        model = AutoModel(
            model=self.model_id,
            device="cuda:0",
            disable_update=True,
        )

        return model

    def cleanup(self) -> None:
        """Release resources."""
        if self._model is not None:
            del self._model
            self._model = None

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def run(self) -> int:
        """Start the worker server."""
        import signal

        import uvicorn

        # Load model
        logger.info(f"Loading model {self.model_id}...")
        start = time.time()
        try:
            self._model = self.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return 1

        self._load_time_ms = int((time.time() - start) * 1000)
        self._ready = True
        self._last_active = time.time()

        from ..utils import get_gpu_memory_mb

        logger.info(
            f"Model loaded in {self._load_time_ms}ms, "
            f"GPU memory: {get_gpu_memory_mb():.1f}MB"
        )

        # Setup signal handler
        def _term_handler(signum, frame):
            try:
                asyncio.get_event_loop().create_task(self._shutdown())
            except Exception:
                os._exit(0)

        try:
            signal.signal(signal.SIGTERM, _term_handler)
        except Exception:
            pass

        # Start server
        logger.info(f"Worker started for {self.task_name}:{self.model_id} on port {self.port}")
        uvicorn.run(self.app, host="127.0.0.1", port=self.port, log_level="info")
        return 0


def main(argv: list[str]) -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="FunASR streaming worker")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--idle-timeout", type=int, default=60, help="Idle seconds before exit")
    parser.add_argument("--language", default="zh", help="Language code")
    args = parser.parse_args(argv)

    worker = FunASRStreamingWorker(
        model_id=args.model_id,
        port=args.port,
        idle_timeout=args.idle_timeout,
        language=args.language,
    )
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
