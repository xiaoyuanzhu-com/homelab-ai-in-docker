"""ASR streaming worker with WebSocket support for live transcription."""

from __future__ import annotations

# Setup CUDA libraries BEFORE importing torch/ML libs
from src.runtime.cuda_paths import setup_cuda_libraries

setup_cuda_libraries()

import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger("asr_streaming_worker")


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


class ASRStreamingWorker:
    """
    ASR worker with WebSocket streaming support.

    Uses WhisperLiveKit for real-time transcription.
    """

    task_name = "asr-streaming"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        language: str = "en",
        diarization: bool = False,
    ):
        self.model_id = model_id
        self.port = port
        self.idle_timeout = idle_timeout
        self.language = language
        self.diarization = diarization

        self._engine = None
        self._load_time_ms: int = 0
        self._ready = False
        self._last_active = time.time()
        self._idle_task: Optional[asyncio.Task] = None
        self._active_streams: int = 0

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
        from whisperlivekit import AudioProcessor

        await websocket.accept()
        self._active_streams += 1
        self._last_active = time.time()
        self._cancel_idle_shutdown()

        logger.info(
            f"Stream started (model={self.model_id}, language={self.language}, "
            f"active_streams={self._active_streams})"
        )

        # Create per-connection audio processor
        audio_processor = AudioProcessor(transcription_engine=self._engine)

        # Send config to client
        try:
            await websocket.send_json({"type": "config", "useAudioWorklet": False})
        except Exception as e:
            logger.warning(f"Failed to send config: {e}")

        # Start result generator
        results_generator = await audio_processor.create_tasks()

        # Task to forward results to client
        async def forward_results():
            try:
                async for response in results_generator:
                    try:
                        if hasattr(response, "to_dict"):
                            data = response.to_dict()
                        elif hasattr(response, "model_dump"):
                            data = response.model_dump()
                        elif isinstance(response, dict):
                            data = response
                        else:
                            import dataclasses

                            if dataclasses.is_dataclass(response):
                                data = dataclasses.asdict(response)
                            else:
                                data = {"raw": str(response)}

                        await websocket.send_json(data)
                    except Exception as e:
                        logger.warning(f"Failed to serialize response: {e}")

                await websocket.send_json({"type": "ready_to_stop"})
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"Error forwarding results: {e}")

        results_task = asyncio.create_task(forward_results())

        try:
            while True:
                message = await websocket.receive_bytes()
                await audio_processor.process_audio(message)
        except WebSocketDisconnect:
            logger.info("Stream disconnected by client")
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
        finally:
            self._active_streams -= 1
            self._last_active = time.time()

            # Cancel results task
            if not results_task.done():
                results_task.cancel()
                try:
                    await results_task
                except asyncio.CancelledError:
                    pass

            # Cleanup audio processor
            try:
                await audio_processor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up audio processor: {e}")

            # Schedule idle shutdown if no active streams
            if self._active_streams == 0:
                self._schedule_idle_shutdown()

            logger.info(f"Stream ended (active_streams={self._active_streams})")

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

    def load_engine(self) -> Any:
        """Load WhisperLiveKit transcription engine."""
        from whisperlivekit import TranscriptionEngine

        from src.config import get_hf_endpoint

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        effective_language = self.language if self.language else "auto"

        logger.info(
            f"Loading TranscriptionEngine (model={self.model_id}, "
            f"language={effective_language}, diarization={self.diarization})"
        )

        engine = TranscriptionEngine(
            model=self.model_id,
            lan=effective_language,
            transcription=True,
            diarization=self.diarization,
            buffer_trimming="sentence",
        )

        return engine

    def cleanup(self) -> None:
        """Release resources."""
        if self._engine is not None:
            try:
                if hasattr(self._engine, "cleanup"):
                    self._engine.cleanup()
            except Exception:
                pass
            del self._engine
            self._engine = None

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

        # Load engine
        logger.info(f"Loading engine for {self.model_id}...")
        start = time.time()
        try:
            self._engine = self.load_engine()
        except Exception as e:
            logger.error(f"Failed to load engine: {e}", exc_info=True)
            return 1

        self._load_time_ms = int((time.time() - start) * 1000)
        self._ready = True
        self._last_active = time.time()

        from ..utils import get_gpu_memory_mb

        logger.info(
            f"Engine loaded in {self._load_time_ms}ms, "
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

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        if v.lower() in ("no", "false", "f", "0"):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")

    parser = argparse.ArgumentParser(description="ASR streaming worker")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--idle-timeout", type=int, default=60, help="Idle seconds before exit")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--diarization", type=str_to_bool, default=False, help="Enable diarization")
    args = parser.parse_args(argv)

    worker = ASRStreamingWorker(
        model_id=args.model_id,
        port=args.port,
        idle_timeout=args.idle_timeout,
        language=args.language,
        diarization=args.diarization,
    )
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
