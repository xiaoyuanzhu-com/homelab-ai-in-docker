"""Isolated worker process for OCR inference.

Runs a lightweight FastAPI server to serve OCR for a single model.
Exits automatically after an idle timeout to free GPU memory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import signal
import sys
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from src.inference.ocr import OCRInferenceEngine
from src.db.catalog import get_model_dict, get_lib_dict


logger = logging.getLogger("ocr_worker")


class InferRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image or data URL")
    output_format: str = Field(default="text", description="Output format: 'text' or 'markdown'")


class InferResponse(BaseModel):
    text: str
    model: str
    processing_time_ms: int
    output_format: str


app = FastAPI()


# Global state for the worker
ENGINE: Optional[OCRInferenceEngine] = None
MODEL_ID: str = ""
MODEL_CONFIG: Optional[Dict[str, Any]] = None
LANGUAGE: Optional[str] = None
IDLE_TIMEOUT: int = 5
LAST_ACCESS: Optional[float] = None
_idle_task: Optional[asyncio.Task] = None


def _get_model_config(model_id: str) -> Dict[str, Any]:
    cfg = get_model_dict(model_id) or get_lib_dict(model_id)
    if cfg is None:
        raise RuntimeError(f"Engine '{model_id}' not found in catalog")
    return cfg


def _decode_image(image_data: str) -> Image.Image:
    if image_data.startswith("data:image"):
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB")


def _schedule_idle_shutdown() -> None:
    global _idle_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _idle_task and not _idle_task.done():
        _idle_task.cancel()

    async def _watchdog(timeout_s: int):
        try:
            await asyncio.sleep(timeout_s)
            if LAST_ACCESS is None:
                return
            idle = time.time() - LAST_ACCESS
            if idle >= timeout_s:
                logger.info("Worker idle for %.1fs; shutting down", idle)
                await _shutdown()
        except asyncio.CancelledError:
            pass

    _idle_task = loop.create_task(_watchdog(IDLE_TIMEOUT))


async def _shutdown() -> None:
    global ENGINE
    try:
        if ENGINE is not None:
            ENGINE.cleanup()
    finally:
        # Terminate the process to fully release CUDA context
        os._exit(0)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    global LAST_ACCESS, ENGINE
    start = time.time()
    try:
        # Update engine's output format for this request
        if ENGINE:  # type: ignore[truthy-bool]
            ENGINE.output_format = req.output_format  # type: ignore[union-attr]

        image = await asyncio.to_thread(_decode_image, req.image)
        text = await asyncio.to_thread(ENGINE.predict, image)  # type: ignore[union-attr]
    except Exception as e:
        # Log the full exception for debugging
        logger.error(f"Inference failed: {e}", exc_info=True)

        # Explicitly handle CUDA OOM: free caches then exit the process to release context
        msg = str(e)
        is_oom = "out of memory" in msg.lower()
        if is_oom:
            try:
                import torch
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
            finally:
                os._exit(1)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    proc_ms = int((time.time() - start) * 1000)
    LAST_ACCESS = time.time()
    _schedule_idle_shutdown()
    return InferResponse(
        text=text,
        model=MODEL_ID,
        processing_time_ms=proc_ms,
        output_format=req.output_format
    )


@app.post("/shutdown")
async def shutdown_endpoint():
    await _shutdown()
    return {"status": "shutting_down"}


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="OCR worker")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--idle-timeout", type=int, default=5, help="Idle seconds before exit")
    parser.add_argument("--language", default=None, help="Language hint (optional)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    global ENGINE, MODEL_ID, MODEL_CONFIG, LANGUAGE, IDLE_TIMEOUT, LAST_ACCESS
    args = _parse_args(argv)
    MODEL_ID = args.model_id
    LANGUAGE = args.language
    IDLE_TIMEOUT = args.idle_timeout
    LAST_ACCESS = time.time()

    # Load model
    MODEL_CONFIG = _get_model_config(MODEL_ID)
    ENGINE = OCRInferenceEngine(
        model_id=MODEL_ID,
        architecture=MODEL_CONFIG.get("architecture", "paddleocr"),
        model_config=MODEL_CONFIG,
        language=LANGUAGE or MODEL_CONFIG.get("language") or "ch",
    )
    ENGINE.load()
    logger.info("Worker started for model %s on port %d", MODEL_ID, args.port)

    # Run uvicorn programmatically
    import uvicorn

    # Install SIGTERM handler to exit fast
    def _term_handler(signum, frame):
        try:
            asyncio.get_event_loop().create_task(_shutdown())
        except Exception:
            os._exit(0)

    try:
        signal.signal(signal.SIGTERM, _term_handler)
    except Exception:
        pass

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
