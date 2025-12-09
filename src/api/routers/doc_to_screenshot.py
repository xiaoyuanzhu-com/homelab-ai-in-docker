"""Doc to Screenshot API router using ScreenItShot."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..models.doc_to_screenshot import DocToScreenshotRequest, DocToScreenshotResponse
from ...storage.history import history_storage
from ...db.catalog import list_libs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["doc-to-screenshot"])

# Thread pool for running synchronous screenshot operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="screenitshot")


def _available_doc_to_screenshot_libs() -> list[str]:
    try:
        return [l["id"] for l in list_libs(task="doc-to-screenshot")]
    except Exception:
        return []


def _decode_data_url(data_url_or_b64: str) -> bytes:
    """Decode base64 data from a raw base64 string or data URL."""
    s = data_url_or_b64
    if "," in s and s.strip().lower().startswith("data:"):
        # Split after the comma: data:<mime>;base64,<payload>
        try:
            s = s.split(",", 1)[1]
        except Exception:
            pass
    try:
        return base64.b64decode(s)
    except Exception as e:
        raise ValueError(f"Invalid base64 payload: {e}")


def _guess_format(filename: Optional[str]) -> str:
    """Guess the format from filename extension."""
    if not filename:
        return "pdf"  # default
    ext = Path(filename).suffix.lower().lstrip(".")
    # screenitshot supports: pdf, epub, docx, xlsx, pptx, ipynb, md, html, tex, rtf, csv, geojson, gpx, mmd
    supported = {"pdf", "epub", "docx", "xlsx", "pptx", "ipynb", "md", "html", "tex", "rtf", "csv", "geojson", "gpx", "mmd"}
    return ext if ext in supported else "pdf"


@router.post("/doc-to-screenshot", response_model=DocToScreenshotResponse)
async def convert_document(request: DocToScreenshotRequest) -> DocToScreenshotResponse:
    """Convert an uploaded document to a screenshot using ScreenItShot.

    Accepts common document types such as PDF, DOCX, PPTX, XLSX, EPUB, Markdown, HTML, etc.
    Returns a base64-encoded PNG screenshot of the document.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Ensure catalog lists the supported lib
    supported = _available_doc_to_screenshot_libs()
    if request.lib not in supported:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ENGINE",
                "message": f"Engine '{request.lib}' is not available. Available: {', '.join(supported) if supported else 'none'}",
                "request_id": request_id,
            },
        )

    # Decode input bytes
    try:
        raw_bytes = _decode_data_url(request.file)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_INPUT", "message": str(e), "request_id": request_id},
        )

    # Determine format from filename
    doc_format = _guess_format(request.filename)

    # Run screenshot conversion
    try:
        # Import screenitshot
        try:
            from screenitshot import screenshot  # type: ignore
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "ENGINE_NOT_INSTALLED",
                    "message": f"ScreenItShot is not installed: {e}",
                    "request_id": request_id,
                },
            )

        # Run the conversion in a thread pool to avoid blocking the event loop
        # screenitshot uses Playwright internally which has its own event loop
        def _do_screenshot():
            return screenshot(raw_bytes, doc_format)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_executor, _do_screenshot)
            screenshot_bytes = result.data
        except Exception as e:
            logger.exception(f"Screenshot conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "CONVERSION_FAILED",
                    "message": f"Failed to convert document to screenshot: {e}",
                    "request_id": request_id,
                },
            )

        # Encode screenshot as base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        processing_time_ms = int((time.time() - start_time) * 1000)
        response = DocToScreenshotResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            screenshot=screenshot_b64,
            model=request.lib or "screenitshot/screenitshot",
        )

        # Save minimal history (avoid storing raw file)
        history_storage.add_request(
            service="doc-to-screenshot",
            request_id=request_id,
            request_data={"filename": request.filename, "lib": request.lib},
            response_data={"request_id": request_id, "processing_time_ms": processing_time_ms, "model": response.model},
            status="success",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in doc-to-screenshot: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"Unexpected error: {e}",
                "request_id": request_id,
            },
        )
