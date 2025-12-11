"""Doc to Screenshot API router using ScreenItShot worker."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..models.doc_to_screenshot import DocToScreenshotRequest, DocToScreenshotResponse
from ...storage.history import history_storage
from ...db.catalog import list_libs, get_lib_dict
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["doc-to-screenshot"])


def _available_doc_to_screenshot_libs() -> list[str]:
    try:
        return [lib["id"] for lib in list_libs(task="doc-to-screenshot")]
    except Exception:
        return []


def _guess_format(filename: Optional[str]) -> str:
    """Guess the format from filename extension."""
    if not filename:
        return "pdf"
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext else "pdf"


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
    lib_id = request.lib or "screenitshot/screenitshot"

    if lib_id not in supported:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ENGINE",
                "message": f"Engine '{lib_id}' is not available. Available: {', '.join(supported) if supported else 'none'}",
                "request_id": request_id,
            },
        )

    # Get lib config for python_env
    lib_config = get_lib_dict(lib_id)
    if not lib_config:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ENGINE",
                "message": f"Library '{lib_id}' not found in catalog",
                "request_id": request_id,
            },
        )

    python_env = lib_config.get("python_env")
    doc_format = _guess_format(request.filename)

    try:
        # Delegate to worker
        result = await coordinator_infer(
            task="doc-to-screenshot",
            model_id=lib_id,
            payload={
                "file": request.file,
                "format": doc_format,
                "filename": request.filename,
            },
            request_id=request_id,
            python_env=python_env,
        )

        screenshot_b64 = result.get("screenshot", "")
        processing_time_ms = int((time.time() - start_time) * 1000)

        response = DocToScreenshotResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            screenshot=screenshot_b64,
            model=lib_id,
        )

        # Save minimal history
        history_storage.add_request(
            service="doc-to-screenshot",
            request_id=request_id,
            request_data={"filename": request.filename, "lib": lib_id},
            response_data={
                "request_id": request_id,
                "processing_time_ms": processing_time_ms,
                "model": response.model,
            },
            status="success",
        )

        return response

    except HTTPException:
        raise
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
