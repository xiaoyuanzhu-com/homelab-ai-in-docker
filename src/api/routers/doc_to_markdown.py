"""Doc to Markdown API router using Microsoft MarkItDown."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..models.doc_to_markdown import DocToMarkdownRequest, DocToMarkdownResponse
from ...storage.history import history_storage
from ...db.catalog import list_libs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["doc-to-markdown"])


def _available_doc_to_md_libs() -> list[str]:
    try:
        return [l["id"] for l in list_libs(task="doc-to-markdown")]
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


def _guess_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".bin"
    suf = Path(filename).suffix
    return suf if suf else ".bin"


@router.post("/doc-to-markdown", response_model=DocToMarkdownResponse)
async def convert_document(request: DocToMarkdownRequest) -> DocToMarkdownResponse:
    """Convert an uploaded document to Markdown using MarkItDown.

    Accepts common document types such as PDF, DOCX, PPTX, XLSX, HTML, etc.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Ensure catalog lists the supported lib
    supported = _available_doc_to_md_libs()
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

    # Create a temporary file with appropriate suffix for MarkItDown
    tmp_path: Optional[Path] = None
    try:
        suffix = _guess_suffix(request.filename)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(raw_bytes)
        tmp.flush()
        tmp.close()
        tmp_path = Path(tmp.name)

        # Import and run MarkItDown conversion
        try:
            from markitdown import MarkItDown  # type: ignore
        except Exception as e:
            # Make it explicit for the user
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "ENGINE_NOT_INSTALLED",
                    "message": f"MarkItDown is not installed: {e}",
                    "request_id": request_id,
                },
            )

        md = MarkItDown()
        try:
            result = md.convert(str(tmp_path))  # pass a file path so type detection works
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "CONVERSION_FAILED",
                    "message": f"Failed to convert document: {e}",
                    "request_id": request_id,
                },
            )

        # Try to extract markdown-like text from result; be flexible across versions
        markdown: Optional[str] = None
        try:
            # Some versions expose .text_content or .markdown, others .text
            for attr in ("markdown", "text_content", "text"):
                if hasattr(result, attr):
                    markdown = getattr(result, attr)
                    break
            # Some may return a dict-like
            if markdown is None and isinstance(result, dict):
                markdown = result.get("markdown") or result.get("text_content") or result.get("text")
        except Exception:
            markdown = None

        if not isinstance(markdown, str) or not markdown:
            # Fallback to stringifying
            try:
                markdown = str(result)
            except Exception:
                markdown = ""

        processing_time_ms = int((time.time() - start_time) * 1000)
        response = DocToMarkdownResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            markdown=markdown or "",
            model=request.lib or "microsoft/markitdown",
        )

        # Save minimal history (avoid storing raw file)
        history_storage.add_request(
            service="doc-to-markdown",
            request_id=request_id,
            request_data={"filename": request.filename, "lib": request.lib},
            response_data=response.model_dump(),
            status="success",
        )

        return response

    finally:
        # Clean up temp file
        try:
            if tmp_path and tmp_path.exists():
                os.unlink(tmp_path)
        except Exception:
            pass

