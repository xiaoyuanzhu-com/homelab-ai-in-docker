"""Doc to Markdown API router using MarkItDown worker."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from ..models.doc_to_markdown import DocToMarkdownRequest, DocToMarkdownResponse
from ...storage.history import history_storage
from ...db.catalog import list_libs, get_lib_dict
from ...worker import coordinator_infer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["doc-to-markdown"])


def _available_doc_to_md_libs() -> list[str]:
    try:
        return [lib["id"] for lib in list_libs(task="doc-to-markdown")]
    except Exception:
        return []


@router.post("/doc-to-markdown", response_model=DocToMarkdownResponse)
async def convert_document(request: DocToMarkdownRequest) -> DocToMarkdownResponse:
    """Convert an uploaded document to Markdown using MarkItDown.

    Accepts common document types such as PDF, DOCX, PPTX, XLSX, HTML, etc.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Ensure catalog lists the supported lib
    supported = _available_doc_to_md_libs()
    lib_id = request.lib or "microsoft/markitdown"

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

    try:
        # Delegate to worker
        result = await coordinator_infer(
            task="doc-to-markdown",
            model_id=lib_id,
            payload={
                "file": request.file,
                "filename": request.filename,
            },
            request_id=request_id,
            python_env=python_env,
        )

        markdown = result.get("markdown", "")
        processing_time_ms = int((time.time() - start_time) * 1000)

        response = DocToMarkdownResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            markdown=markdown,
            model=lib_id,
        )

        # Save minimal history
        history_storage.add_request(
            service="doc-to-markdown",
            request_id=request_id,
            request_data={"filename": request.filename, "lib": lib_id},
            response_data=response.model_dump(),
            status="success",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Document conversion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CONVERSION_FAILED",
                "message": f"Failed to convert document: {e}",
                "request_id": request_id,
            },
        )
