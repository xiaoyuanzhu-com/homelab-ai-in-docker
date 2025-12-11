"""MarkItDown worker for document-to-markdown conversion.

Converts various document formats (PDF, DOCX, PPTX, XLSX, HTML, etc.)
to Markdown using Microsoft's MarkItDown library.
"""

from __future__ import annotations

import base64
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger(__name__)


class MarkItDownWorker(BaseWorker):
    """Worker for document-to-markdown conversion using MarkItDown."""

    task_name = "doc-to-markdown"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._md_converter = None

    def load_model(self) -> Any:
        """Initialize MarkItDown library."""
        from markitdown import MarkItDown

        md = MarkItDown()
        logger.info("MarkItDown library loaded")
        self._md_converter = md
        return md

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert document to markdown.

        Args:
            payload: {
                "file": base64-encoded document bytes,
                "filename": original filename (used for type detection)
            }

        Returns:
            {"markdown": converted markdown text}
        """
        # Decode input
        file_b64 = payload.get("file", "")
        filename = payload.get("filename")

        # Handle data URL format
        if "," in file_b64 and file_b64.strip().lower().startswith("data:"):
            file_b64 = file_b64.split(",", 1)[1]

        raw_bytes = base64.b64decode(file_b64)

        # Determine file suffix for type detection
        suffix = self._guess_suffix(filename)

        # Create temp file for MarkItDown (it needs a file path)
        tmp_path: Optional[Path] = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(raw_bytes)
            tmp.flush()
            tmp.close()
            tmp_path = Path(tmp.name)

            # Run conversion
            result = self._md_converter.convert(str(tmp_path))

            # Extract markdown from result (handle different versions)
            markdown = self._extract_markdown(result)

            return {"markdown": markdown}

        finally:
            # Clean up temp file
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _guess_suffix(self, filename: Optional[str]) -> str:
        """Guess file suffix from filename."""
        if not filename:
            return ".bin"
        suffix = Path(filename).suffix
        return suffix if suffix else ".bin"

    def _extract_markdown(self, result: Any) -> str:
        """Extract markdown text from MarkItDown result.

        Handles different versions of the library that may expose
        the result in different ways.
        """
        markdown: Optional[str] = None

        try:
            # Try different attribute names across versions
            for attr in ("markdown", "text_content", "text"):
                if hasattr(result, attr):
                    markdown = getattr(result, attr)
                    break

            # Some versions may return a dict-like object
            if markdown is None and isinstance(result, dict):
                markdown = (
                    result.get("markdown")
                    or result.get("text_content")
                    or result.get("text")
                )
        except Exception:
            markdown = None

        # Fallback to stringifying
        if not isinstance(markdown, str) or not markdown:
            try:
                markdown = str(result)
            except Exception:
                markdown = ""

        return markdown or ""

    def cleanup(self) -> None:
        """Clean up resources."""
        self._md_converter = None
        super().cleanup()


main = create_worker_main(MarkItDownWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
