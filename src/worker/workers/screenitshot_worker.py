"""ScreenItShot worker for document-to-screenshot conversion.

Converts documents (PDF, DOCX, PPTX, etc.) to high-quality screenshots
using browser-based rendering via Playwright.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger(__name__)

# Formats that need LibreOffice conversion to PDF first
_LIBREOFFICE_FORMATS = {"doc", "ppt", "xls", "pptx", "docx", "xlsx"}


class ScreenItShotWorker(BaseWorker):
    """Worker for document-to-screenshot conversion using ScreenItShot."""

    task_name = "doc-to-screenshot"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._screenshot_fn = None

    def load_model(self) -> Any:
        """Initialize ScreenItShot library."""
        from screenitshot import screenshot

        logger.info("ScreenItShot library loaded")
        self._screenshot_fn = screenshot
        return screenshot

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert document to screenshot.

        Args:
            payload: {
                "file": base64-encoded document bytes,
                "format": document format (e.g., "pdf", "docx"),
                "filename": optional original filename
            }

        Returns:
            {"screenshot": base64-encoded PNG image}
        """
        # Decode input
        file_b64 = payload.get("file", "")
        doc_format = payload.get("format", "pdf")
        filename = payload.get("filename")

        # Handle data URL format
        if "," in file_b64 and file_b64.strip().lower().startswith("data:"):
            file_b64 = file_b64.split(",", 1)[1]

        raw_bytes = base64.b64decode(file_b64)

        # For Office formats, convert to PDF first using LibreOffice
        if doc_format.lower() in _LIBREOFFICE_FORMATS:
            logger.info(f"Converting {doc_format} to PDF using LibreOffice")
            raw_bytes = self._convert_to_pdf(raw_bytes, doc_format)
            doc_format = "pdf"

        # Run screenshot conversion
        result = self._screenshot_fn(raw_bytes, doc_format)
        screenshot_bytes = result.data

        # Encode as base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        return {"screenshot": screenshot_b64}

    def _convert_to_pdf(self, input_bytes: bytes, input_ext: str) -> bytes:
        """Convert document to PDF using LibreOffice.

        Args:
            input_bytes: Raw document bytes
            input_ext: File extension (e.g., 'doc', 'pptx')

        Returns:
            PDF bytes

        Raises:
            RuntimeError: If conversion fails
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write input file
            input_filename = f"input.{input_ext}"
            input_path = Path(tmpdir) / input_filename
            input_path.write_bytes(input_bytes)

            # Try different LibreOffice binary names
            libreoffice_bins = ["libreoffice25.8", "libreoffice", "soffice"]
            result = None

            for lo_bin in libreoffice_bins:
                try:
                    result = subprocess.run(
                        [
                            lo_bin,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            tmpdir,
                            str(input_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        break
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    raise RuntimeError("LibreOffice conversion timed out")

            if result is None or result.returncode != 0:
                error = result.stderr if result else "LibreOffice not found"
                raise RuntimeError(f"LibreOffice conversion failed: {error}")

            # Find the output PDF
            pdf_path = Path(tmpdir) / "input.pdf"
            if not pdf_path.exists():
                pdf_files = list(Path(tmpdir).glob("*.pdf"))
                if pdf_files:
                    pdf_path = pdf_files[0]
                else:
                    raise RuntimeError("LibreOffice did not produce a PDF output")

            return pdf_path.read_bytes()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._screenshot_fn = None
        super().cleanup()


main = create_worker_main(ScreenItShotWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
