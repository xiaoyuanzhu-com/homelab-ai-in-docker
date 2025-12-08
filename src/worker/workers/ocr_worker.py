"""OCR worker using unified OCRInferenceEngine.

Supports multiple architectures:
- PaddleOCR (paddleocr)
- MinerU2.5 (mineru)
- DeepSeek-OCR (deepseek)
- IBM Granite Docling (granite-docling)
- HunyuanOCR (hunyuan-ocr)
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from typing import Any, Dict

from PIL import Image

from ..base import BaseWorker, create_worker_main
from src.inference.ocr import OCRInferenceEngine
from src.db.catalog import get_model_dict, get_lib_dict

logger = logging.getLogger("ocr_worker")


def _get_model_config(model_id: str) -> Dict[str, Any]:
    """Get model configuration from catalog."""
    cfg = get_model_dict(model_id) or get_lib_dict(model_id)
    if cfg is None:
        raise RuntimeError(f"Engine '{model_id}' not found in catalog")
    return cfg


def _decode_image(image_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    if image_data.startswith("data:image"):
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB")


class OCRWorker(BaseWorker):
    """OCR inference worker."""

    task_name = "ocr"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)

        # Extract language from model_config (passed via extra_args)
        self.language = model_config.get("language") if model_config else None
        self._engine: OCRInferenceEngine | None = None
        self._model_cfg: Dict[str, Any] = {}

    def load_model(self) -> OCRInferenceEngine:
        """Load OCR model."""
        self._model_cfg = _get_model_config(self.model_id)

        # Get language: config arg > model config > default 'ch'
        language = self.language or self._model_cfg.get("language") or "ch"

        engine = OCRInferenceEngine(
            model_id=self.model_id,
            architecture=self._model_cfg.get("architecture", "paddleocr"),
            model_config=self._model_cfg,
            language=language,
            output_format="text",  # Will be set per-request
        )
        engine.load()

        self._engine = engine
        return engine

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR inference."""
        if self._engine is None:
            raise RuntimeError("Model not loaded")

        # Extract request parameters
        image_data = payload.get("image", "")
        output_format = payload.get("output_format", "text")

        # Update engine's output format for this request
        self._engine.output_format = output_format

        # Decode and process image
        image = _decode_image(image_data)
        text = self._engine.predict(image)

        return {
            "text": text,
            "output_format": output_format,
        }

    def cleanup(self) -> None:
        """Clean up OCR engine resources."""
        if self._engine is not None:
            self._engine.cleanup()
            self._engine = None
        super().cleanup()


# Main entry point
main = create_worker_main(OCRWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
