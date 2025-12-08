"""Image captioning worker using transformers vision models.

Supports multiple architectures:
- BLIP (blip)
- BLIP-2 (blip2)
- LLaVA (llava)
- LLaVA-NeXT (llava_next)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import platform
import sys
from typing import Any, Dict, Tuple

from PIL import Image

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("captioning_worker")

# Check bitsandbytes availability
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


def _decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    if image_data.startswith("data:image"):
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class CaptioningWorker(BaseWorker):
    """Image captioning inference worker."""

    task_name = "captioning"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._processor = None
        self._model_cfg: Dict[str, Any] = {}

    def load_model(self) -> Any:
        """Load image captioning model."""
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForVision2Seq,
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
            Blip2ForConditionalGeneration,
        )

        from src.config import get_hf_endpoint, get_hf_model_cache_path
        from src.db.catalog import get_model_dict

        # Get model config
        self._model_cfg = get_model_dict(self.model_id)
        if self._model_cfg is None:
            raise ValueError(f"Model '{self.model_id}' not found in catalog")

        # Check quantization requirements
        if self._model_cfg.get("requires_quantization") and not HAS_BITSANDBYTES:
            platform_req = self._model_cfg.get("platform_requirements", "Linux")
            raise ValueError(
                f"Model '{self.model_id}' requires bitsandbytes which is not available. "
                f"Platform requirements: {platform_req}. "
                f"Current platform: {platform.system()}."
            )

        # Set HuggingFace endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Check for local model
        local_model_dir = get_hf_model_cache_path(self.model_id)
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            extra_kwargs = {"local_files_only": True}
        else:
            model_path = self.model_id
            logger.info(f"Model not found locally, will download from HuggingFace: {model_path}")
            extra_kwargs = {}

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, **extra_kwargs
        )

        # Common loading kwargs
        load_kwargs = {
            "low_cpu_mem_usage": True,
            **extra_kwargs,
        }

        # Handle quantization
        if self._model_cfg.get("requires_quantization"):
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16

        # Load model based on architecture
        architecture = self._model_cfg.get("architecture", "").lower()

        if architecture == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
        elif architecture == "llava_next":
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
        elif architecture == "blip2":
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                model_path, **load_kwargs
            )

        return model

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate caption for image."""
        import torch

        image_data = payload.get("image", "")
        prompt = payload.get("prompt")

        # Use default prompt from config if not provided
        if prompt is None and self._model_cfg.get("default_prompt"):
            prompt = self._model_cfg["default_prompt"]

        # Decode image
        image = _decode_image(image_data)

        # Process inputs
        if prompt:
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = self._processor(images=image, return_tensors="pt")

        # Handle device placement for dispatched models
        if hasattr(self._model, "hf_device_map") and self._model.hf_device_map:
            first_device = self._model.hf_device_map[next(iter(self._model.hf_device_map))]
            inputs = {
                k: v.to(first_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        else:
            model_device = next(self._model.parameters()).device
            inputs = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        # Generate caption
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=150)

        # Decode output
        caption = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Clean up caption
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()
        elif prompt and caption.startswith(prompt):
            caption = caption[len(prompt):].strip()

        return {"caption": caption}

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._processor is not None:
            del self._processor
            self._processor = None
        super().cleanup()


# Main entry point
main = create_worker_main(CaptioningWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
