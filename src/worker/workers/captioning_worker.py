"""Image captioning worker using transformers vision models.

Supports multiple architectures:
- BLIP (blip)
- BLIP-2 (blip2)
- LLaVA (llava)
- LLaVA-NeXT (llava_next)
- DeepSeek VL (deepseek)
- Moondream (moondream)
- Jina VLM (jina-vlm)
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Any, Dict, Tuple

from PIL import Image

from ..base import BaseWorker, create_worker_main
from ..image_utils import decode_image
from src.inference.deepseek_vl import DeepSeekVLEngine
from src.inference.jina_vlm import JinaVLMEngine

logger = logging.getLogger("captioning_worker")

# Check bitsandbytes availability
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


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
        self._deepseek_engine: DeepSeekVLEngine | None = None
        self._jina_vlm_engine: JinaVLMEngine | None = None

    def load_model(self) -> Any:
        """Load image captioning model."""
        from src.db.catalog import get_model_dict

        # Get model config
        self._model_cfg = get_model_dict(self.model_id)
        if self._model_cfg is None:
            raise ValueError(f"Model '{self.model_id}' not found in catalog")

        # Check architecture first - some models use their own loading path
        architecture = self._model_cfg.get("architecture", "").lower()

        if architecture == "deepseek":
            return self._load_deepseek()

        if architecture == "moondream":
            return self._load_moondream()

        if architecture == "jina-vlm":
            return self._load_jina_vlm()

        # Other models use standard transformers loading
        return self._load_transformers_model()

    def _load_deepseek(self) -> Any:
        """Load DeepSeek VL model using the shared engine."""
        logger.info(f"Loading DeepSeek VL model for captioning: {self.model_id}")

        self._deepseek_engine = DeepSeekVLEngine(
            model_id=self.model_id,
            model_config=self._model_cfg,
        )
        self._deepseek_engine.load()

        # Return the model for compatibility with base worker
        return self._deepseek_engine.model

    def _load_moondream(self) -> Any:
        """Load Moondream model with custom code support.

        Note: Moondream uses trust_remote_code=True which requires the remote
        Python code to be downloaded. We always use the model ID (not local path)
        and let transformers handle caching, as the custom code is stored
        separately from the model weights.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from src.config import get_hf_endpoint

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        logger.info(f"Loading Moondream model: {self.model_id}")

        # Always use model ID - transformers will use cache if available
        # Don't use local_files_only since trust_remote_code needs to fetch/verify code
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
        )

        logger.info(f"Moondream model loaded successfully: {self.model_id}")
        return model

    def _load_jina_vlm(self) -> Any:
        """Load Jina VLM model using the shared engine."""
        logger.info(f"Loading Jina VLM model for captioning: {self.model_id}")

        self._jina_vlm_engine = JinaVLMEngine(
            model_id=self.model_id,
            model_config=self._model_cfg,
        )
        self._jina_vlm_engine.load()

        # Return the model for compatibility with base worker
        return self._jina_vlm_engine.model

    def _load_transformers_model(self) -> Any:
        """Load standard transformers vision model."""
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForVision2Seq,
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
            Blip2ForConditionalGeneration,
        )

        from src.config import get_hf_endpoint, get_hf_model_cache_path

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
        image_data = payload.get("image", "")
        prompt = payload.get("prompt")

        # Use default prompt from config if not provided
        if prompt is None and self._model_cfg.get("default_prompt"):
            prompt = self._model_cfg["default_prompt"]

        # Decode image (supports HEIC/HEIF via pillow-heif)
        image = decode_image(image_data)

        # Use DeepSeek engine if available
        if self._deepseek_engine is not None:
            caption = self._deepseek_engine.predict(image, prompt)
            return {"caption": caption}

        # Use Jina VLM engine if available
        if self._jina_vlm_engine is not None:
            caption = self._jina_vlm_engine.predict(image, prompt)
            return {"caption": caption}

        # Check architecture for moondream
        architecture = self._model_cfg.get("architecture", "").lower()
        if architecture == "moondream":
            return self._infer_moondream(image, prompt)

        # Standard transformers inference
        return self._infer_transformers(image, prompt)

    def _infer_moondream(self, image: Image.Image, prompt: str | None) -> Dict[str, Any]:
        """Run inference with Moondream model.

        Moondream has its own API with caption() and query() methods.
        - caption(): Generate image captions (short, normal, or long)
        - query(): Answer visual questions about images
        """
        # If no prompt or a generic description prompt, use caption()
        if prompt is None or "describe" in prompt.lower():
            result = self._model.caption(image, length="long")
            caption = result["caption"]
        else:
            # Use query() for specific questions
            result = self._model.query(image, prompt)
            caption = result["answer"]

        return {"caption": caption}

    def _infer_transformers(self, image: Image.Image, prompt: str | None) -> Dict[str, Any]:
        """Run inference with standard transformers models."""
        import torch

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
        if self._deepseek_engine is not None:
            self._deepseek_engine.cleanup()
            self._deepseek_engine = None

        if self._jina_vlm_engine is not None:
            self._jina_vlm_engine.cleanup()
            self._jina_vlm_engine = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        super().cleanup()


# Main entry point
main = create_worker_main(CaptioningWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
