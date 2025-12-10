"""Jina VLM (Vision-Language Model) inference engine.

Supports jinaai/jina-vlm - a 2.4B parameter multilingual VLM with
SigLIP2 vision encoder and Qwen3-1.7B language backbone.
"""

import logging
import os
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class JinaVLMEngine:
    """
    Jina Vision-Language Model inference engine.

    Uses the transformers chat template interface for multi-modal conversations.
    """

    def __init__(
        self,
        model_id: str,
        model_config: Dict[str, Any],
    ):
        """
        Initialize Jina VLM engine.

        Args:
            model_id: Model identifier (e.g., "jinaai/jina-vlm")
            model_config: Model configuration from catalog
        """
        self.model_id = model_id
        self.model_config = model_config
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load the Jina VLM model and processor."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        from src.config import get_hf_endpoint, get_hf_model_cache_path

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        hf_model_id = self.model_config.get("hf_model") or self.model_id
        logger.info(f"Loading Jina VLM model '{self.model_id}' (hf_model: {hf_model_id})...")

        # Check for local model
        local_model_dir = get_hf_model_cache_path(hf_model_id)
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            extra_kwargs = {"local_files_only": True}
        else:
            model_path = hf_model_id
            logger.info(f"Model not found locally, will download from HuggingFace: {model_path}")
            extra_kwargs = {}

        # Try to use flash attention if available
        attn_implementation = None
        try:
            import flash_attn

            if torch.cuda.is_available():
                attn_implementation = "flash_attention_2"
                logger.info("Using flash_attention_2 for Jina VLM")
        except ImportError:
            logger.info("flash-attn not installed, using default attention")

        # Load processor (use_fast=False as recommended by model card)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            **extra_kwargs,
        )

        # Determine dtype - prefer float16 for memory efficiency
        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Build load kwargs
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            **extra_kwargs,
        }

        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        logger.info(f"Jina VLM model loaded successfully with dtype={dtype}")

    def predict(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Run inference on an image with optional prompt.

        Args:
            image: PIL Image object
            prompt: Optional prompt/question. If None, uses default captioning prompt.

        Returns:
            Generated text response
        """
        import torch
        from transformers import GenerationConfig

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Use default prompt if not provided
        if prompt is None:
            prompt = "Describe this image"

        # Build conversation in the format expected by jina-vlm
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding="longest",
            return_tensors="pt",
        )

        # Move inputs to model device
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                ),
                return_dict_in_generate=True,
                use_model_defaults=True,
            )

        # Decode response (skip the input tokens)
        response = self.processor.tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def cleanup(self) -> None:
        """Clean up model resources."""
        import gc

        if self.model is not None:
            try:
                if hasattr(self.model, "cpu"):
                    self.model.cpu()
                elif hasattr(self.model, "to"):
                    self.model.to("cpu")
            except Exception:
                pass
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared for Jina VLM")
        except Exception as e:
            logger.warning(f"Error releasing GPU memory: {e}")
