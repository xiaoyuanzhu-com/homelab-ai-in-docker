"""Jina VLM (Vision-Language Model) inference engine.

Supports jinaai/jina-vlm - a 2.4B parameter multilingual VLM with
SigLIP2 vision encoder and Qwen3-1.7B language backbone.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _sync_dynamic_modules_to_cache(model_path: str) -> None:
    """
    Sync all Python files from a local model directory to the transformers modules cache.

    When loading models with trust_remote_code=True and local_files_only=True,
    transformers creates a modules cache directory but may not copy all required
    Python files (especially when files have relative imports to each other).

    This function ensures all Python files are present in the modules cache before
    loading, preventing FileNotFoundError for dynamic module imports.

    Args:
        model_path: Path to the local model directory
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        return

    # Find all Python files in the model directory
    py_files = list(model_dir.glob("*.py"))
    if not py_files:
        return

    # Determine the HF_HOME and modules cache path
    from src.config import get_data_dir

    hf_home = Path(os.getenv("HF_HOME", get_data_dir() / "models"))
    modules_cache = hf_home / "modules" / "transformers_modules"

    # Build the escaped module directory name
    # transformers escapes hyphens as _hyphen_ and slashes as _hyphen__hyphen_
    # For a path like models--jinaai--jina-vlm, it becomes:
    # models_hyphen__hyphen_jinaai_hyphen__hyphen_jina_hyphen_vlm
    model_dir_name = model_dir.name
    escaped_name = model_dir_name.replace("-", "_hyphen_")

    target_dir = modules_cache / escaped_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sync all Python files to the cache
    synced_count = 0
    for py_file in py_files:
        target_file = target_dir / py_file.name
        # Only copy if missing or source is newer
        if not target_file.exists() or py_file.stat().st_mtime > target_file.stat().st_mtime:
            shutil.copy2(py_file, target_file)
            synced_count += 1
            logger.debug(f"Synced {py_file.name} to transformers modules cache")

    # Ensure __init__.py exists
    init_file = target_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        synced_count += 1

    if synced_count > 0:
        logger.info(f"Synced {synced_count} files to transformers modules cache: {target_dir}")


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
        import transformers.dynamic_module_utils as dmu

        from src.config import get_data_dir, get_hf_endpoint, get_hf_model_cache_path

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Ensure HF_HOME is set so transformers uses the correct modules cache
        hf_home = Path(os.getenv("HF_HOME", get_data_dir() / "models"))
        os.environ["HF_HOME"] = str(hf_home)

        # Monkey-patch the modules cache path if it differs from our expected path
        # This is necessary because HF_MODULES_CACHE is determined at import time
        expected_modules_cache = str(hf_home / "modules")
        if dmu.HF_MODULES_CACHE != expected_modules_cache:
            logger.info(f"Updating HF_MODULES_CACHE from {dmu.HF_MODULES_CACHE} to {expected_modules_cache}")
            dmu.HF_MODULES_CACHE = expected_modules_cache

        hf_model_id = self.model_config.get("hf_model") or self.model_id
        logger.info(f"Loading Jina VLM model '{self.model_id}' (hf_model: {hf_model_id})...")

        # Check for local model
        local_model_dir = get_hf_model_cache_path(hf_model_id)
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            extra_kwargs = {"local_files_only": True}

            # Sync dynamic module files to transformers cache before loading
            # This prevents FileNotFoundError when loading models with trust_remote_code
            # that have Python files with relative imports to each other
            _sync_dynamic_modules_to_cache(model_path)
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
