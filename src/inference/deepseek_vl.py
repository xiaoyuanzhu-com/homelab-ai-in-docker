"""Shared DeepSeek Vision-Language model loading and inference.

This module provides a unified interface for DeepSeek VLM that can be used
by both image-captioning and image-ocr tasks.
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class DeepSeekVLEngine:
    """
    DeepSeek Vision-Language inference engine.

    Supports both OCR and image captioning tasks with custom prompts.
    """

    def __init__(
        self,
        model_id: str,
        model_config: Dict[str, Any],
    ):
        """
        Initialize DeepSeek VL engine.

        Args:
            model_id: Model identifier (e.g., "deepseek-ai/DeepSeek-OCR")
            model_config: Model configuration from catalog
        """
        self.model_id = model_id
        self.model_config = model_config
        self.model = None
        self.processor = None
        self._tokenizer = None
        self._image_processor = None

    def load(self) -> None:
        """Load the DeepSeek model and processor."""
        try:
            from transformers import AutoModel, AutoProcessor
            import torch

            # Use hf_model from config if available (for aliased models)
            hf_model_id = self.model_config.get("hf_model") or self.model_id
            logger.info(f"Loading DeepSeek VL model '{self.model_id}' (hf_model: {hf_model_id})...")

            # Check for local download at HF standard cache path
            from src.config import get_hf_model_cache_path

            local_model_dir = get_hf_model_cache_path(hf_model_id)

            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = hf_model_id
                logger.info(f"Model not found locally, will download from HuggingFace: {model_path}")
                extra_kwargs = {}

            # Try to use flash attention if available, fallback to eager
            attn_implementation = "eager"
            try:
                import flash_attn

                if torch.cuda.is_available():
                    attn_implementation = "flash_attention_2"
                    logger.info("Using flash_attention_2 for DeepSeek VL")
                else:
                    logger.info("GPU not available, using eager attention")
            except ImportError:
                logger.info("flash-attn not installed, using eager attention")

            # Load model
            # Use device_map="cuda" instead of "auto" because the DeepSeek model's
            # infer() method explicitly calls .cuda() on tensors and assumes the
            # entire model is on GPU. With "auto", some layers may be placed on CPU
            # causing device mismatch errors during generation.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            load_kwargs = {
                "device_map": device,
                "_attn_implementation": attn_implementation,
                "use_safetensors": True,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **extra_kwargs,
            }

            self.model = AutoModel.from_pretrained(model_path, **load_kwargs)

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                **extra_kwargs,
            )

            # Detect tokenizer-only processors and prepare a multimodal fallback
            self._setup_fallback_processors(model_path, extra_kwargs)

            logger.info(f"DeepSeek VL model loaded successfully with {attn_implementation}")

        except ImportError as e:
            error_msg = str(e)
            if "packages that were not found in your environment:" in error_msg:
                missing_pkgs = error_msg.split("packages that were not found in your environment:")[1].split(".")[0].strip()
            elif "No module named" in error_msg:
                missing_pkgs = error_msg.split("'")[1] if "'" in error_msg else "unknown"
            else:
                missing_pkgs = "unknown"

            raise RuntimeError(
                f"DeepSeek VL dependencies not installed. Missing: {missing_pkgs}. "
                f"Full error: {error_msg}"
            )

    def _setup_fallback_processors(self, model_path: str, extra_kwargs: Dict[str, Any]) -> None:
        """Setup fallback processors for tokenizer-only AutoProcessor."""
        try:
            has_image_support = any(
                hasattr(self.processor, attr)
                for attr in ("image_processor", "image_processor_class")
            )
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, trust_remote_code=True, **extra_kwargs
            )

            if not has_image_support:
                from transformers import AutoImageProcessor

                logger.info(
                    "AutoProcessor resolved to tokenizer-only; "
                    "loading AutoImageProcessor + AutoTokenizer"
                )
                self._image_processor = AutoImageProcessor.from_pretrained(
                    model_path, trust_remote_code=True, **extra_kwargs
                )
        except Exception as probe_err:
            logger.debug(f"DeepSeek processor probe failed: {probe_err}")

    def predict(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Run inference on an image with optional prompt.

        Args:
            image: PIL Image object
            prompt: Optional prompt/question. If None, uses a default captioning prompt.

        Returns:
            Generated text response
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build prompt with image prefix
        if prompt is None:
            prompt = "<image>\nDescribe this image in detail."
        elif not prompt.startswith("<image>"):
            prompt = f"<image>\n{prompt}"

        try:
            # Preferred path: use custom remote-code infer() if available
            if hasattr(self.model, "infer"):
                return self._predict_with_infer(image, prompt)

            # Standard transformers path
            return self._predict_with_generate(image, prompt)

        except TypeError as te:
            # Fallback: AutoProcessor was tokenizer-only
            if "images" in str(te) and self._image_processor is not None and self._tokenizer is not None:
                return self._predict_with_fallback(image, prompt)
            logger.error(f"DeepSeek prediction failed (TypeError): {te}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"DeepSeek prediction failed: {e}", exc_info=True)
            raise

    def _predict_with_infer(self, image: Image.Image, prompt: str) -> str:
        """Use model.infer() method if available (DeepSeek-OCR specific)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            hf_model_id = self.model_config.get("hf_model") or self.model_id
            self._tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id, use_fast=True, trust_remote_code=True
            )

        with tempfile.TemporaryDirectory() as td:
            tmp_img = os.path.join(td, "in.png")
            image.save(tmp_img)

            res = self.model.infer(
                self._tokenizer,
                prompt=prompt,
                image_file=tmp_img,
                output_path=td,
                base_size=640,
                image_size=640,
                crop_mode=False,
                save_results=False,
                test_compress=False,
                eval_mode=True,  # Required to get return value instead of streaming
            )

            # Handle dict return value (official API returns dict with "text" key)
            if isinstance(res, dict):
                text = res.get("text", "")
                if text and isinstance(text, str) and text.strip():
                    return text.strip()

            # Handle string return value (some versions may return string directly)
            if isinstance(res, str) and res.strip():
                return res.strip()

            # Look for saved output files as fallback
            result = self._read_saved_output(td)
            if result:
                return result

            # Stringify res as last resort (for unexpected return types)
            if res is not None:
                s = str(res).strip()
                # Avoid returning dict string representation
                if s and not s.startswith("{"):
                    return s

            raise RuntimeError("DeepSeek model.infer() returned no usable output")

    def _read_saved_output(self, output_dir: str) -> Optional[str]:
        """Read saved output from model.infer()."""
        latest_path = None
        latest_mtime = -1.0

        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                if f.lower().endswith((".md", ".txt", ".mmd")):
                    p = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(p)
                    except Exception:
                        mtime = 0
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_path = p

        if latest_path:
            try:
                with open(latest_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read().strip()
                    if content:
                        return content
            except Exception:
                pass

        return None

    def _predict_with_generate(self, image: Image.Image, prompt: str) -> str:
        """Use standard model.generate() with processor."""
        import torch

        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

        # Move inputs only when model is not dispatched across devices
        hf_device_map = getattr(self.model, "hf_device_map", None) or getattr(self.model, "device_map", None)
        if not hf_device_map:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=2048)

        # Decode output
        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is None and self._tokenizer is not None:
            text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return text

    def _predict_with_fallback(self, image: Image.Image, prompt: str) -> str:
        """Fallback using separate image processor and tokenizer."""
        import torch

        logger.info("Using fallback AutoImageProcessor + AutoTokenizer")

        image_inputs = self._image_processor(images=[image], return_tensors="pt")
        text_inputs = self._tokenizer(text=prompt, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}

        hf_device_map = getattr(self.model, "hf_device_map", None) or getattr(self.model, "device_map", None)
        if not hf_device_map:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return text

    def cleanup(self) -> None:
        """Clean up model resources."""
        import gc

        # Move model to CPU before releasing
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

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._image_processor is not None:
            del self._image_processor
            self._image_processor = None

        gc.collect()

        # PyTorch cleanup
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared for DeepSeek VL")
        except Exception as e:
            logger.warning(f"Error releasing GPU memory: {e}")
