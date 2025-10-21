"""Unified OCR inference module supporting multiple architectures."""

import logging
from typing import Any, Dict, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class OCRInferenceEngine:
    """
    Unified OCR inference engine supporting multiple architectures:
    - PaddleOCR (paddleocr)
    - MinerU2.5 (mineru)
    - DeepSeek-OCR (deepseek)
    - IBM Granite Docling (granite-docling)
    """

    def __init__(
        self,
        model_id: str,
        architecture: str,
        model_config: Dict[str, Any],
        language: Optional[str] = None,
        output_format: str = "text",
    ):
        """
        Initialize OCR inference engine.

        Args:
            model_id: Hugging Face model identifier
            architecture: Model architecture (paddleocr, mineru, deepseek, granite-docling)
            model_config: Model configuration from database
            language: Optional language hint for OCR
            output_format: Output format ('text' or 'markdown')
        """
        self.model_id = model_id
        self.architecture = architecture
        self.model_config = model_config
        self.language = language
        self.output_format = output_format
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load the model based on architecture."""
        if self.architecture == "paddleocr":
            self._load_paddleocr()
        elif self.architecture == "mineru":
            self._load_mineru()
        elif self.architecture == "deepseek":
            self._load_deepseek()
        elif self.architecture == "granite-docling":
            self._load_granite_docling()
        else:
            raise ValueError(f"Unsupported OCR architecture: {self.architecture}")

    def _load_paddleocr(self) -> None:
        """Load PaddleOCR model."""
        try:
            from paddleocr import PaddleOCR

            logger.info(f"Loading PaddleOCR model '{self.model_id}'...")

            # Determine language: request param > model config > default 'ch'
            lang = self.language or self.model_config.get("language", "ch")

            # Check GPU availability
            try:
                import paddle
                has_gpu = (
                    paddle.device.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0
                )
                device = "gpu:0" if has_gpu else "cpu"
            except Exception:
                device = "cpu"

            self.model = PaddleOCR(lang=lang, device=device)
            logger.info(f"PaddleOCR loaded with lang={lang}, device={device}")

        except ImportError:
            raise RuntimeError(
                "PaddleOCR is not installed. Please install: pip install paddlepaddle paddleocr"
            )

    def _load_mineru(self) -> None:
        """Load MinerU2.5 model."""
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from mineru_vl_utils import MinerUClient
            import torch
            from pathlib import Path

            logger.info(f"Loading MinerU2.5 model '{self.model_id}'...")

            # Check if model is already downloaded locally via hfd
            from src.config import get_data_dir
            local_model_dir = get_data_dir() / "models" / self.model_id

            # Determine which path to use for loading
            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace: {self.model_id}")
                extra_kwargs = {}

            # Load model with auto device mapping
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **extra_kwargs,
            )

            processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **extra_kwargs)

            # Initialize MinerU client
            self.model = MinerUClient(
                backend="transformers",
                model=model,
                processor=processor,
            )

            logger.info(f"MinerU2.5 model loaded successfully")

        except ImportError as e:
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise RuntimeError(
                f"MinerU dependencies not installed. Missing: {missing_pkg}. "
                "Please install: pip install transformers>=4.56.0 mineru-vl-utils torch"
            )

    def _load_deepseek(self) -> None:
        """Load DeepSeek-OCR model.

        Some DeepSeek-OCR releases may not ship a multimodal Processor and
        `AutoProcessor.from_pretrained(...)` can resolve to a tokenizer-only
        instance. In that case, passing `images=` will raise a
        `PreTrainedTokenizerFast._batch_encode_plus()` TypeError. To handle
        this robustly, we fall back to loading `AutoImageProcessor` and
        `AutoTokenizer` separately and combine their outputs at inference.
        """
        try:
            from transformers import AutoModel, AutoProcessor
            import torch
            from pathlib import Path

            logger.info(f"Loading DeepSeek-OCR model '{self.model_id}'...")

            # Check if model is already downloaded locally via hfd
            from src.config import get_data_dir
            local_model_dir = get_data_dir() / "models" / self.model_id

            # Determine which path to use for loading
            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                # Use local_files_only to prevent re-downloading
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace: {self.model_id}")
                extra_kwargs = {}

            # Try to use flash attention if available, fallback to eager
            # Note: DeepSeek-OCR doesn't support sdpa, so we use eager as fallback
            attn_implementation = "eager"  # Default fallback for DeepSeek-OCR
            try:
                import flash_attn
                if torch.cuda.is_available():
                    attn_implementation = "flash_attention_2"
                    logger.info("Using flash_attention_2 for DeepSeek-OCR")
                else:
                    logger.info("GPU not available, using eager attention for DeepSeek-OCR")
            except ImportError:
                logger.info("flash-attn not installed, using eager attention for DeepSeek-OCR")

            # Load model with best available attention implementation
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                _attn_implementation=attn_implementation,
                use_safetensors=True,
                trust_remote_code=True,
                **extra_kwargs,
            )

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                **extra_kwargs,
            )

            # Detect tokenizer-only processors and prepare a multimodal fallback
            self._deepseek_image_processor = None
            self._deepseek_tokenizer = None
            try:
                # Heuristic: a proper multimodal processor exposes either
                # `image_processor` or `image_processor_class` attributes.
                has_image_support = any(
                    hasattr(self.processor, attr)
                    for attr in ("image_processor", "image_processor_class")
                )
                from transformers import AutoTokenizer

                # Always load tokenizer per official usage; processor may be tokenizer-only
                self._deepseek_tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, trust_remote_code=True, **extra_kwargs
                )

                if not has_image_support:
                    from transformers import AutoImageProcessor

                    logger.info(
                        "AutoProcessor resolved to tokenizer-only; "
                        "loading AutoImageProcessor + AutoTokenizer for DeepSeek-OCR"
                    )
                    self._deepseek_image_processor = AutoImageProcessor.from_pretrained(
                        model_path, trust_remote_code=True, **extra_kwargs
                    )
            except Exception as probe_err:
                # Non-fatal: continue without fallback; predict() will still try the processor path first
                logger.debug(f"DeepSeek-OCR processor probe failed: {probe_err}")

            logger.info(f"DeepSeek-OCR model loaded successfully with {attn_implementation}")

        except ImportError as e:
            # Extract missing package names from the error message
            error_msg = str(e)
            if "packages that were not found in your environment:" in error_msg:
                # Format: "This modeling file requires the following packages that were not found in your environment: pkg1, pkg2"
                missing_pkgs = error_msg.split("packages that were not found in your environment:")[1].split(".")[0].strip()
            elif "No module named" in error_msg:
                # Format: "No module named 'package'"
                missing_pkgs = error_msg.split("'")[1] if "'" in error_msg else "unknown"
            else:
                missing_pkgs = "unknown"

            raise RuntimeError(
                f"DeepSeek-OCR dependencies not installed. Missing: {missing_pkgs}. "
                f"Full error: {error_msg}"
            )

    def _load_granite_docling(self) -> None:
        """Load IBM Granite Docling model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            from pathlib import Path

            logger.info(f"Loading Granite Docling model '{self.model_id}'...")

            # Check if model is already downloaded locally via hfd
            from src.config import get_data_dir
            local_model_dir = get_data_dir() / "models" / self.model_id

            # Determine which path to use for loading
            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace: {self.model_id}")
                extra_kwargs = {}

            # Determine device and attention implementation
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try flash attention if available on GPU, fallback to sdpa
            attn_impl = "sdpa"  # Default safe option
            if device == "cuda":
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                    logger.info("Using flash_attention_2 for Granite Docling")
                except ImportError:
                    logger.info("flash-attn not installed, using sdpa attention for Granite Docling")
            else:
                logger.info("Using sdpa attention for Granite Docling on CPU")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_path, **extra_kwargs)

            # Load model with appropriate settings
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                _attn_implementation=attn_impl,
                device_map="auto" if device == "cuda" else None,
                **extra_kwargs,
            )

            # Move to device if CPU
            if device == "cpu":
                self.model = self.model.to(device)

            logger.info(f"Granite Docling model loaded successfully on {device} with {attn_impl}")

        except ImportError as e:
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise RuntimeError(
                f"Granite Docling dependencies not installed. Missing: {missing_pkg}. "
                "Please install: pip install transformers>=4.45.0 docling_core torch"
            )

    def predict(self, image: Image.Image) -> str:
        """
        Run OCR prediction on image.

        Args:
            image: PIL Image object

        Returns:
            Extracted text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.architecture == "paddleocr":
            return self._predict_paddleocr(image)
        elif self.architecture == "mineru":
            return self._predict_mineru(image)
        elif self.architecture == "deepseek":
            return self._predict_deepseek(image)
        elif self.architecture == "granite-docling":
            return self._predict_granite_docling(image)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _predict_paddleocr(self, image: Image.Image) -> str:
        """Run PaddleOCR prediction."""
        import tempfile
        import os

        # PaddleOCR expects file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            result = self.model.predict(tmp_path)

            # Parse results
            extracted_text = []
            if result and len(result) > 0:
                page_result = result[0]
                if "rec_texts" in page_result:
                    extracted_text = page_result["rec_texts"]

            return "\n".join(extracted_text) if extracted_text else ""

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _predict_mineru(self, image: Image.Image) -> str:
        """Run MinerU2.5 prediction."""
        # MinerU uses two-step extraction
        extracted_blocks = self.model.two_step_extract(image)

        # Parse extracted blocks into text
        # The format depends on MinerU's output structure
        if isinstance(extracted_blocks, list):
            texts = []
            for block in extracted_blocks:
                if isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)
        elif isinstance(extracted_blocks, str):
            return extracted_blocks
        else:
            return str(extracted_blocks)

    def _predict_deepseek(self, image: Image.Image) -> str:
        """Run DeepSeek-OCR prediction."""
        import torch
        import tempfile
        import os

        # DeepSeek-OCR requires a text prompt; official usage prefixes with "<image>\n"
        if self.output_format == "markdown":
            # Use grounding prompt for structured markdown output
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
            logger.info("Using markdown grounding prompt for DeepSeek-OCR")
        else:
            # "Free OCR." is suggested in official prompts for layout-agnostic text extraction
            prompt = "<image>\nFree OCR."
            logger.info("Using text extraction prompt for DeepSeek-OCR")

        try:
            # Preferred official path: use custom remote-code infer() if available
            if hasattr(self.model, "infer"):
                # Ensure tokenizer is available
                if getattr(self, "_deepseek_tokenizer", None) is None:
                    from transformers import AutoTokenizer
                    # Load tokenizer against self.model_id to respect local/remote
                    self._deepseek_tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id, use_fast=True, trust_remote_code=True
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                try:
                    # Reasonable defaults per README: base_size=1024, image_size=640, crop_mode=True
                    res = self.model.infer(
                        self._deepseek_tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        output_path=os.path.dirname(tmp_path),
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=True,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                # The remote infer() typically returns a string; fallback to str()
                return res if isinstance(res, str) else str(res)

            # Preferred path: multimodal processor handles both text + images
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

            # Move inputs only when the model is not dispatched across devices
            hf_device_map = getattr(self.model, "hf_device_map", None) or getattr(self.model, "device_map", None)
            if not hf_device_map:
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            # Prefer processor for decoding if available, otherwise tokenizer fallback
            decoder = getattr(self.processor, "batch_decode", None)
            if decoder is None and getattr(self, "_deepseek_tokenizer", None) is not None:
                text = self._deepseek_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return text
        except TypeError as te:
            # Fallback: AutoProcessor was tokenizer-only (no image support)
            if "images" in str(te) and (
                getattr(self, "_deepseek_image_processor", None) is not None
                and getattr(self, "_deepseek_tokenizer", None) is not None
            ):
                logger.info("Falling back to AutoImageProcessor + AutoTokenizer for DeepSeek-OCR")

                image_inputs = self._deepseek_image_processor(images=[image], return_tensors="pt")
                text_inputs = self._deepseek_tokenizer(text=prompt, return_tensors="pt")
                inputs = {**text_inputs, **image_inputs}

                hf_device_map = getattr(self.model, "hf_device_map", None) or getattr(self.model, "device_map", None)
                if not hf_device_map:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                outputs = self.model.generate(**inputs, max_new_tokens=2048)
                text = self._deepseek_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                return text
            # Re-raise other type errors
            logger.error(f"DeepSeek-OCR prediction failed (TypeError): {te}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"DeepSeek-OCR prediction failed: {e}", exc_info=True)
            raise

    def _predict_granite_docling(self, image: Image.Image) -> str:
        """Run IBM Granite Docling prediction."""
        import torch

        # Choose prompt based on output format
        if self.output_format == "markdown":
            prompt_text = "Convert this page to markdown."
            logger.info("Using markdown prompt for Granite Docling")
        else:
            prompt_text = "Extract all text from this document."
            logger.info("Using text extraction prompt for Granite Docling")

        # Create input messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ],
            },
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

        # Move inputs only when the model is not dispatched across devices
        hf_device_map = getattr(self.model, "hf_device_map", None) or getattr(self.model, "device_map", None)
        if not hf_device_map:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)

        # Decode output
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Extract the assistant's response (after the last "Assistant:" or similar marker)
        # The output typically includes the conversation history, we want just the response
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()
        elif "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()

        return generated_text

    def cleanup(self) -> None:
        """Clean up model resources."""
        # Try to move model (or wrapped model) to CPU before releasing
        def _move_to_cpu(obj: Any) -> None:
            try:
                if obj is None:
                    return
                if hasattr(obj, "cpu") and callable(getattr(obj, "cpu")):
                    obj.cpu()
                elif hasattr(obj, "to") and callable(getattr(obj, "to")):
                    obj.to("cpu")
                elif hasattr(obj, "model"):
                    inner = getattr(obj, "model")
                    if hasattr(inner, "cpu") and callable(getattr(inner, "cpu")):
                        inner.cpu()
                    elif hasattr(inner, "to") and callable(getattr(inner, "to")):
                        inner.to("cpu")
            except Exception:
                # Best-effort; ignore failures
                pass

        _move_to_cpu(self.model)

        # Drop references
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Force memory release across backends
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        # PyTorch cleanup (DeepSeek/Granite/MinerU backends)
        try:
            import torch
            if torch.cuda.is_available():
                # Collect any dangling IPC handles and cached blocks
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared for OCR (torch)")
        except Exception as e:
            logger.warning(f"Error releasing torch GPU memory: {e}")

        # Paddle cleanup (PaddleOCR backend)
        try:
            import paddle
            try:
                if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                    try:
                        paddle.device.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        paddle.device.cuda.synchronize()
                    except Exception:
                        pass
                    logger.debug("GPU cache cleared for OCR (paddle)")
            except Exception:
                # If device queries fail, skip silently
                pass
        except Exception:
            # paddle not installed; ignore
            pass
