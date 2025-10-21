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
    ):
        """
        Initialize OCR inference engine.

        Args:
            model_id: Hugging Face model identifier
            architecture: Model architecture (paddleocr, mineru, deepseek)
            model_config: Model configuration from database
            language: Optional language hint for OCR
        """
        self.model_id = model_id
        self.architecture = architecture
        self.model_config = model_config
        self.language = language
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

            logger.info(f"Loading MinerU2.5 model '{self.model_id}'...")

            # Load model with auto device mapping
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)

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
        """Load DeepSeek-OCR model."""
        try:
            from transformers import AutoModel, AutoProcessor
            import torch

            logger.info(f"Loading DeepSeek-OCR model '{self.model_id}'...")

            # Try to use flash attention if available, fallback to sdpa
            attn_implementation = "sdpa"  # Default safe option
            try:
                import flash_attn
                if torch.cuda.is_available():
                    attn_implementation = "flash_attention_2"
                    logger.info("Using flash_attention_2 for DeepSeek-OCR")
                else:
                    logger.info("GPU not available, using sdpa attention for DeepSeek-OCR")
            except ImportError:
                logger.info("flash-attn not installed, using sdpa attention for DeepSeek-OCR")

            # Load model with best available attention implementation
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            logger.info(f"DeepSeek-OCR model loaded successfully with {attn_implementation}")

        except ImportError as e:
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise RuntimeError(
                f"DeepSeek-OCR dependencies not installed. Missing: {missing_pkg}. "
                "Please install: pip install transformers>=4.46.3 torch"
            )

    def _load_granite_docling(self) -> None:
        """Load IBM Granite Docling model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            logger.info(f"Loading Granite Docling model '{self.model_id}'...")

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
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Load model with appropriate settings
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                _attn_implementation=attn_impl,
                device_map="auto" if device == "cuda" else None,
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
        # DeepSeek-OCR inference
        inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(**inputs, max_new_tokens=2048)

        # Decode
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return text

    def _predict_granite_docling(self, image: Image.Image) -> str:
        """Run IBM Granite Docling prediction."""
        import torch

        # Get the prompt from model config or use default
        prompt_text = self.model_config.get("default_prompt", "Convert this page to markdown.")

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

        # Move inputs to same device as model
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
