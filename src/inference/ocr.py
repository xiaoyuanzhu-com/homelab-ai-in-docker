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

            # Load model with flash attention and bfloat16
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            logger.info(f"DeepSeek-OCR model loaded successfully")

        except ImportError as e:
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise RuntimeError(
                f"DeepSeek-OCR dependencies not installed. Missing: {missing_pkg}. "
                "Please install: pip install transformers>=4.46.3 flash-attn>=2.7.3 torch"
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

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Force GPU memory release
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared for OCR model")
        except Exception as e:
            logger.warning(f"Error releasing GPU memory: {e}")
