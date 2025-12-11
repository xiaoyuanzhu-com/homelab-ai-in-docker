"""Unified OCR inference module supporting multiple architectures."""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from PIL import Image

from .deepseek_vl import DeepSeekVLEngine

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OCRInferenceEngine:
    """
    Unified OCR inference engine supporting multiple architectures:
    - PaddleOCR (paddleocr)
    - MinerU2.5 (mineru)
    - DeepSeek-OCR (deepseek)
    - IBM Granite Docling (granite-docling)
    - HunyuanOCR (hunyuan-ocr)
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
        self._deepseek_engine: Optional[DeepSeekVLEngine] = None

    def load(self) -> None:
        """Load the model based on architecture."""
        if self.architecture in ("paddleocr", "paddleocr-legacy"):
            self._load_paddleocr()
        elif self.architecture == "mineru":
            self._load_mineru()
        elif self.architecture == "deepseek":
            self._load_deepseek()
        elif self.architecture == "granite-docling":
            self._load_granite_docling()
        elif self.architecture == "hunyuan-ocr":
            self._load_hunyuan_ocr()
        else:
            raise ValueError(f"Unsupported OCR architecture: {self.architecture}")

    def _load_paddleocr(self) -> None:
        """Load PaddleOCR model.

        For PaddleOCR-VL: Uses transformers backend with flash_attention_2 for
        ~12x memory reduction (40GB -> 3.3GB) and ~6x speed improvement.
        For legacy PaddleOCR: Uses native paddleocr package.
        """
        # Determine if this is PaddleOCR-VL (Vision-Language model) or legacy PaddleOCR
        is_vl_model = "PaddleOCR-VL" in self.model_id or "paddleocr-vl" in self.model_id.lower()

        if is_vl_model:
            self._load_paddleocr_vl_transformers()
        else:
            self._load_paddleocr_legacy()

    def _load_paddleocr_vl_transformers(self) -> None:
        """Load PaddleOCR-VL using transformers with flash_attention_2.

        This provides massive memory savings compared to the native paddleocr backend:
        - Memory: ~3.3GB vs ~40GB (12x reduction)
        - Speed: ~19s vs ~2min (6x faster)

        Reference: https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/59
        """
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            logger.info(f"Loading PaddleOCR-VL model '{self.model_id}' via transformers...")

            # Check for local download at HF standard cache path
            from src.config import get_hf_model_cache_path
            local_model_dir = get_hf_model_cache_path(self.model_id)

            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace")
                extra_kwargs = {}

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try flash_attention_2 first (12x memory reduction), fallback to sdpa
            attn_impl = "sdpa"  # Safe default
            if device == "cuda":
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                    logger.info("Using flash_attention_2 for PaddleOCR-VL (12x memory reduction)")
                except ImportError:
                    logger.info("flash-attn not installed, using sdpa attention for PaddleOCR-VL")

            # Load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                **extra_kwargs,
            ).to(dtype=torch.bfloat16, device=device).eval()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                **extra_kwargs,
            )

            logger.info(f"PaddleOCR-VL loaded successfully on {device} with {attn_impl}")

        except ImportError as e:
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise RuntimeError(
                f"PaddleOCR-VL (transformers) dependencies not installed. Missing: {missing_pkg}. "
                "Please install: pip install transformers torch"
            )

    def _load_paddleocr_legacy(self) -> None:
        """Load legacy PaddleOCR model using native paddleocr package."""
        try:
            from paddleocr import PaddleOCR
            logger.info(f"Loading legacy PaddleOCR model '{self.model_id}'...")

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
            logger.info(f"Legacy PaddleOCR loaded with lang={lang}, device={device}")

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

            # Check for local download at HF standard cache path
            from src.config import get_hf_model_cache_path
            local_model_dir = get_hf_model_cache_path(self.model_id)

            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
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
        """Load DeepSeek-OCR model using the shared DeepSeekVLEngine."""
        self._deepseek_engine = DeepSeekVLEngine(
            model_id=self.model_id,
            model_config=self.model_config,
        )
        self._deepseek_engine.load()
        # Keep reference for compatibility
        self.model = self._deepseek_engine.model
        self.processor = self._deepseek_engine.processor

    def _load_granite_docling(self) -> None:
        """Load IBM Granite Docling model."""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            from pathlib import Path

            logger.info(f"Loading Granite Docling model '{self.model_id}'...")

            # Check for local download at HF standard cache path
            from src.config import get_hf_model_cache_path
            local_model_dir = get_hf_model_cache_path(self.model_id)

            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
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
            self.model = AutoModelForImageTextToText.from_pretrained(
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

    def _load_hunyuan_ocr(self) -> None:
        """Load Tencent HunyuanOCR model."""
        try:
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
            import torch

            logger.info(f"Loading HunyuanOCR model '{self.model_id}'...")

            # Check for local download at HF standard cache path
            from src.config import get_hf_model_cache_path
            local_model_dir = get_hf_model_cache_path(self.model_id)

            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = self.model_id
                logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
                extra_kwargs = {}

            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            # Load processor (use_fast=False as per official docs)
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=False,
                **extra_kwargs,
            )

            # Load model with eager attention (HunyuanOCR uses eager by default)
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="eager",
                dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                **extra_kwargs,
            )

            # Move to device if CPU (device_map handles GPU)
            if device == "cpu":
                self.model = self.model.to(device)

            logger.info(f"HunyuanOCR model loaded successfully on {device} with dtype {dtype}")

        except ImportError as e:
            error_msg = str(e)
            if "HunYuanVLForConditionalGeneration" in error_msg:
                raise RuntimeError(
                    "HunyuanOCR requires transformers with HunYuanVL support (not yet in stable release). "
                    "Please install from commit: pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4"
                )
            missing_pkg = error_msg.split("'")[1] if "'" in error_msg else "unknown"
            raise RuntimeError(
                f"HunyuanOCR dependencies not installed. Missing: {missing_pkg}. "
                "Please install: pip install transformers torch"
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

        if self.architecture in ("paddleocr", "paddleocr-legacy"):
            return self._predict_paddleocr(image)
        elif self.architecture == "mineru":
            return self._predict_mineru(image)
        elif self.architecture == "deepseek":
            return self._predict_deepseek(image)
        elif self.architecture == "granite-docling":
            return self._predict_granite_docling(image)
        elif self.architecture == "hunyuan-ocr":
            return self._predict_hunyuan_ocr(image)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _predict_paddleocr(self, image: Image.Image) -> str:
        """Run PaddleOCR prediction."""
        # Determine if this is PaddleOCR-VL or legacy PaddleOCR
        is_vl_model = "PaddleOCR-VL" in self.model_id or "paddleocr-vl" in self.model_id.lower()

        if is_vl_model:
            return self._predict_paddleocr_vl_transformers(image)
        else:
            return self._predict_paddleocr_legacy(image)

    def _predict_paddleocr_vl_transformers(self, image: Image.Image) -> str:
        """Run PaddleOCR-VL prediction using transformers backend.

        Supports task types: ocr, table, chart, formula
        Reference: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
        """
        import torch

        # Map output_format to task prompts
        # PaddleOCR-VL supports: OCR, Table Recognition, Chart Recognition, Formula Recognition
        if self.output_format == "markdown":
            # For markdown, use table recognition which outputs structured markdown
            task_prompt = "Table Recognition:"
        else:
            task_prompt = "OCR:"

        # Build messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": task_prompt}
                ]
            }
        ]

        # Process inputs using the processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                use_cache=True
            )

        # Decode output
        outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract just the response (after the prompt)
        # The output typically includes the conversation, we want the model's response
        if task_prompt in outputs:
            outputs = outputs.split(task_prompt)[-1].strip()

        return outputs

    def _predict_paddleocr_legacy(self, image: Image.Image) -> str:
        """Run legacy PaddleOCR prediction using native paddleocr package."""
        import tempfile
        import os

        # Legacy PaddleOCR API expects file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # PaddleOCR 3.x: Use ocr() method
            # Note: PP-OCRv5 changed API and doesn't accept cls parameter
            result = self.model.ocr(tmp_path)

            # PaddleOCR 3.x returns a list of dicts with 'rec_texts' key
            extracted_text = []

            if isinstance(result, list) and len(result) > 0:
                # PaddleOCR 3.x format: [{'rec_texts': [...], 'dt_polys': [...], ...}]
                first_item = result[0]
                if isinstance(first_item, dict) and 'rec_texts' in first_item:
                    # New PaddleOCR 3.x format
                    rec_texts = first_item['rec_texts']
                    if isinstance(rec_texts, list):
                        for item in rec_texts:
                            # Items are strings
                            if isinstance(item, str):
                                extracted_text.append(item)
                            elif isinstance(item, dict) and 'text' in item:
                                extracted_text.append(item['text'])
                elif isinstance(first_item, list):
                    # Old format: [[[bbox, (text, confidence)], ...]]
                    for line in first_item:
                        if len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else line[1]
                            extracted_text.append(text)
            elif isinstance(result, dict):
                # Direct dict format (fallback)
                if 'rec_texts' in result:
                    rec_texts = result['rec_texts']
                    if isinstance(rec_texts, list):
                        extracted_text.extend([str(t) for t in rec_texts if t])
                elif 'rec_text' in result:
                    rec_texts = result['rec_text']
                    if isinstance(rec_texts, list):
                        extracted_text.extend([str(t) for t in rec_texts if t])

            return "\n".join(extracted_text) if extracted_text else ""

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _predict_mineru(self, image: Image.Image) -> str:
        """Run MinerU2.5 prediction.

        Returns ContentBlock objects with:
        - type: 'text', 'image', 'table', or 'equation'
        - content: recognized text/HTML/LaTeX (None for images)
        - bbox: normalized bounding box [xmin, ymin, xmax, ymax]
        - angle: rotation angle (0, 90, 180, 270, or None)
        """
        # MinerU uses two-step extraction, returns List[ContentBlock]
        extracted_blocks = self.model.two_step_extract(image)

        logger.info(f"MinerU extracted {len(extracted_blocks)} blocks")

        # Parse ContentBlock objects into text
        # Note: MinerU does not support markdown output format natively
        texts = []
        for block in extracted_blocks:
            # ContentBlock has .type and .content attributes
            block_type = getattr(block, 'type', None)
            block_content = getattr(block, 'content', None)

            logger.debug(f"MinerU block type: {block_type}, content length: {len(str(block_content)) if block_content else 0}")

            # Skip image blocks (content is None)
            if block_content is None:
                continue

            # For text blocks, use content directly
            if block_type == 'text':
                texts.append(block_content)
            # For table blocks, content is HTML
            elif block_type == 'table':
                texts.append(block_content)
            # For equation blocks, content is LaTeX
            elif block_type == 'equation':
                texts.append(f"[LaTeX: {block_content}]")

        logger.info(f"MinerU extracted {len(texts)} content blocks with text")
        return "\n\n".join(texts) if texts else ""

    def _predict_deepseek(self, image: Image.Image) -> str:
        """Run DeepSeek-OCR prediction using the shared engine."""
        if self._deepseek_engine is None:
            raise RuntimeError("DeepSeek engine not loaded")

        # Build OCR-specific prompt based on output format
        if self.output_format == "markdown":
            prompt = "<|grounding|>Convert the document to markdown."
            logger.info("Using markdown grounding prompt for DeepSeek-OCR")
        else:
            prompt = "Free OCR."
            logger.info("Using text extraction prompt for DeepSeek-OCR")

        return self._deepseek_engine.predict(image, prompt)

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

        # Move inputs to the model's device
        # For device_map="auto", get the first device; otherwise get the model's device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        # Using max_new_tokens=2048 as a reasonable default for most OCR tasks
        # (official docs show max support of 8192, but most OCR doesn't need that much)
        # Using temperature=0.0 for deterministic outputs as recommended by official docs
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False
            )

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

    def _predict_hunyuan_ocr(self, image: Image.Image) -> str:
        """Run Tencent HunyuanOCR prediction."""
        import torch

        # Choose prompt based on output format
        # HunyuanOCR supports multiple output formats including markdown
        if self.output_format == "markdown":
            # Document parsing with markdown output
            prompt_text = "Parse and convert this document to markdown format, preserving structure, tables, and formulas."
            logger.info("Using markdown prompt for HunyuanOCR")
        else:
            # Basic text extraction with coordinate output
            prompt_text = "Detect and recognize all text in the image."
            logger.info("Using text extraction prompt for HunyuanOCR")

        # Create input messages in HunyuanOCR chat format
        # HunyuanOCR expects system message (can be empty) + user message with image and text
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]

        # Process inputs
        inputs = self.processor(
            text=texts, images=[image], padding=True, return_tensors="pt"
        )

        # Move inputs to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate output
        # HunyuanOCR supports up to 16384 tokens for complex documents
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
            )

        # Decode output
        output_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        generated_text = output_texts[0] if output_texts else ""

        # Clean up repeated substrings (HunyuanOCR has built-in support for this)
        # The model sometimes produces the conversation including the prompt,
        # so extract just the assistant's response
        if prompt_text in generated_text:
            # Find the response after the prompt
            parts = generated_text.split(prompt_text)
            if len(parts) > 1:
                generated_text = parts[-1].strip()

        return generated_text

    def cleanup(self) -> None:
        """Clean up model resources."""
        # Use DeepSeek engine's cleanup if available
        if self._deepseek_engine is not None:
            self._deepseek_engine.cleanup()
            self._deepseek_engine = None
            self.model = None
            self.processor = None
            return

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
