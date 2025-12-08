"""Unified OCR inference module supporting multiple architectures."""

import importlib
import logging
from typing import Any, Dict, Optional
from PIL import Image

logger = logging.getLogger(__name__)


_PADDLEOCR_VL_DTYPE_PATCHED = False
_PADDLEOCR_VL_CLASS = None


def _ensure_paddleocr_vl_dtype_patch() -> bool:
    """
    Ensure PaddleOCR-VL checkpoints in bfloat16 can load on float32 targets.

    Recent PaddleOCR-VL releases publish bfloat16 weights (especially via
    HuggingFace). When PaddleX constructs the DocVLM models on CPU it expects
    float32 tensors and the stock loader does not promote bfloat16 tensors,
    leading to assertions such as:
        AssertionError: Variable dtype not match, Variable [...] need tensor
        with dtype paddle.float32 but load tensor with dtype paddle.bfloat16

    We monkey-patch PaddleX's `_convert_state_dict_dtype_and_shape` helper so
    checkpoints are promoted to the destination dtype when needed.

    Returns:
        bool: True if the patch was applied (or already active), False otherwise.
    """
    global _PADDLEOCR_VL_DTYPE_PATCHED

    if _PADDLEOCR_VL_DTYPE_PATCHED:
        return True

    module_candidates = [
        # PaddleX ≥3.0 ships its loader here.
        "paddlex.inference.models.common.vlm.transformers.model_utils",
        # Older preview builds used PaddleOCR's bundled copy.
        "paddleocr._pipelines.utils.model_utils",
    ]

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        if getattr(module, "_haid_dtype_patch_applied", False):
            _PADDLEOCR_VL_DTYPE_PATCHED = True
            return True

        convert_fn = getattr(module, "_convert_state_dict_dtype_and_shape", None)
        load_fn = getattr(module, "_load_state_dict_into_model", None)
        load_state_fn = getattr(module, "load_state_dict", None)
        if convert_fn is None:
            continue

        try:
            import paddle
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Paddle not available for PaddleOCR-VL dtype patch: %s", exc)
            return False

        def _patched_convert(state_dict, model_to_load, _orig=convert_fn):
            try:
                reference_state = model_to_load.state_dict()
            except Exception:  # pragma: no cover - fallback to original
                return _orig(state_dict, model_to_load)

            for param_name, ref_tensor in reference_state.items():
                candidate = state_dict.get(param_name)
                if candidate is None:
                    continue
                if not isinstance(candidate, paddle.Tensor):
                    continue
                if candidate.dtype != ref_tensor.dtype:
                    try:
                        converted = paddle.cast(candidate, dtype=ref_tensor.dtype)
                    except Exception as cast_err:
                        logger.warning(
                            "PaddleOCR-VL dtype patch failed to cast %s from %s to %s: %s",
                            param_name,
                            candidate.dtype,
                            ref_tensor.dtype,
                            cast_err,
                        )
                    else:
                        if str(candidate.dtype).lower().find("bfloat16") != -1:
                            logger.debug(
                                "PaddleOCR-VL dtype patch promoting %s from %s to %s",
                                param_name,
                                candidate.dtype,
                                ref_tensor.dtype,
                            )
                        state_dict[param_name] = converted
            return _orig(state_dict, model_to_load)

        module._convert_state_dict_dtype_and_shape = _patched_convert
        if load_state_fn is not None:
            orig_load_state = load_state_fn

            def _patched_load_state_dict(*args, _orig=orig_load_state, **kwargs):
                state_dict = _orig(*args, **kwargs)
                try:
                    for key, value in list(state_dict.items()):
                        if not isinstance(value, paddle.Tensor):
                            continue
                        if str(value.dtype).lower().find("bfloat16") != -1:
                            try:
                                state_dict[key] = paddle.cast(value, paddle.float32)
                                logger.debug(
                                    "PaddleOCR-VL dtype patch promoting %s from %s to float32 (load_state)",
                                    key,
                                    value.dtype,
                                )
                            except Exception as cast_err:
                                logger.warning(
                                    "PaddleOCR-VL dtype load_state patch failed to cast %s: %s",
                                    key,
                                    cast_err,
                                )
                except Exception as err:
                    logger.debug(
                        "PaddleOCR-VL dtype load_state patch skipped due to: %s",
                        err,
                    )
                return state_dict

            module.load_state_dict = _patched_load_state_dict
        if load_fn is not None:
            orig_load_fn = load_fn

            def _patched_load(model_to_load, state_dict, start_prefix, *args, _orig=orig_load_fn, **kwargs):
                try:
                    reference_state = model_to_load.state_dict()
                except Exception:
                    reference_state = None

                try:
                    for key, value in list(state_dict.items()):
                        if not isinstance(value, paddle.Tensor):
                            continue
                        target_dtype = None
                        if reference_state and key in reference_state:
                            target_dtype = reference_state[key].dtype
                        if target_dtype is None:
                            target_dtype = paddle.float32
                        if value.dtype != target_dtype:
                            try:
                                if str(value.dtype).lower().find("bfloat16") != -1:
                                    logger.debug(
                                        "PaddleOCR-VL dtype patch promoting %s from %s to %s (load wrapper)",
                                        key,
                                        value.dtype,
                                        target_dtype,
                                    )
                                state_dict[key] = paddle.cast(value, target_dtype)
                            except Exception as cast_err:
                                logger.warning(
                                    "PaddleOCR-VL dtype load wrapper failed to cast %s from %s to %s: %s",
                                    key,
                                    value.dtype,
                                    target_dtype,
                                    cast_err,
                                )
                except Exception as err:
                    logger.debug("PaddleOCR-VL dtype load wrapper skipped due to: %s", err)

                return _orig(model_to_load, state_dict, start_prefix, *args, **kwargs)

            module._load_state_dict_into_model = _patched_load

        module._haid_dtype_patch_applied = True
        _PADDLEOCR_VL_DTYPE_PATCHED = True
        logger.debug(
            "Patched PaddleOCR-VL dtype conversion hook via module '%s'.", module_name
        )
        return True

    logger.debug(
        "Unable to locate PaddleOCR-VL dtype conversion hook; proceeding without patch."
    )
    return False


def _resolve_paddleocr_vl_class():
    """
    Locate the PaddleOCR-VL wrapper class across PaddleOCR releases.

    Returns:
        type: The PaddleOCR-VL class exported by the installed PaddleOCR build.

    Raises:
        ImportError: If the class cannot be located.
    """
    global _PADDLEOCR_VL_CLASS

    if _PADDLEOCR_VL_CLASS is not None:
        return _PADDLEOCR_VL_CLASS

    try:
        module = importlib.import_module("paddleocr")
    except ImportError as exc:
        raise RuntimeError("PaddleOCR is not installed.") from exc

    for attr_name in ("PaddleOCRVL", "DocVLM"):
        candidate = getattr(module, attr_name, None)
        if candidate is not None:
            _PADDLEOCR_VL_CLASS = candidate
            return candidate

    raise RuntimeError(
        "Unable to locate PaddleOCR-VL/DocVLM class in installed paddleocr package."
    )


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
        """Load PaddleOCR model."""
        try:
            # Determine if this is PaddleOCR-VL (Vision-Language model) or legacy PaddleOCR
            # PaddleOCR-VL supports markdown output via save_to_markdown() method
            is_vl_model = "PaddleOCR-VL" in self.model_id or "paddleocr-vl" in self.model_id.lower()

            if is_vl_model:
                patched = _ensure_paddleocr_vl_dtype_patch()
                if patched:
                    logger.debug("PaddleOCR-VL dtype compatibility patch enabled.")

                paddleocr_vl_cls = _resolve_paddleocr_vl_class()
                logger.info(
                    "Loading PaddleOCR-VL model '%s' using %s...",
                    self.model_id,
                    paddleocr_vl_cls.__name__,
                )

                # PaddleOCR-VL models provide built-in multilingual support and
                # expose markdown-friendly outputs via the paddlex pipeline.
                self.model = paddleocr_vl_cls()
                logger.info("PaddleOCR-VL loaded successfully (supports markdown output)")
            else:
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

            # Use hf_model from config if available (for aliased models like DeepSeek-OCR-4bit)
            hf_model_id = self.model_config.get("hf_model") or self.model_id
            logger.info(f"Loading DeepSeek-OCR model '{self.model_id}' (hf_model: {hf_model_id})...")

            # Check for local download at HF standard cache path
            from src.config import get_hf_model_cache_path
            local_model_dir = get_hf_model_cache_path(hf_model_id)

            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                model_path = str(local_model_dir)
                logger.info(f"Using locally downloaded model from {model_path}")
                extra_kwargs = {"local_files_only": True}
            else:
                model_path = hf_model_id
                logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
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

            # Load model in bfloat16 (fits in 8GB VRAM with high utilization)
            # Note: 4-bit quantization has dtype mismatch issues with DeepSeek-OCR
            load_kwargs = {
                "device_map": "auto",
                "_attn_implementation": attn_implementation,
                "use_safetensors": True,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **extra_kwargs,
            }

            self.model = AutoModel.from_pretrained(
                model_path,
                **load_kwargs,
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
        import tempfile
        import os

        # Determine if this is PaddleOCR-VL or legacy PaddleOCR
        is_vl_model = "PaddleOCR-VL" in self.model_id or "paddleocr-vl" in self.model_id.lower()

        # Both APIs expect file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            if is_vl_model:
                # PaddleOCR-VL: Use predict() method which returns result objects
                output = self.model.predict(tmp_path)

                if self.output_format == "markdown":
                    # Extract markdown content from result objects
                    markdown_texts = []
                    for res in output:
                        # Access the markdown property which returns a dict with markdown_texts
                        md_info = res.markdown
                        if isinstance(md_info, dict) and "markdown_texts" in md_info:
                            # markdown_texts can be a list or string
                            md_content = md_info["markdown_texts"]
                            if isinstance(md_content, list):
                                markdown_texts.extend(md_content)
                            else:
                                markdown_texts.append(str(md_content))
                        elif isinstance(md_info, str):
                            markdown_texts.append(md_info)

                    return "\n\n".join(markdown_texts) if markdown_texts else ""
                else:
                    # Text format: Extract text from JSON structure
                    all_text = []
                    for res in output:
                        json_data = res.json
                        if isinstance(json_data, dict):
                            # Try to extract text from common fields
                            if "text" in json_data:
                                all_text.append(json_data["text"])
                            elif "content" in json_data:
                                all_text.append(json_data["content"])
                        elif isinstance(json_data, str):
                            all_text.append(json_data)

                    return "\n".join(all_text) if all_text else ""
            else:
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

                # Use a dedicated temp directory so we can reliably read saved outputs
                import tempfile as _tf
                with _tf.TemporaryDirectory() as td:
                    tmp_img = os.path.join(td, "in.png")
                    image.save(tmp_img)

                    # Reasonable defaults per README: base_size=1024, image_size=640, crop_mode=True
                    res = self.model.infer(
                        self._deepseek_tokenizer,
                        prompt=prompt,
                        image_file=tmp_img,
                        output_path=td,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=True,  # save to tmp dir so we can read if return is empty
                        test_compress=True,
                    )

                    # Prefer return value when it's a non-empty string
                    if isinstance(res, str) and res.strip():
                        return res.strip()

                    # If not a usable string, look for a saved .md/.txt/.mmd in the temp dir
                    # Note: DeepSeek-OCR saves to result.mmd
                    latest_path = None
                    latest_mtime = -1.0
                    for root, _dirs, files in os.walk(td):
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

                    # As a last resort, stringify res if present
                    if res is not None:
                        s = str(res).strip()
                        if s:
                            return s

                    # If we got here, model.infer() ran but produced no usable output
                    raise RuntimeError("DeepSeek-OCR model.infer() returned no usable text output")

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
