"""Segment Anything (SAM3) worker for promptable image segmentation."""

from __future__ import annotations

import base64
import io
import os
from importlib import resources
from pathlib import Path
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ..base import BaseWorker, create_worker_main
from ..image_utils import decode_image
import logging

logger = logging.getLogger("segmentation_worker")


def _encode_mask_png(mask: torch.Tensor, size: Optional[tuple[int, int]] = None) -> str:
    """Encode a single mask tensor as base64 PNG."""
    mask = mask.detach().cpu()
    if mask.is_floating_point():
        mask = mask > 0.5
    mask = mask.to(torch.uint8)

    while mask.dim() > 2:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            mask = mask[0]

    if mask.dim() == 1 and size and mask.numel() == size[0] * size[1]:
        height, width = size
        mask = mask.view(height, width)

    if mask.dim() != 2:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    mask_img = Image.fromarray((mask.numpy() * 255), mode="L")
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _resolve_bpe_path() -> str:
    """Resolve or download the SAM3 BPE vocab asset."""
    env_path = os.getenv("SAM3_BPE_PATH")
    try:
        from src.db.settings import get_setting

        setting_path = get_setting("sam3_bpe_path")
    except Exception:
        setting_path = None

    if setting_path and Path(setting_path).exists():
        return setting_path
    if env_path and Path(env_path).exists():
        return env_path

    try:
        bpe_path = resources.files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz")
        if bpe_path.is_file():
            return str(bpe_path)
    except Exception:
        pass

    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bpe_path = cache_root / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        urllib.request.urlretrieve(url, bpe_path)
    return str(bpe_path)


def _download_modelscope_file(dest: Path, filename: str, revision: str = "master") -> None:
    """Download a SAM3 file from ModelScope if it does not exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = urllib.parse.quote(filename)
    url = (
        "https://www.modelscope.cn/api/v1/models/facebook/sam3/repo"
        f"?FilePath={encoded_path}&Revision={revision}"
    )
    urllib.request.urlretrieve(url, dest)


def _resolve_modelscope_checkpoint() -> str:
    """Download SAM3 checkpoint from ModelScope and return local path."""
    cache_root = Path(os.getenv("MODELSCOPE_CACHE", Path.home() / ".cache/modelscope"))
    model_dir = cache_root / "facebook" / "sam3"
    ckpt_path = model_dir / "sam3.pt"
    cfg_path = model_dir / "config.json"

    _download_modelscope_file(cfg_path, "config.json")
    _download_modelscope_file(ckpt_path, "sam3.pt")

    if not ckpt_path.exists():
        raise ValueError("Failed to download SAM3 checkpoint from ModelScope")
    return str(ckpt_path)


def _is_hf_access_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "cannot access gated repo" in msg
        or "gatedrepoerror" in msg
        or "401 client error" in msg
        or "403 client error" in msg
        or "forbidden" in msg
        or "unauthorized" in msg
    )


class SegmentAnythingWorker(BaseWorker):
    """SAM3 image segmentation worker."""

    task_name = "image-segmentation"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> Any:
        """Load SAM3 model and processor."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from src.config import get_hf_endpoint, get_hf_token
        from src.db.settings import get_setting

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        checkpoint_path = get_setting("sam3_checkpoint_path") or os.getenv("SAM3_CHECKPOINT_PATH")
        if checkpoint_path:
            checkpoint_path = str(checkpoint_path)
            if not Path(checkpoint_path).exists():
                raise ValueError(f"SAM3 checkpoint not found at '{checkpoint_path}'")

        token = (
            get_hf_token()
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        elif not checkpoint_path:
            try:
                checkpoint_path = _resolve_modelscope_checkpoint()
            except Exception as e:
                raise ValueError(
                    "HuggingFace access token is required for facebook/sam3. "
                    "Set 'hf_token' in settings or export HF_TOKEN/HUGGINGFACE_HUB_TOKEN, "
                    "or provide SAM3_CHECKPOINT_PATH for a local checkpoint. "
                    f"ModelScope download failed: {e}"
                ) from e

        bpe_path = _resolve_bpe_path()
        def _build(checkpoint: Optional[str], use_hf: bool) -> Any:
            return build_sam3_image_model(
                device=self._device,
                bpe_path=bpe_path,
                checkpoint_path=checkpoint,
                load_from_HF=use_hf,
            )

        try:
            model = _build(checkpoint_path, checkpoint_path is None)
        except Exception as e:
            if checkpoint_path is None and _is_hf_access_error(e):
                logger.warning("HF access rejected; falling back to ModelScope for SAM3.")
                checkpoint_path = _resolve_modelscope_checkpoint()
                model = _build(checkpoint_path, False)
            else:
                raise
        self._processor = Sam3Processor(model, device=self._device)
        return model

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAM3 segmentation for a text prompt on an image."""
        if self._processor is None:
            raise RuntimeError("Model not loaded")

        image_data = payload.get("image", "")
        prompt = (payload.get("prompt") or "").strip()
        if not image_data:
            raise ValueError("Missing image payload")
        if not prompt:
            raise ValueError("Prompt is required for segmentation")

        confidence_threshold = payload.get("confidence_threshold")
        if confidence_threshold is not None:
            threshold = float(confidence_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence_threshold must be between 0 and 1")
            self._processor.confidence_threshold = threshold

        max_masks = payload.get("max_masks")
        if max_masks is not None:
            max_masks = int(max_masks)
            if max_masks < 1:
                raise ValueError("max_masks must be >= 1")

        image = decode_image(image_data)
        width, height = image.size

        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(prompt, state)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or masks.numel() == 0:
            return {
                "prompt": prompt,
                "image_width": width,
                "image_height": height,
                "masks": [],
            }

        masks_cpu = masks.detach().cpu()
        boxes_cpu = boxes.detach().cpu() if boxes is not None else None
        scores_cpu = scores.detach().cpu() if scores is not None else None

        mask_count = masks_cpu.shape[0]
        limit = min(mask_count, max_masks) if max_masks is not None else mask_count

        items: List[Dict[str, Any]] = []
        for idx in range(limit):
            mask_png = _encode_mask_png(masks_cpu[idx], size=(height, width))
            score = float(scores_cpu[idx]) if scores_cpu is not None else None
            box = (
                [float(x) for x in boxes_cpu[idx].tolist()]
                if boxes_cpu is not None
                else None
            )
            items.append({"mask": mask_png, "score": score, "box": box})

        return {
            "prompt": prompt,
            "image_width": width,
            "image_height": height,
            "masks": items,
        }


main = create_worker_main(SegmentAnythingWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
