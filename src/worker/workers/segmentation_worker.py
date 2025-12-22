"""Segment Anything (SAM3) worker for promptable image segmentation."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..base import BaseWorker, create_worker_main
from ..image_utils import decode_image
import logging

logger = logging.getLogger("segmentation_worker")

DEFAULT_AUTO_PROMPT = "auto"
AUTO_PROMPT_ALIASES = {
    "auto",
    "all",
    "everything",
    "all objects",
    "all object",
    "segment everything",
}
DEFAULT_POINTS_PER_SIDE = 16
DEFAULT_POINTS_PER_BATCH = 16
DEFAULT_AUTO_IOU_THRESHOLD = 0.6
DEFAULT_AUTO_MIN_AREA_RATIO = 0.001
DEFAULT_AUTO_MAX_MASKS = 64


def _normalize_mask(
    mask: np.ndarray | torch.Tensor, size: Optional[tuple[int, int]] = None
) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()
        if mask.is_floating_point():
            mask = mask > 0.5
        mask = mask.to(torch.uint8).numpy()
    else:
        mask = mask.astype(np.uint8)

    while mask.ndim > 2:
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            mask = mask[0]

    if mask.ndim == 1 and size and mask.size == size[0] * size[1]:
        height, width = size
        mask = mask.reshape(height, width)

    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    return mask.astype(bool)


def _mask_to_rle(
    mask: np.ndarray | torch.Tensor, size: Optional[tuple[int, int]] = None
) -> Dict[str, Any]:
    """Encode a mask as COCO RLE (counts list, column-major order)."""
    mask_np = _normalize_mask(mask, size=size)
    height, width = mask_np.shape
    pixels = mask_np.flatten(order="F").astype(np.uint8)
    counts: List[int] = []
    run_length = 0
    prev = 0
    for pixel in pixels:
        if pixel == prev:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            prev = int(pixel)
    counts.append(run_length)
    return {"size": [height, width], "counts": counts}


def _is_auto_prompt(prompt: str) -> bool:
    return prompt.strip().lower() in AUTO_PROMPT_ALIASES


def _generate_grid_points(points_per_side: int, width: int, height: int) -> np.ndarray:
    """Generate a grid of point prompts in absolute pixel coordinates."""
    if points_per_side < 1:
        raise ValueError("points_per_side must be >= 1")
    x_coords = np.linspace(0.5, max(width - 0.5, 0.5), points_per_side, dtype=np.float32)
    y_coords = np.linspace(0.5, max(height - 0.5, 0.5), points_per_side, dtype=np.float32)
    xs, ys = np.meshgrid(x_coords, y_coords)
    return np.stack([xs, ys], axis=-1).reshape(-1, 2)


def _mask_box(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    return [x0, y0, x1, y1]


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


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


def _find_modelscope_checkpoint() -> Optional[str]:
    """Return local ModelScope checkpoint path if already downloaded."""
    cache_root = Path(os.getenv("MODELSCOPE_CACHE", Path.home() / ".cache/modelscope"))
    ckpt_path = cache_root / "facebook" / "sam3" / "sam3.pt"
    return str(ckpt_path) if ckpt_path.exists() else None


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
        self._auto_predictor = None
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
        else:
            checkpoint_path = _find_modelscope_checkpoint()

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
                enable_inst_interactivity=True,
            )

        use_hf = checkpoint_path is None
        try:
            model = _build(checkpoint_path, use_hf)
        except Exception as e:
            if use_hf and _is_hf_access_error(e):
                logger.warning("HF access rejected; falling back to ModelScope for SAM3.")
                checkpoint_path = _resolve_modelscope_checkpoint()
                model = _build(checkpoint_path, False)
            else:
                raise
        if (
            model.inst_interactive_predictor is not None
            and model.inst_interactive_predictor.model.backbone is None
        ):
            model.inst_interactive_predictor.model.backbone = model.backbone
        self._processor = Sam3Processor(model, device=self._device)
        self._auto_predictor = model.inst_interactive_predictor
        return model

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAM3 segmentation for a text prompt on an image."""
        if self._processor is None:
            raise RuntimeError("Model not loaded")

        image_data = payload.get("image", "")
        prompt = (payload.get("prompt") or "").strip()
        if not image_data:
            raise ValueError("Missing image payload")

        confidence_threshold = payload.get("confidence_threshold")
        threshold = self._processor.confidence_threshold
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

        items: List[Dict[str, Any]] = []
        auto_mode = not prompt or _is_auto_prompt(prompt)
        if auto_mode:
            prompt_label = DEFAULT_AUTO_PROMPT
            if self._auto_predictor is None:
                raise RuntimeError("Automatic mask generation is not available")

            points_per_side = payload.get("points_per_side")
            if points_per_side is None:
                points_per_side = DEFAULT_POINTS_PER_SIDE
            points_per_batch = payload.get("points_per_batch")
            if points_per_batch is None:
                points_per_batch = DEFAULT_POINTS_PER_BATCH
            auto_iou_threshold = payload.get("auto_iou_threshold")
            if auto_iou_threshold is None:
                auto_iou_threshold = DEFAULT_AUTO_IOU_THRESHOLD
            auto_min_area_ratio = payload.get("auto_min_area_ratio")
            if auto_min_area_ratio is None:
                auto_min_area_ratio = DEFAULT_AUTO_MIN_AREA_RATIO

            points_per_side = int(points_per_side)
            if points_per_side < 1:
                raise ValueError("points_per_side must be >= 1")
            points_per_batch = int(points_per_batch)
            if points_per_batch < 1:
                raise ValueError("points_per_batch must be >= 1")
            auto_iou_threshold = float(auto_iou_threshold)
            if not 0.0 <= auto_iou_threshold <= 1.0:
                raise ValueError("auto_iou_threshold must be between 0 and 1")
            auto_min_area_ratio = float(auto_min_area_ratio)
            if not 0.0 <= auto_min_area_ratio <= 1.0:
                raise ValueError("auto_min_area_ratio must be between 0 and 1")

            if max_masks is None:
                max_masks = DEFAULT_AUTO_MAX_MASKS

            self._auto_predictor.set_image(image)
            points = _generate_grid_points(points_per_side, width, height)
            labels = np.ones((points.shape[0], 1), dtype=np.int64)

            def _predict_batch(batch_points: np.ndarray, batch_labels: np.ndarray):
                try:
                    return self._auto_predictor.predict(
                        point_coords=batch_points,
                        point_labels=batch_labels,
                        multimask_output=False,
                        normalize_coords=True,
                    )
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if batch_points.shape[0] <= 1:
                        raise
                    mid = batch_points.shape[0] // 2
                    first = _predict_batch(batch_points[:mid], batch_labels[:mid])
                    second = _predict_batch(batch_points[mid:], batch_labels[mid:])
                    masks_a, scores_a, _ = first
                    masks_b, scores_b, _ = second
                    return (
                        np.concatenate([masks_a, masks_b], axis=0),
                        np.concatenate([scores_a, scores_b], axis=0),
                        None,
                    )

            candidates: List[tuple[float, np.ndarray]] = []
            for start in range(0, points.shape[0], points_per_batch):
                batch_points = points[start : start + points_per_batch]
                batch_labels = labels[start : start + points_per_batch]
                if batch_points.size == 0:
                    continue
                batch_points = batch_points[:, None, :]
                masks, scores, _ = _predict_batch(batch_points, batch_labels)
                if masks.ndim == 3:
                    masks = masks[None, ...]
                if scores.ndim == 1:
                    scores = scores[None, ...]

                batch_size = scores.shape[0]
                for idx in range(batch_size):
                    score_row = scores[idx]
                    if score_row.size == 0:
                        continue
                    best_idx = int(np.argmax(score_row))
                    best_score = float(score_row[best_idx])
                    if best_score < threshold:
                        continue
                    mask = masks[idx, best_idx]
                    if mask.dtype != np.bool_:
                        mask = mask > 0.5
                    area_ratio = float(mask.mean())
                    if area_ratio < auto_min_area_ratio:
                        continue
                    candidates.append((best_score, mask))

            candidates.sort(key=lambda item: item[0], reverse=True)
            selected: List[tuple[float, np.ndarray]] = []
            for score, mask in candidates:
                if auto_iou_threshold > 0:
                    if any(
                        _mask_iou(mask, existing) >= auto_iou_threshold
                        for _, existing in selected
                    ):
                        continue
                selected.append((score, mask))
                if max_masks is not None and len(selected) >= max_masks:
                    break

            for score, mask in selected:
                rle = _mask_to_rle(mask, size=(height, width))
                items.append({"rle": rle, "score": score, "box": _mask_box(mask)})
        else:
            prompt_label = prompt
            state = self._processor.set_image(image)
            output = self._processor.set_text_prompt(prompt, state)
            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")

            if masks is not None and masks.numel() > 0:
                masks_cpu = masks.detach().cpu()
                boxes_cpu = boxes.detach().cpu() if boxes is not None else None
                scores_cpu = scores.detach().cpu() if scores is not None else None

                mask_count = masks_cpu.shape[0]
                limit = min(mask_count, max_masks) if max_masks is not None else mask_count

                for idx in range(limit):
                    rle = _mask_to_rle(masks_cpu[idx], size=(height, width))
                    score = float(scores_cpu[idx]) if scores_cpu is not None else None
                    box = (
                        [float(x) for x in boxes_cpu[idx].tolist()]
                        if boxes_cpu is not None
                        else None
                    )
                    items.append({"rle": rle, "score": score, "box": box})

        return {
            "prompt": prompt_label,
            "image_width": width,
            "image_height": height,
            "masks": items,
        }


main = create_worker_main(SegmentAnythingWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
