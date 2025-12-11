"""Text embedding worker using SentenceTransformers."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("embedding_worker")


class EmbeddingWorker(BaseWorker):
    """Text embedding inference worker."""

    task_name = "embedding"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._batch_size = 32  # Will be calculated based on GPU memory

    def load_model(self) -> Any:
        """Load SentenceTransformer model."""
        from sentence_transformers import SentenceTransformer
        import torch

        from src.config import get_hf_endpoint, get_hf_model_cache_path
        from src.db.catalog import get_model_dict

        # Get model info from catalog
        model_info = get_model_dict(self.model_id)
        if model_info is None:
            raise ValueError(f"Model '{self.model_id}' not found in catalog")

        # Set HuggingFace endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Check for local model
        local_model_dir = get_hf_model_cache_path(self.model_id)
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
        else:
            model_path = self.model_id
            logger.info(f"Model not found locally, will download from HuggingFace: {self.model_id}")

        # Load model with trust_remote_code for models like jina-embeddings-v3
        model = SentenceTransformer(model_path, trust_remote_code=True)

        # Calculate batch size based on GPU memory
        self._batch_size = self._calculate_batch_size(torch)

        return model

    def _calculate_batch_size(self, torch_module) -> int:
        """Calculate optimal batch size based on GPU memory."""
        from src.db.settings import get_setting_int

        memory_per_batch = get_setting_int("embedding_memory_per_batch_mb", 2048)
        max_batch = get_setting_int("embedding_max_batch_size", 32)

        if not torch_module.cuda.is_available():
            return 1

        try:
            device = torch_module.cuda.current_device()
            total_memory_bytes = torch_module.cuda.mem_get_info(device)[1]
            total_memory_mb = total_memory_bytes / (1024 * 1024)

            calculated = int(total_memory_mb / memory_per_batch)
            batch_size = max(1, min(calculated, max_batch))

            logger.info(
                f"Calculated batch_size={batch_size} "
                f"(total_memory={total_memory_mb:.0f}MB, "
                f"memory_per_batch={memory_per_batch}MB)"
            )
            return batch_size
        except Exception as e:
            logger.warning(f"Failed to calculate batch size: {e}, using default=1")
            return 1

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for texts."""
        import torch

        texts: List[str] = payload.get("texts", [])
        if not texts:
            return {"embeddings": [], "dimensions": 0}

        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
        )

        embeddings_list = embeddings.tolist()
        dimensions = len(embeddings_list[0]) if embeddings_list else 0

        # Clean up GPU memory
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "embeddings": embeddings_list,
            "dimensions": dimensions,
        }


# Main entry point
main = create_worker_main(EmbeddingWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
