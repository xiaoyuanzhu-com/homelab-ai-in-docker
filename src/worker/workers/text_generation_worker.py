"""Text generation worker using transformers causal LM."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("text_generation_worker")


class TextGenerationWorker(BaseWorker):
    """Text generation inference worker."""

    task_name = "text-generation"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._tokenizer = None
        self._model_cfg: Dict[str, Any] = {}

    def load_model(self) -> Any:
        """Load text generation model."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        from src.config import get_hf_endpoint, get_hf_model_cache_path
        from src.db.catalog import get_model_dict

        # Get model config
        self._model_cfg = get_model_dict(self.model_id)
        if self._model_cfg is None:
            raise ValueError(f"Model '{self.model_id}' not found in catalog")

        # Set HuggingFace endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Check for local model
        local_model_dir = get_hf_model_cache_path(self.model_id)
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            extra_kwargs = {"local_files_only": True}
        else:
            model_path = self.model_id
            logger.info(f"Model not found locally, will download from HuggingFace: {model_path}")
            extra_kwargs = {}

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, **extra_kwargs)

        # Load model
        load_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
            **extra_kwargs,
        }

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")

        return model

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text from prompt."""
        import torch

        prompt = payload.get("prompt", "")
        max_new_tokens = payload.get("max_new_tokens", 256)
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.9)
        do_sample = payload.get("do_sample", True)

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")

        # Move inputs to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode (skip input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][input_length:]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "generated_text": generated_text,
            "tokens_generated": len(generated_tokens),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        super().cleanup()


# Main entry point
main = create_worker_main(TextGenerationWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
