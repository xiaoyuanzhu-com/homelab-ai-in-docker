"""FunASR speech recognition worker."""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("funasr_worker")

# URL for Fun-ASR-Nano model.py (required for FunASRNano class registration)
FUN_ASR_NANO_MODEL_PY_URL = (
    "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/model.py"
)


def _decode_audio(audio_data: str) -> Path:
    """Decode base64 audio and save to temporary file."""
    if audio_data.startswith("data:audio"):
        audio_data = audio_data.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(audio_bytes)
    temp_file.close()
    return Path(temp_file.name)


class FunASRWorker(BaseWorker):
    """FunASR speech recognition worker."""

    task_name = "funasr"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._model_cfg: Dict[str, Any] = {}

    def load_model(self) -> Any:
        """Load FunASR model."""
        from funasr import AutoModel

        from src.config import get_hf_endpoint
        from src.db.catalog import get_model_dict

        # Set HuggingFace endpoint for models from HF
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Get model config from catalog (optional - allows custom config)
        self._model_cfg = get_model_dict(self.model_id) or {}

        # Determine model source
        # FunASR AutoModel can load from:
        # - HuggingFace hub (hf_model from catalog)
        # - ModelScope (modelscope_model from catalog)
        # - Direct model ID (HuggingFace or ModelScope format)
        hf_model = self._model_cfg.get("hf_model")
        ms_model = self._model_cfg.get("modelscope_model")

        if hf_model:
            # Load from HuggingFace (catalog entry)
            model_path = hf_model
            logger.info(f"Loading FunASR model from HuggingFace: {model_path}")
        elif ms_model:
            # Load from ModelScope (catalog entry)
            model_path = ms_model
            logger.info(f"Loading FunASR model from ModelScope: {model_path}")
        else:
            # Use model_id directly - FunASR can auto-detect source
            # Supports: HuggingFace (org/model), ModelScope (org/model), local paths
            model_path = self.model_id
            logger.info(f"Loading FunASR model: {model_path}")

        # Build AutoModel kwargs
        model_kwargs: Dict[str, Any] = {
            "model": model_path,
            "device": "cuda:0",  # Use GPU
            "disable_update": True,  # Don't check for updates
        }

        # Fun-ASR-Nano and similar models require trust_remote_code
        # Check if model needs remote code execution
        if self._needs_remote_code(model_path):
            model_kwargs["trust_remote_code"] = True
            logger.info(f"Enabling trust_remote_code for model: {model_path}")

            # Fun-ASR-Nano needs model.py downloaded from GitHub
            # The file must be in the model directory for FunASR to find it
            if "Fun-ASR-Nano" in model_path:
                # First, trigger the model download to get the model directory
                from funasr.download.download_model_from_hub import download_model

                download_kwargs = download_model(model=model_path, hub="ms")
                model_dir = download_kwargs.get("model_path", "")
                if model_dir and os.path.isdir(model_dir):
                    # Download model.py to the model directory
                    model_py_path = self._ensure_fun_asr_nano_model_code(model_dir)
                    # Point remote_code to the model.py file
                    model_kwargs["remote_code"] = model_py_path
                    logger.info(f"Using remote_code: {model_py_path}")

        # Optional VAD support from model_config
        vad_model = self._model_cfg.get("vad_model") or (
            self.model_config.get("vad_model") if self.model_config else None
        )
        if vad_model:
            model_kwargs["vad_model"] = vad_model
            vad_kwargs = self._model_cfg.get("vad_kwargs") or (
                self.model_config.get("vad_kwargs") if self.model_config else None
            )
            if vad_kwargs:
                model_kwargs["vad_kwargs"] = vad_kwargs
            logger.info(f"Using VAD model: {vad_model}")

        # Load model with FunASR AutoModel
        # AutoModel handles device placement automatically
        model = AutoModel(**model_kwargs)

        logger.info(f"FunASR model loaded: {self.model_id}")
        return model

    def _needs_remote_code(self, model_path: str) -> bool:
        """Check if model requires trust_remote_code."""
        # Models that need remote code execution
        remote_code_models = [
            "Fun-ASR-Nano",
            "FunAudioLLM/Fun-ASR",
        ]
        return any(pattern in model_path for pattern in remote_code_models)

    def _ensure_fun_asr_nano_model_code(self, model_dir: str) -> str:
        """
        Ensure Fun-ASR-Nano model.py exists in the model directory.

        Fun-ASR-Nano requires a model.py file that defines the FunASRNano class,
        but this file is not included in the ModelScope download. We need to
        download it from the Fun-ASR GitHub repo.

        Returns the path to the model.py file.
        """
        model_py_path = Path(model_dir) / "model.py"

        if model_py_path.exists():
            logger.info(f"Fun-ASR-Nano model.py already exists: {model_py_path}")
            return str(model_py_path)

        logger.info(f"Downloading Fun-ASR-Nano model.py from GitHub...")
        try:
            urllib.request.urlretrieve(FUN_ASR_NANO_MODEL_PY_URL, model_py_path)
            logger.info(f"Downloaded model.py to: {model_py_path}")
            return str(model_py_path)
        except Exception as e:
            logger.error(f"Failed to download model.py: {e}")
            raise RuntimeError(
                f"Failed to download Fun-ASR-Nano model.py from {FUN_ASR_NANO_MODEL_PY_URL}: {e}"
            )

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio using FunASR."""
        audio_data = payload.get("audio", "")
        language = payload.get("language")

        # Decode audio
        audio_path = _decode_audio(audio_data)

        try:
            # Run inference
            # FunASR generate() returns a list of results
            result = self._model.generate(
                input=str(audio_path),
                batch_size_s=300,  # Process up to 300s at once
            )

            # Extract transcription from result
            # Result format varies by model, but typically has 'text' key
            if result and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    text = first_result.get("text", "")
                    # SenseVoice may include emotion/event tags
                    # Format: <|LANG|><|EMOTION|><|EVENT|>text
                    # We keep the full output for now
                else:
                    text = str(first_result)
            else:
                text = ""

            return {
                "text": text,
                "language": language,
                "chunks": None,
            }

        finally:
            # Cleanup temp file
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass


# Main entry point
main = create_worker_main(FunASRWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
