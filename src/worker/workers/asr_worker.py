"""Automatic speech recognition worker using Whisper models."""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("asr_worker")


def _decode_audio(audio_data: str) -> Path:
    """Decode base64 audio and save to temporary file."""
    if audio_data.startswith("data:audio"):
        audio_data = audio_data.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    temp_file.write(audio_bytes)
    temp_file.close()
    return Path(temp_file.name)


class ASRWorker(BaseWorker):
    """Automatic speech recognition worker using Whisper."""

    task_name = "asr"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._processor = None
        self._model_cfg: Dict[str, Any] = {}

    def load_model(self) -> Any:
        """Load Whisper ASR model."""
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

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

        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load processor
        self._processor = AutoProcessor.from_pretrained(model_path, **extra_kwargs)

        # Load model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **extra_kwargs,
        )
        model.to(device, dtype=dtype)

        return model

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio."""
        import torch
        import librosa

        audio_data = payload.get("audio", "")
        language = payload.get("language")
        return_timestamps = payload.get("return_timestamps", False)

        # Decode audio
        audio_path = _decode_audio(audio_data)

        try:
            # Get device and dtype from model
            device = next(self._model.parameters()).device
            torch_dtype = next(self._model.parameters()).dtype

            # Load audio
            audio_array, sampling_rate = librosa.load(str(audio_path), sr=16000)

            # Process
            inputs = self._processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            if hasattr(inputs, "input_features"):
                inputs.input_features = inputs.input_features.to(torch_dtype)

            # Prepare generation kwargs
            generate_kwargs = {"input_features": inputs.input_features}

            if language:
                generate_kwargs["language"] = language

            if return_timestamps:
                generate_kwargs["return_timestamps"] = True

            # Generate
            with torch.no_grad():
                predicted_ids = self._model.generate(**generate_kwargs)

            # Decode
            transcription = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            result = {
                "text": transcription,
                "language": language,
                "chunks": None,
            }

            if return_timestamps and hasattr(predicted_ids, "timestamps"):
                result["chunks"] = predicted_ids.timestamps

            return result

        finally:
            # Cleanup temp file
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._processor is not None:
            del self._processor
            self._processor = None
        super().cleanup()


# Main entry point
main = create_worker_main(ASRWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
