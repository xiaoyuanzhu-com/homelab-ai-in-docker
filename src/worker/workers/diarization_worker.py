"""Speaker diarization worker using pyannote.audio."""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("diarization_worker")


def _decode_audio(audio_data: str) -> Path:
    """Decode base64 audio and save to temporary file."""
    if audio_data.startswith("data:audio"):
        audio_data = audio_data.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    temp_file.write(audio_bytes)
    temp_file.close()
    return Path(temp_file.name)


class DiarizationWorker(BaseWorker):
    """Speaker diarization worker using pyannote."""

    task_name = "speaker-diarization"

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
        """Load pyannote speaker diarization pipeline."""
        import torch

        # Ensure torchcodec is available
        from src.compat import torchcodec_stub
        torchcodec_stub.ensure_torchcodec()

        from pyannote.audio import Pipeline

        from src.config import get_hf_endpoint, get_hf_model_cache_path
        from src.db.catalog import get_model_dict
        from src.db.settings import get_setting

        # Get model config
        self._model_cfg = get_model_dict(self.model_id)
        if self._model_cfg is None:
            raise ValueError(f"Model '{self.model_id}' not found in catalog")

        # Set HuggingFace endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Check for local model
        local_model_dir = get_hf_model_cache_path(self.model_id)
        if local_model_dir.exists() and (local_model_dir / "config.yaml").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            pipeline_kwargs = {"local_files_only": True}
        else:
            model_path = self.model_id
            logger.info(f"Model not found locally, will download from HuggingFace: {model_path}")
            pipeline_kwargs = {}

        # Handle HF token for gated models
        hf_token = (
            get_setting("hf_token")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        if model_path == self.model_id and not hf_token:
            raise ValueError(
                "HuggingFace access token is required. Please set 'hf_token' in settings and accept "
                "conditions at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        if hf_token:
            pipeline_kwargs["token"] = hf_token
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Load pipeline
        pipeline = Pipeline.from_pretrained(model_path, **pipeline_kwargs)

        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        return pipeline

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run speaker diarization."""
        audio_data = payload.get("audio", "")
        num_speakers = payload.get("num_speakers")
        min_speakers = payload.get("min_speakers")
        max_speakers = payload.get("max_speakers")

        # Decode audio
        audio_path = _decode_audio(audio_data)

        try:
            # Build kwargs
            kwargs = {}
            if num_speakers:
                kwargs["num_speakers"] = num_speakers
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

            # Run diarization
            diarization_result = self._model(str(audio_path), **kwargs)

            # Extract annotation from result
            # pyannote>=4 returns DiarizeOutput, older versions return Annotation directly
            diarization_annotation = None
            if hasattr(diarization_result, "speaker_diarization"):
                diarization_annotation = (
                    getattr(diarization_result, "exclusive_speaker_diarization", None)
                    or diarization_result.speaker_diarization
                )
            elif hasattr(diarization_result, "itertracks"):
                diarization_annotation = diarization_result
            else:
                raise ValueError("Unexpected diarization output format")

            # Extract segments
            segments: List[Dict[str, Any]] = []
            speakers = set()

            for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker,
                })
                speakers.add(speaker)

            return {
                "segments": segments,
                "num_speakers": len(speakers),
            }

        finally:
            # Cleanup temp file
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass


# Main entry point
main = create_worker_main(DiarizationWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
