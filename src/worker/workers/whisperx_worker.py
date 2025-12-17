"""WhisperX transcription worker with word-level alignment and speaker diarization."""

from __future__ import annotations

import base64
import gc
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("whisperx_worker")


def _decode_audio(audio_data: str) -> Path:
    """Decode base64 audio and save to temporary file."""
    if audio_data.startswith("data:audio"):
        audio_data = audio_data.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    temp_file.write(audio_bytes)
    temp_file.close()
    return Path(temp_file.name)


# Mapping from short model names to full HuggingFace repo IDs
# Must match faster_whisper.utils._MODELS
WHISPER_MODEL_MAPPING = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


class WhisperXWorker(BaseWorker):
    """WhisperX transcription worker with alignment and diarization."""

    task_name = "whisperx"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_id, port, idle_timeout, model_config)
        self._align_cache: Optional[tuple] = None
        self._diar_cache: Optional[Any] = None

    def load_model(self) -> Any:
        """Load WhisperX ASR model."""
        import torch
        import whisperx

        # Workaround for PyTorch 2.6+ torch.load weights_only=True default
        # pyannote-audio checkpoints contain omegaconf objects that aren't safe-loadable
        # See: https://github.com/pyannote/pyannote-audio/issues/1908
        # Solution: Monkey-patch torch.load to default to weights_only=False
        import functools

        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_torch_load(*args, **kwargs):
            # Force weights_only=False for pyannote model compatibility
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

        from src.config import get_hf_endpoint, get_hf_model_cache_path
        from src.db.settings import get_setting

        # Set HuggingFace endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()
        token = get_setting("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        # Determine device and compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        # Enable TF32 for faster inference on Ampere+ GPUs
        if device == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for WhisperX inference")

        # Expand short model name to full HF repo ID for cache lookup
        hf_repo_id = WHISPER_MODEL_MAPPING.get(self.model_id, self.model_id)

        # Check for local model using full HF repo ID
        # HuggingFace cache structure: models--{repo}/snapshots/{commit_hash}/
        # faster-whisper expects a directory containing model.bin directly
        local_dir = get_hf_model_cache_path(hf_repo_id)
        model_source = self.model_id  # Default: let whisperx download

        if local_dir.exists():
            snapshots_dir = local_dir / "snapshots"
            if snapshots_dir.exists():
                # Find the latest snapshot (there's usually only one)
                snapshot_dirs = list(snapshots_dir.iterdir())
                if snapshot_dirs:
                    # Use the first snapshot directory
                    snapshot_path = snapshot_dirs[0]
                    # Verify model.bin exists (as symlink or file)
                    if (snapshot_path / "model.bin").exists():
                        model_source = str(snapshot_path)
                        logger.info(f"Using cached model from {model_source}")

        logger.info(f"Loading WhisperX model '{model_source}' on {device} with {compute_type}")

        model = whisperx.load_model(
            model_source,
            device=device,
            compute_type=compute_type,
            vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
        )

        return model

    def _load_align_model(self, language_code: str, device: str):
        """Load alignment model for the given language."""
        import whisperx
        from src.db.settings import get_setting

        if self._align_cache and self._align_cache[2] == language_code:
            return self._align_cache[0], self._align_cache[1], self._align_cache[3]

        # Get device preference for alignment
        raw_setting = get_setting("whisperx_align_device")
        align_dev_pref = (raw_setting or "cuda").lower()
        align_device = align_dev_pref if align_dev_pref in {"cpu", "cuda"} else "cuda"

        if align_device != device:
            logger.info(f"Loading align model on {align_device} (ASR on {device})")

        align_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=align_device
        )
        self._align_cache = (align_model, metadata, language_code, align_device)
        return align_model, metadata, align_device

    def _load_diar_pipeline(self, device: str):
        """Load diarization pipeline."""
        import whisperx
        from src.db.settings import get_setting

        if self._diar_cache is not None:
            return self._diar_cache

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HuggingFace token is required for diarization. Set 'hf_token' in settings.")

        # Get device preference for diarization
        diar_dev_pref = (get_setting("whisperx_diar_device") or "cuda").lower()
        diar_device = diar_dev_pref if diar_dev_pref in {"cpu", "cuda"} else "cuda"

        if diar_device != device:
            logger.info(f"Loading diarization on {diar_device} (ASR on {device})")

        diar = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=diar_device)

        # Reduce embedding batch size for memory optimization
        if hasattr(diar, 'embedding_batch_size'):
            diar.embedding_batch_size = 8
            logger.info("Set diarization embedding_batch_size=8")

        self._diar_cache = diar
        return diar

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio with WhisperX."""
        import torch
        import whisperx

        audio_data = payload.get("audio", "")
        language = payload.get("language")
        diarization = payload.get("diarization", False)
        batch_size = payload.get("batch_size", 4)
        min_speakers = payload.get("min_speakers")
        max_speakers = payload.get("max_speakers")

        # Decode audio
        audio_path = _decode_audio(audio_data)

        try:
            # Load audio
            audio = whisperx.load_audio(str(audio_path))

            # Get device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Transcribe
            transcribe_kwargs = {"batch_size": batch_size}
            if language:
                transcribe_kwargs["language"] = language

            result = self._model.transcribe(audio, **transcribe_kwargs)
            detected_language = result.get("language") or language or "unknown"

            # Align
            align_model, metadata, align_device = self._load_align_model(detected_language, device)
            aligned = whisperx.align(result["segments"], align_model, metadata, audio, align_device)

            # Cleanup intermediate result
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Diarization
            diar_turns_data = None
            speaker_embeddings_raw = None
            speaker_labels = None

            if diarization:
                diar = self._load_diar_pipeline(device)

                diar_kwargs = {"return_embeddings": True}
                if min_speakers is not None:
                    diar_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diar_kwargs["max_speakers"] = max_speakers

                diar_result = diar(audio, **diar_kwargs)

                if isinstance(diar_result, tuple) and len(diar_result) == 2:
                    diar_segments, speaker_embeddings_raw = diar_result
                else:
                    diar_segments = diar_result
                    speaker_embeddings_raw = None

                # Extract turns
                import pandas as pd
                if isinstance(diar_segments, pd.DataFrame):
                    diar_turns_data = [
                        {"start": row.get("start"), "end": row.get("end"), "speaker": row.get("speaker")}
                        for _, row in diar_segments.iterrows()
                    ]
                else:
                    diar_turns_data = [
                        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
                        for turn, _, speaker in diar_segments.itertracks(yield_label=True)
                    ]

                # Assign speakers to words
                aligned = whisperx.assign_word_speakers(diar_segments, aligned, fill_nearest=True)

                # Fix speaker assignments using diarization ground truth
                corrections = 0
                for seg in aligned.get("segments", []):
                    for i, w in enumerate(seg.get("words", [])):
                        w_start = w.get("start")
                        if w_start is None:
                            continue
                        for turn in diar_turns_data:
                            if turn["start"] <= w_start < turn["end"]:
                                if turn["speaker"] != w.get("speaker"):
                                    seg["words"][i]["speaker"] = turn["speaker"]
                                    corrections += 1
                                break

                if corrections > 0:
                    logger.info(f"Corrected {corrections} speaker assignments")

                if speaker_embeddings_raw is not None:
                    speaker_labels = sorted(diar_segments.labels()) if hasattr(diar_segments, 'labels') else None

                del diar_segments
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Build response segments
            segments_out = []
            full_text_parts = []

            MAX_INTERRUPTION_WORDS = 2
            MAX_INTERRUPTION_DURATION = 1.0
            no_space_languages = {"zh", "ja", "th", "lo", "km", "my"}
            needs_spaces = detected_language not in no_space_languages

            for seg in aligned.get("segments", []):
                if not seg.get("words"):
                    text = seg.get("text", "").strip()
                    full_text_parts.append(text)
                    segments_out.append({
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": text,
                        "speaker": seg.get("speaker"),
                        "words": None,
                    })
                    continue

                words_raw = seg["words"]
                current_speaker = None
                current_words = []
                current_text_parts = []
                current_start = None

                for i, w in enumerate(words_raw):
                    w_speaker = w.get("speaker")
                    w_word = w.get("word", "")
                    w_start = float(w.get("start")) if w.get("start") is not None else None
                    w_end = float(w.get("end")) if w.get("end") is not None else None

                    w_obj = {
                        "word": w_word,
                        "start": w_start,
                        "end": w_end,
                        "speaker": w_speaker,
                    }

                    if current_speaker is None:
                        current_speaker = w_speaker
                        current_start = w_start

                    if w_speaker and w_speaker != current_speaker:
                        # Check for brief interruption
                        interruption_words = 1
                        interruption_duration = (w_end - w_start) if (w_end and w_start) else 0.0

                        for j in range(i + 1, len(words_raw)):
                            next_w = words_raw[j]
                            if next_w.get("speaker") == w_speaker:
                                interruption_words += 1
                                if next_w.get("end"):
                                    interruption_duration = float(next_w["end"]) - w_start
                            else:
                                break

                        if not (interruption_words <= MAX_INTERRUPTION_WORDS and
                                interruption_duration <= MAX_INTERRUPTION_DURATION):
                            # End current segment
                            if current_words:
                                seg_text = (" " if needs_spaces else "").join(current_text_parts).strip()
                                full_text_parts.append(seg_text)
                                segments_out.append({
                                    "start": current_start,
                                    "end": current_words[-1]["end"],
                                    "text": seg_text,
                                    "speaker": current_speaker,
                                    "words": current_words,
                                })

                            current_speaker = w_speaker
                            current_words = [w_obj]
                            current_text_parts = [w_word]
                            current_start = w_start
                            continue

                    current_words.append(w_obj)
                    current_text_parts.append(w_word)

                if current_words:
                    seg_text = (" " if needs_spaces else "").join(current_text_parts).strip()
                    full_text_parts.append(seg_text)
                    segments_out.append({
                        "start": current_start,
                        "end": current_words[-1]["end"],
                        "text": seg_text,
                        "speaker": current_speaker,
                        "words": current_words,
                    })

            # Build speaker profiles
            speakers_out = None
            if speaker_embeddings_raw is not None:
                speaker_stats = {}
                for seg in segments_out:
                    if seg.get("speaker"):
                        spk = seg["speaker"]
                        if spk not in speaker_stats:
                            speaker_stats[spk] = {"duration": 0.0, "count": 0}
                        speaker_stats[spk]["duration"] += (seg["end"] - seg["start"])
                        speaker_stats[spk]["count"] += 1

                speakers_out = []
                speaker_ids = speaker_labels or sorted(speaker_stats.keys())

                for speaker_id in speaker_ids:
                    if speaker_id in speaker_embeddings_raw:
                        embedding_tensor = speaker_embeddings_raw[speaker_id]
                        if hasattr(embedding_tensor, 'cpu'):
                            embedding_list = embedding_tensor.cpu().numpy().tolist()
                        elif hasattr(embedding_tensor, 'tolist'):
                            embedding_list = embedding_tensor.tolist()
                        else:
                            embedding_list = list(embedding_tensor)

                        stats = speaker_stats.get(speaker_id, {"duration": 0.0, "count": 0})
                        speakers_out.append({
                            "speaker_id": speaker_id,
                            "embedding": embedding_list,
                            "total_duration": round(stats["duration"], 3),
                            "segment_count": stats["count"],
                        })

            return {
                "text": " ".join([t for t in full_text_parts if t]),
                "language": detected_language,
                "segments": segments_out,
                "speakers": speakers_out,
                "num_speakers": len(speakers_out) if speakers_out else None,
            }

        finally:
            # Cleanup temp file
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """Clean up resources."""
        import gc

        self._align_cache = None
        self._diar_cache = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        super().cleanup()


# Main entry point
main = create_worker_main(WhisperXWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
