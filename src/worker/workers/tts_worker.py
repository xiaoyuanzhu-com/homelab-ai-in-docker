"""CosyVoice text-to-speech worker."""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..base import BaseWorker, create_worker_main

logger = logging.getLogger("tts_worker")

# CosyVoice model short names to full HuggingFace repo IDs
MODEL_NAME_MAPPING = {
    # CosyVoice 2.0 models
    "cosyvoice2-0.5b": "FunAudioLLM/CosyVoice2-0.5B",
    "cosyvoice2": "FunAudioLLM/CosyVoice2-0.5B",
    # Fun-CosyVoice 3.0 models (Dec 2025 release)
    # Note: Fun-CosyVoice3-0.5B-2512 is the latest stable version
    "fun-cosyvoice3-0.5b": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "fun-cosyvoice3-0.5b-2512": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "cosyvoice3": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    # Legacy CosyVoice3 (deprecated, use -2512 version)
    "fun-cosyvoice3-0.5b-old": "FunAudioLLM/Fun-CosyVoice3-0.5B",
    # CosyVoice 1.0 models (300M)
    "cosyvoice-300m": "FunAudioLLM/CosyVoice-300M",
    "cosyvoice-300m-sft": "FunAudioLLM/CosyVoice-300M-SFT",
    "cosyvoice-300m-instruct": "FunAudioLLM/CosyVoice-300M-Instruct",
}

# Models that use CosyVoice2 API
COSYVOICE2_MODELS = {
    "FunAudioLLM/CosyVoice2-0.5B",
}

# Models that use CosyVoice3 API (latest)
COSYVOICE3_MODELS = {
    "FunAudioLLM/Fun-CosyVoice3-0.5B",
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}

# GitHub repo URL for CosyVoice
COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"


def _expand_model_name(model_id: str) -> str:
    """Expand short model name to full HuggingFace repo ID."""
    return MODEL_NAME_MAPPING.get(model_id.lower(), model_id)


def _get_model_version(model_id: str) -> int:
    """Get the CosyVoice version (1, 2, or 3) for a model."""
    if model_id in COSYVOICE3_MODELS or "CosyVoice3" in model_id:
        return 3
    if model_id in COSYVOICE2_MODELS or "CosyVoice2" in model_id:
        return 2
    return 1


def _ensure_cosyvoice_repo(data_dir: Path) -> Path:
    """Clone or update the CosyVoice repository and return its path."""
    repo_dir = data_dir / "repos" / "CosyVoice"

    if not repo_dir.exists():
        logger.info(f"Cloning CosyVoice repository to {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--recursive", COSYVOICE_REPO_URL, str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone CosyVoice repo: {result.stderr}")
        logger.info("CosyVoice repository cloned successfully")
    else:
        logger.info(f"Using existing CosyVoice repository at {repo_dir}")

    # Ensure submodules are initialized (required for matcha module)
    matcha_dir = repo_dir / "third_party" / "Matcha-TTS"
    matcha_module = matcha_dir / "matcha"
    if not matcha_module.exists():
        logger.info("Initializing git submodules (matcha module not found)...")
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to init submodules: {result.stderr}")
        logger.info("Git submodules initialized successfully")

    # Add CosyVoice to Python path if not already there
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        logger.info(f"Added {repo_str} to Python path")

    # Also add third_party/Matcha-TTS if it exists
    matcha_dir = repo_dir / "third_party" / "Matcha-TTS"
    if matcha_dir.exists():
        matcha_str = str(matcha_dir)
        if matcha_str not in sys.path:
            sys.path.insert(0, matcha_str)
            logger.info(f"Added {matcha_str} to Python path")

    return repo_dir


class TTSWorker(BaseWorker):
    """CosyVoice text-to-speech worker."""

    task_name = "tts"

    def __init__(
        self,
        model_id: str,
        port: int,
        idle_timeout: int = 60,
        model_config: Dict[str, Any] | None = None,
    ):
        # Expand short model names before calling parent
        expanded_model_id = _expand_model_name(model_id)
        super().__init__(expanded_model_id, port, idle_timeout, model_config)
        self._model_cfg: Dict[str, Any] = {}
        self._cosyvoice = None
        self._model_version = 1  # 1, 2, or 3

    def load_model(self) -> Any:
        """Load CosyVoice model."""
        from src.config import get_data_dir, get_hf_endpoint
        from src.db.catalog import get_model_dict

        # Set HuggingFace endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        # Ensure CosyVoice repo is cloned and in path
        data_dir = get_data_dir()
        _ensure_cosyvoice_repo(data_dir)

        # Get model config from catalog (optional)
        self._model_cfg = get_model_dict(self.model_id) or {}

        # Determine model path
        hf_model = self._model_cfg.get("hf_model") or self.model_id
        logger.info(f"Loading CosyVoice model: {hf_model}")

        # Determine CosyVoice version (1, 2, or 3)
        self._model_version = _get_model_version(hf_model)
        logger.info(f"Detected CosyVoice version: {self._model_version}")

        # Download model if needed
        from huggingface_hub import snapshot_download

        models_dir = data_dir / "models"
        model_local_dir = models_dir / "cosyvoice" / hf_model.replace("/", "--")

        if not model_local_dir.exists():
            logger.info(f"Downloading model to {model_local_dir}")
            snapshot_download(
                repo_id=hf_model,
                local_dir=str(model_local_dir),
            )
        else:
            logger.info(f"Using cached model at {model_local_dir}")

        # Now import and load the model (after repo is in path)
        try:
            disable_ttsfrd = os.getenv("COSYVOICE_DISABLE_TTSFRD", "1") == "1"
            if disable_ttsfrd:
                # Force wetext fallback to match upstream demo default.
                sys.modules["ttsfrd"] = None
                logger.info("CosyVoice ttsfrd disabled; using wetext for text normalization")

            # Use AutoModel to auto-detect model type from config files
            from cosyvoice.cli.cosyvoice import AutoModel
            self._cosyvoice = AutoModel(model_dir=str(model_local_dir))
            logger.info(f"CosyVoice model loaded via AutoModel: {self.model_id}")

            return self._cosyvoice

        except Exception as e:
            logger.error(f"CosyVoice load failed: {e}")
            raise RuntimeError(
                f"Failed to load CosyVoice model '{hf_model}'. "
                f"Ensure cosyvoice dependencies are installed correctly. "
                f"Error: {e}"
            )

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speech from text using CosyVoice."""
        import torchaudio

        text = payload.get("text", "")
        if not text:
            raise ValueError("Text is required for TTS")

        # TTS mode configuration
        mode = payload.get("mode", "zero_shot")  # zero_shot, sft, instruct, cross_lingual
        prompt_text = payload.get("prompt_text", "")
        prompt_audio = payload.get("prompt_audio")  # Base64 encoded audio
        instruction = payload.get("instruction", "")
        speaker_id = payload.get("speaker_id")  # For SFT mode
        speed = payload.get("speed", 1.0)

        logger.info(
            "TTS infer inputs: mode=%s model_version=%s text_len=%d prompt_text_len=%d "
            "prompt_audio=%s instruction_len=%d speaker_id=%s speed=%.2f",
            mode,
            self._model_version,
            len(text),
            len(prompt_text or ""),
            "yes" if prompt_audio else "no",
            len(instruction or ""),
            speaker_id or "",
            speed,
        )

        # Decode prompt audio if provided
        prompt_audio_path = None
        if prompt_audio:
            # Extract MIME type and base64 data
            mime_type = "audio/wav"
            if prompt_audio.startswith("data:audio"):
                # Parse data URL: data:audio/m4a;base64,XXXX
                header, prompt_audio = prompt_audio.split(",", 1)
                if "/" in header and ";" in header:
                    mime_type = header.split(":")[1].split(";")[0]

            audio_bytes = base64.b64decode(prompt_audio)

            # Determine file extension from MIME type
            ext_map = {
                "audio/mp4": ".m4a",
                "audio/x-m4a": ".m4a",
                "audio/m4a": ".m4a",
                "audio/mpeg": ".mp3",
                "audio/mp3": ".mp3",
                "audio/webm": ".webm",
                "audio/ogg": ".ogg",
                "audio/wav": ".wav",
                "audio/x-wav": ".wav",
            }
            input_ext = ext_map.get(mime_type, ".wav")

            # Save original audio to temp file
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=input_ext)
            temp_input.write(audio_bytes)
            temp_input.close()
            logger.info(
                "Prompt audio received: mime=%s ext=%s bytes=%d",
                mime_type,
                input_ext,
                len(audio_bytes),
            )

            # Convert non-WAV formats to WAV using ffmpeg
            if input_ext != ".wav":
                logger.info(f"Converting {input_ext} audio to WAV format using ffmpeg")
                try:
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_wav.close()
                    # Use ffmpeg to convert to 16-bit PCM WAV
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", temp_input.name,
                            "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
                            temp_wav.name
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        # Clean up original temp file
                        os.unlink(temp_input.name)
                        prompt_audio_path = temp_wav.name
                        logger.info("Audio converted successfully to 24kHz mono WAV")
                    else:
                        logger.warning(f"ffmpeg conversion failed: {result.stderr}")
                        os.unlink(temp_wav.name)
                        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
                except FileNotFoundError:
                    logger.error("ffmpeg not found - cannot convert audio format")
                    raise ValueError(
                        f"Cannot convert {input_ext} audio. Please upload a WAV file, "
                        "or ensure ffmpeg is installed on the server."
                    )
            else:
                prompt_audio_path = temp_input.name
            if prompt_audio_path:
                try:
                    audio_info = torchaudio.info(prompt_audio_path)
                    duration_seconds = (
                        audio_info.num_frames / audio_info.sample_rate
                        if audio_info.sample_rate
                        else 0.0
                    )
                    logger.info(
                        "Prompt audio info: sr=%s frames=%s channels=%s duration=%.2fs",
                        audio_info.sample_rate,
                        audio_info.num_frames,
                        audio_info.num_channels,
                        duration_seconds,
                    )
                except Exception as exc:
                    logger.warning("Prompt audio info failed: %s", exc)

        try:
            # Generate speech based on mode
            audio_segments = []

            if mode == "zero_shot":
                # Zero-shot voice cloning
                if not prompt_audio_path:
                    raise ValueError("prompt_audio is required for zero_shot mode")

                # CosyVoice3 requires prompt_text with the exact transcript of reference audio
                # Format: "You are a helpful assistant.<|endofprompt|>transcript_of_reference_audio"
                # Without proper transcript, output will be garbled/unintelligible
                effective_prompt_text = prompt_text or ""
                if self._model_version == 3:
                    if not effective_prompt_text.strip():
                        raise ValueError(
                            "prompt_text is required for CosyVoice3 zero_shot mode. "
                            "Provide the exact transcript of the reference audio, or use "
                            "cross_lingual mode if you don't have the transcript."
                        )
                    if "<|endofprompt|>" not in effective_prompt_text:
                        # Add required prefix for CosyVoice3
                        effective_prompt_text = f"You are a helpful assistant.<|endofprompt|>{effective_prompt_text}"

                for result in self._cosyvoice.inference_zero_shot(
                    text,
                    effective_prompt_text,
                    prompt_audio_path,
                    speed=speed,
                ):
                    audio_segments.append(result["tts_speech"])

            elif mode == "cross_lingual":
                # Cross-lingual synthesis (use reference audio for voice)
                # Does NOT require transcript of reference audio
                if not prompt_audio_path:
                    raise ValueError("prompt_audio is required for cross_lingual mode")

                # CosyVoice3 requires prefix in the tts_text itself
                effective_text = text
                if self._model_version == 3:
                    if "<|endofprompt|>" not in effective_text:
                        effective_text = f"You are a helpful assistant.<|endofprompt|>{text}"

                for result in self._cosyvoice.inference_cross_lingual(
                    effective_text,
                    prompt_audio_path,
                    speed=speed,
                ):
                    audio_segments.append(result["tts_speech"])

            elif mode == "instruct" or mode == "instruct2":
                # Instruction-based synthesis
                if not instruction:
                    raise ValueError("instruction is required for instruct mode")

                # CosyVoice2/3 use inference_instruct2 which requires prompt audio
                # CosyVoice1 uses inference_instruct with speaker_id
                if self._model_version >= 2:
                    if not prompt_audio_path:
                        raise ValueError(
                            "prompt_audio is required for instruct mode with CosyVoice2/3. "
                            "Please provide a reference audio sample for voice cloning."
                        )
                    # CosyVoice2/3 expect <|endofprompt|> suffix; CosyVoice3 also uses a system prompt prefix.
                    effective_instruction = instruction.strip()
                    if "<|endofprompt|>" not in effective_instruction:
                        if self._model_version == 3:
                            if "you are a helpful assistant" not in effective_instruction.lower():
                                effective_instruction = (
                                    f"You are a helpful assistant. {effective_instruction}"
                                )
                        effective_instruction = f"{effective_instruction}<|endofprompt|>"
                    for result in self._cosyvoice.inference_instruct2(
                        text,
                        effective_instruction,
                        prompt_audio_path,
                        speed=speed,
                    ):
                        audio_segments.append(result["tts_speech"])
                else:
                    # CosyVoice1 inference_instruct uses pre-trained speakers
                    for result in self._cosyvoice.inference_instruct(
                        text,
                        speaker_id or "中文女",
                        instruction,
                        speed=speed,
                    ):
                        audio_segments.append(result["tts_speech"])

            elif mode == "sft":
                # SFT mode with pre-trained speaker
                if hasattr(self._cosyvoice, "inference_sft"):
                    for result in self._cosyvoice.inference_sft(
                        text,
                        speaker_id or "中文女",
                        speed=speed,
                    ):
                        audio_segments.append(result["tts_speech"])
                else:
                    raise ValueError(f"SFT mode not supported for model: {self.model_id}")

            else:
                raise ValueError(f"Unknown TTS mode: {mode}")

            # Concatenate audio segments
            import torch
            if audio_segments:
                audio_tensor = torch.cat(audio_segments, dim=1)
            else:
                raise RuntimeError("No audio generated")

            # Get sample rate from model
            # CosyVoice2/3 uses 24000 Hz, CosyVoice1 uses 22050 Hz
            default_sample_rate = 24000 if self._model_version >= 2 else 22050
            sample_rate = getattr(self._cosyvoice, "sample_rate", default_sample_rate)

            # Convert to WAV bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            return {
                "audio": f"data:audio/wav;base64,{audio_base64}",
                "sample_rate": sample_rate,
                "duration_seconds": audio_tensor.shape[1] / sample_rate,
                "mode": mode,
            }

        finally:
            # Cleanup temp file
            if prompt_audio_path and os.path.exists(prompt_audio_path):
                try:
                    os.unlink(prompt_audio_path)
                except Exception:
                    pass


# Main entry point
main = create_worker_main(TTSWorker)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
