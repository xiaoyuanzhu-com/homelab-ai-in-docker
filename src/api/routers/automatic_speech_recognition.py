"""Automatic speech recognition API router for transcribing audio files."""

import base64
import io
import logging
import os
import tempfile
import time
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException
import asyncio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)
import torch

from ..models.automatic_speech_recognition import TranscriptionRequest, TranscriptionResponse
from ...storage.history import history_storage
from ...config import get_model_cache_dir
from ...db.skills import get_skill_dict, list_skills

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["automatic-speech-recognition"])

# Global model cache
_model_cache: Optional[Any] = None
_processor_cache: Optional[Any] = None
_current_model_name: str = ""
_current_model_config: Optional[Dict[str, Any]] = None
_last_access_time: Optional[float] = None
_idle_cleanup_task: Optional[asyncio.Task] = None


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get skill configuration from database.

    Args:
        model_id: Skill identifier

    Returns:
        Skill configuration dictionary

    Raises:
        ValueError: If skill not found in database
    """
    skill = get_skill_dict(model_id)

    if skill is None:
        raise ValueError(f"Skill '{model_id}' not found in database")

    return skill


def get_available_models() -> list[str]:
    """
    Load available ASR skills from the database.

    Returns:
        List of skill IDs that can be used (both Whisper and pyannote)
    """
    asr_skills = list_skills(task="automatic-speech-recognition")
    return [skill["id"] for skill in asr_skills]


def validate_model(model_name: str) -> None:
    """
    Validate that the skill is supported.

    Args:
        model_name: Skill identifier to validate

    Raises:
        ValueError: If skill is not supported
    """
    available = get_available_models()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {', '.join(available)}"
        )


def check_and_cleanup_idle_model():
    """Check if model has been idle too long and cleanup if needed."""
    global _model_cache, _last_access_time, _current_model_name

    if _model_cache is None or _last_access_time is None:
        return

    # Get idle timeout from settings
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    # Check if model has been idle too long
    idle_duration = time.time() - _last_access_time
    if idle_duration >= idle_timeout:
        logger.info(
            f"ASR model '{_current_model_name}' idle for {idle_duration:.1f}s "
            f"(timeout: {idle_timeout}s), unloading from GPU..."
        )
        # Preserve name for accurate post-cleanup logging
        unloaded_name = _current_model_name
        cleanup()
        logger.info(f"ASR model '{unloaded_name}' unloaded from GPU")


def schedule_idle_cleanup() -> None:
    """Schedule background cleanup after idle timeout.

    Ensures model unload even if no further transcription requests arrive.
    """
    global _idle_cleanup_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    if _idle_cleanup_task and not _idle_cleanup_task.done():
        _idle_cleanup_task.cancel()

    async def _watchdog(timeout_s: int):
        try:
            await asyncio.sleep(timeout_s)
            if _last_access_time is None:
                return
            idle_duration = time.time() - _last_access_time
            if idle_duration >= timeout_s and _model_cache is not None:
                logger.info(
                    f"ASR model '{_current_model_name}' idle for {idle_duration:.1f}s "
                    f"(timeout: {timeout_s}s), unloading from GPU..."
                )
                cleanup()
        except asyncio.CancelledError:
            pass
        finally:
            # Clear task reference if it's the current task
            global _idle_cleanup_task
            try:
                current = asyncio.current_task()
            except Exception:
                current = None
            if current is not None and _idle_cleanup_task is current:
                _idle_cleanup_task = None

    _idle_cleanup_task = loop.create_task(_watchdog(idle_timeout))


def get_model(model_name: str):
    """
    Get or load the ASR model using Auto classes.

    Args:
        model_name: Model identifier to load. If different from currently
                   loaded model, will reload with the new model.

    Returns:
        Tuple of (processor, model, model_config)

    Raises:
        ValueError: If model is not supported
    """
    global _model_cache, _processor_cache, _current_model_name, _current_model_config, _last_access_time

    # Check if current model should be cleaned up due to idle timeout
    check_and_cleanup_idle_model()

    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Check if we need to reload the model
    if model_name != _current_model_name:
        # Clear existing cache
        if _model_cache is not None:
            del _model_cache
            _model_cache = None
        if _processor_cache is not None:
            del _processor_cache
            _processor_cache = None
        _current_model_name = model_name
        _current_model_config = model_config

    if _model_cache is None or _processor_cache is None:
        # Get custom cache directory
        cache_dir = get_model_cache_dir("automatic-speech-recognition", _current_model_name)

        # Check if model is already downloaded locally via hfd
        from ...config import get_data_dir
        local_model_dir = get_data_dir() / "models" / _current_model_name

        # Determine which path to use for loading
        if local_model_dir.exists() and (local_model_dir / "config.json").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            # Use local_files_only to prevent re-downloading
            extra_kwargs = {"local_files_only": True}
        else:
            model_path = _current_model_name
            logger.info(f"Model not found locally, will download from HuggingFace: {_current_model_name}")
            extra_kwargs = {}

        # Set HuggingFace endpoint for model loading
        from ...config import get_hf_endpoint
        os.environ["HF_ENDPOINT"] = get_hf_endpoint()

        try:
            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Load processor
            _processor_cache = AutoProcessor.from_pretrained(
                model_path,
                **extra_kwargs
            )

            # Load model with dtype parameter (not torch_dtype which is deprecated)
            _model_cache = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                attn_implementation="sdpa",  # Use scaled dot product attention for better performance
                low_cpu_mem_usage=True,
                use_safetensors=True,
                **extra_kwargs
            )
            _model_cache.to(device, dtype=dtype)

        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    # Update last access time
    _last_access_time = time.time()

    return _processor_cache, _model_cache, _current_model_config


def cleanup():
    """
    Release model and processor resources immediately.
    Forces GPU memory cleanup to free resources for other services.
    """
    global _model_cache, _processor_cache, _current_model_name, _current_model_config, _last_access_time

    model_name = _current_model_name  # Save for logging

    if _model_cache is not None:
        # Move model to CPU first (helps with cleanup)
        try:
            if hasattr(_model_cache, 'cpu'):
                _model_cache.cpu()
                logger.debug(f"Moved ASR model '{model_name}' to CPU")
        except Exception as e:
            logger.warning(f"Error moving model to CPU during cleanup: {e}")

        # Remove reference
        del _model_cache
        _model_cache = None

    if _processor_cache is not None:
        del _processor_cache
        _processor_cache = None

    _current_model_name = ""
    _current_model_config = None
    _last_access_time = None

    # Force GPU memory release
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to finish
            logger.debug("GPU cache cleared and synchronized for ASR model")
    except Exception as e:
        logger.warning(f"Error releasing GPU memory: {e}")


def decode_audio(audio_data: str) -> Path:
    """
    Decode base64 audio and save to temporary file.

    Args:
        audio_data: Base64-encoded audio string

    Returns:
        Path to temporary audio file
    """
    try:
        # Remove data URL prefix if present
        if audio_data.startswith('data:audio'):
            audio_data = audio_data.split(',')[1]

        audio_bytes = base64.b64decode(audio_data)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.audio')
        temp_file.write(audio_bytes)
        temp_file.close()

        return Path(temp_file.name)
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {str(e)}")


@router.post("/automatic-speech-recognition", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest) -> TranscriptionResponse:
    """
    Perform automatic speech recognition on audio.

    Supports both transcription (Whisper models) and speaker diarization (pyannote models).

    Args:
        request: ASR request parameters

    Returns:
        Transcription or diarization results based on model type

    Raises:
        HTTPException: If processing fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    audio_path = None

    try:
        # Route to appropriate processing based on output_format
        if request.output_format == "diarization":
            # Speaker diarization processing
            return await _process_diarization(request, request_id, start_time)
        else:
            # Whisper transcription processing
            return await _process_transcription(request, request_id, start_time)

    except ValueError as e:
        error_msg = str(e)
        code = "INVALID_MODEL" if "Model" in error_msg or "model" in error_msg else "INVALID_AUDIO"
        logger.warning(f"{code} for request {request_id}: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail={"code": code, "message": error_msg, "request_id": request_id},
        )
    except Exception as e:
        error_msg = f"Failed to process audio: {str(e)}"
        logger.error(f"Processing failed for request {request_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"code": "PROCESSING_FAILED", "message": error_msg, "request_id": request_id},
        )


async def _process_transcription(request: TranscriptionRequest, request_id: str, start_time: float) -> TranscriptionResponse:
    """Process Whisper transcription."""
    audio_path = None
    try:
        # Decode audio off the event loop
        audio_path = await asyncio.to_thread(decode_audio, request.audio)

        # Load model and get config off the event loop
        processor, model, model_config = await asyncio.to_thread(get_model, request.model)

        # Run the full preprocessing + generation pipeline in a worker thread
        def _run_inference():
            # Determine device and dtype from model
            device = next(model.parameters()).device
            torch_dtype = next(model.parameters()).dtype

            # Load audio with processor
            import librosa
            audio_array, sampling_rate = librosa.load(str(audio_path), sr=16000)

            # Process audio
            inputs = processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )

            # Move inputs to device and convert to model dtype
            inputs = inputs.to(device)
            # Convert input_features to match model dtype
            if hasattr(inputs, 'input_features'):
                inputs.input_features = inputs.input_features.to(torch_dtype)

            # Prepare generation kwargs
            generate_kwargs = {
                "input_features": inputs.input_features,
            }

            # Add language if specified
            if request.language:
                generate_kwargs["language"] = request.language

            # Add timestamp return option
            if request.return_timestamps:
                generate_kwargs["return_timestamps"] = True

            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(**generate_kwargs)

            # Decode transcription
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            # Extract language if it was auto-detected
            detected_language = None
            if hasattr(predicted_ids, 'sequences'):
                # For models that return language information
                try:
                    # This is model-specific and may need adjustment
                    detected_language = request.language
                except:
                    pass

            result = {
                "text": transcription,
                "language": detected_language or request.language,
                "chunks": None
            }

            # Handle timestamps if requested
            if request.return_timestamps and hasattr(predicted_ids, 'timestamps'):
                result["chunks"] = predicted_ids.timestamps

            return result

        result = await asyncio.to_thread(_run_inference)

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = TranscriptionResponse(
            request_id=request_id,
            text=result["text"],
            model=request.model,
            language=result["language"],
            chunks=result["chunks"],
            processing_time_ms=processing_time_ms,
        )

        # Save to history (exclude audio data to save space)
        history_storage.add_request(
            service="automatic-speech-recognition",
            request_id=request_id,
            request_data={
                "model": request.model,
                "language": request.language,
                "return_timestamps": request.return_timestamps,
            },
            response_data=response.model_dump(),
            status="success",
        )

        # Schedule background idle cleanup after request completes
        schedule_idle_cleanup()
        return response

    finally:
        # Clean up temporary audio file
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary audio file: {e}")


async def _process_diarization(request: TranscriptionRequest, request_id: str, start_time: float) -> TranscriptionResponse:
    """Process pyannote speaker diarization."""
    audio_path = None
    try:
        # Decode audio off the event loop
        audio_path = await asyncio.to_thread(decode_audio, request.audio)

        # Load pyannote pipeline
        from ...compat import torchcodec_stub

        torchcodec_stub.ensure_torchcodec()
        from pyannote.audio import Pipeline
        from ...db.settings import get_setting
        from ...config import get_data_dir, get_hf_endpoint

        # Check for local model
        local_model_dir = get_data_dir() / "models" / request.model
        if local_model_dir.exists() and (local_model_dir / "config.yaml").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            # For local path, don't pass use_auth_token
            pipeline_kwargs = {}
        else:
            model_path = request.model
            pipeline_kwargs = {}
            logger.info("Model not found locally, will use HuggingFace hub")

        os.environ["HF_ENDPOINT"] = get_hf_endpoint()
        hf_token = get_setting("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if model_path == request.model and not hf_token:
            raise ValueError(
                "HuggingFace access token is required. Please set 'hf_token' in settings and accept "
                "conditions at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        if hf_token:
            pipeline_kwargs.setdefault("token", hf_token)
            # Ensure downstream libraries see the token (pyannote uses huggingface_hub directly)
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Load and run pipeline
        def _run_diarization():
            pipeline = Pipeline.from_pretrained(model_path, **pipeline_kwargs)
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            # Build kwargs
            kwargs = {}
            if request.num_speakers:
                kwargs["num_speakers"] = request.num_speakers
            if request.min_speakers:
                kwargs["min_speakers"] = request.min_speakers
            if request.max_speakers:
                kwargs["max_speakers"] = request.max_speakers

            # Run diarization
            return pipeline(str(audio_path), **kwargs)

        diarization_result = await asyncio.to_thread(_run_diarization)

        processing_time_ms = int((time.time() - start_time) * 1000)

        # pyannote>=4 returns DiarizeOutput with annotations attached to attributes.
        # For older versions the pipeline itself is an Annotation.
        diarization_annotation = None
        if hasattr(diarization_result, "speaker_diarization"):
            diarization_annotation = getattr(diarization_result, "exclusive_speaker_diarization", None) or diarization_result.speaker_diarization
        elif hasattr(diarization_result, "annotations") and callable(getattr(diarization_result, "itertracks", None)):
            diarization_annotation = diarization_result
        elif callable(getattr(diarization_result, "itertracks", None)):
            diarization_annotation = diarization_result

        if diarization_annotation is None:
            raise ValueError("Unexpected diarization output format returned by pyannote.")

        from ..models.automatic_speech_recognition import SpeakerSegment
        segments = []
        speakers = set()
        for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=speaker
            ))
            speakers.add(speaker)

        response = TranscriptionResponse(
            request_id=request_id,
            model=request.model,
            segments=segments,
            num_speakers=len(speakers),
            processing_time_ms=processing_time_ms,
        )

        # Save to history
        history_storage.add_request(
            service="automatic-speech-recognition",
            request_id=request_id,
            request_data={
                "model": request.model,
                "min_speakers": request.min_speakers,
                "max_speakers": request.max_speakers,
                "num_speakers": request.num_speakers,
            },
            response_data=response.model_dump(),
            status="success",
        )

        schedule_idle_cleanup()
        return response

    finally:
        # Clean up temporary audio file
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary audio file: {e}")
