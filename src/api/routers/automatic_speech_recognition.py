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
from ...db.catalog import get_model_dict, list_models
from ...services.model_coordinator import get_coordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["automatic-speech-recognition"])

# Track current model name (coordinator manages the actual model cache)
_current_model_name: str = ""
_current_model_config: Optional[Dict[str, Any]] = None


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get model configuration from catalog.

    Args:
        model_id: Model identifier

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model not found in catalog
    """
    model = get_model_dict(model_id)

    if model is None:
        raise ValueError(f"Model '{model_id}' not found in catalog")

    return model


def get_available_models() -> list[str]:
    """
    Load available ASR models from the catalog.

    Returns:
        List of model IDs that can be used (both Whisper and pyannote)
    """
    asr_models = list_models(task="automatic-speech-recognition")
    return [m["id"] for m in asr_models]


def validate_model(model_name: str) -> None:
    """
    Validate that the model is supported.

    Args:
        model_name: Model identifier to validate

    Raises:
        ValueError: If model is not supported
    """
    available = get_available_models()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {', '.join(available)}"
        )


def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    This function is called by periodic cleanup in main.py.
    Delegates to the global model coordinator for memory management.
    """
    from ...db.settings import get_setting_int
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    coordinator = get_coordinator()
    # Use asyncio.create_task to run cleanup asynchronously
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coordinator.cleanup_idle_models(idle_timeout, unloader_fn=_unload_model))
    except RuntimeError:
        # No event loop, skip cleanup
        pass


async def _load_model_impl(model_name: str) -> tuple[Any, Any, Dict[str, Any]]:
    """
    Internal async function to load the ASR model and processor.

    Args:
        model_name: Model ID to load

    Returns:
        Tuple of (processor, model, model_config)
    """
    # Validate model is supported and get config
    validate_model(model_name)
    model_config = get_model_config(model_name)

    # Check for local download at HF standard cache path
    from ...config import get_hf_model_cache_path, get_hf_endpoint
    local_model_dir = get_hf_model_cache_path(model_name)

    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        model_path = str(local_model_dir)
        logger.info(f"Using locally downloaded model from {model_path}")
        extra_kwargs = {"local_files_only": True}
    else:
        model_path = model_name
        logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
        extra_kwargs = {}

    # Set HuggingFace endpoint for model loading
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()

    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load processor and model in thread pool to avoid blocking
    def _load():
        processor = AutoProcessor.from_pretrained(model_path, **extra_kwargs)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            attn_implementation="sdpa",  # Use scaled dot product attention for better performance
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **extra_kwargs
        )
        model.to(device, dtype=dtype)
        return processor, model

    processor, model = await asyncio.to_thread(_load)
    return processor, model, model_config


async def _unload_model(model_tuple: tuple[Any, Any, Dict[str, Any]]) -> None:
    """
    Internal async function to unload and cleanup model.

    Args:
        model_tuple: Tuple of (processor, model, config) to unload
    """
    processor, model, config = model_tuple

    # Move model to CPU first (helps with cleanup)
    try:
        if hasattr(model, 'cpu'):
            await asyncio.to_thread(model.cpu)
            logger.debug("Moved ASR model to CPU")
    except Exception as e:
        logger.warning(f"Error moving ASR model to CPU during cleanup: {e}")

    # Delete references
    del processor
    del model
    del config


async def get_model(model_name: str) -> tuple[Any, Any, Dict[str, Any]]:
    """
    Get or load the ASR model via the global coordinator.

    Models are managed by the coordinator to prevent OOM errors.
    The coordinator will preemptively unload other models if needed.

    Args:
        model_name: Model identifier to load

    Returns:
        Tuple of (processor, model, model_config)

    Raises:
        ValueError: If model is not supported
        RuntimeError: If model loading fails
    """
    global _current_model_name, _current_model_config

    _current_model_name = model_name

    # Get model info for memory estimation
    model_info = get_model_dict(model_name)
    estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

    # Load through coordinator (handles preemptive unload)
    coordinator = get_coordinator()
    model_tuple = await coordinator.load_model(
        key=f"asr:{model_name}",
        loader_fn=lambda: _load_model_impl(model_name),
        unloader_fn=_unload_model,
        estimated_memory_mb=estimated_memory_mb,
        model_type="asr",
    )

    # Update current config from the tuple
    _current_model_config = model_tuple[2]

    return model_tuple


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Delegates to the global model coordinator.
    """
    coordinator = get_coordinator()
    # Use asyncio to run cleanup synchronously
    try:
        loop = asyncio.get_running_loop()
        # Create task to unload all ASR models
        if _current_model_name:
            loop.create_task(coordinator.unload_model(f"asr:{_current_model_name}", _unload_model))
    except RuntimeError:
        # No event loop, can't cleanup
        pass


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

        # Load model and get config via coordinator (handles preemptive unload)
        processor, model, model_config = await get_model(request.model)

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

        # Note: Idle cleanup is handled by the global model coordinator
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
        from ...config import get_hf_endpoint, get_hf_model_cache_path

        # Check for local download at HF standard cache path
        local_model_dir = get_hf_model_cache_path(request.model)

        if local_model_dir.exists() and (local_model_dir / "config.yaml").exists():
            model_path = str(local_model_dir)
            logger.info(f"Using locally downloaded model from {model_path}")
            pipeline_kwargs = {"local_files_only": True}
        else:
            model_path = request.model
            logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")
            pipeline_kwargs = {}

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

        # Note: Idle cleanup is handled by the global model coordinator
        return response

    finally:
        # Clean up temporary audio file
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary audio file: {e}")
