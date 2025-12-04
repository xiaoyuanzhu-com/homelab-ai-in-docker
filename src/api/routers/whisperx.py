"""WhisperX-based transcription with optional diarization and alignment."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Any, Callable

from fastapi import APIRouter, HTTPException

from ..models.whisperx import (
    WhisperXTranscriptionRequest,
    WhisperXTranscriptionResponse,
    WhisperXSegment,
    WhisperXWord,
)
from ...db.settings import get_setting, get_setting_int
from ...config import get_hf_endpoint, get_hf_model_cache_path
from ...storage.history import history_storage
from ...services.model_coordinator import use_model, get_coordinator


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/whisperx", tags=["whisperx"])

# Track current ASR model name (coordinator manages the ASR model cache)
_asr_model_name: str = ""

# Caches for auxiliary models (alignment and diarization)
_align_cache: Optional[tuple[Any, Any, str, str]] = None  # (align_model, metadata, language_code, align_device)
_diar_cache: Optional[Any] = None

# Lock for serializing inference to avoid concurrent GPU spikes
_infer_lock: asyncio.Lock = asyncio.Lock()


def _set_hf_env() -> str:
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()
    token = get_setting("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token or ""


def _device_and_dtype(compute_type: Optional[str]) -> tuple[str, Optional[str]]:
    # Import torch lazily to allow CUDA lib setup to run first
    try:
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    device = "cuda" if has_cuda else "cpu"
    # WhisperX uses string compute types; default sensibly
    if compute_type:
        return device, compute_type
    return device, ("float16" if device == "cuda" else "float32")


def _ensure_torchaudio_compat() -> None:
    """Ensure torchaudio.AudioMetaData is available.

    In torchaudio <2.9, AudioMetaData is available directly in torchaudio module.
    This function is a no-op for compatibility, as the class is already accessible.
    """
    import torchaudio  # type: ignore
    # In torchaudio 2.8.0, AudioMetaData is already available as torchaudio.AudioMetaData
    # No aliasing needed, just verify it exists
    if not hasattr(torchaudio, 'AudioMetaData'):
        raise ImportError("torchaudio.AudioMetaData not found. Ensure torchaudio<2.9 is installed.")



def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    This function is called by periodic cleanup in main.py.
    Delegates to the global model coordinator for memory management.
    """
    # WhisperX processing (transcribe + align + diarize) can take several minutes for long audio
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 300)

    coordinator = get_coordinator()
    # Use asyncio.create_task to run cleanup asynchronously
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coordinator.cleanup_idle_models(idle_timeout, unloader_fn=_unload_asr_model))
    except RuntimeError:
        # No event loop, skip cleanup
        pass


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Delegates to the global model coordinator for ASR model,
    and cleans up auxiliary models (align/diar) directly.
    """
    global _align_cache, _diar_cache

    # Clean up auxiliary models
    _align_cache = None
    _diar_cache = None

    # Delegate ASR model cleanup to coordinator
    coordinator = get_coordinator()
    try:
        loop = asyncio.get_running_loop()
        # Create task to unload WhisperX ASR model
        loop.create_task(coordinator.unload_model(f"whisperx:{_asr_model_name}", _unload_asr_model))
    except RuntimeError:
        # No event loop, can't cleanup
        pass


def _decode_audio_to_file(audio_b64: str) -> Path:
    try:
        if audio_b64.startswith("data:audio"):
            audio_b64 = audio_b64.split(",", 1)[1]
        raw = base64.b64decode(audio_b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
        tmp.write(raw)
        tmp.close()
        return Path(tmp.name)
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {e}")


async def _load_asr_model_impl(model_id: str, device: str, compute_type: Optional[str]) -> Any:
    """
    Internal async function to load the WhisperX ASR model.

    Args:
        model_id: Model ID to load
        device: Device to load model on (cuda/cpu)
        compute_type: Compute type for inference (float16/float32)

    Returns:
        Loaded WhisperX ASR model
    """
    _ensure_torchaudio_compat()
    import whisperx  # type: ignore
    import torch  # type: ignore

    # Enable TF32 for faster inference on Ampere+ GPUs
    # TF32 provides ~10-20% speedup with negligible accuracy impact
    # See: https://pytorch.org/docs/stable/notes/numerical_accuracy.html
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.debug("TF32 enabled for WhisperX inference")

    # Try to prefer local models if mirrored into HF cache
    local_dir = get_hf_model_cache_path(model_id)
    model_source: str | None = None
    if local_dir.exists():
        model_source = str(local_dir)
    else:
        model_source = model_id

    # Build kwargs carefully as whisperx may not accept None compute_type
    kwargs: dict[str, Any] = {"device": device}
    if compute_type:
        kwargs["compute_type"] = compute_type

    try:
        logger.info(
            f"WhisperX: loading ASR model '{model_source}' with kwargs={kwargs}"
        )
        # Load in thread pool to avoid blocking
        model = await asyncio.to_thread(whisperx.load_model, model_source, **kwargs)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load WhisperX model '{model_id}': {e}")


async def _unload_asr_model(model: Any) -> None:
    """
    Internal async function to unload and cleanup ASR model.

    Args:
        model: WhisperX model to unload
    """
    # Delete ASR model reference first
    del model

    # Clear GPU cache and force defragmentation
    try:
        import torch  # type: ignore
        import gc

        # Force garbage collection to release references
        gc.collect()

        # Clear GPU cache to defragment memory
        if torch.cuda.is_available():
            await asyncio.to_thread(torch.cuda.empty_cache)
            await asyncio.to_thread(torch.cuda.synchronize)
            logger.debug("GPU cache cleared for WhisperX ASR model")

    except Exception as e:
        logger.warning(f"Error clearing GPU cache during WhisperX cleanup: {e}")


# Helper functions removed - now using use_model() context manager
# - _run_with_periodic_touch: No longer needed, active_refs protects the model
# - get_asr_model: Replaced with inline use_model() call


def _load_align_model(language_code: str, device: str):
    global _align_cache
    _ensure_torchaudio_compat()
    import whisperx  # type: ignore

    if _align_cache and _align_cache[2] == language_code:
        logger.info(
            f"WhisperX: reusing cached align model for '{language_code}' on {_align_cache[3]}"
        )
        return _align_cache[0], _align_cache[1], _align_cache[3]

    try:
        # Prefer running alignment on CPU to avoid GPU OOM with large ASR
        try:
            from ...db.settings import get_setting  # lazy import

            align_dev_pref = (get_setting("whisperx_align_device") or "cpu").lower()
        except Exception:
            align_dev_pref = "cpu"

        align_device = align_dev_pref if align_dev_pref in {"cpu", "cuda"} else "cpu"
        if align_device != device:
            logger.info(
                f"WhisperX: loading align model on {align_device} (ASR on {device})"
            )
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code, device=align_device
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load alignment model for '{language_code}': {e}")

    _align_cache = (align_model, metadata, language_code, align_device)
    return align_model, metadata, align_device


def _load_diar_pipeline(device: str):
    global _diar_cache
    _ensure_torchaudio_compat()
    import whisperx  # type: ignore

    if _diar_cache is not None:
        logger.info("WhisperX: reusing cached diarization pipeline")
        return _diar_cache

    token = _set_hf_env()
    if not token:
        raise RuntimeError(
            "HuggingFace token is required for diarization. Set 'hf_token' in settings."
        )
    try:
        # Prefer diarization on CUDA by default for better performance; configurable via settings
        try:
            from ...db.settings import get_setting  # lazy import

            diar_dev_pref = (get_setting("whisperx_diar_device") or "cuda").lower()
        except Exception:
            diar_dev_pref = "cuda"

        diar_device = diar_dev_pref if diar_dev_pref in {"cpu", "cuda"} else "cuda"
        if diar_device != device:
            logger.info(
                f"WhisperX: loading diarization on {diar_device} (ASR on {device})"
            )
        # WhisperX 3.3.4+ moved DiarizationPipeline to whisperx.diarize submodule
        diar = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=diar_device)
    except Exception as e:
        raise RuntimeError(f"Failed to load diarization pipeline: {e}")

    _diar_cache = diar
    return diar


@router.post("/transcribe", response_model=WhisperXTranscriptionResponse)
async def transcribe(request: WhisperXTranscriptionRequest) -> WhisperXTranscriptionResponse:
    request_id = str(uuid.uuid4())
    start_time = time.time()
    audio_path: Optional[Path] = None

    try:
        # Prepare env + device/dtype
        _set_hf_env()
        device, compute_type = _device_and_dtype(request.compute_type)

        # Log request parameters
        logger.info(
            f"WhisperX transcription request: model={request.asr_model}, "
            f"batch_size={request.batch_size}, language={request.language}, "
            f"diarize={request.diarize}, min_speakers={request.min_speakers}, "
            f"max_speakers={request.max_speakers}, device={device}, compute_type={compute_type}"
        )

        # Decode audio off the event loop
        t0 = time.time()
        audio_path = await asyncio.to_thread(_decode_audio_to_file, request.audio)
        logger.info(f"⏱️  Audio decode: {(time.time() - t0)*1000:.0f}ms")

        # Import whisperx
        import whisperx  # type: ignore

        # Load audio
        t0 = time.time()
        audio = whisperx.load_audio(str(audio_path))
        logger.info(f"⏱️  Audio load: {(time.time() - t0)*1000:.0f}ms")

        # Get model info for memory estimation
        from ...db.catalog import get_model_dict
        model_info = get_model_dict(request.asr_model)
        estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

        # Serialize WhisperX runs to avoid concurrent GPU spikes / OOM
        async with _infer_lock:
            # Use context manager for automatic cleanup
            async with use_model(
                key=f"whisperx:{request.asr_model}",
                loader_fn=lambda: _load_asr_model_impl(request.asr_model, device, compute_type),
                model_type="whisperx",
                estimated_memory_mb=estimated_memory_mb,
                unloader_fn=_unload_asr_model,
            ) as asr_model:
                # Transcribe (no touch_model needed - protected by active_refs)
                transcribe_kwargs = {"batch_size": request.batch_size}
                if request.language:
                    transcribe_kwargs["language"] = request.language

                t0 = time.time()
                result = await asyncio.to_thread(asr_model.transcribe, audio, **transcribe_kwargs)
                logger.info(f"⏱️  Transcribe: {(time.time() - t0)*1000:.0f}ms")

                language = result.get("language") or request.language
                if not language:
                    language = "unknown"

                # Align words
                t0 = time.time()
                align_model, metadata, align_device = await asyncio.to_thread(_load_align_model, language, device)
                logger.info(f"⏱️  Load align model: {(time.time() - t0)*1000:.0f}ms")

                t0 = time.time()
                aligned = await asyncio.to_thread(
                    whisperx.align,
                    result["segments"], align_model, metadata, audio, align_device
                )
                logger.info(f"⏱️  Align: {(time.time() - t0)*1000:.0f}ms")

                # Clean up transcription result to free memory
                del result
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.debug("Cleaned up transcription intermediate tensors")

                # Diarization (if enabled)
                if request.diarize:
                    t0 = time.time()
                    diar = await asyncio.to_thread(_load_diar_pipeline, device)
                    logger.info(f"⏱️  Load diarization pipeline: {(time.time() - t0)*1000:.0f}ms")

                    # Build diarization kwargs with optional speaker constraints
                    diar_kwargs = {}
                    if request.min_speakers is not None:
                        diar_kwargs["min_speakers"] = request.min_speakers
                    if request.max_speakers is not None:
                        diar_kwargs["max_speakers"] = request.max_speakers

                    logger.info(
                        f"WhisperX: starting diarization with kwargs={diar_kwargs}, "
                        f"audio duration={len(audio)/16000:.1f}s"
                    )

                    t0 = time.time()
                    diar_segments = await asyncio.to_thread(diar, audio, **diar_kwargs)
                    logger.info(f"⏱️  Diarization: {(time.time() - t0)*1000:.0f}ms")

                    # Log diarization results
                    if diar_segments is not None:
                        try:
                            speakers = set()
                            for turn, _, speaker in diar_segments.itertracks(yield_label=True):
                                speakers.add(speaker)
                            logger.info(
                                f"WhisperX: diarization detected {len(speakers)} speakers: {sorted(speakers)}"
                            )
                        except Exception as e:
                            logger.warning(f"WhisperX: could not extract speaker info from diarization result: {e}")
                    else:
                        logger.warning("WhisperX: diarization returned None")

                    t0 = time.time()
                    aligned = await asyncio.to_thread(whisperx.assign_word_speakers, diar_segments, aligned)
                    logger.info(f"⏱️  Assign speakers: {(time.time() - t0)*1000:.0f}ms")

                    # Clean up diarization intermediate tensors
                    del diar_segments
                    import torch
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.debug("Cleaned up diarization intermediate tensors")

                # Model automatically released when context exits
                language, aligned = language, aligned

        # Build response segments
        segments_out: list[WhisperXSegment] = []
        full_text_parts: list[str] = []
        for seg in aligned.get("segments", []):
            words = []
            seg_speaker = None
            if seg.get("words"):
                # Determine majority speaker if present at word-level
                speaker_counts: dict[str, int] = {}
                for w in seg["words"]:
                    w_speaker = w.get("speaker")
                    w_obj = WhisperXWord(
                        word=w.get("word", ""),
                        start=float(w.get("start")) if w.get("start") is not None else None,
                        end=float(w.get("end")) if w.get("end") is not None else None,
                        speaker=w_speaker,
                    )
                    words.append(w_obj)
                    if w_speaker:
                        speaker_counts[w_speaker] = speaker_counts.get(w_speaker, 0) + 1
                if speaker_counts:
                    seg_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]

            text = seg.get("text", "").strip()
            full_text_parts.append(text)
            segments_out.append(
                WhisperXSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text,
                    speaker=seg.get("speaker") or seg_speaker,
                    words=words or None,
                )
            )

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = WhisperXTranscriptionResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            text=" ".join([t for t in full_text_parts if t]),
            language=language,
            model=request.asr_model,
            segments=segments_out,
        )

        # Save to history (exclude audio)
        history_storage.add_request(
            service="whisperx",
            request_id=request_id,
            request_data={
                "asr_model": request.asr_model,
                "language": request.language,
                "diarize": request.diarize,
            },
            response_data=response.model_dump(),
            status="success",
        )

        # Note: Idle cleanup is handled by the global model coordinator
        return response

    except RuntimeError as e:
        logger.error(f"WhisperX runtime error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"WhisperX processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe: {e}")
    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass
