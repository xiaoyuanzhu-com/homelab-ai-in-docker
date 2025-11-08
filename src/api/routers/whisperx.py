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
from typing import Optional, Any

import torch
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


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/whisperx", tags=["whisperx"])

# Caches for models/pipelines
_asr_model_cache: Optional[Any] = None
_asr_model_name: str = ""
_align_cache: Optional[tuple[Any, Any, str]] = None  # (align_model, metadata, language_code)
_diar_cache: Optional[Any] = None
_last_access_time: Optional[float] = None
_idle_cleanup_task: Optional[asyncio.Task] = None


def _set_hf_env() -> str:
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()
    token = get_setting("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token or ""


def _device_and_dtype(compute_type: Optional[str]) -> tuple[str, Optional[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    global _last_access_time, _asr_model_cache
    if _asr_model_cache is None or _last_access_time is None:
        return
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)
    idle_duration = time.time() - _last_access_time
    if idle_duration >= idle_timeout:
        logger.info(
            f"WhisperX ASR idle for {idle_duration:.1f}s (timeout: {idle_timeout}s), unloading..."
        )
        cleanup()


def schedule_idle_cleanup() -> None:
    global _idle_cleanup_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)

    if _idle_cleanup_task and not _idle_cleanup_task.done():
        _idle_cleanup_task.cancel()

    async def _watchdog(timeout_s: int):
        try:
            await asyncio.sleep(timeout_s)
            if _last_access_time is None:
                return
            if (time.time() - _last_access_time) >= timeout_s and _asr_model_cache is not None:
                cleanup()
        except asyncio.CancelledError:
            pass
        finally:
            global _idle_cleanup_task
            try:
                current = asyncio.current_task()
            except Exception:
                current = None
            if current is not None and _idle_cleanup_task is current:
                _idle_cleanup_task = None

    _idle_cleanup_task = loop.create_task(_watchdog(idle_timeout))


def cleanup():
    global _asr_model_cache, _align_cache, _diar_cache, _asr_model_name, _last_access_time
    try:
        _asr_model_cache = None
        _align_cache = None
        _diar_cache = None
        _asr_model_name = ""
        _last_access_time = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Error releasing WhisperX resources: {e}")


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


def _load_asr_model(model_id: str, device: str, compute_type: Optional[str]):
    global _asr_model_cache, _asr_model_name
    check_and_cleanup_idle_model()
    if _asr_model_cache is not None and _asr_model_name == model_id:
        return _asr_model_cache

    _ensure_torchaudio_compat()
    import whisperx  # type: ignore

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
        model = whisperx.load_model(model_source, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load WhisperX model '{model_id}': {e}")

    _asr_model_cache = model
    _asr_model_name = model_id
    return model


def _load_align_model(language_code: str, device: str):
    global _align_cache
    _ensure_torchaudio_compat()
    import whisperx  # type: ignore

    if _align_cache and _align_cache[2] == language_code:
        return _align_cache[0], _align_cache[1]

    try:
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load alignment model for '{language_code}': {e}")

    _align_cache = (align_model, metadata, language_code)
    return align_model, metadata


def _load_diar_pipeline(device: str):
    global _diar_cache
    _ensure_torchaudio_compat()
    import whisperx  # type: ignore

    if _diar_cache is not None:
        return _diar_cache

    token = _set_hf_env()
    if not token:
        raise RuntimeError(
            "HuggingFace token is required for diarization. Set 'hf_token' in settings."
        )
    try:
        diar = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
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

        # Decode audio off the event loop
        audio_path = await asyncio.to_thread(_decode_audio_to_file, request.audio)

        # Load WhisperX ASR model
        asr_model = _load_asr_model(request.asr_model, device, compute_type)

        # Import whisperx in the worker thread to avoid blocking
        import whisperx  # type: ignore

        # Load audio
        audio = whisperx.load_audio(str(audio_path))

        def _run_pipeline():
            # Transcribe
            transcribe_kwargs = {"batch_size": request.batch_size}
            if request.language:
                transcribe_kwargs["language"] = request.language
            result = asr_model.transcribe(audio, **transcribe_kwargs)

            language = result.get("language") or request.language
            if not language:
                language = "unknown"

            # Align words
            align_model, metadata = _load_align_model(language, device)
            aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

            diar_segments = None
            if request.diarize:
                diar = _load_diar_pipeline(device)
                diar_segments = diar(audio)
                aligned = whisperx.assign_word_speakers(diar_segments, aligned)

            return language, aligned

        language, aligned = await asyncio.to_thread(_run_pipeline)

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

        # Cache last access
        global _last_access_time
        _last_access_time = time.time()
        schedule_idle_cleanup()

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
