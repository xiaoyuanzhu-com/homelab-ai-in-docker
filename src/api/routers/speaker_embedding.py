"""Speaker embedding API endpoints using pyannote.audio."""

import asyncio
import base64
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from scipy.spatial.distance import cdist

from ..models.speaker_embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
    CompareEmbeddingsRequest,
    CompareEmbeddingsResponse,
)

router = APIRouter(prefix="/api/speaker-embedding", tags=["speaker-embedding"])
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = None
_inference_cache = None
_current_model_name = None
_last_used_time = None

# Model idle timeout (in seconds)
MODEL_IDLE_TIMEOUT = 300  # 5 minutes


def get_model(model_name: str):
    """Load and cache the pyannote embedding model."""
    global _model_cache, _inference_cache, _current_model_name, _last_used_time

    # Update last used time
    _last_used_time = time.time()

    # Return cached model if same model is requested
    if _model_cache is not None and _current_model_name == model_name:
        return _model_cache, _inference_cache

    from pyannote.audio import Model, Inference
    from ...db.settings import get_setting
    from ...config import get_data_dir, get_hf_endpoint

    # Check for local model
    local_model_dir = get_data_dir() / "models" / model_name
    if local_model_dir.exists() and any(local_model_dir.glob("*.bin")):
        model_path = str(local_model_dir)
        logger.info(f"Using locally downloaded model from {model_path}")
        extra_kwargs = {}
    else:
        model_path = model_name
        logger.info(f"Model not found locally, will download from HuggingFace: {model_name}")
        extra_kwargs = {}

    # Set HuggingFace endpoint and token
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()
    hf_token = get_setting("hf_token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    try:
        # Suppress TF32 warnings by setting the backends before loading model
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere+ GPUs
            # This suppresses the pyannote reproducibility warning
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load model
        model = Model.from_pretrained(model_path, use_auth_token=hf_token if hf_token else None)

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))

        # Create inference wrapper
        inference = Inference(model, window="whole")

        # Cache the model and inference
        _model_cache = model
        _inference_cache = inference
        _current_model_name = model_name

        logger.info(f"Successfully loaded model: {model_name}")
        return model, inference

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load embedding model: {str(e)}"
        )


async def decode_audio(base64_audio: str) -> Path:
    """Decode base64 audio and save to temporary file."""
    try:
        audio_data = base64.b64decode(base64_audio)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_data)
        temp_file.close()
        return Path(temp_file.name)
    except Exception as e:
        logger.error(f"Failed to decode audio: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio data: {str(e)}"
        )


@router.post("/extract", response_model=EmbeddingResponse)
async def extract_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Extract speaker embedding from audio.

    Supports two modes:
    - 'whole': Extract embedding from entire audio file
    - 'segment': Extract embedding from specific time range (requires start_time and end_time)
    """
    request_id = f"embed_{int(time.time() * 1000)}"
    start_time = time.time()
    logger.info(f"Processing embedding extraction request {request_id}")

    audio_path = None
    try:
        # Decode audio
        audio_path = await decode_audio(request.audio)

        # Load model
        model, inference = get_model(request.model)

        # Extract embedding based on mode
        def _extract():
            if request.mode == "segment":
                if request.start_time is None or request.end_time is None:
                    raise ValueError("start_time and end_time are required for 'segment' mode")

                from pyannote.core import Segment
                segment = Segment(request.start_time, request.end_time)
                embedding = inference.crop(str(audio_path), segment)
                duration = request.end_time - request.start_time
            else:
                # Whole file mode
                embedding = inference(str(audio_path))
                duration = None

            # Convert to list - embedding is a (1, D) numpy array
            import numpy as np
            if isinstance(embedding, np.ndarray):
                # If it's already a numpy array, get the first row
                embedding_vector = embedding[0] if embedding.ndim > 1 else embedding
                embedding_list = embedding_vector.tolist()
            else:
                # If it's a different type, try to convert
                embedding_list = np.array(embedding).flatten().tolist()

            return embedding_list, len(embedding_list), duration

        # Run in thread pool to avoid blocking
        embedding_list, dimension, duration = await asyncio.to_thread(_extract)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return EmbeddingResponse(
            request_id=request_id,
            embedding=embedding_list,
            dimension=dimension,
            model=request.model,
            duration=duration,
            processing_time_ms=processing_time_ms
        )

    except ValueError as e:
        logger.error(f"Invalid request for {request_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract embedding: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {audio_path}: {e}")


@router.post("/compare", response_model=CompareEmbeddingsResponse)
async def compare_embeddings(request: CompareEmbeddingsRequest) -> CompareEmbeddingsResponse:
    """
    Compare two audio files and return similarity score.

    The distance metric can be:
    - 'cosine': Cosine distance (default, recommended for speaker verification)
    - 'euclidean': Euclidean distance
    - 'cityblock': Manhattan distance
    """
    request_id = f"compare_{int(time.time() * 1000)}"
    start_time = time.time()
    logger.info(f"Processing embedding comparison request {request_id}")

    audio_path1 = None
    audio_path2 = None

    try:
        # Decode both audio files
        audio_path1 = await decode_audio(request.audio1)
        audio_path2 = await decode_audio(request.audio2)

        # Load model
        model, inference = get_model(request.model)

        # Extract embeddings
        def _compare():
            embedding1 = inference(str(audio_path1))
            embedding2 = inference(str(audio_path2))

            # Calculate distance
            distance = cdist(embedding1, embedding2, metric=request.metric)[0, 0]

            # Calculate similarity (inverse of distance for cosine)
            if request.metric == "cosine":
                similarity = 1.0 - distance
            else:
                # For other metrics, normalize to 0-1 range (approximate)
                similarity = 1.0 / (1.0 + distance)

            return float(distance), float(similarity)

        # Run in thread pool
        distance, similarity = await asyncio.to_thread(_compare)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return CompareEmbeddingsResponse(
            request_id=request_id,
            distance=distance,
            similarity=similarity,
            metric=request.metric,
            model=request.model,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Processing failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare embeddings: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        for audio_path in [audio_path1, audio_path2]:
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {audio_path}: {e}")


def cleanup_model():
    """Cleanup cached model to free memory."""
    global _model_cache, _inference_cache, _current_model_name, _last_used_time
    if _model_cache is not None:
        logger.info(f"Cleaning up speaker embedding model: {_current_model_name}")
        _model_cache = None
        _inference_cache = None
        _current_model_name = None
        _last_used_time = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_and_cleanup_idle_model():
    """Check if model has been idle and cleanup if necessary."""
    global _last_used_time, _model_cache

    if _model_cache is None:
        return

    if _last_used_time is None:
        return

    idle_time = time.time() - _last_used_time
    if idle_time > MODEL_IDLE_TIMEOUT:
        logger.info(f"Model has been idle for {idle_time:.1f}s (timeout: {MODEL_IDLE_TIMEOUT}s), cleaning up")
        cleanup_model()
