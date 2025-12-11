"""Speaker embedding API endpoints using pyannote.audio.

NOTE: This module loads pyannote.audio in-process. The pyannote models
are specified to use the 'whisper' environment which includes pyannote.
For now, this will fail at runtime if pyannote is not installed.
TODO: Convert to use worker subprocess pattern for full env isolation.
"""

import asyncio
import base64
import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Lazy imports for ML libraries - will fail at runtime if not available
# These are in the whisper worker environment
try:
    import numpy as np
    from scipy.spatial.distance import cdist
except ImportError:
    np = None  # type: ignore
    cdist = None  # type: ignore

from ..models.speaker_embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
    CompareEmbeddingsRequest,
    CompareEmbeddingsResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    MatchRequest,
    MatchResponse,
    MatchResult,
    MatchCandidate,
)
from ...services.model_coordinator import use_model

router = APIRouter(prefix="/api/speaker-embedding", tags=["speaker-embedding"])
logger = logging.getLogger(__name__)


async def _load_model_impl(model_name: str) -> tuple:
    """
    Internal async function to load the speaker embedding model.

    Args:
        model_name: Model ID to load

    Returns:
        Tuple of (model, inference)
    """
    # Lazy import - will fail if pyannote not installed
    from pyannote.audio import Model, Inference
    import torch
    from ...db.settings import get_setting
    from ...config import get_hf_endpoint, get_hf_model_cache_path

    # Check for local download at HF standard cache path
    local_model_dir = get_hf_model_cache_path(model_name)

    if local_model_dir.exists() and list(local_model_dir.glob("*.bin")):
        model_path = str(local_model_dir)
        logger.info(f"Using locally downloaded model from {model_path}")
    else:
        model_path = model_name
        logger.info(f"Model not found locally, will download from HuggingFace to cache: {model_path}")

    # Set HuggingFace endpoint and token
    os.environ["HF_ENDPOINT"] = get_hf_endpoint()
    hf_token = get_setting("hf_token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    try:
        # Load model in thread pool to avoid blocking
        def _load():
            # Load model (TF32 is configured globally in main.py)
            model = Model.from_pretrained(model_path, use_auth_token=hf_token if hf_token else None)

            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda"))

            # Create inference wrapper
            inference = Inference(model, window="whole")

            return model, inference

        model, inference = await asyncio.to_thread(_load)
        logger.info(f"Successfully loaded speaker embedding model: {model_name}")
        return model, inference

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load embedding model: {str(e)}"
        )


async def _unload_model(model_tuple: tuple) -> None:
    """
    Internal async function to unload and cleanup model.

    Args:
        model_tuple: Tuple of (model, inference) to unload
    """
    model, inference = model_tuple

    # Move model to CPU first (helps with cleanup)
    try:
        if hasattr(model, 'cpu'):
            await asyncio.to_thread(model.cpu)
            logger.debug("Moved speaker embedding model to CPU")
    except Exception as e:
        logger.warning(f"Error moving speaker embedding model to CPU during cleanup: {e}")

    # Delete references
    del model
    del inference




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

        # Get model info for memory estimation
        from ...db.catalog import get_model_dict
        model_info = get_model_dict(request.model)
        estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

        # Use context manager for automatic cleanup
        async with use_model(
            key=f"speaker:{request.model}",
            loader_fn=lambda: _load_model_impl(request.model),
            model_type="speaker-embedding",
            estimated_memory_mb=estimated_memory_mb,
            unloader_fn=_unload_model,
        ) as model_tuple:
            model, inference = model_tuple

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

            # Model automatically released when context exits

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

        # Get model info for memory estimation
        from ...db.catalog import get_model_dict
        model_info = get_model_dict(request.model)
        estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

        # Use context manager for automatic cleanup
        async with use_model(
            key=f"speaker:{request.model}",
            loader_fn=lambda: _load_model_impl(request.model),
            model_type="speaker-embedding",
            estimated_memory_mb=estimated_memory_mb,
            unloader_fn=_unload_model,
        ) as model_tuple:
            model, inference = model_tuple

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

            # Model automatically released when context exits

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


@router.post("/batch-extract", response_model=BatchEmbeddingResponse)
async def batch_extract_embeddings(request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
    """Extract speaker embeddings for multiple time segments from one audio file."""
    request_id = f"batch_embed_{int(time.time() * 1000)}"
    start_time = time.time()
    audio_path: Path | None = None

    try:
        audio_path = await decode_audio(request.audio)

        # Get model info for memory estimation
        from ...db.catalog import get_model_dict
        model_info = get_model_dict(request.model)
        estimated_memory_mb = model_info.get("gpu_memory_mb") if model_info else None

        # Use context manager for automatic cleanup
        async with use_model(
            key=f"speaker:{request.model}",
            loader_fn=lambda: _load_model_impl(request.model),
            model_type="speaker-embedding",
            estimated_memory_mb=estimated_memory_mb,
            unloader_fn=_unload_model,
        ) as model_tuple:
            model, inference = model_tuple

            def _batch():
                from pyannote.core import Segment
                embs: list[list[float]] = []
                dim = None
                for seg in request.segments:
                    s = Segment(seg.start, seg.end)
                    emb = inference.crop(str(audio_path), s)
                    # emb is (1, D)
                    arr = emb[0] if hasattr(emb, "ndim") and getattr(emb, "ndim", 1) > 1 else emb
                    vec = arr.tolist()
                    if dim is None:
                        dim = len(vec)
                    embs.append(vec)
                return embs, dim or 0

            embeddings, dimension = await asyncio.to_thread(_batch)

            # Model automatically released when context exits

        processing_time_ms = int((time.time() - start_time) * 1000)

        return BatchEmbeddingResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            embeddings=embeddings,
            dimension=dimension,
            count=len(embeddings),
            model=request.model,
        )
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to batch extract embeddings: {e}")
    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass


@router.post("/match", response_model=MatchResponse)
async def match_embeddings(request: MatchRequest) -> MatchResponse:
    """Match query embeddings to a provided registry (stateless identification)."""
    request_id = f"match_{int(time.time() * 1000)}"
    start_time = time.time()

    try:
        import numpy as np
        from scipy.spatial.distance import cdist

        # Normalize embeddings for cosine if metric is cosine
        def _normalize(arr: np.ndarray) -> np.ndarray:
            if request.metric != "cosine":
                return arr
            norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            return arr / norm

        # Prepare query matrix
        Q = np.array(request.query_embeddings, dtype=np.float32)
        Q = _normalize(Q)

        # Prepare registry per speaker
        speaker_names: list[str] = []
        speaker_vectors: list[np.ndarray] = []

        for entry in request.registry:
            speaker_names.append(entry.name)
            S = np.array(entry.embeddings, dtype=np.float32)
            if S.ndim == 1:
                S = S.reshape(1, -1)
            S = _normalize(S)
            if request.strategy == "centroid":
                speaker_vectors.append(np.mean(S, axis=0, keepdims=True))
            else:  # 'best' strategy compares against all samples and takes max similarity
                speaker_vectors.append(S)

        # Compute scores for each query
        results: list[MatchResult] = []
        for q in Q:
            # Build a flat list of candidates for top-k ranking
            cand_names: list[str] = []
            cand_sims: list[float] = []
            for name, vecs in zip(speaker_names, speaker_vectors):
                # Distance then convert to similarity
                d = cdist(q.reshape(1, -1), vecs, metric=request.metric)
                if request.metric == "cosine":
                    sims = 1.0 - d.flatten()
                else:
                    sims = (1.0 / (1.0 + d.flatten()))
                # Best score for this speaker
                best = float(np.max(sims))
                cand_names.append(name)
                cand_sims.append(best)

            # Rank
            order = np.argsort(cand_sims)[::-1]
            top_idx = order[: max(1, request.top_k)]
            candidates = [
                MatchCandidate(name=cand_names[i], similarity=float(cand_sims[i])) for i in top_idx
            ]

            best = candidates[0]
            best_out = None
            if request.threshold is None or best.similarity >= float(request.threshold):
                best_out = best

            results.append(MatchResult(best=best_out, candidates=candidates))

        processing_time_ms = int((time.time() - start_time) * 1000)
        return MatchResponse(
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            results=results,
        )
    except Exception as e:
        logger.error(f"Match computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to match embeddings: {e}")


def cleanup():
    """
    Release model resources immediately.

    This function is called during app shutdown.
    Cleanup now handled entirely by the global model coordinator.
    """
    pass


def check_and_cleanup_idle_model():
    """
    Check if model has been idle too long and cleanup if needed.

    This function is called by periodic cleanup in main.py.
    Cleanup now handled entirely by the global model coordinator.
    """
    pass
