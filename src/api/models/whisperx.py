"""Pydantic models for WhisperX transcription API."""

from typing import Optional, List
from pydantic import BaseModel, Field

from .base import BaseResponse


class WhisperXWord(BaseModel):
    word: str = Field(..., description="Recognized token/word")
    start: Optional[float] = Field(None, description="Start time in seconds")
    end: Optional[float] = Field(None, description="End time in seconds")
    speaker: Optional[str] = Field(None, description="Assigned speaker label if diarization enabled")


class WhisperXSegment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for the segment")
    speaker: Optional[str] = Field(None, description="Assigned speaker label if diarization enabled")
    words: Optional[List[WhisperXWord]] = Field(None, description="Aligned words for the segment")


class WhisperXSpeaker(BaseModel):
    """Speaker metadata including voice embedding for cross-session recognition."""

    speaker_id: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00')")
    embedding: List[float] = Field(..., description="512-dimensional speaker embedding vector for voice fingerprinting")
    total_duration: float = Field(..., description="Total speaking duration in seconds")
    segment_count: int = Field(..., description="Number of segments/turns by this speaker")


class WhisperXTranscriptionRequest(BaseModel):
    """Request for WhisperX transcription & optional diarization."""

    audio: str = Field(
        ..., description="Base64-encoded audio (mp3/mp4/mpeg/mpga/m4a/wav/webm)"
    )
    asr_model: str = Field(
        default="large-v3",
        description=(
            "Whisper model identifier for WhisperX (e.g., 'large-v3', 'small.en'). "
            "This is independent of the catalog DB."
        ),
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en'). If None, WhisperX will detect.",
    )
    diarize: bool = Field(
        default=False,
        description="Enable speaker diarization via WhisperX's pyannote integration",
    )
    min_speakers: Optional[int] = Field(
        default=1,
        description="Minimum number of speakers for diarization (default: 1)",
    )
    max_speakers: Optional[int] = Field(
        default=5,
        description="Maximum number of speakers for diarization (default: 5)",
    )
    batch_size: int = Field(
        default=16, description="Batch size for ASR model inference"
    )
    compute_type: Optional[str] = Field(
        default=None,
        description="Compute type hint for WhisperX (e.g., 'float16', 'float32'). If None, auto-select.",
    )


class WhisperXTranscriptionResponse(BaseResponse):
    """Response with aligned segments and optional speakers."""

    text: str = Field(..., description="Full transcription text")
    language: Optional[str] = Field(None, description="Detected or forced language")
    model: str = Field(..., description="ASR model used by WhisperX")
    segments: list[WhisperXSegment] = Field(
        default_factory=list,
        description="Aligned segments with optional speaker and word info",
    )
    speakers: Optional[List[WhisperXSpeaker]] = Field(
        None,
        description="Speaker embeddings and metadata (only when diarization is enabled)",
    )
