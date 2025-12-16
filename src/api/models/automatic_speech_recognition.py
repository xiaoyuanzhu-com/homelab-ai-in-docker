"""Pydantic models for unified automatic speech recognition API.

Supports multiple ASR backends:
- whisper: Basic Whisper transcription via transformers pipeline
- whisperx: WhisperX with word-level alignment and speaker diarization
- funasr: Alibaba FunASR with emotion/event detection
- whisperlivekit: Real-time streaming transcription
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from .base import BaseResponse


# =============================================================================
# Common Models
# =============================================================================


class SpeakerSegment(BaseModel):
    """A segment with speaker information (basic diarization)."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00', 'SPEAKER_01')")


class Word(BaseModel):
    """Word-level timing and speaker info."""

    word: str = Field(..., description="Recognized token/word")
    start: Optional[float] = Field(None, description="Start time in seconds")
    end: Optional[float] = Field(None, description="End time in seconds")
    speaker: Optional[str] = Field(None, description="Assigned speaker label if diarization enabled")


class Segment(BaseModel):
    """A transcription segment with optional word-level detail."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for the segment")
    speaker: Optional[str] = Field(None, description="Assigned speaker label if diarization enabled")
    words: Optional[List[Word]] = Field(None, description="Word-level alignment (whisperx only)")


class Speaker(BaseModel):
    """Speaker metadata including voice embedding for cross-session recognition."""

    speaker_id: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00')")
    embedding: Optional[List[float]] = Field(
        None, description="Speaker embedding vector for voice fingerprinting (whisperx only)"
    )
    total_duration: Optional[float] = Field(None, description="Total speaking duration in seconds")
    segment_count: Optional[int] = Field(None, description="Number of segments/turns by this speaker")


# =============================================================================
# Unified Request Model
# =============================================================================


class TranscriptionRequest(BaseModel):
    """Unified request model for automatic speech recognition.

    The `lib` parameter determines which backend is used:
    - whisper: Basic Whisper via transformers (default for catalog models)
    - whisperx: WhisperX with alignment and diarization support
    - funasr: FunASR with emotion/event detection
    """

    audio: str = Field(
        ...,
        description="Base64-encoded audio file (supports: mp3, mp4, mpeg, mpga, m4a, wav, webm)",
    )
    model: str = Field(
        default="openai/whisper-large-v3-turbo",
        description="Model identifier. For whisperx, use short names like 'large-v3'. For funasr, use full model IDs.",
    )
    lib: Optional[Literal["whisper", "whisperx", "funasr"]] = Field(
        default=None,
        description=(
            "ASR library to use. If not specified, inferred from model catalog. "
            "Options: 'whisper' (basic), 'whisperx' (alignment+diarization), 'funasr' (emotion/event)"
        ),
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'zh'). If None, auto-detect.",
    )

    # Diarization options
    diarization: bool = Field(
        default=False,
        description="Enable speaker diarization. Supported by whisperx and whisper (with pyannote model).",
    )
    min_speakers: Optional[int] = Field(
        default=None,
        description="Minimum number of speakers for diarization.",
        ge=1,
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="Maximum number of speakers for diarization.",
        ge=1,
    )
    num_speakers: Optional[int] = Field(
        default=None,
        description="Exact number of speakers if known (whisper diarization only).",
        ge=1,
    )

    # Whisper-specific options
    return_timestamps: bool = Field(
        default=False,
        description="Return word-level timestamps (whisper only, for basic mode).",
    )

    # WhisperX-specific options
    batch_size: int = Field(
        default=4,
        description="Batch size for inference (whisperx only). Lower = less GPU memory.",
    )
    compute_type: Optional[str] = Field(
        default=None,
        description="Compute type for whisperx (e.g., 'float16', 'float32'). Auto-select if None.",
    )


# =============================================================================
# Unified Response Model
# =============================================================================


class TranscriptionResponse(BaseResponse):
    """Unified response model for automatic speech recognition.

    Fields are populated based on which library was used:
    - All libs: text, model, language
    - whisperx: segments with words, speakers with embeddings
    - funasr: text_clean, emotion, event
    - whisper diarization: segments (without words)
    """

    # Core fields (always present)
    text: Optional[str] = Field(None, description="Full transcription text")
    model: str = Field(..., description="Model used for processing")
    lib: str = Field(..., description="Library used: whisper, whisperx, or funasr")
    language: Optional[str] = Field(None, description="Detected or specified language")

    # Segment-level output
    segments: Optional[List[Segment]] = Field(
        None,
        description="Transcription segments with timing and optional speaker/word info",
    )

    # Speaker information (diarization)
    speakers: Optional[List[Speaker]] = Field(
        None,
        description="Speaker profiles with embeddings (whisperx diarization only)",
    )
    num_speakers: Optional[int] = Field(
        None,
        description="Number of unique speakers detected",
    )

    # Legacy whisper fields
    chunks: Optional[List[dict]] = Field(
        None,
        description="Timestamp chunks for whisper with return_timestamps=True",
    )

    # FunASR-specific fields
    text_clean: Optional[str] = Field(
        None, description="Text with emotion/event tags removed (funasr/SenseVoice only)"
    )
    emotion: Optional[str] = Field(
        None, description="Detected emotion (funasr/SenseVoice only)"
    )
    event: Optional[str] = Field(
        None, description="Detected audio event (funasr/SenseVoice only)"
    )


# =============================================================================
# Backwards Compatibility Aliases (deprecated, will be removed)
# =============================================================================

# These are kept for internal use during migration
WhisperXWord = Word
WhisperXSegment = Segment
WhisperXSpeaker = Speaker
