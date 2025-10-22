"""Pydantic models for automatic speech recognition API."""

from typing import Optional, Union, Literal
from pydantic import BaseModel, Field

from .base import BaseResponse


class SpeakerSegment(BaseModel):
    """A segment with speaker information."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00', 'SPEAKER_01')")


class TranscriptionRequest(BaseModel):
    """Request model for automatic speech recognition."""

    audio: str = Field(
        ...,
        description="Base64-encoded audio file (supports formats: mp3, mp4, mpeg, mpga, m4a, wav, webm)",
    )
    model: str = Field(
        default="openai/whisper-large-v3-turbo",
        description="Model to use for transcription or speaker diarization",
    )
    output_format: Literal["transcription", "diarization"] = Field(
        default="transcription",
        description="Output format: 'transcription' for text transcription (Whisper), 'diarization' for speaker segmentation (pyannote)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'zh', 'es'). For transcription format only.",
    )
    return_timestamps: bool = Field(
        default=False,
        description="Whether to return word-level timestamps. For transcription format only.",
    )
    min_speakers: Optional[int] = Field(
        default=None,
        description="Minimum number of speakers. For diarization format only.",
        ge=1,
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="Maximum number of speakers. For diarization format only.",
        ge=1,
    )
    num_speakers: Optional[int] = Field(
        default=None,
        description="Exact number of speakers if known. For diarization format only.",
        ge=1,
    )


class TranscriptionResponse(BaseResponse):
    """Response model for automatic speech recognition."""

    text: Optional[str] = Field(None, description="Transcribed text from the audio (Whisper models)")
    model: str = Field(..., description="Model used for processing")
    language: Optional[str] = Field(
        None,
        description="Detected or specified language code (Whisper models)",
    )
    chunks: Optional[list[dict]] = Field(
        None,
        description="Timestamp chunks if return_timestamps=True (Whisper models)",
    )
    segments: Optional[list[SpeakerSegment]] = Field(
        None,
        description="Speaker segments with timestamps (pyannote models)",
    )
    num_speakers: Optional[int] = Field(
        None,
        description="Number of unique speakers detected (pyannote models)",
    )
