"""Pydantic models for automatic speech recognition API."""

from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseResponse


class TranscriptionRequest(BaseModel):
    """Request model for automatic speech recognition."""

    audio: str = Field(
        ...,
        description="Base64-encoded audio file (supports formats: mp3, mp4, mpeg, mpga, m4a, wav, webm)",
    )
    model: str = Field(
        default="openai/whisper-large-v3-turbo",
        description="Model to use for transcription",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'zh', 'es'). If not specified, the model will auto-detect.",
    )
    return_timestamps: bool = Field(
        default=False,
        description="Whether to return word-level timestamps",
    )


class TranscriptionResponse(BaseResponse):
    """Response model for automatic speech recognition."""

    text: str = Field(..., description="Transcribed text from the audio")
    model: str = Field(..., description="Model used for transcription")
    language: Optional[str] = Field(
        None,
        description="Detected or specified language code",
    )
    chunks: Optional[list[dict]] = Field(
        None,
        description="Timestamp chunks if return_timestamps=True",
    )
