"""Pydantic models for FunASR transcription API."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from .base import BaseResponse


class FunASRTranscriptionRequest(BaseModel):
    """Request for FunASR transcription."""

    audio: str = Field(
        ..., description="Base64-encoded audio (mp3/mp4/mpeg/mpga/m4a/wav/webm)"
    )
    model: str = Field(
        default="FunAudioLLM/SenseVoiceSmall",
        description=(
            "FunASR model identifier. Options: "
            "'FunAudioLLM/SenseVoiceSmall' (multi-language with emotion), "
            "'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch' (Mandarin)"
        ),
    )
    language: Optional[str] = Field(
        default=None,
        description="Language hint (e.g., 'zh', 'en'). If None, auto-detect.",
    )


class FunASRTranscriptionResponse(BaseResponse):
    """Response from FunASR transcription."""

    text: str = Field(..., description="Full transcription text")
    text_clean: Optional[str] = Field(
        None, description="Transcription with emotion/event tags removed"
    )
    language: Optional[str] = Field(None, description="Detected language")
    model: str = Field(..., description="Model used for transcription")
    emotion: Optional[str] = Field(
        None, description="Detected emotion (SenseVoice only)"
    )
    event: Optional[str] = Field(
        None, description="Detected audio event (SenseVoice only)"
    )
