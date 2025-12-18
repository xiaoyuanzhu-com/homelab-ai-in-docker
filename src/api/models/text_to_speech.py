"""Pydantic models for text-to-speech API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import BaseResponse


class TextToSpeechRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    text: str = Field(
        ...,
        description="Text to synthesize into speech",
        examples=["Hello, welcome to the AI voice synthesis demo."],
    )
    model: str = Field(
        default="FunAudioLLM/CosyVoice2-0.5B",
        description="Model identifier. Supports short names like 'cosyvoice2' or full IDs like 'FunAudioLLM/CosyVoice2-0.5B'",
    )
    mode: Literal["zero_shot", "cross_lingual", "instruct", "sft"] = Field(
        default="zero_shot",
        description=(
            "TTS synthesis mode:\n"
            "- zero_shot: Voice cloning from prompt audio (requires prompt_audio)\n"
            "- cross_lingual: Cross-lingual synthesis using reference voice\n"
            "- instruct: Instruction-based synthesis (emotions, speed, etc.)\n"
            "- sft: Use pre-trained speaker voices"
        ),
    )
    prompt_text: Optional[str] = Field(
        default=None,
        description="Text transcript of the prompt audio (helps with voice cloning quality)",
    )
    prompt_audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded reference audio for voice cloning (WAV format recommended)",
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Natural language instruction for speech style (e.g., 'Speak happily', 'Use a Sichuan dialect')",
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description="Pre-trained speaker ID for SFT mode (e.g., '中文女', '中文男')",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5 = half speed, 2.0 = double speed)",
    )


class TextToSpeechResponse(BaseResponse):
    """Response model for text-to-speech synthesis."""

    audio: str = Field(
        ...,
        description="Base64-encoded audio data with data URI prefix (e.g., 'data:audio/wav;base64,...')",
    )
    sample_rate: int = Field(
        ...,
        description="Audio sample rate in Hz",
    )
    duration_seconds: float = Field(
        ...,
        description="Duration of generated audio in seconds",
    )
    model: str = Field(
        ...,
        description="Model used for synthesis",
    )
    mode: str = Field(
        ...,
        description="TTS mode used for synthesis",
    )
