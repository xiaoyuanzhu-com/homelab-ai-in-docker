"""Fallback implementation for environments without native torchcodec support."""

from __future__ import annotations

import importlib.machinery
import io
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import os

_STUB_WARNING = (
    "torchcodec native extension could not be loaded. Falling back to a lightweight "
    "Python implementation that reads entire audio files into memory. Install the "
    "official torchcodec wheels with matching FFmpeg and PyTorch libraries for the "
    "most efficient speaker diarization."
)


@dataclass
class AudioStreamMetadata:
    """Simplified metadata container mimicking torchcodec's structure."""

    sample_rate: int
    num_channels: int
    bits_per_sample: int
    duration_seconds_from_header: float


class AudioSamples:
    """Minimal torchcodec AudioSamples replacement."""

    def __init__(self, data: torch.Tensor, sample_rate: int) -> None:
        self.data = data
        self.sample_rate = int(sample_rate)


class AudioDecoder:
    """Decode audio using librosa when native torchcodec is unavailable."""

    def __init__(self, audio: Any) -> None:
        self._waveform, self._sample_rate = _load_waveform(audio)
        self.metadata = AudioStreamMetadata(
            sample_rate=self._sample_rate,
            num_channels=self._waveform.shape[0],
            bits_per_sample=32,
            duration_seconds_from_header=self._waveform.shape[1] / self._sample_rate,
        )

    def get_samples_played_in_range(self, start: float, end: float) -> AudioSamples:
        if end < start:
            raise ValueError("end must be greater than or equal to start.")

        start_index = max(0, int(round(start * self._sample_rate)))
        end_index = max(start_index, int(round(end * self._sample_rate)))
        data = self._waveform[:, start_index:end_index].clone()
        return AudioSamples(data, self._sample_rate)


def ensure_torchcodec() -> None:
    """Make sure a usable torchcodec module is present on sys.modules."""

    try:
        import torchcodec  # type: ignore  # noqa: F401

        return
    except Exception as exc:  # pragma: no cover - only triggered in problematic envs
        install_stub(exc)


def install_stub(error: Exception | None = None) -> None:
    """Register this module and its submodules as `torchcodec` fallbacks."""

    if "torchcodec" in sys.modules:
        return

    module = types.ModuleType("torchcodec")
    module.AudioDecoder = AudioDecoder
    module.AudioSamples = AudioSamples
    module.AudioStreamMetadata = AudioStreamMetadata
    module.__all__ = ["AudioDecoder", "AudioSamples", "AudioStreamMetadata"]
    module.__doc__ = __doc__
    module.__spec__ = importlib.machinery.ModuleSpec(name="torchcodec", loader=None)

    decoders_module = types.ModuleType("torchcodec.decoders")
    decoders_module.AudioDecoder = AudioDecoder
    decoders_module.AudioStreamMetadata = AudioStreamMetadata
    decoders_module.__all__ = ["AudioDecoder", "AudioStreamMetadata"]
    decoders_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchcodec.decoders", loader=None
    )

    samplers_module = types.ModuleType("torchcodec.samplers")
    samplers_module.__all__ = []
    samplers_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchcodec.samplers", loader=None
    )

    core_module = types.ModuleType("torchcodec._core")
    core_module.AudioStreamMetadata = AudioStreamMetadata
    core_module.AudioSamples = AudioSamples
    core_module.__all__ = ["AudioStreamMetadata", "AudioSamples"]
    core_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchcodec._core", loader=None
    )

    core_ops_module = types.ModuleType("torchcodec._core.ops")

    def _noop_loader(*_: Any, **__: Any) -> bool:
        return False

    core_ops_module.load_torchcodec_shared_libraries = _noop_loader
    core_ops_module.__all__ = ["load_torchcodec_shared_libraries"]
    core_ops_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchcodec._core.ops", loader=None
    )

    module.decoders = decoders_module
    module.samplers = samplers_module

    sys.modules["torchcodec"] = module
    sys.modules["torchcodec.decoders"] = decoders_module
    sys.modules["torchcodec.samplers"] = samplers_module
    sys.modules["torchcodec._core"] = core_module
    sys.modules["torchcodec._core.ops"] = core_ops_module

    warnings.warn(_STUB_WARNING, UserWarning, stacklevel=2)
    if error is not None:
        warnings.warn(
            f"Original torchcodec import error: {error!r}", UserWarning, stacklevel=2
        )


def _load_waveform(audio: Any) -> tuple[torch.Tensor, int]:
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")
        source: Any = str(path)
    elif isinstance(audio, (bytes, bytearray)):
        source = io.BytesIO(audio)
    elif hasattr(audio, "read"):
        current_position = None
        if hasattr(audio, "tell"):
            try:
                current_position = audio.tell()
            except Exception:
                current_position = None

        try:
            if hasattr(audio, "seek"):
                audio.seek(0)
            raw = audio.read()
        finally:
            if current_position is not None and hasattr(audio, "seek"):
                try:
                    audio.seek(current_position)
                except Exception:
                    pass
        source = io.BytesIO(raw)
    else:
        raise TypeError("AudioDecoder expects a filesystem path or a file-like object.")

    waveform, sample_rate = librosa.load(source, sr=None, mono=False)
    if waveform.ndim == 1:
        waveform = np.expand_dims(waveform, 0)

    tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
    return tensor, int(sample_rate)


__all__ = [
    "AudioDecoder",
    "AudioSamples",
    "AudioStreamMetadata",
    "ensure_torchcodec",
    "install_stub",
]
