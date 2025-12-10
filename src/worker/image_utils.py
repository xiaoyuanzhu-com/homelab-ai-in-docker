"""Image utilities for worker preprocessing.

Provides HEIC/HEIF support and standardized image decoding.
"""

from __future__ import annotations

import base64
import io
import logging

from PIL import Image

# Register HEIC/HEIF support with Pillow
# This must be done before any Image.open() calls
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

logger = logging.getLogger(__name__)


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image data to PIL Image.

    Supports all PIL-supported formats including HEIC/HEIF when pillow-heif
    is installed. The image is converted to RGB for consistency across models.

    Args:
        image_data: Base64-encoded image string, optionally with data URI prefix.

    Returns:
        PIL Image in RGB mode.
    """
    # Strip data URI prefix if present
    if image_data.startswith("data:image"):
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB for model compatibility
    # HEIC images may be in various modes (RGB, RGBA, etc.)
    return img.convert("RGB")
