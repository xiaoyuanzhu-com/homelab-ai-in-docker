"""Scene text preprocessing for real-world photos.

Applies perspective correction and image enhancement to improve OCR accuracy
on photos taken at angles (signs, documents on tables, etc.).

Pipeline:
1. Edge detection (Canny)
2. Contour detection to find text region
3. Perspective transform to top-down view
4. Deskew (optional rotation correction)
5. Adaptive binarization for clean text

This is similar to what scanner apps (CamScanner, Apple Notes) do.
"""

import logging
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2  # Fail fast if opencv not installed

logger = logging.getLogger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in: top-left, top-right, bottom-right, bottom-left order.

    Args:
        pts: Array of 4 points (x, y)

    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: top-left has smallest, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Difference: top-right has smallest, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to obtain top-down view.

    Args:
        image: Input image (BGR or grayscale)
        pts: 4 corner points of the region to transform

    Returns:
        Warped image with top-down perspective
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image (max of top and bottom edge)
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image (max of left and right edge)
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points for top-down view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Compute perspective transform and apply
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def detect_document_contour(
    image: np.ndarray,
    min_area_ratio: float = 0.1,
    max_area_ratio: float = 0.95,
) -> Optional[np.ndarray]:
    """Detect the largest quadrilateral contour (document/sign boundary).

    Args:
        image: Input image (BGR)
        min_area_ratio: Minimum contour area as ratio of image area
        max_area_ratio: Maximum contour area as ratio of image area

    Returns:
        4 corner points if found, None otherwise
    """
    height, width = image.shape[:2]
    image_area = height * width

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with Canny
    # Use adaptive thresholds based on median pixel intensity
    median = np.median(blurred)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:  # Check top 10 largest contours
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        # Skip if too small or too large
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If it's a quadrilateral, we found our document
        if len(approx) == 4:
            logger.debug(f"Found document contour with area ratio {area_ratio:.2%}")
            return approx.reshape(4, 2)

    return None


def compute_skew_angle(image: np.ndarray) -> float:
    """Compute skew angle of text lines using Hough transform.

    Args:
        image: Grayscale or BGR image

    Returns:
        Skew angle in degrees (positive = clockwise rotation needed)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 8,  # Min line length = 1/8 of width
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        return 0.0

    # Compute angles of all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider nearly horizontal lines (within 45 degrees)
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    # Return median angle (robust to outliers)
    return float(np.median(angles))


def deskew_image(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """Rotate image to correct skew.

    Args:
        image: Input image
        angle: Skew angle in degrees. If None, computed automatically.

    Returns:
        Deskewed image
    """
    if angle is None:
        angle = compute_skew_angle(image)

    # Skip if angle is negligible
    if abs(angle) < 0.5:
        return image

    logger.debug(f"Deskewing by {angle:.2f} degrees")

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Rotate around center
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust rotation matrix for translation
    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]

    # Apply rotation with white background
    rotated = cv2.warpAffine(
        image, M, (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )

    return rotated


def adaptive_binarize(image: np.ndarray) -> np.ndarray:
    """Apply adaptive binarization for clean text on white background.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Binary image (black text on white background)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply adaptive thresholding
    # ADAPTIVE_THRESH_GAUSSIAN_C works better for varying lighting
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,  # Size of neighborhood for threshold calculation
        C=2  # Constant subtracted from mean
    )

    return binary


def enhance_for_ocr(
    image: np.ndarray,
    target_dpi: int = 300,
    current_dpi: int = 72,
) -> np.ndarray:
    """Enhance image resolution and contrast for OCR.

    Args:
        image: Input image
        target_dpi: Target DPI for OCR (300 recommended)
        current_dpi: Estimated current DPI

    Returns:
        Enhanced image
    """
    # Scale up if needed
    scale = target_dpi / current_dpi
    if scale > 1.0 and scale < 4.0:  # Don't scale too much
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Scaled image by {scale:.2f}x to {new_width}x{new_height}")

    # Denoise
    if len(image.shape) == 3:
        image = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)
    else:
        image = cv2.fastNlMeansDenoising(image, h=10)

    return image


def preprocess_scene_image(
    image: Image.Image,
    apply_perspective: bool = True,
    apply_deskew: bool = True,
    apply_binarize: bool = False,
    apply_enhance: bool = True,
) -> Tuple[Image.Image, dict]:
    """Full preprocessing pipeline for scene text images.

    Args:
        image: Input PIL Image
        apply_perspective: Whether to detect and correct perspective
        apply_deskew: Whether to correct text line skew
        apply_binarize: Whether to binarize (black/white).
                        Set False if OCR model handles color well.
        apply_enhance: Whether to enhance resolution/contrast

    Returns:
        Tuple of (processed PIL Image, metadata dict with applied transforms)
    """
    # Convert PIL to OpenCV format (BGR)
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    metadata = {
        "original_size": image.size,
        "perspective_corrected": False,
        "deskew_angle": 0.0,
        "binarized": False,
        "enhanced": False,
    }

    processed = img_bgr

    # Step 1: Perspective correction
    if apply_perspective:
        contour = detect_document_contour(processed)
        if contour is not None:
            processed = four_point_transform(processed, contour)
            metadata["perspective_corrected"] = True
            logger.info("Applied perspective correction")
        else:
            logger.debug("No document contour detected, skipping perspective correction")

    # Step 2: Deskew
    if apply_deskew:
        angle = compute_skew_angle(processed)
        if abs(angle) >= 0.5:
            processed = deskew_image(processed, angle)
            metadata["deskew_angle"] = angle
            logger.info(f"Applied deskew correction: {angle:.2f}°")

    # Step 3: Enhancement
    if apply_enhance:
        processed = enhance_for_ocr(processed)
        metadata["enhanced"] = True

    # Step 4: Binarization (optional - many modern OCR models work better with color)
    if apply_binarize:
        processed = adaptive_binarize(processed)
        metadata["binarized"] = True
        logger.info("Applied adaptive binarization")

    # Convert back to PIL
    if len(processed.shape) == 2:
        # Grayscale/binary
        result = Image.fromarray(processed, mode="L")
    else:
        # Color (BGR to RGB)
        result = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    metadata["processed_size"] = result.size
    logger.info(f"Preprocessing complete: {metadata}")

    return result, metadata


def is_scene_image(image: Image.Image) -> bool:
    """Heuristic to detect if image is a scene photo vs clean document.

    Scene photos typically have:
    - Color variation in background
    - Non-rectangular text regions
    - Variable lighting

    Args:
        image: Input PIL Image

    Returns:
        True if image appears to be a scene photo
    """
    # Convert to OpenCV
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Check 1: Color variance in border regions
    # Clean documents have uniform white/light borders
    height, width = img_bgr.shape[:2]
    border_size = min(height, width) // 10

    # Sample borders
    top = img_bgr[:border_size, :, :]
    bottom = img_bgr[-border_size:, :, :]
    left = img_bgr[:, :border_size, :]
    right = img_bgr[:, -border_size:, :]

    # Compute variance
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])
    variance = np.var(border_pixels)

    # High variance suggests scene photo
    # Clean documents typically have variance < 500
    is_scene = variance > 1000

    logger.debug(f"Border variance: {variance:.0f}, is_scene: {is_scene}")

    return is_scene
