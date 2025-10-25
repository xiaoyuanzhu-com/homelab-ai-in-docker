"""MCP (Model Context Protocol) router for remote AI tool access.

This module exposes the homelab AI capabilities as MCP tools, allowing
Claude Code and other MCP clients to access them remotely via Streamable HTTP.

The MCP server provides the following tools:
- crawl_web: Scrape web pages with JavaScript rendering
- embed_text: Generate embeddings from text
- generate_text: Generate text from prompts using LLMs
- caption_image: Generate captions for images
- ocr_image: Extract text from images using OCR
- transcribe_audio: Transcribe speech from audio files
- get_hardware_info: Get GPU and system hardware information
"""

import base64
import logging
from typing import Optional, List

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp_server = FastMCP(
    name="Homelab AI Services",
    version="0.1.0",
)


@mcp_server.tool()
async def crawl_web(
    url: str,
    screenshot: bool = False,
    wait_for_js: bool = True,
) -> dict:
    """
    Crawl a web page and extract its content as Markdown.

    This tool uses a headless browser with JavaScript rendering to scrape web pages,
    making it ideal for modern SPAs and JavaScript-heavy sites.

    Args:
        url: The URL to crawl
        screenshot: Whether to capture a screenshot (base64-encoded)
        wait_for_js: Whether to wait for JavaScript to execute

    Returns:
        Dictionary with:
        - url: The crawled URL
        - title: Page title
        - markdown: Content in Markdown format
        - screenshot: Base64-encoded screenshot (if requested)
        - success: Whether the crawl succeeded
    """
    from . import crawl

    try:
        result = await crawl.crawl_url(
            url=url,
            screenshot=screenshot,
            wait_for_js=wait_for_js,
        )
        return {
            "url": result["url"],
            "title": result.get("title"),
            "markdown": result["markdown"],
            "screenshot": result.get("screenshot") if screenshot else None,
            "success": result["success"],
        }
    except Exception as e:
        logger.error(f"MCP crawl_web error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp_server.tool()
async def embed_text(
    text: str | List[str],
    model: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Generate embeddings (vector representations) from text.

    Useful for semantic search, similarity comparison, and RAG applications.

    Args:
        text: Text to embed (single string or list of strings)
        model: Embedding model to use (default: all-MiniLM-L6-v2)

    Returns:
        Dictionary with:
        - embeddings: List of embedding vectors
        - model: Model used
        - dimensions: Embedding dimensionality
    """
    from . import text_to_embedding

    try:
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text

        # Get the model
        embedding_model = text_to_embedding.get_model(model)

        # Generate embeddings
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)

        return {
            "embeddings": embeddings.tolist(),
            "model": model,
            "dimensions": len(embeddings[0]) if len(embeddings) > 0 else 0,
        }
    except Exception as e:
        logger.error(f"MCP embed_text error: {e}", exc_info=True)
        return {
            "error": str(e),
        }


@mcp_server.tool()
async def generate_text(
    prompt: str,
    model: str = "qwen-0.5b",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """
    Generate text from a prompt using a language model.

    Args:
        prompt: Input prompt for text generation
        model: Model ID to use (e.g., "qwen-0.5b", "qwen-1.5b")
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling threshold

    Returns:
        Dictionary with:
        - generated_text: The generated text
        - model: Model used
        - prompt_tokens: Number of input tokens
        - completion_tokens: Number of generated tokens
    """
    from . import text_generation

    try:
        # Validate model
        text_generation.validate_model(model)

        # Get model
        model_obj, tokenizer = text_generation.get_model(model)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]

        # Move to same device as model
        import torch
        device = next(model_obj.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion_length = outputs.shape[1] - input_length

        return {
            "generated_text": generated_text,
            "model": model,
            "prompt_tokens": input_length,
            "completion_tokens": completion_length,
        }
    except Exception as e:
        logger.error(f"MCP generate_text error: {e}", exc_info=True)
        return {
            "error": str(e),
        }


@mcp_server.tool()
async def caption_image(
    image_base64: str,
    model: str = "blip-base",
    prompt: Optional[str] = None,
) -> dict:
    """
    Generate a caption for an image.

    Args:
        image_base64: Base64-encoded image data
        model: Model ID to use (e.g., "blip-base", "blip-large")
        prompt: Optional prompt to guide caption generation

    Returns:
        Dictionary with:
        - caption: Generated caption text
        - model: Model used
    """
    from . import image_captioning
    from io import BytesIO
    from PIL import Image

    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Get model
        processor, model_obj = image_captioning.get_model(model)

        # Generate caption
        if prompt:
            # Conditional generation
            inputs = processor(image, prompt, return_tensors="pt")
        else:
            # Unconditional generation
            inputs = processor(image, return_tensors="pt")

        # Move to same device as model
        import torch
        device = next(model_obj.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model_obj.generate(**inputs, max_new_tokens=50)

        # Decode
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return {
            "caption": caption,
            "model": model,
        }
    except Exception as e:
        logger.error(f"MCP caption_image error: {e}", exc_info=True)
        return {
            "error": str(e),
        }


@mcp_server.tool()
async def ocr_image(
    image_base64: str,
    model: str = "paddleocr",
) -> dict:
    """
    Extract text from an image using OCR (Optical Character Recognition).

    Args:
        image_base64: Base64-encoded image data
        model: Model ID to use (default: "paddleocr")

    Returns:
        Dictionary with:
        - text: Extracted text
        - boxes: List of bounding boxes with text and coordinates
        - model: Model used
    """
    from . import image_ocr
    from io import BytesIO
    from PIL import Image
    import numpy as np

    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Convert to numpy array
        image_array = np.array(image)

        # Get model
        ocr_model = image_ocr.get_model(model)

        # Run OCR
        result = ocr_model.ocr(image_array, cls=True)

        # Extract text and boxes
        boxes = []
        all_text = []
        if result and result[0]:
            for line in result[0]:
                box_coords, (text, confidence) = line
                boxes.append({
                    "text": text,
                    "confidence": float(confidence),
                    "box": box_coords,
                })
                all_text.append(text)

        return {
            "text": "\n".join(all_text),
            "boxes": boxes,
            "model": model,
        }
    except Exception as e:
        logger.error(f"MCP ocr_image error: {e}", exc_info=True)
        return {
            "error": str(e),
        }


@mcp_server.tool()
async def transcribe_audio(
    audio_base64: str,
    model: str = "whisper-tiny",
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe speech from an audio file.

    Args:
        audio_base64: Base64-encoded audio data (WAV, MP3, etc.)
        model: Model ID to use (e.g., "whisper-tiny", "whisper-base")
        language: Optional language code (e.g., "en", "zh")

    Returns:
        Dictionary with:
        - text: Transcribed text
        - language: Detected or specified language
        - model: Model used
    """
    from . import automatic_speech_recognition
    import tempfile
    import os

    try:
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_data)
            audio_path = f.name

        try:
            # Get model
            asr_model = automatic_speech_recognition.get_model(model)

            # Transcribe
            if language:
                result = asr_model(audio_path, language=language)
            else:
                result = asr_model(audio_path)

            return {
                "text": result["text"],
                "language": language or "auto",
                "model": model,
            }
        finally:
            # Clean up temporary file
            os.unlink(audio_path)

    except Exception as e:
        logger.error(f"MCP transcribe_audio error: {e}", exc_info=True)
        return {
            "error": str(e),
        }


@mcp_server.tool()
async def get_hardware_info() -> dict:
    """
    Get hardware information including GPU stats, memory, and system metrics.

    Returns:
        Dictionary with:
        - gpu: GPU information (if available)
        - cpu: CPU usage
        - memory: Memory usage
        - disk: Disk usage
    """
    from . import hardware

    try:
        # Get hardware stats
        stats = await hardware.get_hardware_stats()
        return stats
    except Exception as e:
        logger.error(f"MCP get_hardware_info error: {e}", exc_info=True)
        return {
            "error": str(e),
        }


# Export the FastAPI app for mounting
def get_mcp_app():
    """Get the MCP server's FastAPI application for mounting."""
    return mcp_server.get_asgi_app()
