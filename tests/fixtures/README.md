# Test Fixtures

This directory contains test assets (images, audio files) used across integration tests.

## Directory Structure

- `images/` - Sample images for OCR and captioning tests
- `audio/` - Sample audio files for ASR and speaker diarization tests

## Adding Test Fixtures

### Images

For OCR and image captioning tests, add sample images to the `images/` directory:

**Recommended images:**
- `sample-ocr.jpg` - A page with text (invoice, receipt, document page, etc.)
- `sample-doc.png` - A formatted document with headings and tables
- `sample-scene.jpg` - A photo with objects/people for captioning

You can:
1. Use your own images
2. Download from free stock photo sites (Unsplash, Pexels)
3. Create simple test images with text

**Example command to create a simple test image with text:**
```bash
# Using ImageMagick (if installed)
convert -size 800x600 xc:white \
  -pointsize 24 -draw "text 50,100 'Sample Invoice'" \
  -pointsize 16 -draw "text 50,150 'Date: 2025-01-15'" \
  -pointsize 16 -draw "text 50,180 'Total: $123.45'" \
  images/sample-ocr.jpg
```

### Audio

For ASR and speaker diarization tests, add sample audio files to the `audio/` directory:

**Recommended audio files:**
- `sample-asr.wav` - Short speech recording (10-30 seconds)
- `sample-conversation.wav` - Multi-speaker conversation (for diarization)

You can:
1. Record your own audio (using any voice recorder app)
2. Use text-to-speech to generate sample audio
3. Download from free audio sites

**Example using espeak (if installed):**
```bash
espeak "This is a test recording for automatic speech recognition" \
  -w audio/sample-asr.wav
```

**Example using ffmpeg to convert mp3 to wav:**
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 audio/sample-asr.wav
```

## Current Fixtures

<!-- Update this list as you add fixtures -->
- [ ] `images/sample-ocr.jpg` - Not yet added
- [ ] `images/sample-doc.png` - Not yet added
- [ ] `images/sample-scene.jpg` - Not yet added
- [ ] `audio/sample-asr.wav` - Not yet added
- [ ] `audio/sample-conversation.wav` - Not yet added

## Notes

- Keep file sizes reasonable (< 5 MB each)
- Use common formats: JPG/PNG for images, WAV/MP3 for audio
- Tests are designed to be "loose" - they check for reasonable output, not exact matches
- You can use the same image across multiple tests (e.g., all OCR tests can use the same image)
