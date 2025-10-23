#!/bin/bash

# Test: Automatic Speech Recognition - Whisper Turbo
# Tests openai/whisper-large-v3-turbo model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "asr/501-whisper-turbo" "Whisper Turbo transcription"

# Check if test audio exists
AUDIO_PATH="${SCRIPT_DIR}/../../fixtures/audio/sample-asr.wav"
if [ ! -f "$AUDIO_PATH" ]; then
    print_result "SKIP" "Test audio not found: $AUDIO_PATH"
    echo "Please add a sample audio file for ASR testing. See fixtures/README.md"
    exit 0
fi

echo "Test 1: Transcribe audio with Whisper Turbo"
upload_file "/api/automatic-speech-recognition" "$AUDIO_PATH" "audio" \
    "model_id=openai/whisper-large-v3-turbo"

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "text" && \
assert_min_length "text" 5 "transcribed text" && \
assert_field_not_empty "request_id"

# Display transcription
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Transcribed text:"
    echo "$RESPONSE_BODY" | grep -o '"text":"[^"]*"' | cut -d'"' -f4
fi

print_summary
exit $?
