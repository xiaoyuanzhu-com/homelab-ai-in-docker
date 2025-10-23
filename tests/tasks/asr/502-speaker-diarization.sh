#!/bin/bash

# Test: Speaker Diarization
# Tests pyannote/speaker-diarization-3.1 model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../test-utils.sh"

print_test_header "asr/502-speaker-diarization" "Speaker diarization with pyannote"

# Check if test audio exists
AUDIO_PATH="${SCRIPT_DIR}/../../fixtures/audio/sample-conversation.wav"
if [ ! -f "$AUDIO_PATH" ]; then
    # Fallback to regular ASR audio
    AUDIO_PATH="${SCRIPT_DIR}/../../fixtures/audio/sample-asr.wav"
    if [ ! -f "$AUDIO_PATH" ]; then
        print_result "SKIP" "Test audio not found"
        echo "Please add a sample audio file for diarization testing. See fixtures/README.md"
        exit 0
    fi
fi

echo "Test 1: Speaker diarization with pyannote"
upload_file "/api/automatic-speech-recognition" "$AUDIO_PATH" "audio" \
    "model_id=pyannote/speaker-diarization-3.1"

assert_status_200 && \
assert_field_exists "request_id" && \
assert_field_exists "segments" && \
assert_array_not_empty "segments" && \
assert_field_not_empty "request_id"

# Display segments preview
if [ -n "$RESPONSE_BODY" ]; then
    echo ""
    echo "Diarization segments found:"
    echo "$RESPONSE_BODY" | grep -o '"segments":\[[^]]*\]' | head -c 300
    echo "..."
fi

print_summary
exit $?
