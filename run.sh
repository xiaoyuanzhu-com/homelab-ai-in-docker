#!/bin/bash

# Development server runner with proper data directory exclusion
# This prevents server reloads when models are downloaded to data/

# Skip GPU extras on macOS/Windows since flash-attn is Linux-only
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # macOS or Windows - skip GPU extras
    PADDLE_PDX_CACHE_HOME="$(pwd)/data/paddlex" uv run --no-extra gpu uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src
else
    # Linux - include all extras
    PADDLE_PDX_CACHE_HOME="$(pwd)/data/paddlex" uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src
fi
