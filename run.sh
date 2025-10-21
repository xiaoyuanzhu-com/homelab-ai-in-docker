#!/bin/bash

# Development server runner with proper data directory exclusion
# This prevents server reloads when models are downloaded to data/

uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src
