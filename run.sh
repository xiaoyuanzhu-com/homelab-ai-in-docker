#!/bin/bash

# Development server runner with proper data directory exclusion
# This prevents server reloads when models are downloaded to data/

uv run uvicorn main:app --reload --reload-dir src
