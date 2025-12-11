# Worker Architecture Design

## Goals

1. **All tasks in worker subprocesses** - Uniform architecture, clean separation (ML and non-ML alike)
2. **GPU memory protection** - Coordinator ensures only one GPU-heavy operation at a time
3. **Worker lingering** - Keep workers alive for 60s after task completion
4. **Multi-env support** - Workers use isolated Python environments per framework/constraint
5. **Lean main process** - API server has no ML dependencies, just coordination

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Main Process (main env - lean)                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Application                         │   │
│  │  /api/image-ocr, /api/text-to-embedding, /api/asr, etc.         │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              WorkerCoordinator + EnvironmentManager              │   │
│  │                                                                  │   │
│  │  - Serializes GPU access (one worker active at a time)          │   │
│  │  - Ensures worker env is installed before spawning              │   │
│  │  - Manages worker lifecycle (spawn, reuse, terminate)           │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
└──────────────────────────────────┼──────────────────────────────────────┘
                                   │ spawns / HTTP
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  OCR Worker      │    │  Embedding       │    │  ASR Worker      │
│  (paddle env)    │    │  Worker          │    │  (whisper env)   │
│                  │    │  (transformers)  │    │                  │
│  localhost:PORT1 │    │  localhost:PORT2 │    │  localhost:PORT3 │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Environment System

### Main Env (API only)

The main environment is lean - just the API server and coordinator, no task execution:

- `fastapi`, `uvicorn`, `pydantic`, `httpx`
- `psutil`, `nvidia-ml-py` (hardware info)
- `mcp` (MCP server)

**No task dependencies** - no torch, transformers, paddleocr, whisperx, crawl4ai, markitdown, screenitshot, etc. All tasks run in workers.

### Worker Environments

Each environment has an ID that models reference in their manifest:

| Env ID | Models/Libs | Key Dependencies | Reason |
|--------|-------------|------------------|--------|
| `transformers` | MinerU, Jina-VLM, Moondream, Jina-Embed, Qwen-Embed, Gemma, Qwen-Text | torch, transformers>=4.51, sentence-transformers, accelerate, flash-attn | Standard HF/PyTorch stack |
| `deepseek` | DeepSeek-OCR (+ future DeepSeek models) | torch, transformers==4.47.1, flash-attn, bitsandbytes | Pinned old transformers |
| `paddle` | PaddleOCR-VL, PP-OCRv5 | paddlepaddle-gpu, paddleocr, paddlex, custom safetensors | PaddlePaddle framework |
| `whisper` | Whisper-v3, Whisper-turbo, WhisperX, pyannote/* | torch, whisperx, pyannote.audio, librosa, torchaudio | Audio processing stack |
| `hunyuan` | HunyuanOCR | torch, transformers@git-commit | Unreleased transformers feature |
| `crawl4ai` | Crawl4AI | crawl4ai, playwright, playwright-stealth | Web crawling |
| `markitdown` | MarkItDown | markitdown[all] | Document to markdown |
| `screenitshot` | ScreenItShot | screenitshot, pillow | Document to screenshot |

### Model/Lib → Env Mapping

Models declare their environment in `models.json`, libs in `libs.json`:

```json
{
  "id": "deepseek-ai/DeepSeek-OCR",
  "python_env": "deepseek"
}
```

Models without `python_env` default to `transformers`. Libs must always specify `python_env`.

### Directory Layout

```
homelab-ai-in-docker/
├── pyproject.toml              # Main API (lean, no ML deps)
├── .venv/                      # Main env
│
└── envs/
    ├── transformers/           # Default HF/PyTorch worker env
    │   ├── pyproject.toml
    │   ├── .python-version     # 3.13
    │   └── uv.lock
    │
    ├── deepseek/               # DeepSeek models
    │   ├── pyproject.toml
    │   ├── .python-version     # 3.12 (flash-attn wheel compat)
    │   └── uv.lock
    │
    ├── paddle/                 # PaddlePaddle models
    │   ├── pyproject.toml
    │   ├── .python-version
    │   └── uv.lock
    │
    ├── whisper/                # Audio/ASR models
    │   ├── pyproject.toml
    │   ├── .python-version
    │   └── uv.lock
    │
    ├── hunyuan/                # HunyuanOCR
    │   ├── pyproject.toml
    │   ├── .python-version
    │   └── uv.lock
    │
    ├── crawl4ai/               # Web crawling
    │   ├── pyproject.toml
    │   ├── .python-version
    │   └── uv.lock
    │
    ├── markitdown/             # Document to markdown
    │   ├── pyproject.toml
    │   ├── .python-version
    │   └── uv.lock
    │
    └── screenitshot/           # Document to screenshot
        ├── pyproject.toml
        ├── .python-version
        └── uv.lock
```

### Why Standalone Projects (Not uv Workspaces)

We use standalone projects because:
- Workspaces share a single lockfile and require consistent dependencies
- We need **conflicting versions** of the same package (e.g., different `transformers` versions)
- See [uv docs](https://docs.astral.sh/uv/concepts/projects/workspaces/): "Workspaces are not suited for cases in which members have conflicting requirements"

### On-Demand Environment Installation

Environments are installed on first use:

1. Request arrives for model requiring `whisper` env
2. Coordinator checks if `envs/whisper/.venv` exists
3. If not, runs `uv sync --frozen` in that directory
4. If `post_install.sh` exists, runs it with venv activated
5. Spawns worker once env is ready

This enables:
- Smaller Docker images (ship templates, not installed envs)
- Pay-as-you-go disk usage
- Fresh containers start fast, install envs on demand

### Post-Install Scripts

Some environments require additional setup beyond `uv sync` (e.g., Playwright browser installation). Add a `post_install.sh` script to the environment template.

**Important:** Scripts must be **idempotent** - they run on every install/update, so they must check for existing installations before running expensive operations.

Use the `ensure_xxx` pattern:
```bash
#!/bin/bash
# envs/crawl4ai/post_install.sh
set -e

# 1. Check if already installed
BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
if ls "$BROWSERS_PATH"/chromium*/chrome-linux/chrome 2>/dev/null; then
    echo "Playwright chromium already installed, skipping"
    exit 0
fi

# 2. Install only if not present
echo "Installing Playwright chromium..."
playwright install chromium
```

**Do NOT** just run install commands blindly:
```bash
# BAD - always reinstalls, slow and wasteful
playwright install chromium

# GOOD - check first, install only if needed
if [ ! -f "$EXPECTED_PATH" ]; then
    playwright install chromium
fi
```

The script runs with:
- Working directory: environment install location
- `PATH` includes `.venv/bin` (venv activated)
- Must exit 0 on success

Environments with post-install scripts:
- `crawl4ai` - Ensures Playwright browsers installed
- `screenitshot` - Ensures Playwright browsers installed

### Automatic Environment Updates

When you update an environment template (e.g., add a dependency to `envs/transformers/pyproject.toml` or modify `post_install.sh`), deployed Docker services auto-sync on next use:

1. New Docker image deployed with updated template files
2. Request arrives for model in that env
3. Coordinator compares template files with installed copies
4. If different → status is `OUTDATED` → triggers reinstall
5. `uv sync --frozen` runs, then `post_install.sh` if present
6. Worker spawns with new dependencies

**No user interaction required.** The comparison is a simple byte comparison of tracked files (`pyproject.toml`, `post_install.sh`) - the installed copy in the data volume vs the template in the Docker image.

## Worker Protocol

All workers expose the same HTTP interface:

```
GET  /healthz     -> {"status": "ok", "model": "...", "ready": true}
POST /infer       -> Task-specific request/response JSON
POST /shutdown    -> Graceful shutdown request
GET  /info        -> {"model": "...", "memory_mb": ..., "load_time_ms": ...}
```

## Worker Lifecycle

```
                    spawn
    IDLE ─────────────────────► STARTING
     ▲                              │
     │                              │ /healthz OK
     │                              ▼
     │    idle_timeout          READY ◄──────────────┐
     └────────────────────────────  │                │
                                    │ /infer         │
                                    ▼                │
                                BUSY ────────────────┘
                                    │                done
                                    │
                          /shutdown or crash
                                    ▼
                               TERMINATED
```

**Worker lingering:** Workers self-terminate after 60s idle. The coordinator observes worker state and spawns new workers when needed.

## GPU Memory Protection

**Problem:** Multiple workers loading models simultaneously causes OOM.

**Solution:** Single `asyncio.Lock` in coordinator. Acquired before:
- Spawning a new worker (model loading)
- Sending inference request

**Constraint:** Requires single Uvicorn worker (`--workers 1`). Acceptable because:
- Single GPU = single bottleneck
- Model loading is expensive, can't parallelize
- Homelab app doesn't need distributed locking complexity

## File Structure

```
src/
  worker/
    __init__.py
    base.py                    # BaseWorker abstract class
    coordinator.py             # WorkerCoordinator (GPU serialization)
    env_manager.py             # EnvironmentManager (on-demand install)
    protocol.py                # Shared request/response models
    utils.py                   # Port finding, process management

    workers/                   # Individual worker implementations
      __init__.py
      embedding_worker.py
      captioning_worker.py
      text_generation_worker.py
      asr_worker.py
      ocr_worker.py
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `worker_idle_timeout_seconds` | 60 | How long workers linger after last request |
| `worker_startup_timeout_seconds` | 120 | Max time to wait for worker `/healthz` |
| `worker_request_timeout_seconds` | 300 | Max time for a single inference request |

## API Endpoints

### Environment Management

```
GET  /api/environments              # List all envs and status
GET  /api/environments/{env_id}     # Get specific env status
POST /api/environments/{env_id}/install  # Pre-install an env
DELETE /api/environments/{env_id}   # Delete to free disk space
```

### Health Endpoint

```json
GET /api/health
{
    "status": "healthy",
    "workers": {
        "embedding:jina-embeddings-v3": {
            "state": "READY",
            "env": "transformers",
            "port": 50001,
            "idle_seconds": 12.5
        }
    },
    "environments": {
        "transformers": {"status": "ready", "size_mb": 2500},
        "whisper": {"status": "not_installed"},
        "paddle": {"status": "installing"},
        "deepseek": {"status": "outdated", "size_mb": 1800}
    }
}
```

## Docker Strategy

**Lean image approach:**
- Docker image contains only main env + env templates (~1GB vs ~10GB+)
- Worker environments installed on-demand to data volume
- First request for new env takes 1-2 min (install), subsequent requests are fast

```dockerfile
# Copy only API server dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy environment templates (pyproject.toml + uv.lock only, no .venv)
COPY envs/ ./envs/

# Envs installed to data volume on demand
ENV HAID_ENVS_DIR=/haid/data/envs
```

## When to Add a New Environment

Add a new env when:
1. Model has **hard conflicts** with existing envs (different version of same package)
2. Conflict **cannot be resolved** by updating an existing env
3. Model is **worth supporting** despite the overhead

Don't add a new env for:
- Models that work with an existing env
- Models with optional dependencies (mark as unavailable if deps missing)
- Experimental models you're just testing
