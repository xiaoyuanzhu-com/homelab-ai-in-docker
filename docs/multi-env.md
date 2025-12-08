# Multi-Environment Architecture

## Problem

Different AI models have conflicting Python dependency requirements:

| Model | transformers requirement |
|-------|-------------------------|
| DeepSeek-OCR | ==4.47.1 (uses `LlamaFlashAttention2`) |
| MinerU | >=4.56.0 |
| HunyuanOCR | git commit (not in stable release) |
| Most others | >=4.51.0 |

Python cannot have multiple versions of the same package in one environment.

## Solution

Use multiple virtual environments:

```
.venv/                  # Main env - used by most tasks
.venv-deepseek-ocr/     # DeepSeek-OCR specific (transformers==4.47.1)
.venv-hunyuan/          # HunyuanOCR specific (transformers@git)
```

### Why this works

1. **Workers are subprocesses** - The app already spawns model workers as separate Python processes
2. **Workers communicate via HTTP** - No shared memory, just localhost REST calls
3. **Easy to specify interpreter** - Just change which `python` to run

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Main Process                         │
│                    (.venv)                              │
│                                                         │
│  FastAPI server, routing, worker management             │
└─────────────────────┬───────────────────────────────────┘
                      │ spawns workers
          ┌───────────┼───────────┐
          ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Worker    │ │   Worker    │ │   Worker    │
│  (.venv)    │ │(.venv-deepseek)│(.venv-hunyuan)│
│             │ │             │ │             │
│ MinerU, etc │ │ DeepSeek-OCR│ │ HunyuanOCR  │
└─────────────┘ └─────────────┘ └─────────────┘
      ▲               ▲               ▲
      └───────────────┴───────────────┘
              HTTP on localhost
```

## Implementation

### 1. Create additional environments

```bash
# DeepSeek-OCR environment
uv venv .venv-deepseek-ocr --python 3.13
source .venv-deepseek-ocr/bin/activate
uv pip install torch==2.6.0 transformers==4.47.1 flash-attn==2.7.3 \
    bitsandbytes accelerate einops addict easydict safetensors pillow
deactivate

# HunyuanOCR environment (when needed)
uv venv .venv-hunyuan --python 3.13
source .venv-hunyuan/bin/activate
uv pip install torch==2.6.0 \
    git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4 \
    pillow accelerate
deactivate
```

### 2. Add env mapping to model config

In `models.json`, add `python_env` field:

```json
{
  "id": "deepseek-ai/DeepSeek-OCR",
  "architecture": "deepseek",
  "python_env": "deepseek-ocr",
  ...
}
```

### 3. Update worker manager

In `src/worker/manager.py`, resolve Python interpreter:

```python
def _get_python_for_model(self, model_config: dict) -> str:
    """Get Python interpreter path for model's environment."""
    env_name = model_config.get("python_env")
    if env_name:
        # Check for model-specific env
        env_path = Path(f".venv-{env_name}/bin/python")
        if env_path.exists():
            return str(env_path.resolve())
    # Default to main env
    return sys.executable
```

Then use it when spawning:

```python
python_path = self._get_python_for_model(model_config)
proc = subprocess.Popen(
    [python_path, "-m", "src.worker.image_ocr_worker", ...],
    ...
)
```

### 4. Docker: single container, multiple venvs

In `Dockerfile`:

```dockerfile
# Main environment
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e .

# DeepSeek-OCR environment
RUN uv venv /app/.venv-deepseek-ocr && \
    . /app/.venv-deepseek-ocr/bin/activate && \
    uv pip install torch==2.6.0 transformers==4.47.1 ...

# Set main env as default
ENV PATH="/app/.venv/bin:$PATH"
```

## When to add a new environment

Add a new env only when:

1. A model has **hard conflicts** with the main env (different version of same package)
2. The conflict **cannot be resolved** by updating the main env
3. The model is **worth supporting** despite the overhead

Don't add a new env for:

- Models that work with the main env
- Models with optional dependencies (just mark as unavailable if deps missing)
- Experimental models you're just testing

## Tradeoffs

### Pros
- Support models with conflicting deps
- No changes to worker protocol (still HTTP)
- Single Docker image
- Environments created on-demand

### Cons
- Disk space (each env ~2-5GB with PyTorch)
- Build time (multiple pip installs)
- Maintenance (keep envs in sync for shared deps)
- Complexity (more things to debug)

## Alternative: Patch model code

For models with minor incompatibilities (like DeepSeek-OCR's import issue), patching the cached model code might be simpler:

```python
# In _load_deepseek(), before loading:
self._patch_deepseek_imports()
```

This avoids multi-env but is fragile (breaks on model updates).
