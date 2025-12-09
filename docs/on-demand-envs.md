# On-Demand Worker Environments

## Overview

Transform the Docker image from a monolithic build with all ML dependencies to a minimal API server that creates worker environments on-demand. This reduces image size from ~10GB+ to ~1GB while enabling pay-as-you-go disk usage.

## Architecture

```
Docker Image (minimal ~1GB):
├── .venv/                     # API server only (no ML deps)
│   └── fastapi, uvicorn, httpx, pydantic, mcp
│
└── env-templates/             # Just pyproject.toml + uv.lock (~10KB each)
    ├── embedding/
    ├── captioning/
    ├── text-generation/
    ├── asr/
    ├── diarization/
    ├── ocr/
    ├── deepseek-ocr/
    └── crawl/                 # Includes Playwright

Data Volume (grows on demand):
/haid/data/
├── models/                    # Model weights (already here)
├── envs/                      # Worker environments (NEW)
│   ├── embedding/
│   │   ├── pyproject.toml     # Copied from template
│   │   ├── uv.lock            # Copied from template
│   │   └── .venv/             # Created by uv sync
│   ├── crawl/
│   │   └── .venv/             # Includes Playwright browsers
│   └── ...
└── cache/
```

## Environment Templates

Each template is a minimal uv project with locked dependencies:

### API Server (main .venv)

```toml
# pyproject.toml - API server only
[project]
name = "haid-server"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "httpx>=0.28.0",
    "pydantic>=2.0.0",
    "mcp>=1.8.0",
    "aiosqlite>=0.20.0",
    "python-multipart>=0.0.9",
    # Lightweight document processing (no ML)
    "markitdown[all]",
    "screenitshot",
]
# NO torch, transformers, paddleocr, whisperx, crawl4ai, etc.
```

### Worker Templates

Located in `env-templates/` directory:

| Template | Key Dependencies | Size Estimate |
|----------|------------------|---------------|
| `embedding` | torch, sentence-transformers | ~2.5GB |
| `captioning` | torch, transformers, accelerate | ~3GB |
| `text-generation` | torch, transformers, accelerate, bitsandbytes | ~3GB |
| `asr` | torch, whisperx, faster-whisper | ~2.5GB |
| `diarization` | torch, pyannote.audio | ~2GB |
| `ocr` | paddlepaddle, paddleocr, paddlex | ~3GB |
| `deepseek-ocr` | torch, transformers==4.47.1, flash-attn | ~4GB |
| `crawl` | crawl4ai, playwright, playwright-stealth | ~1GB |

### Template Structure

```
env-templates/embedding/
├── pyproject.toml    # Dependencies locked
├── uv.lock           # Reproducible installs
└── .python-version   # Pin Python version (3.13 or 3.12)
```

## Implementation

### 1. Environment Manager

New module: `src/worker/env_manager.py`

```python
class EnvironmentManager:
    """Manages on-demand worker environments."""

    def __init__(
        self,
        templates_dir: Path = Path("env-templates"),
        envs_dir: Path = Path("/haid/data/envs"),
    ):
        self.templates_dir = templates_dir
        self.envs_dir = envs_dir
        self._install_locks: Dict[str, asyncio.Lock] = {}

    async def ensure_environment(self, env_name: str) -> Path:
        """Ensure environment exists, creating if needed."""
        env_path = self.envs_dir / env_name
        venv_path = env_path / ".venv"

        if venv_path.exists():
            return env_path

        # Serialize installs per environment
        if env_name not in self._install_locks:
            self._install_locks[env_name] = asyncio.Lock()

        async with self._install_locks[env_name]:
            # Double-check after acquiring lock
            if venv_path.exists():
                return env_path

            await self._install_environment(env_name)

        return env_path

    async def _install_environment(self, env_name: str) -> None:
        """Install environment from template."""
        template_path = self.templates_dir / env_name
        env_path = self.envs_dir / env_name

        # Copy template files
        env_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(template_path / "pyproject.toml", env_path)
        shutil.copy(template_path / "uv.lock", env_path)
        if (template_path / ".python-version").exists():
            shutil.copy(template_path / ".python-version", env_path)

        # Run uv sync (creates .venv and installs deps)
        proc = await asyncio.create_subprocess_exec(
            "uv", "sync", "--frozen",
            cwd=env_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install {env_name}: {stderr.decode()}")

        # Post-install hooks (e.g., Playwright browsers)
        await self._post_install(env_name, env_path)

    async def _post_install(self, env_name: str, env_path: Path) -> None:
        """Run post-install hooks for specific environments."""
        if env_name == "crawl":
            # Install Playwright browsers
            await asyncio.create_subprocess_exec(
                "uv", "run", "playwright", "install", "chromium",
                cwd=env_path,
            )

    def get_status(self, env_name: str) -> dict:
        """Get environment status."""
        env_path = self.envs_dir / env_name
        venv_path = env_path / ".venv"

        if not venv_path.exists():
            return {"status": "not_installed"}

        # Calculate size
        size_bytes = sum(
            f.stat().st_size for f in venv_path.rglob("*") if f.is_file()
        )

        return {
            "status": "ready",
            "size_mb": size_bytes // (1024 * 1024),
            "path": str(env_path),
        }

    def list_environments(self) -> dict:
        """List all available environments and their status."""
        templates = [d.name for d in self.templates_dir.iterdir() if d.is_dir()]
        return {
            name: self.get_status(name)
            for name in templates
        }
```

### 2. Coordinator Integration

Update `src/worker/coordinator.py`:

```python
class WorkerCoordinator:
    def __init__(self):
        self._env_manager = EnvironmentManager()
        # ... existing init

    async def _spawn_worker(self, task: str, model_config: dict) -> WorkerHandle:
        # Determine which environment to use
        env_name = self._get_env_for_task(task, model_config)

        # Ensure environment is installed (may take minutes first time)
        env_path = await self._env_manager.ensure_environment(env_name)

        # Spawn worker in that environment
        worker_module = f"src.worker.workers.{task.replace('-', '_')}_worker"
        cmd = ["uv", "run", "python", "-m", worker_module, ...]

        proc = subprocess.Popen(cmd, cwd=env_path, ...)
        # ... rest of spawn logic

    def _get_env_for_task(self, task: str, model_config: dict) -> str:
        """Map task + model to environment name."""
        # Check if model specifies a custom env
        if python_env := model_config.get("python_env"):
            return python_env

        # Default mapping
        return {
            "embedding": "embedding",
            "captioning": "captioning",
            "text-generation": "text-generation",
            "asr": "asr",
            "speaker-diarization": "diarization",
            "ocr": "ocr",
            "crawl": "crawl",
        }.get(task, "embedding")  # fallback
```

### 3. Crawl Worker

Convert crawl from direct execution to worker pattern:

New file: `src/worker/workers/crawl_worker.py`

```python
class CrawlWorker(BaseWorker):
    """Web crawling worker using crawl4ai."""

    task_name = "crawl"

    def load_model(self) -> Any:
        """Initialize browser (lazy - actual browser starts per request)."""
        from crawl4ai import AsyncWebCrawler, BrowserConfig

        self.browser_config = BrowserConfig(
            headless=True,
            viewport_width=1280,
            viewport_height=800,
        )
        return None  # No persistent model

    def infer(self, payload: dict) -> dict:
        """Run crawl request."""
        # Run async crawl in sync context
        return asyncio.run(self._crawl(payload))

    async def _crawl(self, payload: dict) -> dict:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

        url = payload["url"]
        config = CrawlerRunConfig(
            wait_until=payload.get("wait_until", "networkidle"),
            page_timeout=payload.get("timeout_ms", 30000),
            # ... other options from payload
        )

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(url=url, config=config)

        return {
            "markdown": result.markdown,
            "html": result.html if payload.get("include_html") else None,
            "screenshot": result.screenshot if payload.get("screenshot") else None,
            "links": result.links,
            "metadata": result.metadata,
        }

main = create_worker_main(CrawlWorker)
```

Update router `src/api/routers/crawl.py`:

```python
@router.post("/crawl")
async def crawl_url(request: CrawlRequest):
    # Use worker instead of direct execution
    result = await coordinator_infer(
        task="crawl",
        model_id="crawl4ai",  # Single "model" for crawl
        payload={
            "url": str(request.url),
            "wait_until": request.wait_until,
            "timeout_ms": request.timeout_ms,
            "screenshot": request.screenshot,
            # ... other options
        },
    )
    return CrawlResponse(**result)
```

### 4. API Endpoints

New router: `src/api/routers/environments.py`

```python
@router.get("/environments")
async def list_environments():
    """List all available environments and their status."""
    return env_manager.list_environments()

@router.get("/environments/{env_name}")
async def get_environment(env_name: str):
    """Get status of a specific environment."""
    return env_manager.get_status(env_name)

@router.post("/environments/{env_name}/install")
async def install_environment(env_name: str, background_tasks: BackgroundTasks):
    """Pre-install an environment (async, returns immediately)."""
    status = env_manager.get_status(env_name)

    if status["status"] == "ready":
        return {"message": "Already installed", **status}

    if status["status"] == "installing":
        return {"message": "Installation in progress"}

    # Start background installation
    background_tasks.add_task(env_manager.ensure_environment, env_name)

    return {"message": "Installation started", "status": "installing"}

@router.delete("/environments/{env_name}")
async def delete_environment(env_name: str):
    """Delete an installed environment to free disk space."""
    env_path = env_manager.envs_dir / env_name
    if env_path.exists():
        shutil.rmtree(env_path)
        return {"message": f"Deleted {env_name}", "freed_mb": "..."}
    return {"message": "Not installed"}
```

### 5. Dockerfile Changes

```dockerfile
# Stage 1: UI Builder (unchanged)
FROM node:lts AS ui-builder
# ... existing UI build

# Stage 2: Minimal Python Runtime
FROM python:3.13-slim

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    # For LibreOffice doc conversion (optional, could also be on-demand)
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /haid

# Copy only API server dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy environment templates
COPY env-templates/ ./env-templates/

# Copy application code
COPY src/ ./src/
COPY main.py ./

# Copy UI build
COPY --from=ui-builder /app/ui/dist ./ui/dist

# Data directory
ENV HAID_DATA_DIR=/haid/data
VOLUME /haid/data

# Minimal environment
ENV PYTHONPATH=/haid
ENV HF_HOME=/haid/data/models

EXPOSE 12310
HEALTHCHECK CMD curl -f http://localhost:12310/api/health || exit 1

CMD ["uv", "run", "python", "main.py"]
```

### 6. Template Generation Script

Script to generate/update templates from current dependencies:

```bash
#!/bin/bash
# scripts/generate-env-templates.sh

set -e

TEMPLATES_DIR="env-templates"

# Embedding environment
mkdir -p "$TEMPLATES_DIR/embedding"
cat > "$TEMPLATES_DIR/embedding/pyproject.toml" << 'EOF'
[project]
name = "haid-embedding-worker"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "torch==2.8.0",
    "sentence-transformers>=3.0.0",
    "transformers>=4.51.0",
    "accelerate>=0.20.0",
    "numpy<2",
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
EOF
echo "3.13" > "$TEMPLATES_DIR/embedding/.python-version"
(cd "$TEMPLATES_DIR/embedding" && uv lock)

# Crawl environment
mkdir -p "$TEMPLATES_DIR/crawl"
cat > "$TEMPLATES_DIR/crawl/pyproject.toml" << 'EOF'
[project]
name = "haid-crawl-worker"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "crawl4ai>=0.7.0",
    "playwright>=1.43.0",
    "playwright-stealth>=1.8.0",
]
EOF
echo "3.13" > "$TEMPLATES_DIR/crawl/.python-version"
(cd "$TEMPLATES_DIR/crawl" && uv lock)

# ... similar for other environments
```

## Migration Plan

### Phase 1: Infrastructure
1. Create `EnvironmentManager` class
2. Create `env-templates/` directory structure
3. Generate templates for all worker types
4. Update coordinator to use EnvironmentManager

### Phase 2: Crawl Worker
1. Create `crawl_worker.py`
2. Update crawl router to use worker
3. Create crawl environment template
4. Test end-to-end

### Phase 3: Slim Docker Image
1. Split pyproject.toml (server vs workers)
2. Update Dockerfile
3. Test fresh container with on-demand installs
4. Measure image size reduction

### Phase 4: UI Integration
1. Add environments page to UI
2. Show install status and progress
3. Pre-install buttons
4. Disk usage monitoring

## User Experience

### First Request Flow

```
User: POST /api/text-to-embedding {"text": "hello"}

Response (if env not installed):
{
  "status": "installing",
  "message": "Setting up embedding environment (first time only)",
  "estimated_time_seconds": 120
}

... installation happens ...

Response (after install):
{
  "embedding": [0.1, 0.2, ...],
  "model": "all-MiniLM-L6-v2"
}
```

### Alternative: Blocking with Progress

Could use SSE or WebSocket to stream installation progress:

```
POST /api/text-to-embedding
Accept: text/event-stream

event: installing
data: {"step": "Creating environment", "progress": 0}

event: installing
data: {"step": "Installing torch", "progress": 30}

event: installing
data: {"step": "Installing sentence-transformers", "progress": 60}

event: ready
data: {"embedding": [...]}
```

### Pre-warm Option

UI could show available capabilities with "Enable" buttons:

```
┌─────────────────────────────────────────────────┐
│ AI Capabilities                                 │
├─────────────────────────────────────────────────┤
│ ✅ Text Embedding      [2.5 GB] [Ready]         │
│ ⬜ Image Captioning    [3.0 GB] [Enable]        │
│ ⬜ Text Generation     [3.0 GB] [Enable]        │
│ 🔄 Speech Recognition  [2.5 GB] [Installing...] │
│ ⬜ OCR                  [3.0 GB] [Enable]        │
│ ✅ Web Crawling        [1.0 GB] [Ready]         │
├─────────────────────────────────────────────────┤
│ Total: 6.0 GB used / 50 GB available            │
└─────────────────────────────────────────────────┘
```

## Benefits

| Metric | Before | After |
|--------|--------|-------|
| Docker image size | ~10GB+ | ~1GB |
| Initial disk usage | ~10GB | ~1GB |
| Max disk usage | ~10GB | ~15GB (if all installed) |
| First request (new env) | Fast | Slow (1-2 min) |
| Subsequent requests | Fast | Fast |
| Offline capability | Full | Partial (needs internet for first install) |

## Considerations

### Offline Mode
- Templates include `uv.lock` for reproducibility
- Could pre-download wheel cache for fully offline installs
- Or accept that first install needs internet

### Disk Space Management
- Add endpoint to delete unused environments
- Show disk usage per environment
- Auto-cleanup after N days of inactivity (optional)

### Version Updates
- Templates updated with new image versions
- Existing environments not auto-updated
- Add "update available" indicator in UI
- Manual or automatic update option

### Error Recovery
- If install fails, cleanup partial state
- Retry logic for transient network errors
- Clear error messages in UI
