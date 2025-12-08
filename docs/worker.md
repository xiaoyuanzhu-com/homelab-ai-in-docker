# Worker Architecture Design

## Goals

1. **Migrate all AI tasks to worker subprocess pattern** - Uniform architecture
2. **GPU memory protection** - Coordinator ensures only one GPU-heavy operation at a time
3. **Worker lingering** - Keep workers alive for 60s after task completion for better perf
4. **Multi-env support** - Workers can use different Python environments for conflicting dependencies

## Current State

| Task | Current Pattern | Worker Needed |
|------|----------------|---------------|
| image-ocr | Worker subprocess | ✅ Already done |
| text-to-embedding | In-process + coordinator | ✅ Migrate |
| image-captioning | In-process + coordinator | ✅ Migrate |
| text-generation | In-process + coordinator | ✅ Migrate |
| automatic-speech-recognition | In-process + coordinator | ✅ Migrate |
| speaker-diarization | In-process + coordinator | ✅ Migrate |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Main Process (.venv)                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Application                          │   │
│  │  /api/image-ocr, /api/text-to-embedding, /api/asr, etc.          │   │
│  └───────────────────────────────┬──────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    WorkerCoordinator                             │   │
│  │                                                                   │   │
│  │  - Serializes GPU access (only one worker active at a time)      │   │
│  │  - Manages worker lifecycle (spawn, reuse, terminate)            │   │
│  │  - Routes requests to appropriate workers                        │   │
│  │  - Handles worker health monitoring                              │   │
│  └───────────────────────────────┬──────────────────────────────────┘   │
│                                  │                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │ spawns / HTTP
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  OCR Worker      │    │  Embedding       │    │  ASR Worker      │
│  (.venv-deepseek)│    │  Worker (.venv)  │    │  (.venv)         │
│                  │    │                  │    │                  │
│  localhost:PORT1 │    │  localhost:PORT2 │    │  localhost:PORT3 │
│  /healthz        │    │  /healthz        │    │  /healthz        │
│  /infer          │    │  /infer          │    │  /infer          │
│  /shutdown       │    │  /shutdown       │    │  /shutdown       │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Key Components

### 1. WorkerCoordinator (replaces ModelCoordinator)

Location: `src/services/worker_coordinator.py`

```python
class WorkerCoordinator:
    """
    Centralized coordinator for all AI workers.

    Responsibilities:
    - Serialize GPU access (mutex across all workers)
    - Manage worker lifecycle
    - Route requests to correct worker
    - Handle worker pool and reuse
    """

    def __init__(
        self,
        max_concurrent_workers: int = 1,  # GPU serialization
        worker_idle_timeout: int = 60,    # Keep warm for 60s
        worker_startup_timeout: int = 120,
    ):
        self._gpu_lock = asyncio.Lock()  # Only one GPU operation at a time
        self._workers: Dict[str, WorkerHandle] = {}
        self._worker_lock = asyncio.Lock()
        ...
```

**GPU Serialization Strategy:**

The coordinator holds a `_gpu_lock` that MUST be acquired before:
1. Spawning a new worker (which loads a model)
2. Sending inference request to a worker

This ensures only one GPU-heavy operation happens at a time, preventing OOM.

### 2. Worker Protocol

All workers expose the same HTTP interface:

```
GET  /healthz     -> {"status": "ok", "model": "...", "ready": true}
POST /infer       -> Task-specific request/response JSON
POST /shutdown    -> Graceful shutdown request
GET  /info        -> {"model": "...", "memory_mb": ..., "load_time_ms": ...}
```

### 3. Worker Lifecycle

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

**Worker lingering behavior (worker self-terminates):**

Workers manage their own idle timeout (same pattern as existing OCR worker in `src/worker/image_ocr_worker.py`):

1. Worker updates `last_active` timestamp after each inference
2. Worker runs an internal watchdog timer
3. If new request arrives before timeout → reset timer, process request
4. If timeout expires → worker calls `os._exit(0)` to fully release GPU context

The coordinator does NOT manage idle timeouts. It only:
- Observes worker state (`proc.poll()` to detect exit)
- Cleans up stale worker handles
- Spawns new workers when needed

**Coordinator side:**

1. Before inference: acquire `_gpu_lock`
2. Check if worker exists and is alive (`proc.poll() is None`)
3. If not, spawn new worker (still holding lock)
4. Send `/infer` request
5. Release `_gpu_lock` after response received

### 4. Base Worker Class

Location: `src/worker/base_worker.py`

```python
class BaseWorker(ABC):
    """Base class for all inference workers."""

    def __init__(self, model_id: str, port: int, idle_timeout: int = 60):
        self.model_id = model_id
        self.port = port
        self.idle_timeout = idle_timeout
        self._last_active = time.time()
        self._idle_task: Optional[asyncio.Task] = None
        self._model = None

    @abstractmethod
    async def load_model(self) -> None:
        """Load the model into memory. Called once at startup."""
        pass

    @abstractmethod
    async def infer(self, request: dict) -> dict:
        """Run inference on the loaded model."""
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Called before shutdown."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 5. Worker Manager (per-task type)

Each task type has a manager that knows how to spawn and configure workers:

```python
# src/worker/managers/embedding_worker_manager.py
class EmbeddingWorkerManager(BaseWorkerManager):
    """Manages embedding workers."""

    def get_worker_module(self) -> str:
        return "src.worker.workers.embedding_worker"

    def get_python_env(self, model_config: dict) -> str:
        # Most embedding models use main env
        return model_config.get("python_env", None)
```

## Interface Contracts

### Main Process → Coordinator

```python
# Usage from routers
from src.services.worker_coordinator import coordinator

@router.post("/text-to-embedding")
async def embed_text(request: EmbeddingRequest):
    result = await coordinator.infer(
        task="embedding",
        model_id=request.model,
        payload={"texts": request.texts},
    )
    return EmbeddingResponse(embeddings=result["embeddings"], ...)
```

### Coordinator → Worker

```python
@dataclass
class InferRequest:
    """Standard inference request to worker."""
    payload: dict       # Task-specific data
    request_id: str     # For tracking

@dataclass
class InferResponse:
    """Standard inference response from worker."""
    result: dict        # Task-specific result
    request_id: str
    processing_time_ms: int
    model: str
```

### Worker Internal

```python
# Worker receives HTTP POST to /infer
{
    "payload": {"texts": ["hello", "world"]},
    "request_id": "uuid"
}

# Worker responds
{
    "result": {"embeddings": [[0.1, 0.2, ...], ...]},
    "request_id": "uuid",
    "processing_time_ms": 150,
    "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

## File Structure

```
src/
  worker/
    __init__.py
    base.py                    # BaseWorker abstract class
    coordinator.py             # WorkerCoordinator (GPU serialization)
    protocol.py                # Shared request/response models
    utils.py                   # Port finding, process management

    workers/                   # Individual worker implementations
      __init__.py
      embedding_worker.py
      captioning_worker.py
      text_generation_worker.py
      asr_worker.py
      ocr_worker.py           # Migrate from image_ocr_worker.py
```

**Migration mapping from current files:**

| Current | New Location | Notes |
|---------|--------------|-------|
| `src/worker/manager.py` | `src/worker/coordinator.py` | Generalize OCRWorkerManager → WorkerCoordinator |
| `src/worker/image_ocr_worker.py` | `src/worker/workers/ocr_worker.py` | Refactor to use BaseWorker |
| `src/services/model_coordinator.py` | (removed) | Replaced by WorkerCoordinator |
| Router in-process loading | (removed) | Routers call coordinator.infer() instead |

## Migration Plan

### Phase 1: Infrastructure

1. Create `BaseWorker` and worker protocol
2. Create `WorkerCoordinator` with GPU serialization
3. Migrate existing `OCRWorkerManager` to new pattern

### Phase 2: Migrate Tasks (one at a time)

1. **text-to-embedding** (simplest, good test case)
2. **image-captioning**
3. **text-generation**
4. **automatic-speech-recognition**
5. **speaker-diarization**

For each:
1. Create worker in `src/worker/workers/`
2. Update router to use coordinator
3. Remove old in-process loading code
4. Test thoroughly

### Phase 3: Multi-env Support

1. Add `python_env` to model configs in catalog
2. Update worker spawning to use correct interpreter
3. Create additional venvs for conflicting models

See [Multi-Environment Support](#multi-environment-support) section for details.

## Worker Idle Behavior (60s linger)

**Why 60s?**

- Most interactive use cases have < 60s between requests
- Keeps model "warm" for better response times
- Avoids repeated load times during active sessions
- 60s is long enough for user workflows, short enough to free resources

**Implementation:**

```python
# In worker
async def _idle_watchdog(self):
    while True:
        await asyncio.sleep(1)
        idle_time = time.time() - self._last_active
        if idle_time >= self.idle_timeout:
            logger.info(f"Worker idle for {idle_time:.1f}s, shutting down")
            await self._shutdown()
            break

# After each inference
self._last_active = time.time()
```

**Coordinator tracking:**

```python
@dataclass
class WorkerHandle:
    model_key: str
    port: int
    proc: subprocess.Popen
    state: WorkerState  # STARTING, READY, BUSY, TERMINATED
    last_active: float

    def is_alive(self) -> bool:
        return self.proc.poll() is None
```

## GPU Memory Protection

**Problem:** If two workers try to load models simultaneously, GPU runs out of memory.

**Solution:** In-process `asyncio.Lock` in the coordinator. Since ALL worker access goes through the single coordinator instance in the main process, a simple async lock is sufficient:

```python
class WorkerCoordinator:
    def __init__(self):
        self._gpu_lock = asyncio.Lock()  # Simple in-process lock
        self._workers: Dict[str, WorkerHandle] = {}

    async def infer(self, task: str, model_id: str, payload: dict) -> dict:
        async with self._gpu_lock:
            # Only one coroutine can be here at a time
            # This serializes ALL GPU operations across ALL workers

            # 1. Get or spawn worker (may load model → uses GPU)
            worker = await self._ensure_worker(task, model_id)

            # 2. Send inference request (uses GPU)
            result = await self._send_request(worker, payload)

            return result
```

**Why this works:**

1. **Single coordinator instance** - All API routes call the same coordinator
2. **asyncio.Lock is sufficient** - FastAPI runs in a single event loop, so async lock serializes all concurrent requests
3. **No distributed locking needed** - Workers are subprocesses of main process, not separate services
4. **Simple to reason about** - Same pattern as current `ModelCoordinator`

**Deployment constraint:**

This requires running with a single Uvicorn worker (`--workers 1`, which is the default). Multi-worker deployment (e.g., `--workers 4`) would break the in-process lock. This is acceptable because:
- Single GPU = single bottleneck (no benefit from multiple API workers)
- Model loading is expensive, can't have N workers each loading models
- Complexity of cross-process locking isn't worth it for a homelab app

**Lock scope:**

The lock is held during:
- Worker spawn (model loading)
- Inference request

This prevents:
- Two models loading simultaneously → OOM
- Inference while another model is loading → OOM
- Memory fragmentation from concurrent operations

**Future optimization:** Could release lock after spawn if model supports concurrent inference (worker handles its own batching).

## Error Handling

### Worker Crash

```python
async def _ensure_worker(self, task: str, model_id: str) -> WorkerHandle:
    key = f"{task}:{model_id}"
    worker = self._workers.get(key)

    # Check if worker is still alive
    if worker and not worker.is_alive():
        logger.warning(f"Worker {key} crashed, removing")
        self._workers.pop(key)
        worker = None

    if worker is None:
        worker = await self._spawn_worker(task, model_id)
        self._workers[key] = worker

    return worker
```

### Request Timeout

Use stdlib `urllib` (like existing OCR manager) or `httpx` for async HTTP:

```python
async def _send_request(self, worker: WorkerHandle, payload: dict) -> dict:
    """Send inference request to worker via HTTP."""
    import json
    from urllib import request as urlrequest
    from urllib.error import HTTPError, URLError

    url = f"http://127.0.0.1:{worker.port}/infer"
    data = json.dumps(payload).encode()

    def _do_request():
        req = urlrequest.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urlrequest.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())

    try:
        return await asyncio.to_thread(_do_request)
    except TimeoutError:
        # Kill the worker - it's stuck
        await self._terminate_worker(worker)
        raise TimeoutError(f"Worker inference timed out after 300s")
```

This matches the existing pattern in `src/worker/manager.py` and avoids adding new dependencies.

## Configuration

Settings stored via `src/db/settings.py` (SQLite-backed):

```python
from src.db.settings import get_setting_int

# Read settings (with defaults)
idle_timeout = get_setting_int("worker_idle_timeout_seconds", 60)
startup_timeout = get_setting_int("worker_startup_timeout_seconds", 120)
request_timeout = get_setting_int("worker_request_timeout_seconds", 300)
```

| Setting | Default | Description |
|---------|---------|-------------|
| `worker_idle_timeout_seconds` | 60 | How long workers linger after last request |
| `worker_startup_timeout_seconds` | 120 | Max time to wait for worker `/healthz` |
| `worker_request_timeout_seconds` | 300 | Max time for a single inference request |

## Monitoring

### Health endpoint enhancement

```json
GET /api/health
{
    "status": "healthy",
    "workers": {
        "embedding:all-MiniLM-L6-v2": {
            "state": "READY",
            "port": 50001,
            "idle_seconds": 12.5,
            "memory_mb": 450
        },
        "ocr:DeepSeek-OCR": {
            "state": "READY",
            "port": 50002,
            "idle_seconds": 3.2,
            "memory_mb": 8500
        }
    },
    "gpu_lock_held": false
}
```

Workers can report memory via `/info` endpoint using:

```python
def get_gpu_memory_mb() -> float:
    """Get GPU memory used by this process."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0
```

## Testing

1. **Unit tests:** Worker base class, coordinator logic
2. **Integration tests:** Full request flow through coordinator
3. **Load tests:** Concurrent requests to verify serialization
4. **Memory tests:** Verify OOM doesn't happen with sequential loads

## Multi-Environment Support

Some AI models have conflicting Python dependency requirements:

| Model | transformers requirement |
|-------|-------------------------|
| DeepSeek-OCR | ==4.47.1 (uses `LlamaFlashAttention2`) |
| MinerU | >=4.56.0 |
| HunyuanOCR | git commit (not in stable release) |
| Most others | >=4.51.0 |

Since workers are subprocesses communicating via HTTP, each can use a different Python interpreter.

### Environment Layout

```
.venv/                  # Main env - used by most tasks
.venv-deepseek-ocr/     # DeepSeek-OCR specific (transformers==4.47.1)
.venv-hunyuan/          # HunyuanOCR specific (transformers@git)
```

### Model Configuration

In the models database (via `src/db/catalog.py`), add `python_env` field:

```json
{
  "id": "deepseek-ai/DeepSeek-OCR",
  "architecture": "deepseek",
  "python_env": "deepseek-ocr"
}
```

### Interpreter Selection

The coordinator resolves the Python interpreter when spawning:

```python
def _get_python_for_model(self, model_config: dict) -> str:
    """Get Python interpreter path for model's environment."""
    env_name = model_config.get("python_env")
    if env_name:
        env_path = Path(f".venv-{env_name}/bin/python")
        if env_path.exists():
            return str(env_path.resolve())
    return sys.executable  # Default to main env
```

### Creating Additional Environments

> **Note:** Linux-only. The project runs in Docker or on Linux hosts.

```bash
# PyTorch CUDA index (must match main env's CUDA 12.6)
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"

# DeepSeek-OCR environment
uv venv .venv-deepseek-ocr --python 3.13
source .venv-deepseek-ocr/bin/activate
uv pip install --index-url $PYTORCH_INDEX \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
uv pip install transformers==4.47.1 flash-attn>=2.8.3 \
    bitsandbytes accelerate einops addict easydict safetensors pillow
deactivate
```

### Docker Strategy

**Option A: Host/dev only (recommended initially)**

Keep Docker simple (single env). Models requiring alternate envs marked as "host-only" in catalog.

**Option B: Multi-venv in Docker**

```dockerfile
# Additional environments (adds ~2-5GB per env)
RUN uv venv /app/.venv-deepseek-ocr --python 3.13 && \
    . /app/.venv-deepseek-ocr/bin/activate && \
    uv pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 && \
    uv pip install transformers==4.47.1 flash-attn>=2.8.3 ...
```

### When to Add a New Environment

Add a new env only when:

1. A model has **hard conflicts** with the main env (different version of same package)
2. The conflict **cannot be resolved** by updating the main env
3. The model is **worth supporting** despite the overhead

Don't add a new env for:

- Models that work with the main env
- Models with optional dependencies (mark as unavailable if deps missing)
- Experimental models you're just testing
