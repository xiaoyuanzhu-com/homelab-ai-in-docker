# Model Loading, Unloading, and Lifecycle Management Analysis

## Executive Summary

The codebase implements a **distributed, timeout-based model management system** with per-router singleton caching and background cleanup tasks. Each model type (ASR, embedding, captioning, OCR, text generation, speaker embedding, WhisperX) manages its own lifecycle independently with similar patterns.

---

## 1. WHERE MODELS ARE LOADED

### 1.1 Model Loading Locations by Service

| Service | Router | Load Function | Cache Pattern |
|---------|--------|---------------|----------------|
| **Text-to-Embedding** | `src/api/routers/text_to_embedding.py:93` | `get_model()` | Singleton `_model_cache` |
| **Image Captioning** | `src/api/routers/image_captioning.py:168` | `get_model()` | Singleton `_model_cache`, `_processor_cache` |
| **Image OCR** | `src/api/routers/image_ocr.py:178` | `get_model()` | Singleton `_model_cache` + Worker manager |
| **ASR (Whisper)** | `src/api/routers/automatic_speech_recognition.py:156` | `get_model()` | Singleton `_model_cache`, `_processor_cache` |
| **Speaker Embedding** | `src/api/routers/speaker_embedding.py:42` | `get_model()` | Singleton `_model_cache`, `_inference_cache` |
| **Text Generation** | `src/api/routers/text_generation.py:150` | `get_model()` | Singleton `_model_cache`, `_tokenizer_cache` |
| **WhisperX** | `src/api/routers/whisperx.py:158` | `_load_asr_model()` | Singleton `_asr_model_cache` + alignment + diarization models |

### 1.2 Loading Process Flow

```python
# Typical pattern across all routers:
1. get_model(model_name) is called
2. check_and_cleanup_idle_model()  # Check if current model should be unloaded
3. If different model requested: delete old model references, set new model name
4. If model not in cache:
   a. Check HuggingFace local cache at HF_HOME/hub/models--{org}--{model}/
   b. If not found locally, download from HuggingFace (sets HF_ENDPOINT)
   c. Load with architecture-specific loader:
      - sentence-transformers.SentenceTransformer (embeddings)
      - transformers.AutoModelFor* (image captioning, ASR, text generation)
      - pyannote.audio.Model (speaker embedding)
      - whisperx (ASR with alignment)
      - paddle-ocr or easyocr (OCR, via worker process)
5. Update _last_access_time = time.time()
6. Return model(s)
```

### 1.3 Model Storage Structure

```
HF_HOME=/haid/data/models
├── hub/models--sentence-transformers--all-MiniLM-L6-v2/
├── hub/models--openai--whisper-large-v2/
├── hub/models--microsoft--phi-3-mini-4k-instruct/
└── ...
```

### 1.4 Model Loading Parameters

**Common Configuration:**
- `local_files_only=True` when model exists locally (no HF download)
- `device="cuda"` if GPU available, else `"cpu"`
- `dtype=torch.float16` on CUDA (fp32 on CPU)
- `low_cpu_mem_usage=True` for efficient loading
- `attn_implementation="sdpa"` for ASR (scaled dot-product attention)
- `use_safetensors=True` for ASR (faster loading)

**Architecture-Specific:**
- **Image Captioning**: Supports BLIP, BLIP-2, LLaVA, LLaVA-NeXT with `device_map="auto"` for quantized models
- **Text Generation**: `dtype=torch.float16` always, GPU placement via `.to("cuda")`
- **ASR**: Converts input features to match model dtype after loading

---

## 2. CURRENT IDLE TIMEOUT UNLOAD IMPLEMENTATION

### 2.1 Settings-Based Configuration

**Source**: `src/db/settings.py:24`

```python
defaults = [
    ("model_idle_timeout_seconds", "5", "Seconds of inactivity before unloading models from GPU memory"),
]
```

**Access Pattern**: Every router calls `get_setting_int("model_idle_timeout_seconds", default_value)`
- Default: 5 seconds
- Overridable per-router: text_generation uses 60s, speaker_embedding uses 300s (5 min)
- Dynamically read from database each check (allows runtime changes)

### 2.2 Dual-Level Cleanup Mechanism

#### Level 1: Per-Request Polling (Every Request)

**Location**: `main.py:78-141` - `periodic_model_cleanup()`

```python
async def periodic_model_cleanup():
    # Runs every 1 second, checks all model services
    while True:
        await asyncio.sleep(1)
        
        # For each service:
        image_captioning.check_and_cleanup_idle_model()
        text_to_embedding.check_and_cleanup_idle_model()
        # ... etc
```

**Implementation in Each Router**: `check_and_cleanup_idle_model()`

```python
def check_and_cleanup_idle_model():
    if _model_cache is None or _last_access_time is None:
        return
    
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)
    idle_duration = time.time() - _last_access_time
    
    if idle_duration >= idle_timeout:
        logger.info(f"Model idle for {idle_duration:.1f}s (timeout: {idle_timeout}s), unloading...")
        cleanup()
```

**Pros**: 
- Synchronous, no async overhead
- Catches idle models proactively every second
- Works even if no new requests arrive

**Cons**: 
- Requires event loop to be running
- 1-second polling interval may miss brief idle windows

#### Level 2: Background Task Scheduling (Per-Request)

**Location**: Called after every successful request in each router

**Implementation**:

```python
def schedule_idle_cleanup() -> None:
    global _idle_cleanup_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    
    idle_timeout = get_setting_int("model_idle_timeout_seconds", 5)
    
    # Cancel any previous watchdog task
    if _idle_cleanup_task and not _idle_cleanup_task.done():
        _idle_cleanup_task.cancel()
    
    async def _watchdog(timeout_s: int):
        try:
            await asyncio.sleep(timeout_s)
            if _last_access_time is None:
                return
            idle_duration = time.time() - _last_access_time
            if idle_duration >= timeout_s and _model_cache is not None:
                logger.info(f"Model idle for {idle_duration:.1f}s, unloading...")
                cleanup()
        except asyncio.CancelledError:
            pass
        finally:
            global _idle_cleanup_task
            _idle_cleanup_task = None
    
    _idle_cleanup_task = loop.create_task(_watchdog(idle_timeout))
```

**Pros**:
- Precise: unloads exactly at timeout
- Cancels previous tasks (no memory leak)
- Works for long-idle models between polling intervals

**Cons**:
- Creates/cancels tasks per request (overhead)
- Text generation has special `_model_in_use` flag to prevent cleanup during generation

### 2.3 Cleanup Function Pattern

**Location**: Each router has a `cleanup()` function

```python
def cleanup():
    global _model_cache, _processor_cache, _current_model_name, _last_access_time
    
    # 1. Move to CPU if possible (helps GPU memory cleanup)
    if _model_cache is not None:
        try:
            _model_cache.cpu()
        except Exception as e:
            logger.warning(f"Error moving to CPU: {e}")
    
    # 2. Delete references
    del _model_cache
    _model_cache = None
    del _processor_cache
    _processor_cache = None
    
    # 3. Reset state
    _current_model_name = ""
    _last_access_time = None
    
    # 4. Force GPU memory release
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for GPU ops to finish
```

### 2.4 Router-Specific Timeout Variations

| Router | Default Timeout | Notes |
|--------|-----------------|-------|
| text_to_embedding | 5s | Uses get_setting_int with default 5 |
| image_captioning | 5s | Same pattern |
| image_ocr | 5s | Uses worker manager (isolated processes) |
| automatic_speech_recognition | 5s | Handles both Whisper + pyannote |
| speaker_embedding | 300s (5 min) | Hardcoded MODEL_IDLE_TIMEOUT = 300 |
| text_generation | 60s | Uses get_setting_int with default 60 |
| whisperx | 5s | Handles ASR + alignment + diarization |

**Issue**: Speaker embedding doesn't read from settings, uses hardcoded timeout

---

## 3. EXISTING MODEL LIFECYCLE MANAGEMENT

### 3.1 Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION STARTUP                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│   main.py lifespan.__aenter__()                                │
│   - Load models/libs catalog from JSON                         │
│   - Initialize database schema                                 │
│   - Start periodic_model_cleanup() background task (every 1s) │
│   - Start MCP server                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST ARRIVES                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│   API Endpoint Handler (e.g., /api/text-to-embedding)         │
│   - get_model(model_name)                                      │
│     ├─ check_and_cleanup_idle_model() (polling check)        │
│     ├─ Validate model from catalog                            │
│     ├─ Check local HF cache, download if needed              │
│     ├─ Load model (SentenceTransformer, etc.)                │
│     └─ _last_access_time = time.time()                       │
│   - Run inference                                              │
│   - schedule_idle_cleanup() (background task)                │
│   - Save to history                                           │
│   - Return response                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
   ┌─────────────────┐                    ┌──────────────────┐
   │  Polling Check  │                    │ Background Task  │
   │  (every 1s)     │                    │   (async sleep)  │
   │                 │                    │                  │
   │ If idle >=      │                    │ If idle >=       │
   │ timeout:        │                    │ timeout:         │
   │   cleanup()     │                    │   cleanup()      │
   └─────────────────┘                    └──────────────────┘
         │                                       │
         └────────────────────┬────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   cleanup()      │
                    │ ─────────────    │
                    │ 1. cpu()         │
                    │ 2. del refs      │
                    │ 3. reset vars    │
                    │ 4. empty_cache() │
                    │ 5. synchronize() │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  GPU MEMORY      │
                    │   FREED          │
                    └──────────────────┘
```

### 3.2 Application Lifecycle (main.py)

**Startup** (`lifespan.__aenter__`):
1. Load models/libs catalog (JSON manifest)
2. Initialize database tables (models, libs, settings, history)
3. Create `periodic_model_cleanup()` background task
4. Start MCP server lifespan

**Shutdown** (`lifespan.__aexit__`):
1. Cancel periodic cleanup task
2. Call `cleanup()` on ALL model services:
   - text_to_embedding
   - text_generation
   - image_captioning
   - image_ocr
   - automatic_speech_recognition
   - whisperx
   - speaker_embedding
3. Shutdown joblib/loky executor (from PaddleOCR)
4. Stop MCP server

### 3.3 Service-Level Model Tracking

Each router maintains:
```python
_model_cache: Optional[Any] = None              # Actual model object
_current_model_name: str = ""                   # Which model is loaded
_last_access_time: Optional[float] = None       # For timeout calculation
_idle_cleanup_task: Optional[asyncio.Task] = None  # Background cleanup
_processor_cache: Optional[Any] = None          # For transformers models
_tokenizer_cache: Optional[Any] = None          # For text generation
_current_model_config: Optional[Dict] = None    # Model metadata from catalog
```

### 3.4 Model Model Switching

When user requests different model:
1. `get_model(new_model_name)` called
2. If `_current_model_name != new_model_name`:
   - Delete old `_model_cache`
   - Load new model
   - Update `_current_model_name`

**Note**: No preemptive unload - old model stays in GPU until cleanup triggers

---

## 4. MEMORY MANAGEMENT PATTERNS

### 4.1 GPU Memory Cleanup Sequence

```python
# Standard pattern in all cleanup() functions:
if _model_cache is not None:
    # Step 1: Move to CPU (frees GPU memory)
    if hasattr(_model_cache, 'cpu'):
        _model_cache.cpu()
    
    # Step 2: Delete Python reference
    del _model_cache
    _model_cache = None

# Step 3: Force PyTorch to reclaim GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()      # Free unused memory
    torch.cuda.synchronize()      # Wait for GPU ops to complete
```

**Why both steps?**
- `.cpu()`: Moves tensors from GPU to RAM
- `del`: Removes Python reference
- `empty_cache()`: Tells CUDA to reclaim the memory
- `synchronize()`: Ensures GPU has finished all operations before cleanup continues

### 4.2 CPU Memory Optimization

**Text Generation** loads with:
```python
load_kwargs = {
    "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
    "dtype": torch.float16,      # 50% RAM vs float32
}
```

**Image Models** load with:
```python
load_kwargs = {
    "low_cpu_mem_usage": True,
    "dtype": torch.float16,      # For non-quantized
    "device_map": "auto",        # For quantized (distributed)
}
```

### 4.3 Quantization Support

- **bitsandbytes** (4-bit/8-bit) supported on Linux
- **Platform check**: `HAS_BITSANDBYTES = True/False`
- Models marked with `requires_quantization: true` in catalog
- Quantized models use `device_map="auto"` (Accelerate)

### 4.4 Special Cases

**WhisperX**: Manages 3 separate models with pooled cleanup
```python
_asr_model_cache       # Whisper ASR model
_align_cache           # Word-level alignment model
_diar_cache            # Diarization pipeline

# All cleaned up in single cleanup() call
```

**OCR (Image OCR)**: 
- Uses **isolated worker process** per model+language combo
- Worker process has its own idle timeout
- Main process just coordinates via HTTP calls
- Avoids GPU memory conflicts

**Speaker Embedding**: 
- Uses pyannote.audio.Inference wrapper
- Model + Inference both cached separately

### 4.5 Problematic Patterns

1. **Race Condition**: `check_and_cleanup_idle_model()` called at request start, but could unload during long inference if timing aligns with 1s polling interval

2. **Text Generation**: Adds `_model_in_use` flag to prevent cleanup during generation:
   ```python
   # Mark model as in use BEFORE starting inference
   global _model_in_use
   _model_in_use = True
   # ... run inference ...
   _model_in_use = False
   ```
   But this flag is set but **never checked in the watchdog**!

3. **Speaker Embedding**: Uses hardcoded 300s timeout, ignores database settings

4. **Task Cancellation**: Each `schedule_idle_cleanup()` cancels previous task, but if request arrives during cleanup, could miss the cancellation

---

## 5. HOW DIFFERENT MODEL TYPES ARE TRACKED

### 5.1 Centralized Model Catalog (Database)

**Source**: `src/api/catalog/models.json` (loaded into SQLite at startup)

**Fields per model**:
```json
{
  "id": "all-MiniLM-L6-v2",
  "label": "All MiniLM-L6-v2",
  "tasks": ["text-to-embedding"],
  "architecture": "sentence-transformers",
  "gpu_memory_mb": 150,
  "parameters_m": 22.7,
  "dimensions": 384,
  "requires_quantization": false,
  "requires_download": true,
  "hf_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Query Functions**:
- `get_model_dict(model_id)` - Get single model config
- `list_models(task="text-to-embedding")` - Get all models for task
- `get_lib_dict(lib_id)` - Get library config (for OCR)

### 5.2 Per-Router Tracking

Each router independently tracks its current state:

```
text_to_embedding router:
├─ _current_model_name = "all-MiniLM-L6-v2"
├─ _model_cache = SentenceTransformer(...)
├─ _last_access_time = 1699684500.123
└─ _idle_cleanup_task = <asyncio.Task>

image_captioning router:
├─ _current_model_name = "blip-large"
├─ _model_cache = Blip2ForConditionalGeneration(...)
├─ _processor_cache = AutoProcessor(...)
├─ _current_model_config = {...}
└─ _last_access_time = 1699684505.456
```

### 5.3 No Central Registry

**Current State**: No global tracking across routers
- Can't query "which models are loaded?"
- Can't unload all models at once
- Each router is independent

**Status Tracking**: Database tracks model metadata, not runtime state

### 5.4 Request History Tracking

**Location**: `src/storage/history.py`

Each request logged with:
```python
history_storage.add_request(
    service="text-to-embedding",
    request_id=request_id,
    request_data=request.model_dump(),
    response_data=response.model_dump(),
    status="success"
)
```

**Use Case**: Audit trail, not active memory management

---

## 6. CURRENT ARCHITECTURE STRENGTHS & WEAKNESSES

### Strengths

1. **Dual Cleanup Layers**: Polling (1s) + background tasks (per-request) = reliable unload
2. **Configurable Timeout**: Database settings allow runtime changes
3. **Consistent Pattern**: All routers follow similar cleanup structure
4. **Graceful Unload**: CPU move + empty_cache + synchronize is thorough
5. **Isolation**: OCR uses separate process to prevent GPU conflicts

### Weaknesses  

1. **No Coordination**: Multiple models can be loaded simultaneously → GPU OOM risk
2. **No Preemptive Unload**: Switching models keeps old one in GPU until timeout
3. **Inconsistent Timeouts**: Hardcoded values override database settings
4. **1-Second Polling**: May miss brief idle windows, overhead for 24/7 polling
5. **Task Overhead**: Every request creates/cancels async task
6. **Text Gen Bug**: `_model_in_use` flag never checked in watchdog
7. **No Global State**: Can't monitor/control all models at once
8. **Worker Complexity**: OCR worker process adds management overhead

---

## 7. DESIGN RECOMMENDATIONS FOR AGGRESSIVE UNLOAD

### Phase 1: Immediate Improvements

1. **Fix text_generation watchdog**: Check `_model_in_use` flag
2. **Consolidate timeouts**: Speaker embedding + all routers read from settings
3. **Global model registry**: Track which models are loaded where
4. **Model preemption**: Unload least-recently-used (LRU) when new model requested

### Phase 2: Aggressive Unload Strategy

1. **Memory-Aware Cleanup**: 
   - Monitor GPU memory usage
   - Trigger cleanup if > 80% utilized
   - Unload oldest idle model first

2. **Coordinated Unload**:
   - When one model hits timeout, don't just unload that router
   - Check if other routers have idle models too
   - Unload oldest first

3. **Rapid Timeout Options**:
   - Ultra-aggressive: 1-2 second timeouts
   - Medium: 10-30 seconds
   - Configurable per model type

4. **Preemptive Model Switching**:
   - Before loading new model, check GPU usage
   - If > 70%, unload ALL idle models
   - Model switching doesn't wait for timeout

5. **Request-Level Cleanup**:
   - After request completes, immediately check for cleanup opportunities
   - Don't wait for next polling interval

---

## Key Files to Understand

1. **main.py**: Application lifespan, periodic cleanup orchestration
2. **src/db/settings.py**: Configuration management
3. **src/api/routers/*.py**: Individual model loaders and cleaners
4. **src/config.py**: Model cache path resolution
5. **src/worker/manager.py**: OCR worker process management

