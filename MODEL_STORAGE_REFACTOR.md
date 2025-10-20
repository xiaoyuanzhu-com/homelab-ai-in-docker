# Model Storage Refactoring Plan

## Current State
- Embedding models: `data/embedding/models/{org}/{model}` (custom structure)
- Caption models: `data/image-caption/Salesforce--blip-image-captioning-base/` (custom cache_dir)
- Each service manages its own cache directories independently

## Proposed Solution: Use HF_HOME

### Why HF_HOME?
1. **Standard HuggingFace convention**: All HF libraries respect HF_HOME
2. **Automatic organization**: HF creates structure: `HF_HOME/hub/models--{org}--{model}/`
3. **No custom code needed**: Libraries handle caching automatically
4. **Easy to inspect**: Standard structure across all model types

### Implementation

#### 1. Set HF_HOME in main.py
```python
# Set HuggingFace cache directory for all models
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(HAID_DATA_DIR / "models")
```

**Result**: All models will be stored in:
```
data/
  models/
    hub/
      models--BAAI--bge-large-en-v1.5/
      models--Alibaba-NLP--gte-large-en-v1.5/
      models--Salesforce--blip-image-captioning-base/
      models--sentence-transformers--all-MiniLM-L6-v2/
```

#### 2. Create Unified Models Manifest

Create `src/api/models/models_manifest.json`:
```json
{
  "embedding": [
    {
      "id": "BAAI/bge-large-en-v1.5",
      "name": "bge-large-en-v1.5",
      "team": "BAAI",
      "dimensions": 1024,
      "size_mb": 1340,
      "link": "https://huggingface.co/BAAI/bge-large-en-v1.5"
    }
  ],
  "caption": [
    {
      "id": "Salesforce/blip-image-captioning-base",
      "name": "BLIP Base",
      "team": "Salesforce",
      "size_mb": 990,
      "link": "https://huggingface.co/Salesforce/blip-image-captioning-base"
    }
  ]
}
```

#### 3. Update Models API

New endpoint: `GET /api/models` - List ALL models (not just embedding)

```python
def check_model_downloaded(model_id: str) -> tuple[bool, Optional[int]]:
    """Check if model exists in HF cache."""
    from ...config import get_data_dir

    # Convert BAAI/bge-large -> models--BAAI--bge-large
    hf_model_dir = f"models--{model_id.replace('/', '--')}"
    cache_path = get_data_dir() / "models" / "hub" / hf_model_dir

    if cache_path.exists():
        # Calculate size
        total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        size_mb = total_size // (1024 * 1024)
        return True, size_mb

    return False, None
```

#### 4. Migration Strategy

**Option A: Fresh start (recommended for development)**
- Set HF_HOME
- Let models re-download to new location
- Old dirs can be cleaned up manually

**Option B: Move existing models**
- Write migration script to move from old structure to HF structure
- More complex but preserves downloads

## Benefits

1. **Centralized**: All models in one place
2. **Standard**: Uses HuggingFace conventions
3. **Discoverable**: Easy to list all downloaded models
4. **Future-proof**: Works with any HF model library
5. **Simpler code**: Less custom cache management

## Models Page Implementation

With unified manifest:
- Group models by type (embedding, caption, etc.)
- Check download status using HF cache structure
- Download using `huggingface-cli` or library methods
- Delete by removing HF cache directory

## API Changes

### Current
- `GET /api/models/embedding` - Only embedding models

### New
- `GET /api/models` - All models, grouped by type
- `GET /api/models/{type}` - Models of specific type
- `POST /api/models/download` - Download any model
- `DELETE /api/models/{model_id}` - Delete any model

## Next Steps

1. ✅ Set HF_HOME in main.py
2. ✅ Create unified models manifest
3. ✅ Update models router to list all models
4. ✅ Update check_model_downloaded to use HF cache structure
5. ✅ Update UI to show all model types
6. Test with fresh model downloads
7. Optional: Write migration script for existing models
