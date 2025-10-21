#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting Homelab AI container"

# Defaults
# - Best-effort flash-attn install when GPU-capable torch is present
# - Optional GPU PyTorch upgrade if a GPU is detected but torch is CPU-only
: "${HAID_AUTO_FLASH_ATTN:=auto}"
: "${HAID_AUTO_TORCH_CUDA:=off}"
: "${HAID_TORCH_CUDA_CHANNEL:=cu121}"

has_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >/dev/null 2>&1 && return 0 || true
  fi
  # Fallback: environment hints provided by NVIDIA Container Toolkit
  [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]] && return 0 || return 1
}

should_try_install() {
  # Returns 0 (true) if we should attempt install, 1 otherwise
  case "${HAID_AUTO_FLASH_ATTN}" in
    0|false|off|disable) return 1 ;;
    *) ;;
  esac

  python - <<'PY'
import importlib.util
try:
    import torch  # noqa
except Exception:
    # torch not available, nothing to do
    print("NO")
else:
    has_cuda = False
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    has_fa = importlib.util.find_spec("flash_attn") is not None
    print("YES" if (has_cuda and not has_fa) else "NO")
PY
}

# Optional: if GPU detected but torch is CPU-only, try upgrading to a CUDA wheel
maybe_upgrade_torch_cuda() {
  case "${HAID_AUTO_TORCH_CUDA}" in
    1|true|on|auto) : ;;
    *) return 0 ;;
  esac

  if ! has_gpu; then
    return 0
  fi

  # If torch is not installed, skip (our base image installs torch already)
  if ! python -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"; then
    return 0
  fi

  # If torch already has CUDA, skip
  if python - <<'PY'
import torch
print("YES" if getattr(torch.version, "cuda", None) else "NO")
PY
  | grep -q YES; then
    return 0
  fi

  echo "[entrypoint] GPU detected but torch CPU build found; attempting CUDA wheel install from ${HAID_TORCH_CUDA_CHANNEL}"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${HAID_TORCH_CUDA_CHANNEL}"
  if command -v pip >/dev/null 2>&1; then
    pip install -q --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio -U || echo "[entrypoint] CUDA torch install failed (continuing with CPU)"
  else
    python -m pip install -q --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio -U || echo "[entrypoint] CUDA torch install failed (continuing with CPU)"
  fi
}

maybe_upgrade_torch_cuda

if [[ "$(should_try_install)" == "YES" ]]; then
  echo "[entrypoint] CUDA detected and flash-attn missing; attempting install (best-effort)"
  # Prefer system pip; ignore failure and continue
  if command -v pip >/dev/null 2>&1; then
    pip install -q "flash-attn==2.7.3" --no-build-isolation || echo "[entrypoint] flash-attn install failed (continuing without it)"
  else
    python -m pip install -q "flash-attn==2.7.3" --no-build-isolation || echo "[entrypoint] flash-attn install failed (continuing without it)"
  fi
else
  echo "[entrypoint] Skipping flash-attn install (not needed or disabled)"
fi

echo "[entrypoint] Launching: $*"
exec "$@"
