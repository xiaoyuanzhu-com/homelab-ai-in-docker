#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting Homelab AI container"

# Default: best-effort flash-attn install when GPU-capable torch is present
: "${HAID_AUTO_FLASH_ATTN:=auto}"

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

