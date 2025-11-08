#!/usr/bin/env bash
# Ensure CUDA shared libraries from pip (nvidia-*) are on LD_LIBRARY_PATH
# Usage: source scripts/setup_cuda_env.sh

set -euo pipefail

python_bin="${PYTHON:-python}"
site_pkgs_dir="$($python_bin - <<'PY'
import site, sys
paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    p = site.getusersitepackages()
    if p:
        paths.append(p)
except Exception:
    pass
for p in sys.path:
    if p and 'site-packages' in p:
        paths.append(p)
seen = []
for p in paths:
    if p not in seen:
        seen.append(p)
print(seen[0] if seen else '')
PY
)"

if [[ -z "$site_pkgs_dir" ]]; then
  echo "Could not locate site-packages. Ensure your venv is active."
  return 1 2>/dev/null || exit 1
fi

declare -a lib_paths=(
  "$site_pkgs_dir/torch/lib"
  "$site_pkgs_dir/nvidia/cuda_runtime/lib"
  "$site_pkgs_dir/nvidia/cublas/lib"
  "$site_pkgs_dir/nvidia/cudnn/lib"
  "$site_pkgs_dir/nvidia/cufft/lib"
  "$site_pkgs_dir/nvidia/cuda_nvrtc/lib"
  "$site_pkgs_dir/nvidia/nvjitlink/lib"
  "$site_pkgs_dir/nvidia/cuda_cupti/lib"
)

to_add=""
for d in "${lib_paths[@]}"; do
  if [[ -d "$d" ]]; then
    if [[ -z "$to_add" ]]; then
      to_add="$d"
    else
      to_add="$to_add:$d"
    fi
  fi
done

if [[ -z "$to_add" ]]; then
  echo "No NVIDIA/CUDA lib directories found under $site_pkgs_dir"
  return 0 2>/dev/null || exit 0
fi

export LD_LIBRARY_PATH="$to_add:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH updated with: $to_add"

