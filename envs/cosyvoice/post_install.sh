#!/bin/bash
# Post-install script for cosyvoice environment
# Installs PyTorch with CUDA 12.1 support (cuDNN bundled)
#
# Why this is needed:
# - PyTorch cu124 wheels use non-standard platform tags (linux_x86_64 vs manylinux_2_28_x86_64)
# - uv sync rejects these wheels due to strict platform checking
# - uv pip install handles them correctly
# - onnxruntime-gpu 1.19+ requires cuDNN 9.x, which PyTorch cu124 bundles

set -e

echo "Installing PyTorch with CUDA 12.1 support..."

# Check if torch is already installed with correct version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")

if [[ "$TORCH_VERSION" == "2.3.1"*"+cu121" ]]; then
    echo "PyTorch $TORCH_VERSION already installed, skipping"
else
    echo "Installing torch and torchaudio from PyTorch cu121 index..."
    uv pip install "torch==2.3.1" "torchaudio==2.3.1" \
        --index-url https://download.pytorch.org/whl/cu121
fi

# Verify CUDA is working
echo "Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'torch={torch.__version__}')
print(f'cuda={torch.version.cuda}')
print(f'cudnn={torch.backends.cudnn.version()}')
print(f'cuda_available={torch.cuda.is_available()}')
"

echo "CosyVoice post-install complete"
