#!/usr/bin/env bash
set -euo pipefail

# Run on the target Ubuntu server after confirming nvidia-smi works.
# PyTorch wheels include the CUDA runtime; a matching system CUDA toolkit is
# still needed to compile this project's custom CUDA extensions.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Conservative match for the verified server (driver 550.78).
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r gpu_setup/requirements-base.txt

export TORCH_CUDA_ARCH_LIST="8.9"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

