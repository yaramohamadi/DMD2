#!/usr/bin/env bash
set -e

# ---------------- User-tunable settings ----------------
ENV_NAME="unidad"        # name of the conda env
PY_VER="3.10.13"         # Python version
TORCH_CUDA_TAG="cu118"   # "cu118" for CUDA 11.8, or "cpu" for CPU-only
# -------------------------------------------------------

echo ">>> Setting up local conda environment: $ENV_NAME (Python $PY_VER, torch $TORCH_CUDA_TAG)"

# 1) Conda activate
if ! command -v conda &> /dev/null; then
  echo "conda not found in PATH. Please load/initialize conda first."
  exit 1
fi
eval "$(conda shell.bash hook)"

# 2) Create env if missing
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> Creating conda env '$ENV_NAME' with Python $PY_VER"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

echo ">>> Activating env '$ENV_NAME'"
conda activate "$ENV_NAME"

# 3) Basic tools
echo ">>> Upgrading pip/setuptools/wheel"
pip install -U pip setuptools wheel

# 4) PyTorch
echo ">>> Installing PyTorch and torchvision ($TORCH_CUDA_TAG)"
if [[ "$TORCH_CUDA_TAG" == "cpu" ]]; then
  pip install torch==2.0.1 torchvision==0.15.2
else
  pip install \
    "torch==2.0.1+${TORCH_CUDA_TAG}" \
    "torchvision==0.15.2+${TORCH_CUDA_TAG}" \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
fi

# 5) Repo deps (run from repo root)
echo ">>> Installing repo requirements"
pip install -r requirements.txt
python setup.py develop

# 6) guided-diffusion (Dhariwal)
echo ">>> Setting up guided-diffusion (Dhariwal)"
if [ ! -d third_party/dhariwal ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/yaramohamadi/guided-diffusion third_party/dhariwal
else
  echo ">>> third_party/dhariwal already exists, skipping clone"
fi

python -m pip install -e third_party/dhariwal --no-deps --no-build-isolation

echo "✅ Done (local setup complete)."
echo "Next time, just run:  conda activate $ENV_NAME"
