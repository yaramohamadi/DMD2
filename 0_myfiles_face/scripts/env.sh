#!/usr/bin/env bash
set -e

MODE="${1:-}"
if [[ "$MODE" != "local" && "$MODE" != "cc" ]]; then
  echo "Usage: $0 {local|cc}"
  exit 1
fi

# --- versions/paths you can tweak ---
ENV_NAME="dmd2"
PY_VER="3.11"
TORCH_CUDA_TAG="cu118"  # use "cpu" for CPU-only
VENV_DIR="${PROJECT:-$HOME}/dmd2_env"  # used only for cc mode

# ---------- LOCAL (conda) ----------
if [[ "$MODE" == "local" ]]; then
  # 1) Conda activate
  eval "$(conda shell.bash hook)"

  # 2) Create env if missing
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -y -n "$ENV_NAME" "python=$PY_VER"
  conda activate "$ENV_NAME"

  # 3) Tools + anyio
  pip install -U pip setuptools wheel anyio

  # 4) PyTorch
  if [[ "$TORCH_CUDA_TAG" == "cpu" ]]; then
    pip install torch==2.0.1 torchvision==0.15.2
  else
    pip install \
      "torch==2.0.1+${TORCH_CUDA_TAG}" \
      "torchvision==0.15.2+${TORCH_CUDA_TAG}" \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
  fi

  # 5) Repo deps (run from repo root)
  pip install -r requirements.txt
  python setup.py develop
  pip install lpips scikit-learn
  pip install opencv-python 
  pip install datasets 

  # 6) guided-diffusion submodule
  if [ ! -d third_party/dhariwal ]; then
    git submodule add https://github.com/yaramohamadi/guided-diffusion third_party/dhariwal
  fi
  git submodule update --init --recursive third_party/dhariwal
  python -m pip install -e third_party/dhariwal --no-deps --no-build-isolation

  echo "✅ Done (local). Next time: conda activate $ENV_NAME"
  exit 0
fi

# ---------- COMPUTE CANADA (virtualenv) ----------
if [[ "$MODE" == "cc" ]]; then
  module load python/$PY_VER
  module load rust/1.85.0
  module load gcc opencv/4.9.0
  module load arrow/15.0.1

  # 1) venv
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"

  # 2) Tools + anyio
  pip install -U pip setuptools wheel anyio

  # 3) PyTorch
  pip install \
        "torch==2.0.1+${TORCH_CUDA_TAG}" \
        "torchvision==0.15.2+${TORCH_CUDA_TAG}" \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

  # 4) Repo deps
  pip install -r requirements.txt
  pip install -e .
  pip install lpips scikit-learn
  pip install --no-deps "datasets==4.0.0"

  # from repo root
  rm -rf third_party/dhariwal
  mkdir -p third_party
  git clone --depth 1 https://github.com/yaramohamadi/guided-diffusion third_party/dhariwal
  pip install -e third_party/dhariwal --no-deps --no-build-isolation

  echo "✅ Done (Compute Canada). Activate later with: source \"$VENV_DIR/bin/activate\""
fi
