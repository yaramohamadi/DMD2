#!/bin/bash
set -euo pipefail

# ---- paths & logging ----
CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# ---- optional common hparams (edit as needed) ----
export WANDB_PROJECT="DMD_GAN_HEADS"
export GEN_LR="2e-6"
export GRAD_ACCUM_STEPS=1
export BATCH_SIZE=1
export NUM_DENOISING_STEP=3
export EVAL_BATCH_SIZE=4
export CUDA_VISIBLE_DEVICES=1
export TRAIN_GPUS=1
export TEST_GPUS=1
export NPROC_PER_NODE=1
export NNODES=1

# If your training script needs these to enable the GAN classifier:
# export GAN_CLASSIFIER=true

# ---- the sweep grid ----
# Four configurations:
#  1) (true,  all)
#  2) (true,  last_pct:0.3)
#  3) (true,  last_pct:0.7)
#  4) (false, <ignored>)
MULTIHEAD=(true          true          false) # true  
HEADSPEC=("last_pct:0.3" "last_pct:0.7" "none") # "all" 

# ---- choose runner: 'local' or 'sbatch' ----
RUNNER="${RUNNER:-local}"   # export RUNNER=sbatch to submit to Slurm

for i in "${!MULTIHEAD[@]}"; do
  mh="${MULTIHEAD[$i]}"
  hl="${HEADSPEC[$i]}"

  # tag for logs / wandb
  tag="mh${mh}"
  if [[ "$mh" == "true" ]]; then
    # sanitize ':' for filenames
    tag="${tag}_hl$(echo "$hl" | tr ':' '_')"
  fi

  echo "=== Running config: GAN_MULTIHEAD=$mh  GAN_HEAD_LAYERS=$hl  tag=$tag ==="

  # export to child
  export GAN_MULTIHEAD="$mh"
  # if false, set a harmless placeholder
  if [[ "$mh" == "true" ]]; then
    export GAN_HEAD_LAYERS="$hl"
  else
    export GAN_HEAD_LAYERS="none"
  fi

  # nice to have: pass an extra run tag
  export EXTRA_TAG="_${tag}"

  # local/in-place run
  bash "$CHILD"
done
