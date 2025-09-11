#!/bin/bash

# Mother sweep for V0/V1/V2 variants
# - V0: Unconditional train → Unconditional test
# - V1: Conditional (no NULL, no dropout) train → Conditional test
# - V2: Conditional (+NULL, with dropout) train → Conditional + Marginal(NULL) tests

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"   # your sbatch child
SLURM_LOG_DIR="0_myfiles_face/slurm"
mkdir -p "$SLURM_LOG_DIR"

# ---- Global knobs you may override when calling this script ----
K="${K:-10}"                                 # number of real pseudo-classes
NUM_DENOISING_STEP="${NUM_DENOISING_STEP:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-DMD_babies_V_sweep}"

# For V2 only: try one or more dropout probabilities
# Example to sweep: export V2_DROPS="0.2 0.3 0.5"
V2_DROPS=(${V2_DROPS:-0.30})

# Optional: choose a single LR/BS inside the child, or let the child sweep.
# Here we just tag runs; child controls the actual LR/BS arrays.

run_child() {
  local mode="$1" tag="$2"
  echo "[RUN] $mode ($tag)"
  export EXP_TRAIN_MODE="$mode"
  export K
  export NUM_DENOISING_STEP
  export WANDB_PROJECT
  export LABEL_DROPOUT_P="${3:-0.0}"
  export EXTRA_TAG="_${tag}"
  export SERVER="local"
  export CUDA_VISIBLE_DEVICES=0,2
  export TRAIN_GPUS=0
  export TEST_GPUS=2
  export NPROC_PER_NODE=1
  export NNODES=1
  # Optional: capture logs like sbatch would
  bash "$CHILD"
}

# V0: Uncond → Uncond
# run_child V0 "V0_uncond" 0.0
run_child V1 "V1_cond_noNull" 0.0

# ---------- V0: Uncond → Uncond ----------
# tag="V0_uncond"
# bash \
#   --job-name="dmd2_babies_${tag}" \
#   --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
#   --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
#   --export=ALL,EXP_TRAIN_MODE=V0,K="$K",LABEL_DROPOUT_P="0.0",NUM_DENOISING_STEP="$NUM_DENOISING_STEP",WANDB_PROJECT="$WANDB_PROJECT",EXTRA_TAG="_${tag}" \
#   "$CHILD"

# ---------- V1: Cond(no NULL) → Cond ----------
# tag="V1_cond_noNull"
# sbatch \
#   --job-name="dmd2_babies_${tag}" \
#   --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
#   --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
#   --export=ALL,EXP_TRAIN_MODE=V1,K="$K",LABEL_DROPOUT_P="0.0",NUM_DENOISING_STEP="$NUM_DENOISING_STEP",WANDB_PROJECT="$WANDB_PROJECT",EXTRA_TAG="_${tag}" \
#   "$CHILD"
# 
# # ---------- V2: Cond(+NULL, dropout) → Cond + Marginal(NULL) ----------
# for p in "${V2_DROPS[@]}"; do
#   ptag="$(printf '%s' "$p" | sed 's/\./p/g')"   # 0.30 -> 0p30 for filenames
#   tag="V2_cond_withNull_drop${ptag}"
#   sbatch \
#     --job-name="dmd2_babies_${tag}" \
#     --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
#     --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
#     --export=ALL,EXP_TRAIN_MODE=V2,K="$K",LABEL_DROPOUT_P="$p",NUM_DENOISING_STEP="$NUM_DENOISING_STEP",WANDB_PROJECT="$WANDB_PROJECT",EXTRA_TAG="_${tag}" \
#     "$CHILD"
# done