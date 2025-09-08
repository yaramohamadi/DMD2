#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"   # <-- change to your .sbatch path
SLURM_LOG_DIR="0_myfiles_face/slurm"
mkdir -p "$SLURM_LOG_DIR"

for w in 1 0.01 0.001; do
  tag="dmd$(printf '%s' "$w" | sed 's/\./p/g')"   # e.g., dmd0p5

  sbatch \
    --job-name="dmd2_babies_${tag}" \
    --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
    --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
    --export=ALL,DMD_LOSS_WEIGHT="$w",WANDB_PROJECT="DMD_unconditional_babies_dmd_weight_ablation",EXTRA_TAG="_${tag}" \
    "$CHILD"
  # bash $CHILD

done