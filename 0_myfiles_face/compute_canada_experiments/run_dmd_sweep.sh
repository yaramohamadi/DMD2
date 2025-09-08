#!/bin/bash
set -euo pipefail

CHILD="0_myfiles_face/experiments/run_config_babies.sh"   # <-- change to your .sbatch path
SLURM_LOG_DIR="0_myfiles_face/slurm"
mkdir -p "$SLURM_LOG_DIR"


for w in 1 0.5 0.25 0.1 0.05; do
  tag="dmd$(printf '%s' "$w" | sed 's/\./p/g')"   # e.g., dmd0p5

  sbatch \
    --job-name="dmd2_babies_${tag}" \
    --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
    --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
    --export=ALL,DMD_LOSS_WEIGHT="$w",EXTRA_TAG="_${tag}" \
    "$CHILD"
done