#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"   # <- point to your .sbatch file
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

WEIGHTS=(5e-2 1e-1 5e-1)

for w in "${WEIGHTS[@]}"; do
  tag="gcls$(printf '%s' "$w" | sed -e 's/\./p/g' -e 's/-/m/g')"
  sbatch \
    --job-name="dmd2_babies_${tag}" \
    --output="${LOGDIR}/dmd2_babies_${tag}-%j.out" \
    --error="${LOGDIR}/dmd2_babies_${tag}-%j.err" \
    --export=ALL,GEN_CLS_LOSS_WEIGHT="$w",EXTRA_TAG="_${tag}" \
    "$CHILD"
done