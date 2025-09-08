#!/bin/bash
set -euo pipefail

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# pick the sweep you want
STEPS=(1 2 3)

for dn in "${STEPS[@]}"; do
  tag="dn${dn}"
  sbatch \
    --job-name="dmd2_babies_${tag}" \
    --output="${LOGDIR}/dmd2_babies_${tag}-%j.out" \
    --error="${LOGDIR}/dmd2_babies_${tag}-%j.err" \
    --export=ALL,NUM_DENOISING_STEP="$dn",EXTRA_TAG="_${tag}" \
    "$CHILD"
done
