#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# pick the sweep you want

GEN_CLS_LOSS_WEIGHT=(3e-3 15e-3)
CLS_LOSS_WEIGHT=(1e-3 5e-3)
GEN_LR=(5e-8)

export WANDB_PROJECT=DMD_ABLATE_LR_CLSLOSS"}"

for dn in "${STEPS[@]}"; do
  for dn in "${STEPS[@]}"; do
    tag="dn${dn}"
    sbatch \
      --job-name="dmd2_babies_${tag}" \
      --output="${LOGDIR}/dmd2_babies_${tag}-%j.out" \
      --error="${LOGDIR}/dmd2_babies_${tag}-%j.err" \
      --export=ALL,EXTRA_TAG="_${tag}" \
      "$CHILD"
  done
done



# #!/bin/bash
# 
# CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"   # point to your script
# 
# STEPS=(4)
# 
# for dn in "${STEPS[@]}"; do
#   tag="dn${dn}"
#   echo "Running with NUM_DENOISING_STEP=$dn"
# 
#   # Run locally, export variables, redirect logs
#   NUM_DENOISING_STEP="$dn" EXTRA_TAG="$tag" bash "$CHILD"
# done

