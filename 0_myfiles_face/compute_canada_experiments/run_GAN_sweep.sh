#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
SLURM_LOG_DIR="0_myfiles_face/slurm"
mkdir -p "$SLURM_LOG_DIR"

LOSSES=("hinge" "bce" "wgan" "lsgan")
WANDB_PROJ="DMD_unconditional_babies_gan_sweep"

for loss in "${LOSSES[@]}"; do
  tag="gan_${loss}"

  extra_exports=""
  if [[ "$loss" == "wgan" ]]; then
    extra_exports=",WGAN_GP_LAMBDA=10.0"
  fi

  if [[ -n "$SLURM_JOB_ID" ]]; then
    # HPC / SLURM
    sbatch \
      --job-name="dmd2_babies_${tag}" \
      --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
      --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
      --export=ALL,\
GAN_ADV_LOSS="$loss",\
WANDB_PROJECT="$WANDB_PROJ",\
EXTRA_TAG="_${tag}"\
${extra_exports} \
      "$CHILD"
  else
    # Local run
    echo "[LOCAL] running GAN loss = $loss"
    GAN_ADV_LOSS="$loss" \
    WANDB_PROJECT="$WANDB_PROJ" \
    EXTRA_TAG="_${tag}" \
    ${extra_exports:+${extra_exports//,/ }} \
    bash "$CHILD"
  fi

done
