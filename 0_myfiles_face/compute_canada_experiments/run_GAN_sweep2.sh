#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
SLURM_LOG_DIR="0_myfiles_face/slurm"
mkdir -p "$SLURM_LOG_DIR"

LOSSES=("wgan") # "bce" 

WANDB_PROJECT="DMD_babies_gan_sweep"

for loss in "${LOSSES[@]}"; do
  tag="gan_${loss}"

  if [[ -n "$SLURM_JOB_ID" ]]; then
    # HPC / SLURM
    sbatch \
      --job-name="dmd2_babies_${tag}" \
      --output="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.out" \
      --error="${SLURM_LOG_DIR}/dmd2_babies_${tag}-%j.err" \
      --export=ALL,GAN_ADV_LOSS="$loss",WANDB_PROJECT="$WANDB_PROJ",EXTRA_TAG="_${tag}" \
      "$CHILD"
  else
    # Local run
    echo "[LOCAL] running GAN loss = $loss"
    export GAN_ADV_LOSS="$loss" 
    export WANDB_PROJECT="$WANDB_PROJECT" 
    export EXTRA_TAG="_${tag}" 
    export CUDA_VISIBLE_DEVICES=2,3
    export TRAIN_GPUS=2
    export TEST_GPUS=3
    export NPROC_PER_NODE=1
    export NNODES=1
    bash "$CHILD"
  fi
done