#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
SLURM_LOG_DIR="0_myfiles_face/slurm"
mkdir -p "$SLURM_LOG_DIR"

LOSSES=("hinge" "wgan" "LSGAN" "BCE")  #  "wgan"

export NO_LPIPS="--no_lpips"

WANDB_PROJECT="DMD_sunglasses_gan_sweep"

for loss in "${LOSSES[@]}"; do
  tag="gan_${loss}"

  if [[ -n "$SLURM_JOB_ID" ]]; then
    # HPC / SLURM
    sbatch \
      --job-name="dmd2_sunglasses_${tag}" \
      --output="${SLURM_LOG_DIR}/dmd2_sunglasses_${tag}-%j.out" \
      --error="${SLURM_LOG_DIR}/dmd2_sunglasses_${tag}-%j.err" \
      --export=ALL,GAN_ADV_LOSS="$loss",WANDB_PROJECT="$WANDB_PROJ",EXTRA_TAG="_${tag}" \
      "$CHILD"
  else
    # Local run
    echo "[LOCAL] running GAN loss = $loss"
    export DATASET_NAME="sunglasses"
    export GAN_ADV_LOSS="$loss" 
    export WANDB_PROJECT="$WANDB_PROJECT" 
    export EXTRA_TAG="_${tag}" 
    export CUDA_VISIBLE_DEVICES=0,1
    export TRAIN_GPUS=0,1
    export TEST_GPUS=2
    export NPROC_PER_NODE=1
    export GRAD_ACCUM_STEPS=2
    export NNODES=1
    bash "$CHILD"
  fi
done