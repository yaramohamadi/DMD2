#!/usr/bin/env bash
set -euo pipefail

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# you can keep these as 1-element arrays or add more values later
GEN_CLS_LOSS_WEIGHTS=(15e-3)
CLS_LOSS_WEIGHTS=(5e-2)
GEN_LRS=(2e-6)

# cross product: all steps × all sizes
NUM_DENOISING_STEPS=(3 2 1) #  2 1
DATASET_SIZES=(10 5 1) # 10 5 
 
export WANDB_PROJECT="METFACES_DATASET_SIZES_NUM_DENOISING_SWEEP"

for lr in "${GEN_LRS[@]}"; do
  for glw in "${GEN_CLS_LOSS_WEIGHTS[@]}"; do
    for clw in "${CLS_LOSS_WEIGHTS[@]}"; do
      for steps in "${NUM_DENOISING_STEPS[@]}"; do
        for dsize in "${DATASET_SIZES[@]}"; do

          tag="steps${steps}_data${dsize}_lr${lr}_glw${glw}_clw${clw}"
          echo "[LOCAL] lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  steps=$steps  data=$dsize  tag=$tag"

          export DATASET_NAME="metfaces"
          export TRAIN_ITERS=80000
          export GEN_LR="$lr"                 # use the value directly
          export GEN_CLS_LOSS_WEIGHT="$glw"
          export CLS_LOSS_WEIGHT="$clw"
          export DATASET_SIZE="$dsize"
          export NUM_DENOISING_STEP="$steps"  # set once
          export GRAD_ACCUM_STEPS=8
          export BATCH_SIZE=1

          export CUDA_VISIBLE_DEVICES="0,3"
          export TRAIN_GPUS=0
          export TEST_GPUS=3
          export NPROC_PER_NODE=1
          export NNODES=1
          export EXTRA_TAG="_${tag}"

          bash "$CHILD"

        done
      done
    done
  done
done
