#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"  # <-- sbatch/runner
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

GEN_CLS_LOSS_WEIGHTS=(5e-4)
CLS_LOSS_WEIGHTS=(2e-3)
GEN_LRS=(2e-6)

export DMD_LOSS_WEIGHT=1

export TRAIN_FAKE_ON_REAL="--train_fake_on_real"
export WANDB_PROJECT="REAL_ONLINE_TEACHER"

export GAN_MULTIHEAD=""

DATASETS=("metfaces") #  "cat"

# paired sweep, local runs
for ds in "${DATASETS[@]}"; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHTS[$i]}"
      tag="WITH_DMD_TO_SOURCE"

      echo "[LOCAL] dataset=$ds  lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"

      export DATASET_NAME="$ds"
      export GEN_LR="$lr"
      export GEN_CLS_LOSS_WEIGHT="$glw"
      export CLS_LOSS_WEIGHT="$clw"
      export GRAD_ACCUM_STEPS=1
      export BATCH_SIZE=1
      export NUM_DENOISING_STEP=3
      export CUDA_VISIBLE_DEVICES=0,1
      export TRAIN_GPUS=0
      export TEST_GPUS=1
      export NPROC_PER_NODE=1
      export NNODES=1
      export EXTRA_TAG="_${tag}"

      bash "$CHILD"
    done
  done
done
