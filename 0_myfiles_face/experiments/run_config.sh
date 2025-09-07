#!/bin/bash
set -e

# -----------------------
# Fixed configs
# -----------------------
export CUDA_VISIBLE_DEVICES=1,3
export TRAIN_GPUS=1,3
export TEST_GPUS=3
export NPROC_PER_NODE=2
export NNODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)   # pick a random free port

export PROJECT_PATH="0_myfiles_face"
export CHECKPOINT_INIT="$PROJECT_PATH/checkpoint_path/ffhq.pt"
export REAL_IMAGE_PATH="$PROJECT_PATH/datasets/targets/FFHQ_lmdb"

export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie"
export WANDB_PROJECT="DMD_face"

export TRAIN_ITERS=20000
export SEED=10
export RESOLUTION=256
export LABEL_DIM=0
export DATASET_NAME="FFHQ"
export DENOISING_SIGMA_END=0.5

export DFAKE_GEN_UPDATE_RATIO=5
export CLS_LOSS_WEIGHT=1e-2
export GEN_CLS_LOSS_WEIGHT=3e-3
export DMD_LOSS_WEIGHT=1
export DIFFUSION_GAN_MAX_TIMESTEP=1000

export LOG_ITERS=100
export WANDB_ITERS=100
export MAX_CHECKPOINT=100

export FID_NPZ_ROOT="$PROJECT_PATH/datasets/fid_npz"
export CATEGORY="FFHQ"
export FEWSHOT_DATASET="$PROJECT_PATH/datasets/targets/10_babies/0/"
export EVAL_BATCH_SIZE=3
export TOTAL_EVAL_SAMPLES=5000
export CONDITIONING_SIGMA=80.0
export LPIPS_CLUSTER_SIZE=100
export NO_LPIPS="--no_lpips"

# -----------------------
# Sweep ranges
# -----------------------

export GEN_LRS=(2e-6)
export BATCH_SIZES=(1)
export DENOISE_STEPS=(2)

# -----------------------
# Sweep loop
# -----------------------
for lr in "${GEN_LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for dn in "${DENOISE_STEPS[@]}"; do
      
      export EXPERIMENT_NAME="FFHQ_lr${lr}_bs${bs}_dn${dn}"
      export OUTPUT_PATH="$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME"
      export WANDB_NAME="$EXPERIMENT_NAME"

      echo "[RUN] $EXPERIMENT_NAME"

      GENERATOR_LR=$lr \
      BATCH_SIZE=$bs \
      NUM_DENOISING_STEP=$dn \
      EXPERIMENT_NAME=$EXPERIMENT_NAME \
      OUTPUT_PATH=$OUTPUT_PATH \
      WANDB_NAME=$WANDB_NAME \
      CHECKPOINT_INIT=$CHECKPOINT_INIT \
      REAL_IMAGE_PATH=$REAL_IMAGE_PATH \
      WANDB_ENTITY=$WANDB_ENTITY \
      WANDB_PROJECT=$WANDB_PROJECT \
      TRAIN_ITERS=$TRAIN_ITERS \
      SEED=$SEED \
      RESOLUTION=$RESOLUTION \
      LABEL_DIM=$LABEL_DIM \
      DATASET_NAME=$DATASET_NAME \
      DFAKE_GEN_UPDATE_RATIO=$DFAKE_GEN_UPDATE_RATIO \
      CLS_LOSS_WEIGHT=$CLS_LOSS_WEIGHT \
      GEN_CLS_LOSS_WEIGHT=$GEN_CLS_LOSS_WEIGHT \
      DMD_LOSS_WEIGHT=$DMD_LOSS_WEIGHT \
      DIFFUSION_GAN_MAX_TIMESTEP=$DIFFUSION_GAN_MAX_TIMESTEP \
      LOG_ITERS=$LOG_ITERS \
      WANDB_ITERS=$WANDB_ITERS \
      MAX_CHECKPOINT=$MAX_CHECKPOINT \
      FID_NPZ_ROOT=$FID_NPZ_ROOT \
      CATEGORY=$CATEGORY \
      FEWSHOT_DATASET=$FEWSHOT_DATASET \
      EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
      TOTAL_EVAL_SAMPLES=$TOTAL_EVAL_SAMPLES \
      CONDITIONING_SIGMA=$CONDITIONING_SIGMA \
      LPIPS_CLUSTER_SIZE=$LPIPS_CLUSTER_SIZE \
      NO_LPIPS=$NO_LPIPS \
      bash $PROJECT_PATH/experiments/run_both.sh

    done
  done
done
