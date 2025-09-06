#!/bin/bash

############################
# User Configurations
############################

# GPU selection
CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES

# Project paths
EXPERIMENT_NAME="FFHQ_dmd1_fewstep"
PROJECT_PATH="0_myfiles_face"                                 # checkpoint & dataset folder
CHECKPOINT_FOLDER="$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME/"

# Weights & Biases (W&B) tracking
WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie"
WANDB_PROJECT="DMD_face"
WANDB_NAME="{$EXPERIMENT_NAME}_evaluation"

# Dataset & evaluation
FID_NPZ_ROOT="$PROJECT_PATH/datasets/fid_npz"                 # must contain ${CATEGORY}.npz
CATEGORY="FFHQ"                                               # -> ${FID_NPZ_ROOT}/FFHQ.npz
FEWSHOT_DATASET="/export/datasets/public/diffusion_datasets/adaptation/datasets/targets/10_babies/0/"

# Evaluation parameters
RESOLUTION=256
LABEL_DIM=0
EVAL_BATCH_SIZE=4
TOTAL_EVAL_SAMPLES=5000
CONDITIONING_SIGMA=80.0
LPIPS_CLUSTER_SIZE=100
NO_LPIPS="--no_lpips"    # leave blank if you want LPIPS

############################
# Run Evaluation
############################

python -u main/dhariwal/test_folder_dhariwal.py \
  --folder "$CHECKPOINT_FOLDER" \
  --wandb_name "$WANDB_NAME" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --fid_npz_root "$FID_NPZ_ROOT" \
  --category "$CATEGORY" \
  --resolution $RESOLUTION \
  --label_dim $LABEL_DIM \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --total_eval_samples $TOTAL_EVAL_SAMPLES \
  --conditioning_sigma $CONDITIONING_SIGMA \
  --lpips_cluster_size $LPIPS_CLUSTER_SIZE \
  --fewshotdataset "$FEWSHOT_DATASET" \
  $NO_LPIPS
