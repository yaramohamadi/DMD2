#!/bin/bash

############################
# User Configurations
############################

# Project paths
EXPERIMENT_NAME="FFHQ_dmd1_fewstep"
PROJECT_PATH="0_myfiles_face"
CHECKPOINT_INIT="$PROJECT_PATH/checkpoint_path/ffhq.pt"
REAL_IMAGE_PATH="$PROJECT_PATH/datasets/FFHQ_lmdb"
OUTPUT_PATH="$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME/"


# Weights & Biases (W&B) tracking
WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie"
WANDB_PROJECT="DMD_face"
WANDB_NAME="$EXPERIMENT_NAME"

# Training parameters
GENERATOR_LR=2e-6
GUIDANCE_LR=2e-6
BATCH_SIZE=1
TRAIN_ITERS=50000
SEED=10
RESOLUTION=256
LABEL_DIM=0
DATASET_NAME="FFHQ"

# Diffusion + GAN configs
DFAKE_GEN_UPDATE_RATIO=5
CLS_LOSS_WEIGHT=1e-2
GEN_CLS_LOSS_WEIGHT=3e-3
DMD_LOSS_WEIGHT=1
DIFFUSION_GAN_MAX_TIMESTEP=1000

# Denoising
NUM_DENOISING_STEP=2
DENOISING_SIGMA_END=0.5

# Logging & checkpoints
LOG_ITERS=500
WANDB_ITERS=100
MAX_CHECKPOINT=100

# GPU & distributed
CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(pwd):$PYTHONPATH
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

############################
# Run Training
############################

CUDA_VISIBLE_DEVICES=3 torchrun \
  --nproc_per_node 1 \
  --nnodes 1 \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  main/dhariwal/train_dhariwal.py \
    --generator_lr $GENERATOR_LR \
    --guidance_lr $GUIDANCE_LR \
    --train_iters $TRAIN_ITERS \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --initialie_generator \
    --log_iters $LOG_ITERS \
    --resolution $RESOLUTION \
    --label_dim $LABEL_DIM \
    --dataset_name "$DATASET_NAME" \
    --seed $SEED \
    --model_id "$CHECKPOINT_INIT" \
    --wandb_iters $WANDB_ITERS \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --real_image_path "$REAL_IMAGE_PATH" \
    --dfake_gen_update_ratio $DFAKE_GEN_UPDATE_RATIO \
    --cls_loss_weight $CLS_LOSS_WEIGHT \
    --gan_classifier \
    --gen_cls_loss_weight $GEN_CLS_LOSS_WEIGHT \
    --dmd_loss_weight $DMD_LOSS_WEIGHT \
    --diffusion_gan \
    --diffusion_gan_max_timestep $DIFFUSION_GAN_MAX_TIMESTEP \
    --delete_ckpts \
    --max_checkpoint $MAX_CHECKPOINT \
    --denoising \
    --num_denoising_step $NUM_DENOISING_STEP \
    --denoising_sigma_end $DENOISING_SIGMA_END
