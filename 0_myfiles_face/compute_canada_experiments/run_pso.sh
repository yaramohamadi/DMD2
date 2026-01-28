#!/usr/bin/env bash
set -euo pipefail

CKPT="/path/to/pytorch_model.bin"
INSTANCE_DIR="/path/to/instance_images"
OUTDIR="./pso_lora_out"

INSTANCE_PROMPT="photo of TOK dog"
CLASS_PROMPT="photo of a dog"

UNIQUE_TOKEN="TOK"
CLASS_NAME="dog"

accelerate launch scripts/train_pso_sd15_1step.py \
  --dmd2_unet_ckpt "$CKPT" \
  --instance_dir "$INSTANCE_DIR" \
  --instance_prompt "$INSTANCE_PROMPT" \
  --class_prompt "$CLASS_PROMPT" \
  --unique_token "$UNIQUE_TOKEN" \
  --class_name "$CLASS_NAME" \
  --output_dir "$OUTDIR" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 600 \
  --learning_rate 2e-4 \
  --mixed_precision fp16 \
  --num_negatives 20 \
  --beta_pso 5.0 \
  --neg_defactor 0.1 \
  --prior_loss_weight 0.5 \
  --validation_steps 200 \
  --num_validation_images 1
