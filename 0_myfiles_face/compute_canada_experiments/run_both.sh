#!/bin/bash

# -----------------------
# Training
# -----------------------
train() {
  echo "[train] Starting training..."
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    main/dhariwal/train_dhariwal.py \
      --generator_lr $GENERATOR_LR \
      --guidance_lr $GENERATOR_LR \
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
      --denoising_sigma_end $DENOISING_SIGMA_END \
      --label_dropout_p $LABEL_DROPOUT_P \
      $GAN_MULTIHEAD \
      --gan_head_type "$GAN_HEAD_TYPE" \
      --gan_head_layers "$GAN_HEAD_LAYERS" \
      --gan_adv_loss "$GAN_ADV_LOSS" \
      $USE_BF16 \
      --grad_accum_steps "$GRAD_ACCUM_STEPS"
}

# -----------------------
# Testing (streaming conditional)
# -----------------------
test_stream_conditional() {
  echo "[test] Starting streaming conditional (uniform) evaluation..."
  CUDA_VISIBLE_DEVICES=$TEST_GPUS python -u main/dhariwal/test_folder_dhariwal.py \
    --folder "$OUTPUT_PATH" \
    --wandb_name "${WANDB_NAME}_eval_uniform" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --fid_npz_root "$FID_NPZ_ROOT" \
    --category "$CATEGORY" \
    --resolution $RESOLUTION \
    --label_dim $LABEL_DIM \
    --label_mode uniform \
    $HAS_NULL \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --total_eval_samples $TOTAL_EVAL_SAMPLES \
    --conditioning_sigma $CONDITIONING_SIGMA \
    --lpips_cluster_size $LPIPS_CLUSTER_SIZE \
    --fewshotdataset "$FEWSHOT_DATASET" \
    $DEN_FLAG \
    --num_denoising_step $NUM_DENOISING_STEP \
    $BEST_FLAG \
    $NO_LPIPS \
    $USE_BF16
}

# -----------------------
# Testing (one-shot marginal/NULL on best ckpt)
# -----------------------
test_null() {
  echo "[test] Evaluating marginal (NULL) on best checkpoint..."
  CUDA_VISIBLE_DEVICES=$TEST_GPUS python -u main/dhariwal/test_folder_dhariwal.py \
    --folder "$OUTPUT_PATH" \
    --wandb_name "${WANDB_NAME}_eval_null_best" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --fid_npz_root "$FID_NPZ_ROOT" \
    --category "$CATEGORY" \
    --resolution $RESOLUTION \
    --label_dim $LABEL_DIM \
    --label_mode null \
    $HAS_NULL \
    --eval_best_once \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --total_eval_samples $TOTAL_EVAL_SAMPLES \
    --conditioning_sigma $CONDITIONING_SIGMA \
    --lpips_cluster_size $LPIPS_CLUSTER_SIZE \
    --fewshotdataset "$FEWSHOT_DATASET" \
    $DEN_FLAG \
    --num_denoising_step $NUM_DENOISING_STEP \
    $NO_LPIPS
}

# -----------------------
# Orchestration
# -----------------------
#train &                  # start training
#TRAIN_PID=$!

test_stream_conditional &  # start streaming conditional eval
TEST_PID=$!

# Wait for both to finish 
wait $TEST_PID
#wait $TRAIN_PID

# After training finishes, evaluate best checkpoint with NULL sampling
# test_best_null
