#!/bin/bash
set -Eeuo pipefail  # -E makes ERR traps propagate out of functions

# -----------------------
# Training
# -----------------------
train() {
  echo "[train] Starting training..."
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    main/dhariwal/train_dhariwal.py \
      --generator_lr "$GEN_LR" \
      --guidance_lr "$GEN_LR" \
      --train_iters "$TRAIN_ITERS" \
      --output_path "$OUTPUT_PATH" \
      --batch_size "$BATCH_SIZE" \
      --initialie_generator \
      --log_iters "$LOG_ITERS" \
      --resolution "$RESOLUTION" \
      --label_dim "$LABEL_DIM" \
      --dataset_name "$DATASET_NAME" \
      --seed "$SEED" \
      --model_id "$CHECKPOINT_INIT" \
      --wandb_iters "$WANDB_ITERS" \
      --wandb_entity "$WANDB_ENTITY" \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_name "$WANDB_NAME" \
      --real_image_path "$REAL_IMAGE_PATH" \
      --dfake_gen_update_ratio "$DFAKE_GEN_UPDATE_RATIO" \
      --cls_loss_weight "$CLS_LOSS_WEIGHT" \
      ${GAN_CLASSIFIER-} \
      --gen_cls_loss_weight "$GEN_CLS_LOSS_WEIGHT" \
      --dmd_loss_weight "$DMD_LOSS_WEIGHT" \
      --diffusion_gan \
      --diffusion_gan_max_timestep "$DIFFUSION_GAN_MAX_TIMESTEP" \
      --delete_ckpts \
      --max_checkpoint "$MAX_CHECKPOINT" \
      --denoising \
      --num_denoising_step "$NUM_DENOISING_STEP" \
      --denoising_sigma_end "$DENOISING_SIGMA_END" \
      --label_dropout_p "$LABEL_DROPOUT_P" \
      ${GAN_MULTIHEAD:-} \
      --gan_head_type "${GAN_HEAD_TYPE:-}" \
      --gan_head_layers "${GAN_HEAD_LAYERS:-}" \
      --gan_adv_loss "${GAN_ADV_LOSS:-}" \
      ${USE_BF16:-} \
      --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
      ${REVERSE_DMD-} \
      ${TRAIN_FAKE_ON_REAL-} \
      --use_source_teacher "$USE_SOURCE_TEACHER" \
      --use_target_teacher "$USE_TARGET_TEACHER" \
      --train_target_teacher "$TRAIN_TARGET_TEACHER"
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
    --resolution "$RESOLUTION" \
    --label_dim "$LABEL_DIM" \
    --label_mode "uniform" \
    ${HAS_NULL:-} \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --total_eval_samples "$TOTAL_EVAL_SAMPLES" \
    --conditioning_sigma "$CONDITIONING_SIGMA" \
    --lpips_cluster_size "$LPIPS_CLUSTER_SIZE" \
    --fewshotdataset "$FEWSHOT_DATASET" \
    ${DEN_FLAG:-} \
    --num_denoising_step "$NUM_DENOISING_STEP" \
    ${BEST_FLAG:-} \
    ${NO_LPIPS:-} \
    ${USE_BF16:-}
}

TEST_PID=""
TEST_PGID=""

teardown() {
  local code=$?
  # prevent re-entry
  trap - EXIT INT TERM ERR

  if [[ -n "${TEST_PID:-}" ]] && kill -0 "$TEST_PID" 2>/dev/null; then
    # determine process group and kill the entire group so children die too
    if [[ -z "${TEST_PGID:-}" ]]; then
      TEST_PGID="$(ps -o pgid= "$TEST_PID" | tr -d ' ')"
    fi
    echo "[orchestrator] stopping test (pid=$TEST_PID, pgid=${TEST_PGID:-?})"
    if [[ -n "${TEST_PGID:-}" ]]; then
      kill -TERM "-$TEST_PGID" 2>/dev/null || true
    else
      kill -TERM "$TEST_PID" 2>/dev/null || true
    fi
    # wait up to 10s, then force kill if still alive
    for _ in {1..10}; do
      kill -0 "$TEST_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$TEST_PID" 2>/dev/null; then
      if [[ -n "${TEST_PGID:-}" ]]; then
        kill -KILL "-$TEST_PGID" 2>/dev/null || true
      else
        kill -KILL "$TEST_PID" 2>/dev/null || true
      fi
    fi
    # reap
    wait "$TEST_PID" 2>/dev/null || true
  fi

  exit "$code"
}

trap teardown EXIT INT TERM ERR

# -----------------------
# Orchestration
# -----------------------

# Start test in its own process group so children share the PGID

# export -f test_stream_conditional
# ( setsid bash -c 'test_stream_conditional' ) &
# TEST_PID=$!
# TEST_PGID="$(ps -o pgid= "$TEST_PID" | tr -d ' ')" || true

# Run training in foreground; on exit (success or error), EXIT trap runs teardown()
train
# end of script — teardown() will run via the EXIT trap with train’s exit code

test_stream_conditional
