#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"  # sbatch-ready (has #SBATCH lines)
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# --------- mode switch: local vs Compute Canada (sbatch) ----------
MODE="${MODE:-"local"}"   # set MODE=cc to use sbatch
submit_run () {
  local tag="$1"

  if [[ "$MODE" == "cc" ]]; then
    # Submit as a Slurm job with per-run exports
    sbatch \
      --job-name="$tag" \
      --output="$LOGDIR/%x-%j.out" \
      --export=ALL,\
DATASET_NAME="$DATASET_NAME",\
GEN_LR="$GEN_LR",\
GEN_CLS_LOSS_WEIGHT="$GEN_CLS_LOSS_WEIGHT",\
CLS_LOSS_WEIGHT="$CLS_LOSS_WEIGHT",\
DMD_LOSS_WEIGHT="$DMD_LOSS_WEIGHT",\
GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS",\
BATCH_SIZE="$BATCH_SIZE",\
NUM_DENOISING_STEP="$NUM_DENOISING_STEP",\
TRAIN_GPUS="$TRAIN_GPUS",\
TEST_GPUS="$TEST_GPUS",\
NPROC_PER_NODE="$NPROC_PER_NODE",\
NNODES="$NNODES",\
TRAIN_FAKE_ON_REAL="$TRAIN_FAKE_ON_REAL",\
WANDB_PROJECT="$WANDB_PROJECT",\
EXTRA_TAG="$EXTRA_TAG",\
USE_SOURCE_TEACHER="$USE_SOURCE_TEACHER",\
USE_TARGET_TEACHER="$USE_TARGET_TEACHER",\
TRAIN_TARGET_TEACHER="$TRAIN_TARGET_TEACHER",\
GAN_CLASSIFIER="$GAN_CLASSIFIER" \
      "$CHILD"
  else
    # Local run
    bash "$CHILD"
  fi
}
# ------------------------------------------------------------------

# sweeps
GEN_CLS_LOSS_WEIGHTS=(0)
CLS_LOSS_WEIGHTS=(0)
GEN_LRS=(2e-6)
DMD_LOSS_WEIGHTS=(1)

# GAN: disabled (child should conditionally add --gan_classifier only if GAN_CLASSIFIER is non-empty)
export GAN_CLASSIFIER=""

# fixed flags
export WANDB_PROJECT="METFACES_TARGET_TEACHER_SWEEP"

# Target Teacher switches (match child usage exactly)
# child passes: --use_source_teacher $USE_SOURCE_TEACHER
#               --use_target_teacher $USE_TARGET_TEACHER
#               --train_target_teacher $TRAIN_TARGET_TEACHER   # you said this expects a value now
export USE_SOURCE_TEACHER=0
export USE_TARGET_TEACHER=1
export TRAIN_TARGET_TEACHER=1

DATASETS=("metfaces")

for ds in "${DATASETS[@]}"; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHTS[$i]}"

      for dmdw in "${DMD_LOSS_WEIGHTS[@]}"; do
        # per-run vars (exported for CHILD)
        export DATASET_NAME="$ds"
        export GEN_LR="$lr"
        export GEN_CLS_LOSS_WEIGHT="$glw"
        export CLS_LOSS_WEIGHT="$clw"
        export DMD_LOSS_WEIGHT="$dmdw"

        export GRAD_ACCUM_STEPS=1
        export BATCH_SIZE=1
        export NUM_DENOISING_STEP=3

        # Let Slurm handle GPU binding; for local you can still set CUDA_VISIBLE_DEVICES inside CHILD if needed
        export TRAIN_GPUS=0
        export TEST_GPUS=1
        export NPROC_PER_NODE=1
        export NNODES=1

        # Tag (no reverse flag now). Include TT settings & GAN state.
        tt_tag="src${USE_SOURCE_TEACHER}_tgt${USE_TARGET_TEACHER}_trainTT${TRAIN_TARGET_TEACHER}"
        gan_tag="gan$([[ -n "$GAN_CLASSIFIER" ]] && echo 1 || echo 0)"
        tag="TT_${ds}_DMDW${dmdw}_lr${lr}_clw${clw}_glw${glw}_${tt_tag}_${gan_tag}"

        echo "[$MODE] dataset=$ds  lr=$lr  DMD_LOSS_WEIGHT=$dmdw  TT:$tt_tag  $gan_tag  tag=$tag"

        export EXTRA_TAG="_${tag}"

        submit_run "$tag"
      done
    done
  done
done
