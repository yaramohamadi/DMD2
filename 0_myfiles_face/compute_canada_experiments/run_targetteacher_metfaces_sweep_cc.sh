#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"  # sbatch-ready (has #SBATCH lines)
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# --------- mode switch: local vs Compute Canada (sbatch) ----------
MODE="${MODE:-cc}"   # set MODE=cc to use sbatch
export SERVER="${SERVER:-cc}"
submit_run () {
  local tag="$1"

  if [[ "$MODE" == "cc" ]]; then
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
GAN_CLASSIFIER="$GAN_CLASSIFIER",\
TT_MATCH_GUIDANCE="$TT_MATCH_GUIDANCE" \
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

# GAN off unless either cls loss is non-zero (child should use ${GAN_CLASSIFIER-})
export GAN_CLASSIFIER=""

# TT cadence knob:
#   ""  -> update TT only on generator steps (default behavior)
#   "--tt_match_guidance" -> update TT every optimizer step (same cadence as guidance)
export TT_MATCH_GUIDANCE=""   # set "" to disable # --tt_match_guidance

# fixed flags
export WANDB_PROJECT="METFACES_TARGET_TEACHER_SWEEP"

# Target Teacher switches
export USE_SOURCE_TEACHER=0
export USE_TARGET_TEACHER=1
export TRAIN_TARGET_TEACHER=1    # if your child expects a value; else make it empty and use ${...:+--train_target_teacher}

DATASETS=("metfaces")

for ds in "${DATASETS[@]}"; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHTS[$i]}"

      for dmdw in "${DMD_LOSS_WEIGHTS[@]}"; do
        export DATASET_NAME="$ds"
        export GEN_LR="$lr"
        export GEN_CLS_LOSS_WEIGHT="$glw"
        export CLS_LOSS_WEIGHT="$clw"
        export DMD_LOSS_WEIGHT="$dmdw"

        export GRAD_ACCUM_STEPS=1
        export BATCH_SIZE=1
        export NUM_DENOISING_STEP=3

        export TRAIN_GPUS=0
        export TEST_GPUS=1
        export NPROC_PER_NODE=1
        export NNODES=1

        # If you ever sweep GAN on/off based on weights:
        if [[ "$glw" == "0" && "$clw" == "0" ]]; then
          export GAN_CLASSIFIER=""
        else
          export GAN_CLASSIFIER="--gan_classifier"
        fi

        # Cadence tag for clarity
        cadence_tag="$([ -n "$TT_MATCH_GUIDANCE" ] && echo tt_guid || echo tt_gen)"
        tt_tag="src${USE_SOURCE_TEACHER}_tgt${USE_TARGET_TEACHER}_trainTT${TRAIN_TARGET_TEACHER}_${cadence_tag}"
        gan_tag="gan$([[ -n "$GAN_CLASSIFIER" ]] && echo 1 || echo 0)"
        tag="TT_${ds}_DMDW${dmdw}_lr${lr}_clw${clw}_glw${glw}_${tt_tag}_${gan_tag}"

        echo "[$MODE] dataset=$ds  lr=$lr  DMD_LOSS_WEIGHT=$dmdw  TT:$tt_tag  $gan_tag  tag=$tag"

        export EXTRA_TAG="_${tag}"

        submit_run "$tag"
      done
    done
  done
done
