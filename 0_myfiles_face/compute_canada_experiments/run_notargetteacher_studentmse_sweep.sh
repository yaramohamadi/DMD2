#!/bin/bash
# Exactly 10 experiments = 5 rows (table) × 2 datasets
# Rows:
#  (1) GAN single-head only
#  (2) DMDsrc only
#  (3) DMDtrg only
#  (4) DMDtrg + GAN multi-head
#  (5) DMDsrc + DMDtrg + GAN single-head
# Skipped (already done): [DMDsrc + DMDtrg + GAN multi-head], [DMDsrc + GAN multi-head]

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

MODE="${MODE:-cc}"
export SERVER="${SERVER:-cc}"
export WANDB_PROJECT="${WANDB_PROJECT:-NOTARGETTEACHER_STUDENTMSE_SWEEP}"

submit_run () {
  local tag="$1"
  if [[ "$MODE" == "cc" ]]; then
    sbatch \
      --job-name="$tag" \
      --output="$LOGDIR/%x-%j.out" \
      --export=ALL,\
DATASET_NAME="$DATASET_NAME",\
DATASET_SIZE="$DATASET_SIZE",\
GEN_LR="$GEN_LR",\
GEN_CLS_LOSS_WEIGHT="$GEN_CLS_LOSS_WEIGHT",\
CLS_LOSS_WEIGHT="$CLS_LOSS_WEIGHT",\
DMD_LOSS_WEIGHT="$DMD_LOSS_WEIGHT",\
DMD_SOURCE_WEIGHT="$DMD_SOURCE_WEIGHT",\
DMD_TARGET_WEIGHT="$DMD_TARGET_WEIGHT",\
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
GAN_MULTIHEAD="$GAN_MULTIHEAD",\
TT_MATCH_GUIDANCE="$TT_MATCH_GUIDANCE",\
GEN_DENOISE_WEIGHT="$GEN_DENOISE_WEIGHT",\
DISABLE_TARGET_TEACHER="$DISABLE_TARGET_TEACHER" \
      "$CHILD"
  else
    bash "$CHILD"
  fi
}

# --------- Base hparams for this sweep ---------
export GEN_LR="2e-6"
export GEN_CLS_LOSS_WEIGHT="1e-2"
export CLS_LOSS_WEIGHT="3e-3"
export DMD_LOSS_WEIGHT="1"
export DATASET_SIZE="10"
export NUM_DENOISING_STEP="3"
export GRAD_ACCUM_STEPS=1
export BATCH_SIZE=1
export TRAIN_GPUS=0
export TEST_GPUS=0
export NPROC_PER_NODE=1
export NNODES=1

# new method: no target teacher at all
export USE_SOURCE_TEACHER="1.0"
export USE_TARGET_TEACHER="0.0"
export TRAIN_TARGET_TEACHER="0.0"
export DMD_SOURCE_WEIGHT="1.0"
export DMD_TARGET_WEIGHT="0.0"
export DISABLE_TARGET_TEACHER="--disable_target_teacher"

# helpers
fmtw () { echo "$1" | sed 's/\./p/g'; }

# SIMPLE SWEEP over generator denoising weight
export DATASET_NAME="metfaces"

# choose whatever weights you want to try
for w in 0.0 0.05 0.1 0.2 0.5 1.0; do 
  export GEN_DENOISE_WEIGHT="$w"
  tag="Gden_${DATASET_NAME}_w$(fmtw "$w")"
  echo "[SWEEP] dataset=$DATASET_NAME | GEN_DENOISE_WEIGHT=$w | tag=$tag"

  export EXTRA_TAG="_${tag}"
  submit_run "$tag"
done