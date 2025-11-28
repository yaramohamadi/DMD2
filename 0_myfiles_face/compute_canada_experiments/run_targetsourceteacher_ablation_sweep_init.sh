#!/bin/bash
# 2 datasets × 3 cases = 6 runs
# Both teachers ON; TT is frozen iff TT ckpt provided. GAN multi-head ON.

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

MODE="${MODE:-cc}"
export SERVER="${SERVER:-cc}"
export WANDB_PROJECT="${WANDB_PROJECT:-RED_ABLATIONS}"

# Default base checkpoint path (raw path; runner wraps it as --checkpoint_path ...)
export CHECKPOINT_PATH_DEFAULT="${CHECKPOINT_PATH_DEFAULT:-0_myfiles_face/checkpoint_path/FFHQ_distilled_weights/checkpoint_best}"

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
TARGET_TEACHER_CHECKPOINT_PATH="$TARGET_TEACHER_CHECKPOINT_PATH",\
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
      "$CHILD"
  else
    bash "$CHILD"
  fi
}

# ----- Base hparams -----
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

# Teachers always on
export USE_SOURCE_TEACHER=1
export USE_TARGET_TEACHER=1
export TT_MATCH_GUIDANCE=""

# GAN multi-head ON by default
export GAN_CLASSIFIER="--gan_classifier"
export GAN_MULTIHEAD="--gan_multihead"

fmtw () { echo "$1" | sed 's/\./p/g'; }

# Per-dataset setup
setup_dataset () {
  local ds="$1"
  export DATASET_NAME="$ds"
  if [[ "$ds" == "babies" ]]; then
    export DMD_SOURCE_WEIGHT=0.75
    export DMD_TARGET_WEIGHT=0.25
    export TT_CKPT_FILE="0_myfiles_face/checkpoints/babies_finetune.pt"
  else
    export DMD_SOURCE_WEIGHT=0.25
    export DMD_TARGET_WEIGHT=0.75
    export TT_CKPT_FILE="0_myfiles_face/checkpoints/metfaces_finetune.pt"
  fi
}

# One dataset × three cases
run_cases () {
  local ds="$1"; setup_dataset "$ds"
  local s_tag="$(fmtw "$DMD_SOURCE_WEIGHT")"
  local t_tag="$(fmtw "$DMD_TARGET_WEIGHT")"

  # ---------- Case 1: base_only (base ckpt yes, TT ckpt no) ----------
  export CHECKPOINT_PATH="$CHECKPOINT_PATH_DEFAULT"      # raw path; runner wraps with --checkpoint_path
  export TARGET_TEACHER_CHECKPOINT_PATH=""               # no TT flag
  export TRAIN_TARGET_TEACHER=1                          # train TT when TT ckpt is absent
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "[WARN] base_only: CHECKPOINT_PATH empty → base init skipped."
  elif [[ ! -e "$CHECKPOINT_PATH" ]]; then
    echo "[WARN] base_only: CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
  fi
  export EXTRA_TAG="_${ds}_SW${s_tag}_TW${t_tag}_base_only"
  submit_run "SW${s_tag}_TW${t_tag}_${ds}_base_only_ganMH"
# 
  # ---------- Case 2: tt_only (TT ckpt yes, base ckpt no) ----------
  # export CHECKPOINT_PATH=""                              # no base
  # export TARGET_TEACHER_CHECKPOINT_PATH="--target_teacher_ckpt_path ${TT_CKPT_FILE}"
  # export TRAIN_TARGET_TEACHER=0 # freeze TT when TT ckpt is provided
  # if [[ ! -e "$TT_CKPT_FILE" ]]; then
  #   echo "[WARN] tt_only: TT ckpt not found: $TT_CKPT_FILE"
  # fi
  # export EXTRA_TAG="_${ds}_SW${s_tag}_TW${t_tag}_tt_only"
  # submit_run "SW${s_tag}_TW${t_tag}_${ds}_tt_only_ganMH"
# 
  # # ---------- Case 3: both (base ckpt yes, TT ckpt yes) ----------
  # export CHECKPOINT_PATH="$CHECKPOINT_PATH_DEFAULT"
  # export TARGET_TEACHER_CHECKPOINT_PATH="--target_teacher_ckpt_path ${TT_CKPT_FILE}"
  # export TRAIN_TARGET_TEACHER=0
  # if [[ -z "$CHECKPOINT_PATH" ]]; then
  #   echo "[WARN] both: CHECKPOINT_PATH empty → base init skipped."
  # elif [[ ! -e "$CHECKPOINT_PATH" ]]; then
  #   echo "[WARN] both: CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
  # fi
  # if [[ ! -e "$TT_CKPT_FILE" ]]; then
  #   echo "[WARN] both: TT ckpt not found: $TT_CKPT_FILE"
  # fi
  # export EXTRA_TAG="_${ds}_SW${s_tag}_TW${t_tag}_both"
  # submit_run "SW${s_tag}_TW${t_tag}_${ds}_both_ganMH"
}

# ----- RUN: metfaces + babies -----
for ds in metfaces; do
  run_cases "$ds"
done