#!/bin/bash
# Sweep: single-head GAN, source-only (no target) + optional "distill finetuned"

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

MODE="${MODE:-local}"
export SERVER="${SERVER:-local}"
export WANDB_PROJECT="FINETUNE_DISTILL"

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
CHECKPOINT_INIT="$CHECKPOINT_INIT",\
MAKE_DDIM_GRID="$MAKE_DDIM_GRID",\
DDIM_GRID_ONLY="$DDIM_GRID_ONLY",\
EVAL_BEST_ONCE="$EVAL_BEST_ONCE" \
      "$CHILD"
  else
    bash "$CHILD"
  fi
}

# ---- Base hparams ----
export GEN_LR="2e-6"
export GEN_CLS_LOSS_WEIGHT="1e-2"
export CLS_LOSS_WEIGHT="3e-3"
export DMD_LOSS_WEIGHT="1"
export DATASET_SIZE="10"
export NUM_DENOISING_STEP="3"
export GRAD_ACCUM_STEPS=1
export BATCH_SIZE=1
export TRAIN_GPUS=2
export TEST_GPUS=3
export CUDA_VISIBLE_DEVICES=2,3
export NPROC_PER_NODE=1
export NNODES=1
export TRAIN_TARGET_TEACHER=0
export TT_MATCH_GUIDANCE=""

# ---- Fixed: source-only + single-head GAN ----
export USE_SOURCE_TEACHER=1
export USE_TARGET_TEACHER=0
export DMD_SOURCE_WEIGHT=1.0
export DMD_TARGET_WEIGHT=0.0
export GAN_CLASSIFIER="--gan_classifier"
export GAN_MULTIHEAD=""

fmtw () { echo "$1" | sed 's/\./p/g'; }

# ---- Per-dataset checkpoints ----
declare -A CKPT
CKPT[babies]="0_myfiles_face/checkpoints/babies_finetune.pt"
CKPT[cat]="0_myfiles_face/checkpoints/cat_finetune.pt"
CKPT[sunglasses]="0_myfiles_face/checkpoints/sunglasses_finetune.pt"
CKPT[metfaces]="0_myfiles_face/checkpoints/metfaces_finetune.pt"

DATASETS=("babies")

export MAKE_DDIM_GRID="--make_ddim_grid"
export DDIM_GRID_ONLY="--ddim_grid_only"
export EVAL_BEST_ONCE="--eval_best_once"

# ---- NEW: Distill-finetuned axis (set to (0 1) to sweep, or just (0) / (1) to fix) ----
DISTILL_FINETUNED_STATES=(0)

for df in "${DISTILL_FINETUNED_STATES[@]}"; do
  export DISTILL_FINETUNED="$df"
  if [[ "$DISTILL_FINETUNED" == "1" ]]; then
    export DISTILL_FINETUNED_FLAG="--distill_finetuned"
  else
    export DISTILL_FINETUNED_FLAG=""
  fi
  df_tag="df${DISTILL_FINETUNED}"

  for ds in "${DATASETS[@]}"; do
    export DATASET_NAME="$ds"

    # set CHECKPOINT_INIT; warn if missing
    CHECKPOINT_INIT="${CKPT[$ds]}"
    if [[ -z "$CHECKPOINT_INIT" ]]; then
      echo "[WARN] No checkpoint mapping for dataset '$ds'; running without resume."
      export CHECKPOINT_INIT=""
    elif [[ ! -d "$CHECKPOINT_INIT" && ! -f "$CHECKPOINT_INIT" ]]; then
      echo "[WARN] Checkpoint path not found for '$ds': $CHECKPOINT_INIT ; running without resume."
      export CHECKPOINT_INIT=""
    else
      export CHECKPOINT_INIT
    fi

    s_tag="$(fmtw "$DMD_SOURCE_WEIGHT")"
    t_tag="$(fmtw "$DMD_TARGET_WEIGHT")"
    ck_tag="$([[ -n "$CHECKPOINT_INIT" ]] && echo "ckpt1" || echo "ckpt0")"
    tag="SRCONLY_${ds}_SW${s_tag}_TW${t_tag}_gan_single_${ck_tag}_${df_tag}"

    echo "[$MODE] dataset=$ds | SRC=1 TGT=0 | SW=1.0 TW=0.0 | GAN=single-head | resume='${CHECKPOINT_INIT:-none}' | distill_finetuned=${DISTILL_FINETUNED} | tag=$tag"

    export EXTRA_TAG="_${tag}"
    submit_run "$tag"
  done
done
