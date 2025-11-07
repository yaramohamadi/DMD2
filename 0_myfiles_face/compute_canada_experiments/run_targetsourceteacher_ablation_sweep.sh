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
export WANDB_PROJECT="${WANDB_PROJECT:-ABLATION_TABLE}"

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
TT_MATCH_GUIDANCE="$TT_MATCH_GUIDANCE" \
      "$CHILD"
  else
    bash "$CHILD"
  fi
}

# --------- Base hparams (match your previous sweeps unless you override) ---------
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
export TRAIN_TARGET_TEACHER=1
export TT_MATCH_GUIDANCE=""

fmtw () { echo "$1" | sed 's/\./p/g'; }

run_row () {
  # args: ds use_src use_tgt sw tw gan_mode rowname
  local ds="$1" use_src="$2" use_tgt="$3" sw="$4" tw="$5" gan="$6" row="$7"

  # map GAN mode -> flags
  case "$gan" in
    none)   export GAN_CLASSIFIER="";                    export GAN_MULTIHEAD="";;
    single) export GAN_CLASSIFIER="--gan_classifier";    export GAN_MULTIHEAD="";;
    multi)  export GAN_CLASSIFIER="--gan_classifier";    export GAN_MULTIHEAD="--gan_multihead";;
    *) echo "Unknown GAN mode: $gan"; exit 1;;
  esac

  export DATASET_NAME="$ds"
  export USE_SOURCE_TEACHER="$use_src"
  export USE_TARGET_TEACHER="$use_tgt"
  export DMD_SOURCE_WEIGHT="$sw"
  export DMD_TARGET_WEIGHT="$tw"

  local s_tag t_tag
  s_tag="$(fmtw "$sw")"; t_tag="$(fmtw "$tw")"
  local tag="TAB_${ds}_${row}_src${use_src}_tgt${use_tgt}_SW${s_tag}_TW${t_tag}_gan${gan}"
  echo "[$MODE] $ds | row=$row | src=$use_src tgt=$use_tgt SW=$sw TW=$tw | GAN=$gan | tag=$tag"

  export EXTRA_TAG="_${tag}"
  submit_run "$tag"
}

# metfaces
# ------------------ EXACT 10 RUNS ------------------
for ds in babies; do
  # per-dataset weights when both teachers are ON
  if [[ "$ds" == "babies" ]]; then
    SW_BOTH=0.75; TW_BOTH=0.25
  else
    SW_BOTH=0.25; TW_BOTH=0.75
  fi

  # (1) GAN single-head only
  run_row "$ds" 0 1 0.0 0.0 "single" "gan_single_only"

  run_row "$ds" 0 1 0.0 0.0 "multi" "gan_multi_only"

  # (4) DMDtrg + GAN multi-head
  #run_row "$ds" 0 1 0.0 1.0 "multi" "dmd_trg_gan_multi"

  # (5) DMDsrc + DMDtrg + GAN single-head
  run_row "$ds" 1 1 "$SW_BOTH" "$TW_BOTH" "single" "dmd_src_trg_gan_single"

  # (6) DMDsrc + DMDtrg + GAN single-head
  run_row "$ds" 1 1 "$SW_BOTH" "$TW_BOTH" "none" "dmd_src_trg_only"
done