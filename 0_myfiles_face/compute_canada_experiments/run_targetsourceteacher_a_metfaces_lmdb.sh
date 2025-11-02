#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

MODE="${MODE:-cc}"
export SERVER="${SERVER:-cc}"

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
TT_MATCH_GUIDANCE="$TT_MATCH_GUIDANCE" \
      "$CHILD"
  else
    bash "$CHILD"
  fi
}

# ---------- sweeps ----------
GEN_CLS_LOSS_WEIGHTS=(1e-2)
CLS_LOSS_WEIGHTS=(3e-3)
GEN_LRS=(2e-6)
DMD_LOSS_WEIGHTS=(1)   # global multiplier

# PAIRED per-teacher weights (same length!)
SRC_WEIGHTS=(0.25)
TGT_WEIGHTS=(0.75)

if [[ ${#SRC_WEIGHTS[@]} -ne ${#TGT_WEIGHTS[@]} ]]; then
  echo "[ERROR] SRC_WEIGHTS and TGT_WEIGHTS must have the same length." >&2
  exit 1
fi

export TT_MATCH_GUIDANCE=""  # "--tt_match_guidance" to enable
export WANDB_PROJECT="SWEEP_METFACES_SRC75_TGT25_DENOISINGSTEPS_DATASETSIZE"

# Enable both teachers; TT is trainable
export USE_SOURCE_TEACHER=1
export USE_TARGET_TEACHER=1
export TRAIN_TARGET_TEACHER=1

DATASETS=("metfaces")

# NEW: the two sweep axes you asked for
DATASET_SIZES=(10)
DENOISING_STEPS=(2)

fmtw () { echo "$1" | sed 's/\./p/g'; }

for ds in "${DATASETS[@]}"; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHTS[$i]}"

      for dmdw in "${DMD_LOSS_WEIGHTS[@]}"; do
        for j in "${!SRC_WEIGHTS[@]}"; do
          sw="${SRC_WEIGHTS[$j]}"
          tw="${TGT_WEIGHTS[$j]}"

          # >>> NEW nested sweeps <<<
          for K in "${DATASET_SIZES[@]}"; do
            for NSTEP in "${DENOISING_STEPS[@]}"; do

              export DATASET_NAME="$ds"
              export DATASET_SIZE="$K"           # 10, 5, 1
              export GEN_LR="$lr"
              export GEN_CLS_LOSS_WEIGHT="$glw"
              export CLS_LOSS_WEIGHT="$clw"
              export DMD_LOSS_WEIGHT="$dmdw"
              export DMD_SOURCE_WEIGHT="$sw"
              export DMD_TARGET_WEIGHT="$tw"

              export GRAD_ACCUM_STEPS=1
              export BATCH_SIZE=1
              export NUM_DENOISING_STEP="$NSTEP" # 3, 2, 1

              export CUDA_VISIBLE_DEVICES=0
              export TRAIN_GPUS=0
              export TEST_GPUS=0
              export NPROC_PER_NODE=1
              export NNODES=1

              # keep GAN off here (turn on only if you sweep cls losses)
              if [[ "$glw" == "0" && "$clw" == "0" ]]; then
                export GAN_CLASSIFIER=""
              else
                export GAN_CLASSIFIER="--gan_classifier"
              fi

              cadence_tag="$([ -n "$TT_MATCH_GUIDANCE" ] && echo tt_guid || echo tt_gen)"
              s_tag="$(fmtw "$sw")"
              t_tag="$(fmtw "$tw")"
              tt_tag="src${USE_SOURCE_TEACHER}_tgt${USE_TARGET_TEACHER}_trainTT${TRAIN_TARGET_TEACHER}_${cadence_tag}"
              gan_tag="gan$([[ -n "$GAN_CLASSIFIER" ]] && echo 1 || echo 0)"

              # include K and N in the run tag
              tag="TT_${ds}_K${K}_N${NSTEP}_lr${lr}_DMD${dmdw}_SW${s_tag}_TW${t_tag}_clw${clw}_glw${glw}_${tt_tag}_${gan_tag}"
              echo "[$MODE] dataset=$ds K=$K N=$NSTEP lr=$lr DMD=$dmdw SW=$sw TW=$tw TT:$tt_tag $gan_tag tag=$tag"

              export EXTRA_TAG="_${tag}"
              submit_run "$tag"

            done
          done
        done
      done
    done
  done
done