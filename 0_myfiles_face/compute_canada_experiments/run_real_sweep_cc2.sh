#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"  # sbatch-ready (has #SBATCH lines)
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# --------- mode switch: local vs Compute Canada (sbatch) ----------
MODE="${MODE:-"cc"}"   # set MODE=cc to use sbatch
submit_run () {
  local tag="$1"

  if [[ "$MODE" == "cc" ]]; then
    # Submit as a Slurm job with per-run exports
    sbatch \
      --job-name="$tag" \
      --output="$LOGDIR/%x-%j.out" \
      --export=ALL,DATASET_NAME="$DATASET_NAME",GEN_LR="$GEN_LR",GEN_CLS_LOSS_WEIGHT="$GEN_CLS_LOSS_WEIGHT",CLS_LOSS_WEIGHT="$CLS_LOSS_WEIGHT",DMD_LOSS_WEIGHT="$DMD_LOSS_WEIGHT",REVERSE_DMD="$REVERSE_DMD",GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS",BATCH_SIZE="$BATCH_SIZE",NUM_DENOISING_STEP="$NUM_DENOISING_STEP",TRAIN_GPUS="$TRAIN_GPUS",TEST_GPUS="$TEST_GPUS",NPROC_PER_NODE="$NPROC_PER_NODE",NNODES="$NNODES",TRAIN_FAKE_ON_REAL="$TRAIN_FAKE_ON_REAL",WANDB_PROJECT="$WANDB_PROJECT",EXTRA_TAG="$EXTRA_TAG" \
      "$CHILD"
  else
    # Local run
    bash "$CHILD"
  fi
}
# ------------------------------------------------------------------

GEN_CLS_LOSS_WEIGHTS=(0)
CLS_LOSS_WEIGHTS=(0)
GEN_LRS=(2e-6)

# sweep values
DMD_LOSS_WEIGHTS=(1)

# fixed flags you always want
export TRAIN_FAKE_ON_REAL="--train_fake_on_real"
export WANDB_PROJECT="FAKE_ONLINE_TEACHER_SWEEP"

DATASETS=("metfaces")

for ds in "${DATASETS[@]}"; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHTS[$i]}"

      for dmdw in "${DMD_LOSS_WEIGHTS[@]}"; do

        # only allow reverse when dmdw > 0
        if [[ "$dmdw" == "0" ]]; then
          REVERSE_DMD_OPTS=("")
        else
          REVERSE_DMD_OPTS=("--reverse_dmd" "" )
        fi

        for rev in "${REVERSE_DMD_OPTS[@]}"; do
          # per-run vars (exported for CHILD)
          export DATASET_NAME="$ds"
          export GEN_LR="$lr"
          export GEN_CLS_LOSS_WEIGHT="$glw"
          export CLS_LOSS_WEIGHT="$clw"
          export DMD_LOSS_WEIGHT="$dmdw"
          export REVERSE_DMD="$rev"

          export GRAD_ACCUM_STEPS=1
          export BATCH_SIZE=1
          export NUM_DENOISING_STEP=3

          # Let Slurm handle GPU binding; for local you can still set CUDA_VISIBLE_DEVICES inside CHILD if needed
          export TRAIN_GPUS=0
          export TEST_GPUS=0
          export NPROC_PER_NODE=1
          export NNODES=1

          rev_tag="rev0"; [[ -n "$rev" ]] && rev_tag="rev1"
          tag="GAN_MULTIHEAD_${ds}_DMDW${dmdw}_${rev_tag}_lr${lr}_clw${clw}_glw${glw}_trainfakeonreal${TRAIN_FAKE_ON_REAL}"

          echo "[$MODE] dataset=$ds  lr=$lr  DMD_LOSS_WEIGHT=$dmdw  reverse=${rev_tag}  tag=$tag"

          export EXTRA_TAG="_${tag}"

          submit_run "$tag"
        done
      done
    done
  done
done
