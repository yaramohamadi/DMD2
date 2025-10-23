#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"  # <-- sbatch/runner
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

GEN_CLS_LOSS_WEIGHTS=(15e-3)
CLS_LOSS_WEIGHTS=(5e-2)
GEN_LRS=(2e-6)

# sweep values
DMD_LOSS_WEIGHTS=(0 0.03 0.06 0.10 0.20 1)

# fixed flags you always want
export TRAIN_FAKE_ON_REAL="--train_fake_on_real"
export WANDB_PROJECT="REAL_ONLINE_TEACHER_SWEEP"

DATASETS=("metfaces")

# paired sweep, local runs
for ds in "${DATASETS[@]}"; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHTS[$i]}"

      for dmdw in "${DMD_LOSS_WEIGHTS[@]}"; do

        # choose reverse options: only allow reverse when dmdw > 0
        if [[ "$dmdw" == "0" ]]; then
          REVERSE_DMD_OPTS=("")                # no reverse at weight 0
        else
          REVERSE_DMD_OPTS=("" "--reverse_dmd")
        fi

        for rev in "${REVERSE_DMD_OPTS[@]}"; do
          # Export per-run env vars
          export DATASET_NAME="$ds"
          export GEN_LR="$lr"
          export GEN_CLS_LOSS_WEIGHT="$glw"
          export CLS_LOSS_WEIGHT="$clw"
          export DMD_LOSS_WEIGHT="$dmdw"     # consumed by CHILD
          export REVERSE_DMD="$rev"          # "" or "--reverse_dmd"

          export GRAD_ACCUM_STEPS=1
          export BATCH_SIZE=1
          export NUM_DENOISING_STEP=3

          export CUDA_VISIBLE_DEVICES=0
          export TRAIN_GPUS=0
          export TEST_GPUS=0
          export NPROC_PER_NODE=1
          export NNODES=1

          # Build a readable tag
          rev_tag="rev0"
          if [[ -n "$rev" ]]; then rev_tag="rev1"; fi
          tag="GAN_MULTIHEAD_${ds}_DMDW${dmdw}_${rev_tag}_lr${lr}_clw${clw}_glw${glw}"

          echo "[LOCAL] dataset=$ds  lr=$lr  DMD_LOSS_WEIGHT=$dmdw  reverse=${rev_tag}  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"

          export EXTRA_TAG="_${tag}"

          bash "$CHILD"
        done
      done
    done
  done
done
