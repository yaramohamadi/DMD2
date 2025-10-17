#!/bin/bash
set -euo pipefail

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

DATASETS=(metfaces cat)
GEN_CLS_LOSS_WEIGHTS=(15e-3)
CLS_LOSS_WEIGHTS=(5e-2)
GEN_LRS=(2e-6)

export DDPM_STEPS=all
export SAMPLER="ddim"
export WANDB_PROJECT="Babies_FINETUNE_no_distill"

for ds in "${DATASETS[@]}"; do
  for dd in all; do
    for lr in "${GEN_LRS[@]}"; do
      for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
        glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
        clw="${CLS_LOSS_WEIGHTS[$i]}"

        tag="ds${ds}_lr${lr}_clw${clw}_glw${glw}"
        echo "[LOCAL] dataset=$ds  lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"

        # Finetune baseline settings -------
        export FT_MODE="naive"
        export DDPM_STEPS="$dd"
        export TOTAL_EVAL_SAMPLES=1000
        export LOG_ITERS=100
        # ----------------------------------
        export DATASET_NAME="$ds"
        export GEN_LR="$lr"
        export GEN_CLS_LOSS_WEIGHT="$glw"
        export CLS_LOSS_WEIGHT="$clw"
        export GRAD_ACCUM_STEPS=4
        export BATCH_SIZE=1
        if [[ "$dd" == "few" ]]; then export NUM_DENOISING_STEP=3; fi
        export EXTRA_TAG="_naive_${dd}"
        export CUDA_VISIBLE_DEVICES=2
        export TRAIN_GPUS=2
        export TEST_GPUS=2
        export NPROC_PER_NODE=1
        export NNODES=1

        bash "$CHILD"
      done
    done
  done
done
