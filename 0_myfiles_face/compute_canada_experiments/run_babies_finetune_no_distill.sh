#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# ---- Datasets to sweep (edit this list) ----
DATASETS=(babies sunglasses metfaces cat)   # <--- put all your dataset names here
# -------------------------------------------

GEN_CLS_LOSS_WEIGHTS=(15e-3)
CLS_LOSS_WEIGHTS=(5e-2)
GEN_LRS=(2e-6)

export DDPM_STEPS=all
export SAMPLER="ddim"
export WANDB_PROJECT="Babies_FINETUNE_no_distill"

# paired sweep, local runs
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
        export TOTAL_EVAL_SAMPLES=5000
        export LOG_ITERS=100
        export TRAIN_ITERS=50000
        # ----------------------------------

        export DATASET_NAME="$ds"
        export GEN_LR="$lr"
        export GEN_CLS_LOSS_WEIGHT="$glw"
        export CLS_LOSS_WEIGHT="$clw"
        export GRAD_ACCUM_STEPS=4
        export BATCH_SIZE=1

        # handle few-step special case cleanly
        if [[ "$dd" == "few" ]]; then
          export NUM_DENOISING_STEP=3
        else
          unset NUM_DENOISING_STEP
        fi

        # tag gets extended inside run_config_babies.sh
        export EXTRA_TAG="_${ds}_nodistill_naive_${dd}"

        export CUDA_VISIBLE_DEVICES=0,1
        export TRAIN_GPUS=2
        export TEST_GPUS=3
        export NPROC_PER_NODE=1
        export NNODES=1

        bash "$CHILD"
      done
    done
  done
done
