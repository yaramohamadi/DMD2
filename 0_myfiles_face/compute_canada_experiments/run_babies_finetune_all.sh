#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"


GEN_CLS_LOSS_WEIGHTS=(15e-3)
CLS_LOSS_WEIGHTS=(5e-2)
GEN_LRS=(2e-7)

export WANDB_PROJECT="Babies_FINETUNE"

# paired sweep, local runs
for dd in all; do
  for lr in "${GEN_LRS[@]}"; do
    for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
      glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
      clw="${CLS_LOSS_WEIGHT[$i]}"
      tag="lr${lr}_clw${clw}_glw${glw}"

      echo "[LOCAL] lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"

      # Finetune baseline settings -------
      export FT_MODE="naive"
      export DDPM_STEPS="$dd"
      export TOTAL_EVAL_SAMPLES=5000
      export LOG_ITERS=100
      # ----------------------------------
      export CHECKPOINT_PATH="0_myfiles_face/checkpoint_path/FFHQ_distilled_weights/checkpoint_model_037200/"
      export DATASET_NAME="babies"
      export GEN_LR="$lr" 
      export GEN_CLS_LOSS_WEIGHT="$glw" 
      export CLS_LOSS_WEIGHT="$clw" 
      export GRAD_ACCUM_STEPS=4
      export BATCH_SIZE=1
      if [[ "$dd" == "few" ]]; then export NUM_DENOISING_STEP=3; fi
      export EXTRA_TAG="_naive_${dd}"  # gets extended inside run_config_babies.sh
      export CUDA_VISIBLE_DEVICES=0,1
      export TRAIN_GPUS=0
      export TEST_GPUS=1
      export NPROC_PER_NODE=1
      export NNODES=1 
      bash "$CHILD"
    done
  done
done

# 
# CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"   # point to your script
# 
# STEPS=(4)
# 
# for dn in "${STEPS[@]}"; do
#   tag="dn${dn}"
#   echo "Running with NUM_DENOISING_STEP=$dn"
# 
#   # Run locally, export variables, redirect logs
#   NUM_DENOISING_STEP="$dn" EXTRA_TAG="$tag" bash "$CHILD"
# done

