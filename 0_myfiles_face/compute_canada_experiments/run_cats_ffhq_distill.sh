#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"


GEN_CLS_LOSS_WEIGHT=(15e-3)
CLS_LOSS_WEIGHT=(5e-2)
GEN_LR=(5e-8)

export WANDB_PROJECT="CAT_FFHQ_distilled"

# paired sweep, local runs\
for lr in "${GEN_LR[@]}"; do
  for i in "${!GEN_CLS_LOSS_WEIGHT[@]}"; do
    glw="${GEN_CLS_LOSS_WEIGHT[$i]}"
    clw="${CLS_LOSS_WEIGHT[$i]}"
    tag="lr${lr}_clw${clw}_glw${glw}"

    echo "[LOCAL] lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"

    export GEN_LR="2e-6"
    export DMD_LOSS_WEIGHT="1"
    export CHECKPOINT_PATH="0_myfiles_face/checkpoint_path/FFHQ_distilled_weights/checkpoint_model_050000"
    export DATASET_NAME="cat"
    export GEN_LR="$lr" 
    export GEN_CLS_LOSS_WEIGHT="$glw" 
    export CLS_LOSS_WEIGHT="$clw" 
    export GEN_LR="${GEN_LR[0]}" 
    export GRAD_ACCUM_STEPS=4
    export BATCH_SIZE=1
    export NUM_DENOISING_STEP=3 
    export CUDA_VISIBLE_DEVICES=0,1 
    export TRAIN_GPUS=0,1
    export TEST_GPUS=0
    export NPROC_PER_NODE=2 
    export NNODES=1 
    export EXTRA_TAG="_${tag}" 
    bash "$CHILD"

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

