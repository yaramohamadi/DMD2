#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"


GEN_CLS_LOSS_WEIGHTS=(15e-3 5e-2 7.5e-2) # previously this was 15e-3 #  15e-2
CLS_LOSS_WEIGHTS=(5e-3 1e-2 5e-2)
GEN_LRS=(2e-6)
 
export WANDB_PROJECT="CAT_from_scratch"
#     export CHECKPOINT_PATH="0_myfiles_face/checkpoint_path/cat_lr5e-8_bs1_dn3_DMD1_GClsw15e-3__lr5e-8_clw5e-2_glw15e-3/checkpoint_model_131600"
# paired sweep, local runs\
for lr in "${GEN_LRS[@]}"; do
  for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
    glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
    clw="${CLS_LOSS_WEIGHTS[$i]}"
    tag="lr${lr}_clw${clw}_glw${glw}"

    echo "[LOCAL] lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"
    export DATASET_NAME="cat"
    export GEN_LR="$lr" 
    export GEN_CLS_LOSS_WEIGHT=$glw
    export CLS_LOSS_WEIGHT=$clw
    export GRAD_ACCUM_STEPS=4
    export BATCH_SIZE=1
    export NUM_DENOISING_STEP=3 
    export CUDA_VISIBLE_DEVICES=0,1
    export TRAIN_GPUS=1
    export TEST_GPUS=0
    export NPROC_PER_NODE=1 
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

