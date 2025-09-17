#!/bin/bash

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# pick the sweep you want

GEN_CLS_LOSS_WEIGHT=(3e-3 15e-3 3e-3)
CLS_LOSS_WEIGHT=(1e-2 5e-2 5e-2)
GEN_LR=(5e-8 5e-7)

export WANDB_PROJECT="DMD_ABLATE_LR_CLSLOSS"

# paired sweep, local runs
for lr in "${GEN_LR[@]}"; do
  for i in "${!GEN_CLS_LOSS_WEIGHT[@]}"; do
    glw="${GEN_CLS_LOSS_WEIGHT[$i]}"
    clw="${CLS_LOSS_WEIGHT[$i]}"
    tag="lr${lr}_clw${clw}_glw${glw}"

    echo "[LOCAL] lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"
    
    sbatch \
      --job-name="dmd2_babies_${tag}" \
      --output="${LOGDIR}/dmd2_babies_${tag}-%j.out" \
      --error="${LOGDIR}/dmd2_babies_${tag}-%j.err" \
      --export=ALL,GEN_LR="$lr",GEN_CLS_LOSS_WEIGHT="$glw",CLS_LOSS_WEIGHT="$clw",GRAD_ACCUM_STEPS=2,BATCH_SIZE=2,NUM_DENOISING_STEP=4,EXTRA_TAG="_${tag}" \
      "$CHILD"
  done
done

# GEN_CLS_LOSS_WEIGHT=(3e-3 15e-3)
# CLS_LOSS_WEIGHT=(1e-3 5e-3)
# GEN_LR=(5e-8)
# 
# export WANDB_PROJECT="DMD_ABLATE_LR_CLSLOSS"
# 
# # paired sweep, local runs
# for lr in "${GEN_LR[@]}"; do
#   for i in "${!GEN_CLS_LOSS_WEIGHT[@]}"; do
#     glw="${GEN_CLS_LOSS_WEIGHT[$i]}"
#     clw="${CLS_LOSS_WEIGHT[$i]}"
#     tag="lr${lr}_clw${clw}_glw${glw}"
# 
#     echo "[LOCAL] lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"
# 
#     GEN_LR="$lr" \
#     GEN_CLS_LOSS_WEIGHT="$glw" \
#     CLS_LOSS_WEIGHT="$clw" \
#     GEN_LR="${GEN_LR[0]}" \
#     GRAD_ACCUM_STEPS=2 \
#     BATCH_SIZE=2 \
#     NUM_DENOISING_STEP=4 \
#     CUDA_VISIBLE_DEVICES=0,1 \
#     TRAIN_GPUS=0,1 \
#     TEST_GPUS=1 \
#     NPROC_PER_NODE=2 \
#     NNODES=1 \
#     EXTRA_TAG="_${tag}" \
#     bash "$CHILD"
# 
#     exit 0
#   done
# done

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

