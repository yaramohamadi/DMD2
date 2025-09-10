#!/bin/bash
# 
CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"       # <-- point to the sbatch file above
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# CUDA_VISIBLE_DEVICES=0,TRAIN_GPUS=0,TEST_GPUS=0,NPROC_PER_NODE=1,NNODES=1

# pick the sweep you want
STEPS=(4)

for dn in "${STEPS[@]}"; do
  tag="dn${dn}"
  sbatch \
    --job-name="dmd2_babies_${tag}" \
    --output="${LOGDIR}/dmd2_babies_${tag}-%j.out" \
    --error="${LOGDIR}/dmd2_babies_${tag}-%j.err" \
    --export=ALL,NUM_DENOISING_STEP="$dn",EXTRA_TAG="_${tag}" \
    "$CHILD"
done

#!/bin/bash

# CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
# 
# STEPS=(4)
# 
# export CUDA_VISIBLE_DEVICES=0 # ,1,2,3,4,5,6,7
# export TRAIN_GPUS=0 # ,1,2,3,4,5,6,7
# export TEST_GPUS=0 # 7
# export NPROC_PER_NODE=1 #8
# export NNODES=1
# 
# 
# for dn in "${STEPS[@]}"; do
# 
#   echo "[*] Launch ${tag} -> ${out}"
#   export NUM_DENOISING_STEP="$dn" 
#   bash "$CHILD"
# done