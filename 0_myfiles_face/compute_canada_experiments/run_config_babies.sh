#!/bin/bash
#SBATCH --job-name=dmd2_babies_bs3_1gpu
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --mail-user=yara.mohammadi-bahram.1@ens.etsmtl.ca   # <-- put your email here
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=0_myfiles_face/slurm/%x-%j.out
#SBATCH --error=0_myfiles_face/slurm/%x-%j.err

set -e

# Compute canada mode:
ENV_NAME="dmd2"
PY_VER="3.10.13"
module load StdEnv/2023 python/$PY_VER
module load rust/1.85.0
module load gcc opencv/4.9.0
module load arrow/15.0.1
# --- versions/paths you can tweak ---
VENV_DIR="${PROJECT:-$HOME}/dmd2_env"  # used only for cc mode
# 1) venv
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

REPO_ROOT="/home/ymbahram/projects/def-hadi87/ymbahram/DMD2/DMD2/"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
# python -m pip check

# -----------------------
# Fixed configs
# -----------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export TRAIN_GPUS=0,1,2,3
export TEST_GPUS=3
export NPROC_PER_NODE=4
export NNODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)   # pick a random free port

export PROJECT_PATH="0_myfiles_face"
export CHECKPOINT_INIT="$PROJECT_PATH/checkpoints/ffhq.pt"
export REAL_IMAGE_PATH="$PROJECT_PATH/datasets/targets/10_babies_lmdb"

export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie"
export WANDB_PROJECT="DMD_unconditional_babies_dmd_weight_ablation"

export TRAIN_ITERS=100000
export SEED=10
export RESOLUTION=256
export LABEL_DIM=0
export DATASET_NAME="babies"
export DENOISING_SIGMA_END=0.5

export DFAKE_GEN_UPDATE_RATIO=5
export CLS_LOSS_WEIGHT=5e-2 # 1e-2
export GEN_CLS_LOSS_WEIGHT=15e-3 # 3e-3
export DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-1}"
export DIFFUSION_GAN_MAX_TIMESTEP=1000

export LOG_ITERS=2500
export WANDB_ITERS=100
export MAX_CHECKPOINT=100

export FID_NPZ_ROOT="$PROJECT_PATH/datasets/fid_npz"
export CATEGORY="babies"
export FEWSHOT_DATASET="$PROJECT_PATH/datasets/targets/10_babies/0"
export EVAL_BATCH_SIZE=4
export TOTAL_EVAL_SAMPLES=5000
export CONDITIONING_SIGMA=80.0
export LPIPS_CLUSTER_SIZE=100
export NO_LPIPS="" # --no_lpips

export GAN_HEAD_TYPE="global"
export GAN_HEAD_LAYERS="all"
export GAN_ADV_LOSS="bce"
export GAN_MULTIHEAD="--gan_multihead"

export ACCELERATE_LOG_LEVEL=error   # or error
export TRANSFORMERS_VERBOSITY=error   # optional: quiet transformers too
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export DEN_FLAG="--denoising"
export BEST_FLAG="" # --eval_best_once

# -----------------------
# Sweep ranges
# -----------------------

# LR 2e-6 -> Batch-size 40 -> Iterations 400k
# Using linear scaling: 
# LR 2e-7 -> Batch-size 4 -> Iterations 4M
# LR 1.5e-7 -> Batch-size 3 -> Iterations 6M
# LR 1e-7 -> Batch-size 2 -> Iterations 8M
# LR 5e-8 -> Batch-size 1 -> Iterations 16M

export GEN_LRS=(5e-8)
export BATCH_SIZES=(2)
export DENOISE_STEPS=(3)

# -----------------------
# Sweep loop
# -----------------------
for lr in "${GEN_LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for dn in "${DENOISE_STEPS[@]}"; do
      
      export EXPERIMENT_NAME="babies_lr${lr}_bs${bs}_dn${dn}_DMD_LOSS_WEIGHT${DMD_LOSS_WEIGHT}"
      export OUTPUT_PATH="$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME"
      export WANDB_NAME="$EXPERIMENT_NAME"

      echo "[RUN] $EXPERIMENT_NAME"

      GENERATOR_LR=$lr \
      BATCH_SIZE=$bs \
      NUM_DENOISING_STEP=$dn \
      EXPERIMENT_NAME=$EXPERIMENT_NAME \
      OUTPUT_PATH=$OUTPUT_PATH \
      WANDB_NAME=$WANDB_NAME \
      CHECKPOINT_INIT=$CHECKPOINT_INIT \
      REAL_IMAGE_PATH=$REAL_IMAGE_PATH \
      WANDB_ENTITY=$WANDB_ENTITY \
      WANDB_PROJECT=$WANDB_PROJECT \
      TRAIN_ITERS=$TRAIN_ITERS \
      SEED=$SEED \
      RESOLUTION=$RESOLUTION \
      LABEL_DIM=$LABEL_DIM \
      DATASET_NAME=$DATASET_NAME \
      DFAKE_GEN_UPDATE_RATIO=$DFAKE_GEN_UPDATE_RATIO \
      CLS_LOSS_WEIGHT=$CLS_LOSS_WEIGHT \
      GEN_CLS_LOSS_WEIGHT=$GEN_CLS_LOSS_WEIGHT \
      DMD_LOSS_WEIGHT=$DMD_LOSS_WEIGHT \
      DIFFUSION_GAN_MAX_TIMESTEP=$DIFFUSION_GAN_MAX_TIMESTEP \
      LOG_ITERS=$LOG_ITERS \
      WANDB_ITERS=$WANDB_ITERS \
      MAX_CHECKPOINT=$MAX_CHECKPOINT \
      FID_NPZ_ROOT=$FID_NPZ_ROOT \
      CATEGORY=$CATEGORY \
      FEWSHOT_DATASET=$FEWSHOT_DATASET \
      EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
      TOTAL_EVAL_SAMPLES=$TOTAL_EVAL_SAMPLES \
      CONDITIONING_SIGMA=$CONDITIONING_SIGMA \
      LPIPS_CLUSTER_SIZE=$LPIPS_CLUSTER_SIZE \
      NO_LPIPS=$NO_LPIPS \
      bash $PROJECT_PATH/experiments/run_both.sh

    done
  done
done
