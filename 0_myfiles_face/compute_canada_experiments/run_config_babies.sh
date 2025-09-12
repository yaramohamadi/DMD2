#!/bin/bash
#SBATCH --job-name=dmd2_babies_bs3_1gpu
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --mail-user=yara.mohammadi-bahram.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=0_myfiles_face/slurm/%x-%j.out
#SBATCH --error=0_myfiles_face/slurm/%x-%j.err


export SERVER="${SERVER:-"cc"}"

if [[ "$SERVER" != "local" && "$SERVER" != "cc" ]]; then
  echo "Usage: $0 {local|cc}"
  exit 1
fi
# ---------- LOCAL (conda) ----------
if [[ "$SERVER" == "cc" ]]; then

  ENV_NAME="dmd2"
  PY_VER="3.10.13"
  module load StdEnv/2023 python/$PY_VER
  module load rust/1.85.0
  module load gcc opencv/4.9.0
  module load arrow/15.0.1

  VENV_DIR="${PROJECT:-$HOME}/dmd2_env"
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"

  # REPO_ROOT="/home/ymbahram/projects/def-hadi87/ymbahram/DMD2/DMD2/"
  # cd "$REPO_ROOT"
  # export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
fi

if [[ "$SERVER" == "local" ]]; then
  conda init bash
  conda activate dmd2
fi


# -----------------------
# Fixed configs
# -----------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 0,1,2,3
export TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}" # 
export TEST_GPUS="${TEST_GPUS:-7}" #3
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" #4
export NNODES="${NNODES:-1}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

export PROJECT_PATH="0_myfiles_face"
export DATASET_NAME="${DATASET_NAME:-"babies"}"
export CHECKPOINT_INIT="$PROJECT_PATH/checkpoints/ffhq.pt"
export REAL_IMAGE_PATH="$PROJECT_PATH/datasets/targets/10_${DATASET_NAME}_lmdb"

export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie"
export WANDB_PROJECT="${WANDB_PROJECT:-"DMD_unconditional_${DATASET_NAME}_dmd_weight_ablation"}"
export WANDB_API_KEY=37efdaf78afc776eece6c9207e21caaff0ede2c3

export TRAIN_ITERS=100000
export SEED=10
export RESOLUTION=256

# For label handling ------------------------------------------------------
# ---- intent switches (set these per run) ----
export EXP_TRAIN_MODE="${EXP_TRAIN_MODE:-V0}"   # one of: V0, V1, V2
export K="${K:-10}"                              # number of real pseudo-classes
export LABEL_DROPOUT_P="${LABEL_DROPOUT_P:-0.30}"

# V1: Unconditional training and unconditional sampling 
# V2: Conditional training and conditional sampling
# V3: Conditional training with null and conditional sampling (But only sampling from classes and not null)
# Extra option:? sample from null during sampling

# ---- derive training flags ----
case "$EXP_TRAIN_MODE" in
  V0)  LABEL_DIM=0;  HAS_NULL_ENABLED=0; LABEL_DROPOUT_P=0.0 ;;
  V1)  LABEL_DIM=$K; HAS_NULL_ENABLED=0; LABEL_DROPOUT_P=0.0 ;;             # IMPORTANT: keep 0.0 here
  V2)  LABEL_DIM=$((K+1)); HAS_NULL_ENABLED=1 ;;                             # dropout P used as given
  *) echo "Unknown EXP_TRAIN_MODE=$EXP_TRAIN_MODE"; exit 1 ;;
esac
export LABEL_DIM LABEL_DROPOUT_P

# pass --has_null only when you truly reserved a NULL index
if [[ "${HAS_NULL_ENABLED}" == "1" ]]; then
  export HAS_NULL="--has_null"
else
  export HAS_NULL=""
fi
# ------------------------------------------------------------------------

export DENOISING_SIGMA_END=0.5

export DFAKE_GEN_UPDATE_RATIO=5
export CLS_LOSS_WEIGHT=5e-2
export GEN_CLS_LOSS_WEIGHT="${GEN_CLS_LOSS_WEIGHT:-15e-3}"
export DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-0.1}"
export DIFFUSION_GAN_MAX_TIMESTEP=1000

export LOG_ITERS=100
export WANDB_ITERS=100
export MAX_CHECKPOINT=100

export FID_NPZ_ROOT="$PROJECT_PATH/datasets/fid_npz"
export FEWSHOT_DATASET="$PROJECT_PATH/datasets/targets/10_${DATASET_NAME}/0"
export EVAL_BATCH_SIZE=8
export TOTAL_EVAL_SAMPLES=5000
export CONDITIONING_SIGMA=80.0
export LPIPS_CLUSTER_SIZE=100
export NO_LPIPS=""  # --no_lpips
export USE_BF16="--use_bf16" # --use_bf16

export GAN_HEAD_TYPE="global"
export GAN_HEAD_LAYERS="all"
export GAN_ADV_LOSS="bce"
export GAN_MULTIHEAD="--gan_multihead"

export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export DEN_FLAG="--denoising"
export BEST_FLAG="" # --eval_best_once
export NUM_DENOISING_STEP="${NUM_DENOISING_STEP:-2}"

# -----------------------
# Sweep ranges
# -----------------------
export GEN_LRS=(5e-8)
export BATCH_SIZES=(1)
export DENOISE_STEPS=("$NUM_DENOISING_STEP")

# -----------------------
# Sweep loop
# -----------------------
for lr in "${GEN_LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for dn in "${DENOISE_STEPS[@]}"; do
      export EXPERIMENT_NAME="$BF16_{DATASET_NAME}_LABELDIM${LABEL_DIM}_LABELDROPOUTP${LABEL_DROPOUT_P}_lr${lr}_bs${bs}_dn${dn}_drop${LABEL_DROPOUT_P}_DMD${DMD_LOSS_WEIGHT}_GClsw${GEN_CLS_LOSS_WEIGHT}${EXTRA_TAG}"
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
      CATEGORY=$DATASET_NAME \
      FEWSHOT_DATASET=$FEWSHOT_DATASET \
      EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
      TOTAL_EVAL_SAMPLES=$TOTAL_EVAL_SAMPLES \
      CONDITIONING_SIGMA=$CONDITIONING_SIGMA \
      LPIPS_CLUSTER_SIZE=$LPIPS_CLUSTER_SIZE \
      NO_LPIPS=$NO_LPIPS \
      LABEL_DROPOUT_P=$LABEL_DROPOUT_P \
      HAS_NULL="$HAS_NULL" \
      bash $PROJECT_PATH/compute_canada_experiments/run_both.sh

    done
  done
done
