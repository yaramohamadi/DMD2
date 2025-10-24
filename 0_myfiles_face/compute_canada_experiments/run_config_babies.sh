#!/bin/bash
#SBATCH --job-name=dmd2_babies_bs3_1gpu
#SBATCH --account=def-hadi87
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=07:00:00
#SBATCH --mail-user=yara.mohammadi-bahram.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=0_myfiles_face/slurm/%x-%j.out
#SBATCH --error=0_myfiles_face/slurm/%x-%j.err

export SERVER="${SERVER:-"local"}"

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

  # Only for FIR not for NIBI
  REPO_ROOT="/home/ymbahram/projects/def-hadi87/ymbahram/DMD2/DMD2/"
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

  echo "Compute canada activated"
fi

if [[ "$SERVER" == "local" ]]; then
  conda init bash
  conda activate dmd2

  export PYTHONPATH="$PWD/third_party/dhariwal:$PYTHONPATH" 

fi

# -----------------------
# Fixed configs
# -----------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" # 0,1,2,3
export TRAIN_GPUS="${TRAIN_GPUS:-0,1}" # 
export TEST_GPUS="${TEST_GPUS:-1}" #3
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}" #4
export NNODES="${NNODES:-1}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE=24
export NUM_DENOISING_STEP="${NUM_DENOISING_STEP:-3}"
export TRAIN_ITERS=20000
export DATASET_SIZE="${DATASET_SIZE:-10}"  # 10 5 1

export PROJECT_PATH="0_myfiles_face"
export DATASET_NAME="${DATASET_NAME:-"babies"}"
export CHECKPOINT_INIT="$PROJECT_PATH/checkpoints/ffhq.pt"
export REAL_IMAGE_PATH="$PROJECT_PATH/datasets/targets/${DATASET_SIZE}_${DATASET_NAME}_lmdb"

export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie"
export WANDB_PROJECT="${WANDB_PROJECT:-"DMD_unconditional_${DATASET_NAME}_dmd_weight_ablation"}"
export WANDB_API_KEY=37efdaf78afc776eece6c9207e21caaff0ede2c3

export SEED=10
export RESOLUTION=256

export TT_MATCH_GUIDANCE="${TT_MATCH_GUIDANCE-}" # --tt_match_guidance

export DMD_SOURCE_WEIGHT="${DMD_SOURCE_WEIGHT:-1.0}"
export DMD_TARGET_WEIGHT="${DMD_TARGET_WEIGHT:-1.0}"

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
export CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-5e-2}" # 1e-2
export GEN_CLS_LOSS_WEIGHT="${GEN_CLS_LOSS_WEIGHT:-15e-3}" #-3e-3
export DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-1}"
export DIFFUSION_GAN_MAX_TIMESTEP=1000

export LOG_ITERS=50
export WANDB_ITERS=50
export MAX_CHECKPOINT=100

export FID_NPZ_ROOT="$PROJECT_PATH/datasets/fid_npz"
export FEWSHOT_DATASET="$PROJECT_PATH/datasets/targets/${DATASET_SIZE}_${DATASET_NAME}/0"
export TOTAL_EVAL_SAMPLES=5000
export CONDITIONING_SIGMA=80.0
export LPIPS_CLUSTER_SIZE=100
export NO_LPIPS=${NO_LPIPS-}  # --no_lpips
export USE_BF16="--use_bf16" # --use_bf16

export GAN_HEAD_TYPE="global"
export GAN_HEAD_LAYERS="${GAN_HEAD_LAYERS:-"all"}"
export GAN_ADV_LOSS="${GAN_ADV_LOSS:-bce}"
export GAN_MULTIHEAD="${GAN_MULTIHEAD-"--gan_multihead"}" # "--gan_multihead"

export REVERSE_DMD="${REVERSE_DMD-}"

export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GEN_LR="${GEN_LR:-2e-6}"  # 2e-6 5e-8

export DEN_FLAG="--denoising"
export BEST_FLAG=""
export CHECKPOINT_PATH="${CHECKPOINT_PATH-}"

export TRAIN_FAKE_ON_REAL="${TRAIN_FAKE_ON_REAL-}" #  --train_fake_on_real

export USE_SOURCE_TEACHER="${USE_SOURCE_TEACHER:-1.0}"
export USE_TARGET_TEACHER="${USE_TARGET_TEACHER:-0.0}"
export TRAIN_TARGET_TEACHER="${TRAIN_TARGET_TEACHER:-1.0}"

export GAN_CLASSIFIER="${GAN_CLASSIFIER-"--gan_classifier"}" # --gan_classifier

export EXPERIMENT_NAME="${DATASET_NAME}_lr${GEN_LR}_bs${BATCH_SIZE}_dn${NUM_DENOISING_STEP}_${TRAIN_FAKE_ON_REAL}_DMD${DMD_LOSS_WEIGHT}_GClsw${GEN_CLS_LOSS_WEIGHT}_${EXTRA_TAG}"
export OUTPUT_PATH="0_myfiles_face/checkpoint_path/$EXPERIMENT_NAME"
# "$PROJECT_PATH/checkpoint_path/$EXPERIMENT_NAME"
export WANDB_NAME="$EXPERIMENT_NAME"


echo "DATASET_NAME ${DATASET_NAME}"
echo "TRAIN_ITERS ${TRAIN_ITERS}"
echo "GEN_LR ${GEN_LR}"
echo "GEN_CLS_LOSS_WEIGHT ${GEN_CLS_LOSS_WEIGHT}"
echo "CLS_LOSS_WEIGHT  ${CLS_LOSS_WEIGHT}"
echo "DATASET_SIZE ${DATASET_SIZE}"
echo "NUM_DENOISING_STEP ${NUM_DENOISING_STEP}"
echo "GRAD_ACCUM_STEPS ${GRAD_ACCUM_STEPS}"
echo "BATCH_SIZE ${BATCH_SIZE}"
echo "NUM_DENOISING_STEP ${NUM_DENOISING_STEP}"
echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "TRAIN_GPUS ${TRAIN_GPUS}"
echo "TEST_GPUS ${TEST_GPUS}"
echo "NPROC_PER_NODE ${NPROC_PER_NODE}"
echo "NNODES ${NNODES}"
echo "TRAIN_FAKE_ON_REAL ${TRAIN_FAKE_ON_REAL}"
echo "USE_SOURCE_TEACHER ${USE_SOURCE_TEACHER}"
echo "USE_TARGET_TEACHER ${USE_TARGET_TEACHER}"
echo "TRAIN_TARGET_TEACHER ${TRAIN_TARGET_TEACHER}"
echo "GAN_CLASSIFIER ${GAN_CLASSIFIER}"

echo "_____________________________"

echo "[RUN] $EXPERIMENT_NAME"

GEN_LR="$GEN_LR" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_DENOISING_STEP="$NUM_DENOISING_STEP" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
OUTPUT_PATH="$OUTPUT_PATH" \
WANDB_NAME="$WANDB_NAME" \
CHECKPOINT_INIT="$CHECKPOINT_INIT" \
REAL_IMAGE_PATH="$REAL_IMAGE_PATH" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_PROJECT="$WANDB_PROJECT" \
TRAIN_ITERS="$TRAIN_ITERS" \
SEED="$SEED" \
RESOLUTION="$RESOLUTION" \
LABEL_DIM="$LABEL_DIM" \
DATASET_NAME="$DATASET_NAME" \
DFAKE_GEN_UPDATE_RATIO="$DFAKE_GEN_UPDATE_RATIO" \
CLS_LOSS_WEIGHT="$CLS_LOSS_WEIGHT" \
GEN_CLS_LOSS_WEIGHT="$GEN_CLS_LOSS_WEIGHT" \
DMD_LOSS_WEIGHT="$DMD_LOSS_WEIGHT" \
DIFFUSION_GAN_MAX_TIMESTEP="$DIFFUSION_GAN_MAX_TIMESTEP" \
LOG_ITERS="$LOG_ITERS" \
WANDB_ITERS="$WANDB_ITERS" \
MAX_CHECKPOINT="$MAX_CHECKPOINT" \
FID_NPZ_ROOT="$FID_NPZ_ROOT" \
CATEGORY="$DATASET_NAME" \
FEWSHOT_DATASET="$FEWSHOT_DATASET" \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
TOTAL_EVAL_SAMPLES="$TOTAL_EVAL_SAMPLES" \
CONDITIONING_SIGMA="$CONDITIONING_SIGMA" \
LPIPS_CLUSTER_SIZE="$LPIPS_CLUSTER_SIZE" \
NO_LPIPS="$NO_LPIPS" \
LABEL_DROPOUT_P="$LABEL_DROPOUT_P" \
HAS_NULL="$HAS_NULL" \
bash "$PROJECT_PATH/compute_canada_experiments/run_both.sh"
