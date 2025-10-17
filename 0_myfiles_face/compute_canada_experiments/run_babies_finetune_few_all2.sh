#!/bin/bash
set -Eeuo pipefail

CHILD="0_myfiles_face/compute_canada_experiments/run_config_babies.sh"
LOGDIR="0_myfiles_face/slurm"
mkdir -p "$LOGDIR"

# ---------------- configs to sweep ----------------
DATASETS=(sunglasses babies)   # <— your dataset sweep
GEN_CLS_LOSS_WEIGHTS=(15e-3)
CLS_LOSS_WEIGHTS=(5e-2)
GEN_LRS=(5e-7)

export WANDB_PROJECT="Babies_FINETUNE"
export SAMPLER="karras"

# Optional: per-dataset overrides (uncomment / edit if needed)
# declare -A RES_BY_DS=( [babies]=256 [cat]=256 [metfaces]=256 [sunglasses]=256 )
# declare -A LABELDIM_BY_DS=( [babies]=0 [cat]=0 [metfaces]=0 [sunglasses]=0 )
# declare -A CHECKPT_BY_DS=( 
#   [babies]="0_myfiles_face/checkpoint_path/babies_lr2e-7_bs1_dn3_DMD0_GClsw15e-3__naive_few_ftnaive_ddpmfew_dn3/checkpoint_model_047900"
#   [cat]="0_myfiles_face/checkpoint_path/cat_lr2e-7_bs1_dn3_DMD0_GClsw15e-3__dscat_ddfew_lr2e-7_clw5e-2_glw15e-3_ftnaive_ddpmfew_dn3/checkpoint_model_048000"  
#   [metfaces]="0_myfiles_face/checkpoint_path/metfaces_lr2e-7_bs1_dn3_DMD0_GClsw15e-3__dsmetfaces_ddfew_lr2e-7_clw5e-2_glw15e-3_ftnaive_ddpmfew_dn3/checkpoint_model_047900"
#   [sunglasses]="0_myfiles_face/checkpoint_path/sunglasses_lr2e-7_bs1_dn3_DMD0_GClsw15e-3__dssunglasses_ddfew_lr2e-7_clw5e-2_glw15e-3_ftnaive_ddpmfew_dn3/checkpoint_model_047900"
# )

# paired sweep, local runs
for ds in "${DATASETS[@]}"; do
  for dd in few; do           # or: for dd in all few; do
    for lr in "${GEN_LRS[@]}"; do
      for i in "${!GEN_CLS_LOSS_WEIGHTS[@]}"; do
        glw="${GEN_CLS_LOSS_WEIGHTS[$i]}"
        clw="${CLS_LOSS_WEIGHTS[$i]}"             # <-- fixed typo (was CLS_LOSS_WEIGHT)
        tag="ds${ds}_dd${dd}_lr${lr}_clw${clw}_glw${glw}"

        echo "[LOCAL] ds=$ds dd=$dd lr=$lr  GEN_CLS_LOSS_WEIGHT=$glw  CLS_LOSS_WEIGHT=$clw  tag=$tag"

        # ---------------- Finetune baseline settings ----------------
        export FT_MODE="naive"
        export DDPM_STEPS="$dd"                 # 'all' = 1000-step FT, 'few' = K-step FT
        export TOTAL_EVAL_SAMPLES=5000
        export LOG_ITERS=100
        # ------------------------------------------------------------

        # Per-dataset plumbing
        export DATASET_NAME="$ds"

        # If you keep a single checkpoint path for all, fine; else use CHECKPT_BY_DS["$ds"]
        # export CHECKPOINT_PATH="0_myfiles_face/checkpoint_path/FFHQ_distilled_weights/checkpoint_model_037200/"
        # export CHECKPOINT_PATH="${CHECKPT_BY_DS[$ds]}"

        # Optim/compute knobs
        export GEN_LR="$lr"
        export GEN_CLS_LOSS_WEIGHT="$glw"
        export CLS_LOSS_WEIGHT="$clw"
        export GRAD_ACCUM_STEPS=4
        export BATCH_SIZE=1
        if [[ "$dd" == "few" ]]; then
          export NUM_DENOISING_STEP=3
        else
          unset NUM_DENOISING_STEP || true
        fi

        # Tagging & GPUs
        export EXTRA_TAG="_${tag}"
        export CUDA_VISIBLE_DEVICES=1
        export TRAIN_GPUS=1
        export TEST_GPUS=1
        export NPROC_PER_NODE=1
        export NNODES=1

        # Kick off
        bash "$CHILD"
      done
    done
  done
done
