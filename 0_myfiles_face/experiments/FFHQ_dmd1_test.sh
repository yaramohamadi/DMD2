export PROJECT_PATH="0_myfiles_face" # change this to your own checkpoint folder 
export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie" # change this to your own wandb entity
export WANDB_PROJECT="DMD_face" # change this to your own wandb project
export FID_NPZ_ROOT="0_myfiles_face/datasets/fid_npz"   # must contain ${CATEGORY}.npz
export CATEGORY="FFHQ"                                     # -> ${FID_NPZ_ROOT}/ffhq.npz

# SRC="/export/datasets/public/diffusion_datasets/adaptation/datasets/fid_npz/FFHQ.npz"
# DST="$PROJECT_PATH/datasets/fid_npz/"
# mkdir -p "$DST"
# rsync -avh --info=progress2 "$SRC" "$DST"

# pick a single GPU (recommended: single process to match baseline sampling)
export CUDA_VISIBLE_DEVICES=3

python -u main/dhariwal/test_folder_dhariwal.py \
  --folder /export/livia/home/vision/Ymohammadi/DMD2_checkpoints/FFHQ256_dmd1/time_1756780684_seed10/ \
  --wandb_name FFHQ_dmd1_test \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --fid_npz_root $FID_NPZ_ROOT \
  --category $CATEGORY \
  --resolution 256 \
  --label_dim 0 \
  --eval_batch_size 32 \
  --total_eval_samples 5000 \
  --conditioning_sigma 80.0 \
  --lpips_cluster_size 100 \
  --fewshotdataset $PROJECT_PATH/datasets/10-shot/babies/0 \
  --no_lpips
