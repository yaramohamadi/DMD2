export PROJECT_PATH="0_mystuff_face" # change this to your own checkpoint folder 
export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie" # change this to your own wandb entity
export WANDB_PROJECT="DMD_face" # change this to your own wandb project
export FID_NPZ_ROOT="0_mystuff_face/datasets/fid_npz"   # must contain ${CATEGORY}.npz
export CATEGORY="babies"                                     # -> ${FID_NPZ_ROOT}/ffhq.npz

# pick a single GPU (recommended: single process to match baseline sampling)
export CUDA_VISIBLE_DEVICES=3

python main/dhariwal/test_folder_dhariwal.py \
  --folder $PROJECT_PATH/checkpoint_path/ffhq256_babies_dmd0.1_multiheadGAN/time_1756313057_seed10 \
  --wandb_name ffhq_babies_dmd0.1_multiheadGAN_stest \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --fid_npz_root $FID_NPZ_ROOT \
  --category $CATEGORY \
  --resolution 256 \
  --label_dim 0 \
  --eval_batch_size 8 \
  --total_eval_samples 5000 \
  --conditioning_sigma 50.0 \
  --lpips_cluster_size 100 \
  --fewshotdataset $PROJECT_PATH/datasets/$CATEGORY/0