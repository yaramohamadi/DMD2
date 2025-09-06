# SRC="/export/datasets/public/diffusion_datasets/adaptation/datasets/fid_npz/FFHQ.npz"
# DST="$PROJECT_PATH/datasets/fid_npz/"
# mkdir -p "$DST"
# rsync -avh --info=progress2 "$SRC" "$DST"
# exit 

SRC="/export/livia/home/vision/Ymohammadi/DMD2/datasets/targets/FFHQ_lmdb/"
DST="$PROJECT_PATH/datasets/10-shot/FFHQ_lmdb/"
mkdir -p "$DST"
rsync -avh --info=progress2 "$SRC" "$DST"
SRC="/export/livia/home/vision/Ymohammadi/DMD2/adaptation/checkpoints/ffhq.pt"
DST="$PROJECT_PATH/checkpoint_path/"
mkdir -p "$DST"
rsync -avh --info=progress2 "$SRC" "$DST"
exit 0