python 0_mystuff/scripts_util/make_anchor_pairs.py \
  --real_image_path 0_mystuff/checkpoint_path/pokemon_lmdb \
  --out_path 0_mystuff/checkpoint_path/pokemon_lmdb/pokemon_1000_anchors.pt \
  --label_dim 1000 --resolution 64 --conditioning_sigma 80.0 \
  --anchors_per_image 1000
