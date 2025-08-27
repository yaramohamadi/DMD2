export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)   # pick a random free port


# Buil anchors for regression
python 0_mystuff/scripts_util/make_anchor_pairs.py \
  --real_image_path 0_mystuff/checkpoint_path/pokemon_lmdb \
  --out_path 0_mystuff/checkpoint_path/pokemon_lmdb/pokemon_10_anchors1.pt \
  --label_dim 1 --model_label_dim 1000 --resolution 64 --conditioning_sigma 80.0 \
  --anchors_per_image 1

# start a training with regression on 10 anchors (Without GAN loss)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --nnodes 1 \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" main/edm/train_edm.py \
  --generator_lr 2e-6 \
  --guidance_lr 2e-6 \
  --train_iters 10000000 \
  --output_path $CHECKPOINT_PATH/pokemon_dmd0.1_regression10_0.01lowprob_radius0_1anchor_noGAN \
  --batch_size 20 \
  --initialie_generator --log_iters 500 \
  --resolution 64 \
  --label_dim 1000 \
  --dataset_name "imagenet" \
  --seed 10 \
  --model_id $CHECKPOINT_PATH/edm-imagenet-64x64-cond-adm.pkl \
  --wandb_iters 100 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --use_fp16 \
  --wandb_name "pokemon_dmd0.1_regression10_0.01lowprob_radius0_1anchor_noGAN" \
  --real_image_path $CHECKPOINT_PATH/pokemon_lmdb \
  --dmd_loss_weight 0.1 \
  --delete_ckpts \
  --max_checkpoint 500 \
  --anchor_pairs_path $CHECKPOINT_PATH/pokemon_lmdb/pokemon_10_anchors1.pt \
  --anchor_prob 0.25 \
  --anchor_radius 0 \
  --lambda_regression 0.01
