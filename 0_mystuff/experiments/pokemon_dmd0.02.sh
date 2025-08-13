export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)   # pick a random free port

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --nnodes 1 --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" main/edm/train_edm.py \
    --generator_lr 2e-6  \
    --guidance_lr 2e-6  \
    --train_iters 10000000 \
    --output_path $CHECKPOINT_PATH/pokemon_dmd0.02 \
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
    --wandb_name "pokemon_dmd0.02"   \
    --real_image_path $CHECKPOINT_PATH/pokemon_lmdb \
    --dfake_gen_update_ratio 5 \
    --cls_loss_weight 3e-2 \
    --gan_classifier \
    --gen_cls_loss_weight 2e-2 \
    --dmd_loss_weight 0.1 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --delete_ckpts \
    --max_checkpoint 500 

# dmd loss weight is added by me    