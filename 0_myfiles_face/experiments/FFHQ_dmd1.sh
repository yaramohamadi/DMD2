export PROJECT_PATH="0_myfiles_face" # change this to your own checkpoint folder 
export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie" # change this to your own wandb entity
export WANDB_PROJECT="DMD_face" # change this to your own wandb project
export CUDA_VISIBLE_DEVICES=3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)   # pick a random free port

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --nnodes 1 --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" main/dhariwal/train_dhariwal.py \
    --generator_lr 2e-6  \
    --guidance_lr 2e-6  \
    --train_iters 50000 \
    --output_path /export/livia/home/vision/Ymohammadi/DMD2_checkpoints/FFHQ256_dmd1_bf16_bugfixes \
    --batch_size 1 \
    --initialie_generator --log_iters 500 \
    --resolution 256 \
    --label_dim 0 \
    --dataset_name "FFHQ" \
    --seed 10 \
    --model_id $PROJECT_PATH/checkpoint_path/ffhq.pt \
    --wandb_iters 100 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_name "FFHQ_DMD1_bf16_interpolate"   \
    --real_image_path 0_myfiles_face/datasets/FFHQ_lmdb \
    --dfake_gen_update_ratio 5 \
    --cls_loss_weight 1e-2 \
    --gan_classifier \
    --gen_cls_loss_weight 3e-3 \
    --dmd_loss_weight 1 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --delete_ckpts \
    --max_checkpoint 100 \
    --use_fp16
