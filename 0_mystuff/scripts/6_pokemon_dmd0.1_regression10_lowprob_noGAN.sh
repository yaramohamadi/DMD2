export CHECKPOINT_NAME="imagenet/imagenet_lr2e-6_scratch"  # note that the imagenet/ is necessary
export OUTPUT_PATH="0_mystuff/output"

# bash scripts/download_hf_checkpoint.sh $CHECKPOINT_NAME $OUTPUT_PATH

export CHECKPOINT_PATH="0_mystuff/checkpoint_path" # change this to your own checkpoint folder 
export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie" # change this to your own wandb entity
export WANDB_PROJECT="DMD" # change this to your own wandb project

# mkdir $CHECKPOINT_PATH

# bash scripts/download_imagenet.sh $CHECKPOINT_PATH

# start a training with 7 gpu

export CUDA_VISIBLE_DEVICES=0,1

# start a training with regression on 10 anchors
bash 0_mystuff/experiments/pokemon_dmd0.1_regression10_lowprob_noGAN.sh  $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT