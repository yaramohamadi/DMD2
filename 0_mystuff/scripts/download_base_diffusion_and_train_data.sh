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
bash 0_mystuff/experiments/imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch.sh  $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT

# on the same node, start a testing process that continually reads from the checkpoint folder and evaluate the FID 
# Change TIMESTAMP_TBD to the real one

python main/edm/test_folder_edm.py \
    --folder $CHECKPOINT_PATH/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch/TIMESTAMP_TBD \
    --wandb_name test_imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --resolution 64 --label_dim 1000  \
    --ref_path $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz \
    --detector_url $CHECKPOINT_PATH/inception-2015-12-05.pkl 