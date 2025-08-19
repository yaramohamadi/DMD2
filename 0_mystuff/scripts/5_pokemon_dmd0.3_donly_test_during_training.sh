export CHECKPOINT_NAME="imagenet/imagenet_lr2e-6_scratch"  # note that the imagenet/ is necessary
export OUTPUT_PATH="0_mystuff/output"

# bash scripts/download_hf_checkpoint.sh $CHECKPOINT_NAME $OUTPUT_PATH

export CHECKPOINT_PATH="0_mystuff/checkpoint_path" # change this to your own checkpoint folder 
export WANDB_ENTITY="yara-mohammadi-bahram-1-ecole-superieure-de-technologie" # change this to your own wandb entity
export WANDB_PROJECT="DMD" # change this to your own wandb project

export CUDA_VISIBLE_DEVICES=2,3

python main/edm/test_folder_edm.py \
    --folder $CHECKPOINT_PATH/pokemon_dmd0.3_donly_dmdfrac/time_1755542728_seed10 \
    --wandb_name test_pokemon_dmd0.3_donly_dmdfrac \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --resolution 64 --label_dim 1000  \
    --ref_path $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz \
    --detector_url $CHECKPOINT_PATH/inception-2015-12-05.pkl