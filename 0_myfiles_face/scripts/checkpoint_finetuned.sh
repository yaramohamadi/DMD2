python 0_myfiles_face/scripts/checkpoint_distill_finetuned.py \
  --ckpt 0_myfiles_face/checkpoint_path/babies_finetune/checkpoint_best/pytorch_model.bin \
  --out 0_myfiles_face/checkpoints/sunglasses_finetune.pt \
  --include-prefix unet. \
  --strip-prefix unet. \
  --show-prefixes

# CKPT[babies]="0_myfiles_face/checkpoint_path/babies_lr5e-7_bs1_dn3_DMD0_GClsw15e-3__naive_all_ftnaive_ddpmall/checkpoint_best"
# CKPT[cat]="0_myfiles_face/checkpoint_path/cat_lr5e-7_bs1_dn3_DMD0_GClsw15e-3__naive_all_ftnaive_ddpmall/checkpoint_best"
# CKPT[sunglasses]="0_myfiles_face/checkpoint_path/sunglasses_lr5e-7_bs1_dn3_DMD0_GClsw15e-3__naive_all_ftnaive_ddpmall/checkpoint_best"
# CKPT[metface]="0_myfiles_face/checkpoint_path/metfaces_lr5e-7_bs1_dn3_DMD0_GClsw15e-3__naive_all_ftnaive_ddpmall/checkpoint_best"