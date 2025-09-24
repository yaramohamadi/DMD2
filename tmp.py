import torch, glob
for f in sorted(glob.glob("0_myfiles_face/checkpoint_path/FFHQ_distilled_weights/checkpoint_model_037200/pytorch_model*.bin")):
    try:
        torch.load(f, map_location="cpu")
        print("OK:", f)
    except Exception as e:
        print("CORRUPT:", f, e)