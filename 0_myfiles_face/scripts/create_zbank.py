import torch

# create once
H, W, N = 256, 256, 5000
bank = torch.randn(N, 3, H, W, dtype=torch.float32)
meta = dict(sampler="ddim", eta=0.0, steps=3, schedule="lin", res=(H,W),
            labels="uncond", guidance_scale=0.0)
torch.save({"zT": bank, "meta": meta}, "0_myfiles_face/z_bank/zbank_256_3.pt")