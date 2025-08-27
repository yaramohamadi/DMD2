import torch
from types import SimpleNamespace
from main.dhariwal.dhariwal_network import get_edm_network

args = SimpleNamespace(
    resolution=256,
    label_dim=0,        # FFHQ is unconditional
    use_fp16=False,
    model_id=None       # or a path to your ADM/guided_diffusion UNet checkpoint
)

net = get_edm_network(args)  # DhariwalUNetAdapter

B, C, H, W = 2, 3, 256, 256
x0 = torch.randn(B, C, H, W)
sigma = torch.full((B,), 80.0)  # matches your conditioning_sigma defaults
x_noisy = x0 + sigma.view(B,1,1,1) * torch.randn_like(x0)

# 1) Standard forward → x0_hat
x0_hat = net(x_noisy, sigma, class_labels=None)
print("x0_hat:", x0_hat.shape)

# 2) Bottleneck features for your classifier head
feat = net(x_noisy, sigma, class_labels=None, return_bottleneck=True)
print("feat:", feat.shape)
