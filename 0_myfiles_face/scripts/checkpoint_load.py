
import torch, sys
from collections import Counter

path = "0_myfiles_face/checkpoints/ffhq.pt"  # <-- your file

def choose_state_dict(blob):
    # Priority: ema > model > unet > state_dict > raw mapping
    if isinstance(blob, dict):
        for k in ("ema","model","unet","state_dict"):
            if k in blob and isinstance(blob[k], dict):
                return blob[k]
        # maybe already a raw state_dict-like mapping
        if all(isinstance(v, torch.Tensor) for v in blob.values()):
            return blob
    return blob  # last resort

def prefix_counts(keys, depth):
    return Counter(".".join(k.split(".")[:depth]) for k in keys)

ckpt = torch.load(path, map_location="cpu")
sd = choose_state_dict(ckpt)
if not isinstance(sd, dict):
    print("[ERROR] Not a dict-like state_dict. Got:", type(sd), file=sys.stderr)
    sys.exit(1)

keys = list(sd.keys())
print(f"[INFO] Loaded {len(keys)} parameter keys from {path}", file=sys.stderr)

for d in (1, 2, 3):
    cnt = prefix_counts(keys, d)
    print(f"\n[Depth {d}] {len(cnt)} unique prefixes (total params: {sum(cnt.values())})")
    for p, c in cnt.most_common(100):
        print(f"  {p}: {c}")
