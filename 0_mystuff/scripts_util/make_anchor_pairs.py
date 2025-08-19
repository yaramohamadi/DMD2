import argparse, os, torch, numpy as np
from main.data.lmdb_dataset import LMDBDataset
from torchvision import transforms as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_image_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--label_dim", type=int, required=True)
    ap.add_argument("--resolution", type=int, default=32)
    ap.add_argument("--conditioning_sigma", type=float, default=80.0)
    ap.add_argument("--anchors_per_image", type=int, default=8)   # K per target
    ap.add_argument("--max_images", type=int, default=None)       # if you want to cap
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hflip = T.RandomHorizontalFlip(p=0.5)  # add --hflip_p to argparse if not already
    ds = LMDBDataset(args.real_image_path, transform=hflip)

    # we assume LMDBDataset returns images in [0,1] and labels as Long
    N = len(ds) if args.max_images is None else min(len(ds), args.max_images)
    K = args.anchors_per_image
    M = N * K

    Z = torch.empty(M, 3, args.resolution, args.resolution, dtype=torch.float32)
    X = torch.empty(M, 3, args.resolution, args.resolution, dtype=torch.float32)
    Y = torch.zeros(M, args.label_dim, dtype=torch.float32)
    IMG_IDX = torch.empty(M, dtype=torch.long)

    ptr = 0
    for i in range(N):
        item = ds[i]
        x = item["images"].to(torch.float32) * 2.0 - 1.0     # [-1,1]
        # if dataset resolution differs, you’d resize here (but LMDB is pre-sized)
        y_scalar = int(item["class_labels"].view(-1)[0].item())
        y_onehot = torch.zeros(args.label_dim); y_onehot[y_scalar] = 1.0

        for k in range(K):
            z = torch.randn(3, args.resolution, args.resolution) * args.conditioning_sigma
            Z[ptr] = z
            X[ptr] = x
            Y[ptr] = y_onehot
            IMG_IDX[ptr] = i
            ptr += 1

    payload = dict(
        z=Z,                   # [M,3,H,W], ~N(0, sigma^2)
        x=X,                   # [M,3,H,W], in [-1,1]
        y=Y,                   # [M,label_dim] one-hot
        img_idx=IMG_IDX,       # [M]
        resolution=args.resolution,
        conditioning_sigma=args.conditioning_sigma,
        label_dim=args.label_dim,
        anchors_per_image=K,
        num_images=N,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(payload, args.out_path)
    print(f"[anchor pairs] saved {ptr} anchors to {args.out_path}")

if __name__ == "__main__":
    main()


