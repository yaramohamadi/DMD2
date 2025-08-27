import argparse, os, torch
from main.data.lmdb_dataset import LMDBDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_image_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--label_dim", type=int, required=True, help="Dataset classes (e.g., 10)")
    ap.add_argument("--model_label_dim", type=int, default=None,
                    help="Model's label width (e.g., 1000). If None, uses --label_dim.")
    ap.add_argument("--resolution", type=int, default=32)
    ap.add_argument("--conditioning_sigma", type=float, default=80.0)
    ap.add_argument("--anchors_per_image", type=int, default=8)  # per original image; total becomes 2*K with flip
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    L_data = int(args.label_dim)
    L_model = int(args.model_label_dim) if args.model_label_dim is not None else L_data

    # No random transforms; we bake a deterministic horizontal flip.
    ds = LMDBDataset(args.real_image_path, transform=None)

    N = len(ds) if args.max_images is None else min(len(ds), args.max_images)
    K = args.anchors_per_image
    M = N * K * 2  # original + flipped

    Z = torch.empty(M, 3, args.resolution, args.resolution, dtype=torch.float32)
    X = torch.empty(M, 3, args.resolution, args.resolution, dtype=torch.float32)
    Y = torch.zeros(M, L_model, dtype=torch.float32)          # one-hot @ model width
    IMG_IDX = torch.empty(M, dtype=torch.long)

    ptr = 0
    for i in range(N):
        item = ds[i]
        x0 = item["images"].to(torch.float32) * 2.0 - 1.0     # [-1,1], [3,H,W]
        y_scalar = int(item["class_labels"].view(-1)[0].item())

        # two deterministic views: original and horizontal flip (flip width: dim=2)
        x_views = [x0, torch.flip(x0, dims=[2])]

        for xv in x_views:
            for _ in range(K):
                z = torch.randn(3, args.resolution, args.resolution) * args.conditioning_sigma
                Z[ptr] = z
                X[ptr] = xv

                # map dataset label into model's one-hot space
                # place at the same index (0..L_data-1) within 0..L_model-1
                if y_scalar < L_model:
                    Y[ptr, y_scalar] = 1.0
                else:
                    # safety: if dataset idx >= model width, wrap or clamp (wrap chosen)
                    Y[ptr, y_scalar % L_model] = 1.0

                IMG_IDX[ptr] = i
                ptr += 1

    payload = dict(
        z=Z,                    # [M,3,H,W], ~N(0, sigma^2)
        x=X,                    # [M,3,H,W], in [-1,1]
        y=Y,                    # [M, L_model], one-hot for the MODEL directly
        img_idx=IMG_IDX,        # [M]
        resolution=args.resolution,
        conditioning_sigma=args.conditioning_sigma,
        label_dim_data=L_data,          # dataset label count
        label_dim_model=L_model,        # model label count (matches y.shape[-1])
        anchors_per_image=K*2,
        num_images=N,
        seed=args.seed,
        note="Includes original + horizontal flip; y is one-hot at model width (use directly)."
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(payload, args.out_path)
    print(f"[anchor pairs] saved {ptr} anchors to {args.out_path} (2x with flips), "
          f"y shape = {tuple(Y.shape)}")

if __name__ == "__main__":
    main()
