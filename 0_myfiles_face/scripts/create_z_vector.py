# make_zbank.py
import argparse, os, json, torch
from datetime import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="zbank",
                   help="Folder to save the bank and metadata.")
    p.add_argument("--name", type=str, default="zbank_256.pt",
                   help="Filename for the tensor package.")
    p.add_argument("--n", type=int, default=100, help="Number of z_T samples.")
    p.add_argument("--res", type=int, default=256, help="Image size (H=W).")
    p.add_argument("--channels", type=int, default=3, help="Channels (e.g., 3 for RGB).")
    p.add_argument("--seed", type=int, default=1234, help="Global RNG seed.")
    # Optional sampler metadata—adjust to match your eval setup
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddim","ddpm"])
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0 = deterministic).")
    p.add_argument("--steps", type=int, default=3, help="Number of inference steps (NFE).")
    p.add_argument("--schedule", type=str, default="linear", help="Timestep schedule label.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Deterministic noise creation
    g = torch.Generator().manual_seed(args.seed)
    zT = torch.randn(args.n, args.channels, args.res, args.res, generator=g, dtype=torch.float32)

    meta = {
        "created": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "shape": [args.n, args.channels, args.res, args.res],
        "dtype": "float32",
        "sampler": args.sampler,
        "eta": args.eta,
        "steps": args.steps,
        "schedule": args.schedule,
        "notes": "Reuse these z_T tensors for apples-to-apples comparisons across runs."
    }

    # Save tensor package
    pt_path = os.path.join(args.out_dir, args.name)
    torch.save({"zT": zT, "meta": meta}, pt_path)

    # Also save a tiny JSON sidecar (easy to read without torch)
    with open(os.path.join(args.out_dir, args.name.replace(".pt", ".json")), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved z bank: {pt_path}")
    print(f"Tensor shape: {tuple(zT.shape)}  dtype: {zT.dtype}")

if __name__ == "__main__":
    main()
