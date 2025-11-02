#!/usr/bin/env python3
import os, sys, torch, argparse
from collections import Counter

def guess_state_dict(blob):
    # Common formats: raw state_dict, {"state_dict": ...}, {"model": ...}
    if isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            return blob["state_dict"]
        if "model" in blob and isinstance(blob["model"], dict):
            return blob["model"]
    # Fallback: assume it's already a state_dict-like mapping
    return blob

def strip_prefix(name, prefix):
    return name[len(prefix):] if name.startswith(prefix) else name

def _prefix_counts(keys, depth):
    return Counter(".".join(k.split(".")[:depth]) for k in keys)

def print_prefixes(sd, max_depth=3, topn=None, file=sys.stderr):
    keys = list(sd.keys())
    print("\n=== Key prefix summary ===", file=file)
    for d in range(1, max_depth + 1):
        cnt = _prefix_counts(keys, d)
        total = sum(cnt.values())
        items = cnt.most_common()
        if topn is not None:
            items = items[:topn]
        print(f"\n[Depth {d}] {len(cnt)} unique prefixes ({total} params)", file=file)
        for p, c in items:
            print(f"  {p}: {c}", file=file)
    print("==========================\n", file=file)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to pytorch_model.bin")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--include-prefix", nargs="+", default=["generator."],
                    help="Only keep keys that start with any of these prefixes")
    ap.add_argument("--exclude-prefix", nargs="*", default=[],
                    help="Drop keys that start with any of these prefixes")
    ap.add_argument("--strip-prefix", default="generator.",
                    help="Strip this prefix off kept keys when saving")
    ap.add_argument("--show-prefixes", action="store_true",
                    help="Print prefix summaries (depths 1..3)")
    ap.add_argument("--topn", type=int, default=None,
                    help="If set, only print the top-N prefixes per depth")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = guess_state_dict(ckpt)

    # Optional: show prefix summaries before filtering
    if args.show_prefixes:
        print_prefixes(sd, max_depth=3, topn=args.topn, file=sys.stderr)

    kept = {}
    for k, v in sd.items():
        if not any(k.startswith(p) for p in args.include_prefix):
            continue
        if any(k.startswith(p) for p in args.exclude_prefix):
            continue
        new_k = strip_prefix(k, args.strip_prefix) if args.strip_prefix else k
        kept[new_k] = v

    if not kept:
        # Help debug by showing top-level heads if nothing matched
        heads = sorted(set(k.split(".")[0] for k in sd.keys()))
        print("[WARN] No parameters matched. Top-level prefixes I see:", heads, file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(kept, args.out)
    print(f"[OK] Saved {len(kept)} tensors to {args.out}")

if __name__ == "__main__":
    main()
