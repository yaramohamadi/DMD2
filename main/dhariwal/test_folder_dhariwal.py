# --- NEW: Dhariwal/FFHQ-256 evaluator with the same FID loop you used for EDM ---
# Differences vs test_folder_edm.py:
#   - Uses Dhariwal UNet adapter + FFHQ-256 config (unconditional)
#   - Defaults: resolution=256, label_dim=0, dataset_name='ffhq256'
#   - Leaves single-step EDM-style call path intact (adapter expected to follow that API)

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import argparse
import dnnlib
import pickle
import wandb
import torch
import scipy
import glob
import json
import time
import os
from pathlib import Path

from main.dhariwal.dhariwal_network import get_edm_network  # this builds DhariwalUNetAdapter
# NEW: baseline evaluators
from argparse import Namespace
from main.dhariwal.evaluation_util import Evaluator


def is_checkpoint_ready(ckpt_dir: Path) -> bool:
    # Preferred: the trainer will touch this file when atomic rename is done
    if (ckpt_dir / ".READY").exists():
        return True
    # Backward-compatible fallback: wait until pytorch_model.bin stops changing size
    binp = ckpt_dir / "pytorch_model.bin"
    if not binp.exists():
        return False
    size1 = binp.stat().st_size
    time.sleep(0.8)  # small settle delay
    if not binp.exists():
        return False
    size2 = binp.stat().st_size
    return size1 == size2 and size1 > 0

def try_eval_lock(ckpt_dir: Path) -> bool:
    lock = ckpt_dir / ".EVAL_LOCK"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_eval_lock(ckpt_dir: Path):
    lock = ckpt_dir / ".EVAL_LOCK"
    if lock.exists():
        lock.unlink(missing_ok=True)

def create_generator(checkpoint_path, base_model=None):
    if base_model is None:
        # Build a minimal args namespace for the network factory you already have.
        # get_edm_network returns a DhariwalUNetAdapter in your repo.
        args_like = Namespace(
            # use defaults that match your training; adjust if your builder expects more fields
            resolution=256,
            img_channels=3,
            label_dim=0,          # unconditional FFHQ
            use_fp16=False,
            model_type="DhariwalUNet",
            model_id=None,        # your builder prints a warning if this is None
            # include any other fields your get_edm_network reads; unused ones are fine
        )
        generator = get_edm_network(args_like)
        # if your underlying .model has map_augment, null it (harmless if not present)
        m = getattr(generator, "model", None)
        if m is not None and hasattr(m, "map_augment"):
            try:
                del m.map_augment
                m.map_augment = None
            except Exception:
                pass
    else:
        generator = base_model

    # robust load
    while True:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            break
        except Exception as e:
            print(f"fail to load checkpoint {checkpoint_path} ({e})")
            time.sleep(1)

    print(generator.load_state_dict(state_dict, strict=False))
    return generator


@torch.no_grad()
def sample(accelerator, current_model, args, model_index):
    """
    Assumes your Dhariwal adapter follows the EDM-style forward:
        f(x_noised, sigma, y) -> x_denoised in [-1,1]
    If your adapter instead needs a sampler loop, swap this function out
    with your guided-diffusion sampling util.
    """
    m = getattr(current_model, "module", current_model)
    Lm = getattr(m, "label_dim", 0)  # unconditional if 0
    dev = accelerator.device

    # constant scalar sigma 't' tensor to match your preconditioning interface
    t = torch.full((args.eval_batch_size,), args.conditioning_sigma, device=dev)

    current_model.eval()
    all_images = []
    all_images_tensor = []

    set_seed(args.seed + accelerator.process_index)

    def constant_label_zero(B):
        if Lm == 0:
            return None
        y = torch.zeros(B, Lm, device=dev, dtype=torch.float32)
        y[:, 0] = 1.0
        return y

    # generate until we hit total_eval_samples (same logic as your original)
    while len(all_images_tensor) * args.eval_batch_size * accelerator.num_processes < args.total_eval_samples:
        noise = torch.randn(args.eval_batch_size, 3, args.resolution, args.resolution, device=dev)
        y_const0 = constant_label_zero(args.eval_batch_size)

        # --- KEY CALL (unchanged API expectation) ---
        imgs = current_model(noise * args.conditioning_sigma, t, y_const0)

        # to uint8 NHWC for logging / FID
        imgs_u8 = ((imgs + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

        gathered = accelerator.gather(imgs_u8)
        all_images.append(gathered.cpu().numpy())
        all_images_tensor.append(gathered.cpu())

    # concat & log
    all_images = np.concatenate(all_images, axis=0)[:args.total_eval_samples]
    all_images_tensor = torch.cat(all_images_tensor, dim=0)[:args.total_eval_samples]

    if accelerator.is_main_process:
        grid_size = int(args.test_visual_batch_size ** 0.5)
        grid = all_images[:grid_size * grid_size].reshape(
            grid_size, grid_size, args.resolution, args.resolution, 3
        )
        grid = np.swapaxes(grid, 1, 2).reshape(
            grid_size * args.resolution, grid_size * args.resolution, 3
        )
        wandb.log({
            "generated_image_grid": wandb.Image(grid),
            "image_mean": all_images_tensor.float().mean().item(),
            "image_std": all_images_tensor.float().std().item(),
            "eval/label_mode": "constant_zero",
            "eval/model_label_dim": int(Lm),
        }, step=model_index)

    accelerator.wait_for_everyone()
    return all_images_tensor


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="path to folder containing checkpoint_* dirs")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=8)           # 12 -> 8 (safer VRAM @256)
    parser.add_argument("--resolution", type=int, default=256)               # CHANGED: 256
    parser.add_argument("--total_eval_samples", type=int, default=5000)
    parser.add_argument("--label_dim", type=int, default=0)                  # CHANGED: unconditional
    parser.add_argument("--test_visual_batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="ffhq256")       # CHANGED: ffhq256
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--conditioning_sigma", type=float, default=50.0)    # CHANGED: 50.0 is a good start at 256px
    parser.add_argument("--category", type=str, default="ffhq", help="Category name used to pick FID reference npz")
    parser.add_argument("--fid_npz_root", type=str, required=True, help="Directory that contains category npz files (e.g., .../fid_npz/ffhq.npz)")
    parser.add_argument("--lpips_cluster_size", type=int, default=100, help="Cluster size for intra-LPIPS")
    parser.add_argument("--fewshotdataset", type=str, default="", help="Path to few-shot dataset (intra-LPIPS eval)")

    args = parser.parse_args()

    folder = args.folder
    evaluated_checkpoints = set()
    overall_stats = {}

    # accelerator init (same)
    accelerator_project_config = ProjectConfiguration(logging_dir=args.folder)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config
    )
    print(accelerator.state)

    # resume
    info_path = os.path.join(folder, "stats.json")
    if os.path.isfile(info_path) and not args.no_resume:
        with open(info_path, "r") as f:
            overall_stats = json.load(f)
            evaluated_checkpoints = set(overall_stats.keys())
    if accelerator.is_main_process:
        print(f"folder to evaluate: {folder}")

    # wandb
    if accelerator.is_main_process:
        run = wandb.init(config=args, dir=args.folder, **{"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
        wandb.run.name = args.wandb_name
        print(f"wandb run dir: {run.dir}")

    # define generator before loop
    generator = None

    while True:
        # only directories named 'checkpoint_*'
        new_ckpts = sorted(p for p in Path(folder).glob("checkpoint_*") if p.is_dir())
        new_ckpts = [str(p) for p in new_ckpts if str(p) not in evaluated_checkpoints]
        if not new_ckpts:
            time.sleep(3.0)
            continue

        for checkpoint in new_ckpts:
            ckpt_path = Path(checkpoint)
            ckpt_name = ckpt_path.name  # e.g., 'checkpoint_model_008600'
            parts = ckpt_name.split("_")
            try:
                model_index = int(parts[-1])
            except Exception:
                # skip weird names
                evaluated_checkpoints.add(checkpoint)
                continue

            if accelerator.is_main_process:
                print(f"Evaluating {folder} {checkpoint}")

            # READY / stability gate
            if not is_checkpoint_ready(ckpt_path):
                # Don’t mark as evaluated; we’ll come back
                continue

            # Try to acquire lock; if another evaluator has it, skip
            if not try_eval_lock(ckpt_path):
                continue

            try:
                generator = create_generator(
                    str(ckpt_path / "pytorch_model.bin"),
                    base_model=generator
                ).to(accelerator.device)

                all_images_tensor = sample(accelerator, generator, args, model_index)

                imgs_nchw_f01 = all_images_tensor.permute(0, 3, 1, 2).to(torch.float32) / 255.0

                ref_npz_path = os.path.join(args.fid_npz_root, f"{args.category}.npz")
                if accelerator.is_main_process:
                    print(f"[Evaluator] Using FID reference: {ref_npz_path}")

                eval_args = Namespace(**{
                    "device": str(accelerator.device),
                    "category": args.category,
                    "fewshotdataset": args.fewshotdataset,
                    "normalization": True,
                })

                stats = {}
                if accelerator.is_main_process:
                    evaluator = Evaluator(eval_args, imgs_nchw_f01, ref_npz_path, args.lpips_cluster_size)
                    fid_score = evaluator.calc_fid()
                    intra_lpips = evaluator.calc_intra_lpips()
                    stats["fid"] = float(fid_score)
                    stats["intra_lpips"] = float(intra_lpips)
                    print(f"checkpoint {checkpoint} FID {fid_score:.4f}, Intra-LPIPS {intra_lpips:.4f}")
                    overall_stats[checkpoint] = stats

                if accelerator.is_main_process:
                    wandb.log(stats, step=model_index)

                torch.cuda.empty_cache()
                evaluated_checkpoints.add(checkpoint)
            finally:
                release_eval_lock(ckpt_path)

        if accelerator.is_main_process:
            with open(os.path.join(folder, "stats.json"), "w") as f:
                json.dump(overall_stats, f, indent=2)


if __name__ == "__main__":
    evaluate()