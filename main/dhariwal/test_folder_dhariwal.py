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
import math
from PIL import Image
import shutil
from shutil import copytree
from contextlib import contextmanager
from main.dhariwal.dhariwal_network import get_edm_network  # this builds DhariwalUNetAdapter
from argparse import Namespace
from main.dhariwal.evaluation_util import Evaluator



# Helpers for saving best checkpoints ---------------------------------
BEST_META_NAME = "best_ckpt.json"          # metadata file
BEST_DIR_NAME  = "checkpoint_best"         # stable directory name
BEST_LOCK_NAME = ".BEST_LOCK"              # simple file lock

def _best_paths(root: Path):
    root = Path(root)
    return (root / BEST_DIR_NAME, root / BEST_META_NAME, root / BEST_LOCK_NAME)

def read_best_meta(root: Path) -> dict:
    _, meta_p, _ = _best_paths(root)
    if meta_p.exists():
        try:
            with open(meta_p, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"fid": float("inf"), "iteration": -1, "src": None, "dst": None, "timestamp": None}

def write_best_meta(root: Path, meta: dict) -> None:
    _, meta_p, _ = _best_paths(root)
    tmp = meta_p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_p)

@contextmanager
def best_lock(root: Path):
    _, _, lock_p = _best_paths(root)
    # exclusive create; if taken, just raise so caller skips
    fd = None
    try:
        fd = os.open(str(lock_p), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.remove(lock_p)
        except OSError:
            pass

def copy_checkpoint_as_best(src_ckpt_dir: Path, root: Path, iteration: int, fid_value: float, mode: str = "copy") -> Path:
    """
    Copy/symlink the entire checkpoint folder to <root>/checkpoint_best atomically.
    mode='copy' (default) physically copies; mode='link' makes a symlink.
    Returns the destination Path.
    """
    dst_dir, _, _ = _best_paths(root)
    dst_dir = Path(dst_dir)
    tmp = dst_dir.with_suffix(".tmp")

    if mode == "link":
        # symlink update (zero extra disk)
        if tmp.exists() or tmp.is_symlink():
            try: tmp.unlink()
            except OSError: shutil.rmtree(tmp, ignore_errors=True)
        os.symlink(src_ckpt_dir, tmp)
        os.replace(tmp, dst_dir)   # atomic swap
    else:
        # physical copy (safe atomic swap)
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src_ckpt_dir, tmp)
        os.replace(tmp, dst_dir)   # atomic swap

    # also drop a small marker file inside best dir (optional)
    try:
        with open(dst_dir / ".BEST_INFO", "w") as f:
            f.write(f"iteration={iteration}\nfid={fid_value:.6f}\nsrc={src_ckpt_dir}\n")
    except Exception:
        pass

    return dst_dir
# ----------------------------------------------------------------------

# --------------------------------------------------------------------



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
        print(f"Checkpoint {ckpt_dir} is already being evaluated by another process.")
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
            use_fp16=args.use_fp16,  # True/False
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
    Generate exactly args.total_eval_samples images across all processes,
    with a global tqdm progress bar (on rank 0). Returns NHWC uint8 tensor on CPU.
    """
    dev   = accelerator.device
    world = accelerator.num_processes
    bs    = args.eval_batch_size
    total = args.total_eval_samples

    # How many synchronized steps do we need if every rank makes `bs` images each step?
    steps = math.ceil(total / float(bs * world))

    # Make the model fast for inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    m = getattr(current_model, "module", current_model)
    Lm = getattr(m, "label_dim", 0)
    current_model.eval()
    current_model.float()

    def const_label_zero(B):
        if Lm == 0:
            return None
        y = torch.zeros(B, Lm, device=dev, dtype=torch.float32)
        y[:, 0] = 1.0
        return y

    # only rank 0 collects into CPU memory
    rank0_chunks = []
    if accelerator.is_main_process:
        pbar = tqdm(total=total, desc=f"Sampling {total} @ {args.resolution}", ncols=100)

    for _ in range(steps):
        # All ranks generate the same batch size -> gather has consistent shapes
        cur = bs
        t   = torch.full((cur,), args.conditioning_sigma, device=dev)

        noise = torch.randn(cur, 3, args.resolution, args.resolution, device=dev)

        imgs = current_model(noise * args.conditioning_sigma, t, const_label_zero(cur))  # [-1,1], NCHW

        imgs_u8 = ((imgs + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)     # NCHW
        imgs_u8 = imgs_u8.permute(0, 2, 3, 1).contiguous()                  # NHWC

        # Gather (shape-consistent across ranks), then only rank 0 stores/updates tqdm
        gathered = accelerator.gather(imgs_u8)
        if accelerator.is_main_process:
            rank0_chunks.append(gathered.cpu())
            pbar.update(gathered.size(0))  # global increment by world*bs each step

    if accelerator.is_main_process:
        pbar.close()
        all_images_tensor = torch.cat(rank0_chunks, dim=0)[:total]  # [N, H, W, 3] uint8 on CPU

        # build a preview grid with an auto grid size (near-square)
        n = all_images_tensor.size(0)
        g = int(np.floor(np.sqrt(min(100, n))))  # cap grid to 100 cells
        g = max(1, g)
        grid = all_images_tensor[:g*g].numpy().reshape(g, g, args.resolution, args.resolution, 3)
        grid = np.swapaxes(grid, 1, 2).reshape(g*args.resolution, g*args.resolution, 3)

        wandb.log({
            "generated_image_grid": wandb.Image(grid),
            "image_mean": float(all_images_tensor.float().mean().item()),
            "image_std":  float(all_images_tensor.float().std().item()),
            "eval/label_mode": "constant_zero" if Lm > 0 else "uncond",
            "eval/model_label_dim": int(Lm),
        }, step=model_index)
    else:
        all_images_tensor = torch.empty(0, dtype=torch.uint8)  # non-main returns empty placeholder

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
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)    
    parser.add_argument("--category", type=str, default="ffhq", help="Category name used to pick FID reference npz")
    parser.add_argument("--fid_npz_root", type=str, required=True, help="Directory that contains category npz files (e.g., .../fid_npz/ffhq.npz)")
    parser.add_argument("--lpips_cluster_size", type=int, default=100, help="Cluster size for intra-LPIPS")
    parser.add_argument("--fewshotdataset", type=str, default="", help="Path to few-shot dataset (intra-LPIPS eval)")
    parser.add_argument("--no_lpips", action="store_true", help="Disable LPIPS computation to save time")
    parser.add_argument("--use_fp16", action="store_true", help="Use bf16 for inference (if supported by GPU/driver)")

    args = parser.parse_args()

    folder = args.folder
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
    evaluated_checkpoints = set()            # <- NEW: always define it
    overall_stats = {}                       # keep this too (you already have it above)

    if os.path.isfile(info_path) and not args.no_resume:
        with open(info_path, "r") as f:
            overall_stats = json.load(f)
        # keys are the checkpoint paths you wrote earlier
        evaluated_checkpoints = set(overall_stats.keys())

    # wandb
    if accelerator.is_main_process:
        run = wandb.init(config=args, dir=args.folder, **{"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
        wandb.run.name = args.wandb_name
        print(f"wandb run dir: {run.dir}")

    # define generator before loop
    generator = None

    while True:
        # only directories named 'checkpoint_*'
        new_ckpts = sorted(p for p in Path(folder).glob("checkpoint_model_*") if p.is_dir())
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
# 
                all_images_tensor = sample(accelerator, generator, args, model_index)

                #TMP TODO: save numpy for inspection
                # tmp_npy = os.path.join(folder, f"_tmp_imgs_{model_index:06d}.npy")
                # print('saving', tmp_npy)
                # np.save(tmp_npy, all_images_tensor.numpy())
                # print('saved', tmp_npy)
                # exit()
                # TMP TODO: load numpy for testing
                # Reload mem-mapped to keep RAM down (zero-copy into torch via from_numpy)
                #print('loading', tmp_npy)
                #imgs_memmap = np.load(tmp_npy, mmap_mode='r')   # dtype=uint8, shape [N, H, W, 3]
                # all_images_tensor = torch.from_numpy(imgs_memmap)  # still NHWC uint8, CPU

                # TMP TODO: save locally an optional grid --------------------------
                # build a preview grid with an auto grid size (near-square)
                n = all_images_tensor.size(0)
                g = int(np.floor(np.sqrt(min(100, n))))  # cap grid to 100 cells
                g = max(1, g)
                grid = all_images_tensor[:g*g].numpy().reshape(g, g, args.resolution, args.resolution, 3)
                grid = np.swapaxes(grid, 1, 2).reshape(g*args.resolution, g*args.resolution, 3)

                # save grid locally too
                grid_path = "grid_{model_index:06d}.png"

                # ensure C-contiguous uint8 for PIL
                Image.fromarray(np.ascontiguousarray(grid)).save(grid_path)
                # ------------------------------------------------------------------

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

                    # Save best model if needed
                    if accelerator.is_main_process:
                        with best_lock(Path(folder)):
                            best = read_best_meta(Path(folder))
                            current_fid = float(fid_score)
                            if np.isfinite(current_fid) and (current_fid < float(best.get("fid", float("inf")))):
                                dst_path = copy_checkpoint_as_best(ckpt_path, Path(folder), model_index, current_fid)
                                new_best = {
                                    "fid": current_fid,
                                    "iteration": int(model_index),
                                    "src": str(ckpt_path),
                                    "dst": str(dst_path),
                                    "timestamp": time.time(),
                                }
                                write_best_meta(Path(folder), new_best)
                                print(f"[BEST] New best FID {current_fid:.4f} at iter {model_index}. Saved to: {dst_path}")

                    prec, rec = evaluator.calc_precision_recall(nearest_k=5)
                    if args.no_lpips:
                        intra_lpips = -1.0
                    else:
                        intra_lpips = evaluator.calc_intra_lpips()
                    stats["fid"] = float(fid_score)
                    stats["intra_lpips"] = float(intra_lpips)
                    stats["precision"] = float(prec)
                    stats["recall"] = float(rec)
                    print(f"checkpoint {checkpoint} FID {fid_score:.4f} Precision {prec:.4f} Recall {rec:.4f} Intra-LPIPS {intra_lpips:.4f}")
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