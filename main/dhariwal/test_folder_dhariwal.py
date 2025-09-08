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

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))


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
    Copy src_ckpt_dir to <root>/checkpoint_best directly, NO temp staging.
    """
    dst_dir, _, _ = _best_paths(root)
    dst_dir = Path(dst_dir)

    # Remove existing destination
    if dst_dir.is_symlink():
        dst_dir.unlink(missing_ok=True)
    elif dst_dir.exists():
        shutil.rmtree(dst_dir, ignore_errors=True)

    shutil.copytree(src_ckpt_dir, dst_dir)

    try:
        with open(dst_dir / ".BEST_INFO", "w") as f:
            f.write(f"iteration={iteration}\nfid={fid_value:.6f}\nsrc={src_ckpt_dir}\n")
    except Exception:
        pass

    return dst_dir

def locate_best_checkpoint_dir(folder: str, overall_stats: dict | None = None) -> str | None:
    """
    Prefer, in order:
      1) <folder>/checkpoint_best (copy/symlink maintained by evaluator)
      2) best_ckpt.json's 'dst' (or 'src')
      3) min-FID entry from stats.json
    Returns a path to a directory that looks like 'checkpoint_*' (or 'checkpoint_best'), or None.
    """
    root = Path(folder)
    best_dir, meta_p, _ = _best_paths(root)

    # 1) checkpoint_best directory
    if best_dir.exists() and best_dir.is_dir():
        return str(best_dir)

    # 2) best_ckpt.json
    if meta_p.exists():
        try:
            with open(meta_p, "r") as f:
                meta = json.load(f)
            cand = meta.get("dst") or meta.get("src")
            if cand and Path(cand).exists():
                return cand
        except Exception:
            pass

    # 3) Lowest FID in stats.json
    if overall_stats is None:
        info_path = os.path.join(folder, "stats.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r") as f:
                    overall_stats = json.load(f)
            except Exception:
                overall_stats = None

    if overall_stats:
        try:
            best_ckpt = min(
                overall_stats.items(),
                key=lambda kv: float(kv[1].get("fid", float("inf")))
            )[0]
            if best_ckpt and Path(best_ckpt).exists():
                return best_ckpt
        except Exception:
            pass

    return None

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
            use_fp16=False,  # True/False
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


# For few-step sampling
def _make_sigma_schedule(sigma_start: torch.Tensor, K: int, sigma_end: float) -> torch.Tensor:
    """
    Monotonically decreasing per-sample sigma schedule of length K.
    Mirrors the geometric (log-space) interpolation you use in training.
    Returns: [K, B] tensor.
    """
    if sigma_start.ndim > 1:
        sigma_start = sigma_start.view(sigma_start.shape[0], -1)[:, 0]  # [B]
    B = sigma_start.shape[0]
    s0 = sigma_start.clamp_min(sigma_end + 1e-8)
    sK = torch.full_like(s0, float(sigma_end))
    t0 = torch.log(s0); tK = torch.log(sK)
    grid = torch.linspace(0, 1, steps=K, device=s0.device, dtype=s0.dtype).unsqueeze(1)  # [K,1]
    logs = t0.unsqueeze(0) * (1 - grid) + tK.unsqueeze(0) * grid                         # [K,B]
    return torch.exp(logs)                                                                # [K,B]

def _re_noise(x0: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    """EDM corruption x = x0 + sigma * eps (same as in your training model)."""
    if sigma_next.ndim == 1:
        sigma_next = sigma_next.view(-1, 1, 1, 1)
    return x0 + sigma_next * torch.randn_like(x0)




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
    
    # Supporting both 1 step and multi-step
    for _ in range(steps):
        cur = bs

        if (not args.denoising) or (args.num_denoising_step <= 1):
            # -------- one-step path (existing behavior) --------
            t = torch.full((cur,), args.conditioning_sigma, device=dev)
            noise = torch.randn(cur, 3, args.resolution, args.resolution, device=dev)
            imgs = current_model(noise * args.conditioning_sigma, t, const_label_zero(cur))  # [-1,1], NCHW
        else:
            # -------- few-step (K-step) unrolled path --------
            K = int(max(1, args.num_denoising_step))
            # start at sigma0 = conditioning_sigma (per-sample)
            sigma0 = torch.full((cur,), args.conditioning_sigma, device=dev)
            # initial x is pure noise at sigma0
            x = torch.randn(cur, 3, args.resolution, args.resolution, device=dev) * sigma0.view(-1,1,1,1)
            y = const_label_zero(cur)

            # build decreasing schedule [K, B] from sigma0 -> denoising_sigma_end
            sigmas = _make_sigma_schedule(sigma0, K, args.denoising_sigma_end)

            last_x0 = None
            for i in range(K):
                si = sigmas[i]                     # [B]
                # predict x0 at this sigma (your Dhariwal adapter returns x0)
                x0_hat = current_model(x, si, y)   # [-1,1], NCHW
                last_x0 = x0_hat
                if i + 1 < K:
                    s_next = sigmas[i + 1]         # [B]
                    x = _re_noise(x0_hat, s_next)  # re-noise back up to next sigma
            imgs = last_x0

        imgs_u8 = ((imgs + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)     # NCHW
        imgs_u8 = imgs_u8.permute(0, 2, 3, 1).contiguous()                  # NHWC

        gathered = accelerator.gather(imgs_u8)
        if accelerator.is_main_process:
            rank0_chunks.append(gathered.cpu())
            pbar.update(gathered.size(0))


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
    parser.add_argument("--eval_best_once", action="store_true",
                        help="Evaluate only the best checkpoint once and exit")   
    parser.add_argument("--denoising", action="store_true",
                    help="Enable few-step (K-step) unrolled sampling like training.")
    parser.add_argument("--num_denoising_step", type=int, default=1,
                        help="K steps; 1 = single-step (current behavior).")
    parser.add_argument("--denoising_sigma_end", type=float, default=0.5,
                        help="Terminal sigma for the unrolled schedule.")


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


    model_index = -1
    # --- One-shot best-checkpoint evaluation and exit -----------------
    if args.eval_best_once:
        target_dir = locate_best_checkpoint_dir(folder, overall_stats)
        if target_dir is None:
            if accelerator.is_main_process:
                print("[eval_best_once] No best checkpoint found (checkpoint_best/, best_ckpt.json, or stats.json). Exiting.")
            return

        ckpt_path = Path(target_dir)
        ckpt_name = ckpt_path.name
        if ckpt_name.startswith("checkpoint_model_"):
            try:
                model_index = int(ckpt_name.split("_")[-1])
            except Exception:
                model_index = -1  # not critical

        # If it's the synthetic 'checkpoint_best' dir, don't rely on .READY
        ready = True if ckpt_name == BEST_DIR_NAME else is_checkpoint_ready(ckpt_path)
        if not ready:
            if accelerator.is_main_process:
                print(f"[eval_best_once] Best checkpoint found at {target_dir} but not READY. Exiting.")
            return

        if not try_eval_lock(ckpt_path):
            if accelerator.is_main_process:
                print(f"[eval_best_once] Could not acquire lock for {target_dir}. Exiting.")
            return

        try:
            generator = create_generator(
                str(ckpt_path / "pytorch_model.bin"),
                base_model=None  # fresh construct is fine here
            ).to(accelerator.device)

            all_images_tensor = sample(accelerator, generator, args, model_index)


            # TMP TODO: save numpy for inspection
            # tmp_npy = os.path.join(folder, f"_tmp_imgs_{model_index:06d}.npy")
            #print('saving', tmp_npy)
            #np.save(tmp_npy, all_images_tensor.numpy())
            #print('saved', tmp_npy)
            #exit()
            # TMP TODO: load numpy for testing
            # Reload mem-mapped to keep RAM down (zero-copy into torch via from_numpy)
            # print('loading', tmp_npy)
            # imgs_memmap = np.load(tmp_npy, mmap_mode='r')   # dtype=uint8, shape [N, H, W, 3]
            # all_images_tensor = torch.from_numpy(imgs_memmap)  # still NHWC uint8, CPU



            # (bugfix) ensure f-string here:
            n = all_images_tensor.size(0)
            g = int(np.floor(np.sqrt(min(100, n))))
            g = max(1, g)
            grid = all_images_tensor[:g*g].numpy().reshape(g, g, args.resolution, args.resolution, 3)
            grid = np.swapaxes(grid, 1, 2).reshape(g*args.resolution, g*args.resolution, 3)
            grid_path = f"grid_{model_index:06d}.png"
            Image.fromarray(np.ascontiguousarray(grid)).save(grid_path)

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
                prec, rec = evaluator.calc_precision_recall(nearest_k=5)
                intra_lpips = -1.0 if args.no_lpips else evaluator.calc_intra_lpips()

                stats["fid"] = float(fid_score)
                stats["intra_lpips"] = float(intra_lpips)
                stats["precision"] = float(prec)
                stats["recall"] = float(rec)

                print(f"[eval_best_once] {target_dir} FID {fid_score:.4f} Intra-LPIPS {intra_lpips:.4f}"
                        f"Precision {prec:.4f} Recall {rec:.4f}") 

                # persist/update stats.json under the same key used in the streaming path
                overall_stats[target_dir] = stats
                with open(os.path.join(folder, "stats.json"), "w") as f:
                    json.dump(overall_stats, f, indent=2)

            if accelerator.is_main_process:
                wandb.log(stats, step=model_index if model_index is not None else 0)

            torch.cuda.empty_cache()
        finally:
            release_eval_lock(ckpt_path)

        # EXIT after single evaluation
        return
    # ------------------------------------------------------------------




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
                grid_path = f"grid_{model_index:06d}.png"

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