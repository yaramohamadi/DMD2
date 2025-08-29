import matplotlib
matplotlib.use('Agg')

from main.utils import (
    prepare_images_for_saving, draw_valued_array, cycle,
    draw_probability_histogram, RandomHorizontalFlipTensor,
    _is_locked_or_not_ready, _safe_rmtree,
)
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from main.data.lmdb_dataset import LMDBDataset
from main.dhariwal.dhariwal_unified_model import dhariwalUniModel
from accelerate.utils import set_seed
from accelerate import Accelerator
import argparse 
import shutil
import wandb 
import torch 
import time 
import os
import math
from pathlib import Path
import shutil

class Trainer:
    def __init__(self, args):

        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        accelerator_project_config = ProjectConfiguration(logging_dir=args.output_path)

        accelerator = Accelerator(
            gradient_accumulation_steps=args.grad_accum_steps,
            mixed_precision="bf16" if args.use_fp16 else "no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=None
        )

        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        # Enable checkpoint resuming and saving in the same wandb run
        if accelerator.is_main_process:
            if args.checkpoint_path is not None:
                # Resume into the SAME run folder (parent of the checkpoint directory)
                resume_run_dir = os.path.dirname(args.checkpoint_path.rstrip("/"))
                self.output_path = resume_run_dir
                os.makedirs(self.output_path, exist_ok=True)
            else:
                # fresh run
                self.run_id = int(time.time())
                output_path = os.path.join(args.output_path, f"time_{self.run_id}_seed{args.seed}")
                os.makedirs(output_path, exist_ok=False)
                self.output_path = output_path

            if args.cache_dir != "":
                if args.checkpoint_path is not None:
                    # mirror cache dir name to the resumed run_id/seed pattern if you use it
                    parent = os.path.basename(os.path.dirname(self.output_path))  # e.g., time_1756313057_seed10
                    self.cache_dir = os.path.join(args.cache_dir, parent)
                else:
                    self.cache_dir = os.path.join(args.cache_dir, f"time_{getattr(self, 'run_id', int(time.time()))}_seed{args.seed}")
                os.makedirs(self.cache_dir, exist_ok=True)
                

        self.model = dhariwalUniModel(args, accelerator)
        self.dataset_name = args.dataset_name
        self.real_image_path = args.real_image_path

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.num_train_timesteps = args.num_train_timesteps 

        self.cls_loss_weight = args.cls_loss_weight 

        self.gan_classifier = args.gan_classifier 
        self.gan_adv_loss = args.gan_adv_loss
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        self.dmd_loss_weight = args.dmd_loss_weight # Added by me
        self.no_save = args.no_save
        self.previous_time = None 
        self.step = 0 
        self.global_step = 0
        self.cache_checkpoints = (args.cache_dir != "")
        self.max_checkpoint = args.max_checkpoint

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading checkpoints without optimizer states from {args.ckpt_only_path}")
            # state_dict = torch.load(args.ckpt_only_path, map_location="cpu")
            # print(self.model.load_state_dict(state_dict, strict=False))
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")

            generator_state_dict = torch.load(generator_path, map_location="cpu")
            guidance_state_dict = torch.load(guidance_path, map_location="cpu")

            print(self.model.feedforward_model.load_state_dict(generator_state_dict, strict=False))
            print(self.model.guidance_model.load_state_dict(guidance_state_dict, strict=False))

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator checkpoints from {args.generator_ckpt_path}")
            generator_path = os.path.join(args.generator_ckpt_path, "pytorch_model.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=True))

        # also load the training dataset images, this will be useful for GAN loss 
        real_transform = RandomHorizontalFlipTensor(p=0.5)
        real_dataset = LMDBDataset(args.real_image_path, transform=real_transform)

        real_image_dataloader = torch.utils.data.DataLoader(
            real_dataset, batch_size=1, shuffle=True, 
            drop_last=True, num_workers=0
        )
        real_image_dataloader = accelerator.prepare(real_image_dataloader)
        self.real_image_dataloader = cycle(real_image_dataloader)
            
        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad], 
            lr=args.guidance_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad], 
            lr=args.generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )

        # actually this scheduler is not very useful (it warms up from 0 to max_lr in 500 / num_gpu steps), but we keep it here for consistency
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        # the self.model is not wrapped in ddp, only its two subnetworks are wrapped 
        (
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance, 
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator 
        ) = accelerator.prepare(
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance, 
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator
        ) 

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution 
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.conditioning_sigma = args.conditioning_sigma 

        self.label_dim = args.label_dim
        self.eye_matrix = torch.eye(self.label_dim, device=accelerator.device)
        self.delete_ckpts = args.delete_ckpts
        self.max_grad_norm = args.max_grad_norm

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        if self.accelerator.is_main_process:
            run = wandb.init(config=args, dir=self.output_path, **{"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

    def load(self, checkpoint_path):
        # Expecting directories like .../checkpoint_model_000123
        self.global_step = int(checkpoint_path.rstrip("/").split("_")[-1])
        accum = self.accelerator.gradient_accumulation_steps
        
        print("loading a previous checkpoints including optimizer and random seed")
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")
        self.global_step += 1
        self.step = self.global_step * max(1, accum)  # micro-step counter aligned to optimizer step

    def save(self):
        run_root = Path(self.output_path)
        gs = self.global_step  # snapshot once
        final_dir = run_root / f"checkpoint_model_{gs:06d}"
        tmp_dir   = run_root / f".checkpoint_model_{gs:06d}.tmp"

        print(f"start saving checkpoint to {final_dir}", flush=True)

        if final_dir.exists():
            msg = "already exists (.READY) -> skip" if (final_dir / ".READY").exists() and (final_dir / "pytorch_model.bin").exists() else "exists (incomplete?) -> skip"
            print(f"[save] {final_dir} {msg}.", flush=True)
            if tmp_dir.exists():
                _safe_rmtree(tmp_dir)
        else:
            if tmp_dir.exists():
                _safe_rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            self.accelerator.save_state(str(tmp_dir))

            try:
                os.replace(str(tmp_dir), str(final_dir))
            except OSError as e:
                if final_dir.exists():
                    print(f"[save] {final_dir} appeared during save; skipping this step. ({e})", flush=True)
                    _safe_rmtree(tmp_dir)
                else:
                    raise
            else:
                (final_dir / ".READY").touch()

                if self.delete_ckpts:
                    for folder in os.listdir(self.output_path):
                        if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{gs:06d}":
                            d = run_root / folder
                            if (d / ".READY").exists() and not (d / ".EVAL_LOCK").exists():
                                _safe_rmtree(d)

                if self.cache_checkpoints:
                    cache_ckpt = Path(self.cache_dir) / f"checkpoint_model_{gs:06d}"
                    if cache_ckpt.exists():
                        _safe_rmtree(cache_ckpt)
                    shutil.copytree(str(final_dir), str(cache_ckpt), dirs_exist_ok=False)

                    checkpoints = sorted(
                        p for p in Path(self.cache_dir).iterdir()
                        if p.is_dir() and p.name.startswith("checkpoint_model")
                    )
                    if len(checkpoints) > self.max_checkpoint:
                        for p in checkpoints[:-self.max_checkpoint]:
                            _safe_rmtree(p)

                print("done saving", flush=True)


    def train_one_step(self):

        self.model.train()
        accelerator = self.accelerator
        # For handling larger batch size than GPU memory
        accum = self.accelerator.gradient_accumulation_steps
        self.global_step = self.step // max(1, accum)   # optimizer-step index
        
        # Retrieve a batch of real images from the dataloader.
        real_dict = next(self.real_image_dataloader)
        # Extract the images from the dictionary and normalize them.
        real_image = real_dict["images"] * 2.0 - 1.0 
        if self.label_dim > 0:
            real_label = self.eye_matrix[real_dict["class_labels"].squeeze(dim=1)]
        else:
            # Unconditional: ignore dataset labels
            real_label = None
        real_train_dict = {
            "real_image": real_image,
            "real_label": real_label
        }

        # Generate scaled noise based on the maximum sigma value.
        scaled_noise = torch.randn(
            self.batch_size, 3, self.resolution, self.resolution, 
            device=accelerator.device
        ) * self.conditioning_sigma
        # Set timestep sigma to a preset value for all images in the batch.
        timestep_sigma = torch.ones(self.batch_size, device=accelerator.device) * self.conditioning_sigma
        # For unconditional generation (our case), labels are set to None.
        if self.label_dim > 0:
            labels = torch.randint( low=0, high=self.label_dim, size=(self.batch_size,), device=accelerator.device, dtype=torch.long)
            labels = self.eye_matrix[labels]
        else:
            labels = None

        COMPUTE_GENERATOR_GRADIENT = self.global_step % self.dfake_gen_update_ratio == 0

        generator_grad_norm = torch.tensor(0.0, device=accelerator.device)  # default for logging

        # =========================
        #  Generator micro-step(s)
        # =========================
        with accelerator.accumulate(self.model.feedforward_model):
            gen_loss_dict, gen_log_dict = self.model(
                scaled_noise, timestep_sigma, labels,
                real_train_dict=real_train_dict,
                compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                generator_turn=True, guidance_turn=False
            )

            if COMPUTE_GENERATOR_GRADIENT:
                generator_loss = self.dmd_loss_weight * gen_loss_dict["loss_dm"]
                if self.gan_classifier:
                    generator_loss = generator_loss + gen_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

                accelerator.backward(generator_loss)

                # Only clip/step/zero on the LAST micro-step
                if accelerator.sync_gradients:
                    if accelerator.mixed_precision == "fp16":
                        generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(),
                                                                        self.max_grad_norm)
                    else:
                        generator_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.feedforward_model.parameters(),
                                                                            self.max_grad_norm)
                    self.optimizer_generator.step()
                    self.optimizer_generator.zero_grad()
                    # scheduler steps once per optimizer step
                    self.scheduler_generator.step()
            else:
                # No generator update this optimizer step; still advance scheduler only on sync to keep semantics consistent.
                if accelerator.sync_gradients:
                    self.scheduler_generator.step()

         # =========================
        #  Guidance micro-step(s)
        # =========================
        with accelerator.accumulate(self.model.guidance_model):
            guid_loss_dict, guid_log_dict = self.model(
                scaled_noise, timestep_sigma, labels,
                real_train_dict=real_train_dict,
                compute_generator_gradient=False,
                generator_turn=False, guidance_turn=True,
                guidance_data_dict=gen_log_dict.get('guidance_data_dict', None)
            )

            guidance_loss = guid_loss_dict["loss_fake_mean"]
            if self.gan_classifier:
                guidance_loss = guidance_loss + guid_loss_dict["guidance_cls_loss"] * self.cls_loss_weight

            accelerator.backward(guidance_loss)

            if accelerator.sync_gradients:
                guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(),
                                                                self.max_grad_norm)
                self.optimizer_guidance.step()
                self.optimizer_guidance.zero_grad()
                self.scheduler_guidance.step()
            else:
                guidance_grad_norm = torch.tensor(0.0, device=accelerator.device)

        # ---- merge logs (unchanged except small guards) ----
        loss_dict = {**gen_loss_dict, **guid_loss_dict}
        log_dict  = {**gen_log_dict,  **guid_log_dict}

        # ===== Logging once per optimizer step =====
        if accelerator.sync_gradients:
            # keep your NaN checks as-is
            for k in ["generated_image", "dmtrain_pred_real_image", "dmtrain_pred_fake_image", "dmtrain_noisy_latents"]:
                if k in log_dict and torch.isnan(log_dict[k]).any():
                    print(f"[warn] NaN detected in {k} at step {self.step}")

            if self.global_step % self.wandb_iters == 0:
                log_dict['generated_image'] = accelerator.gather(log_dict['generated_image'])
                log_dict['dmtrain_grad'] = accelerator.gather(log_dict['dmtrain_grad'])
                log_dict['dmtrain_timesteps'] = accelerator.gather(log_dict['dmtrain_timesteps'])
                log_dict['dmtrain_pred_real_image'] = accelerator.gather(log_dict['dmtrain_pred_real_image'])
                log_dict['dmtrain_pred_fake_image'] = accelerator.gather(log_dict['dmtrain_pred_fake_image'])

            if accelerator.is_main_process and self.global_step % self.wandb_iters == 0:
                # TODO: Need more refactoring here 
                with torch.no_grad():
                    generated_image = log_dict['generated_image']
                    generated_image_brightness = (generated_image*0.5+0.5).clamp(0, 1).mean() 
                    generated_image_std =  (generated_image*0.5+0.5).clamp(0, 1).std()

                    generated_image_grid = prepare_images_for_saving(generated_image, resolution=self.resolution)

                    data_dict = {
                        "generated_image": wandb.Image(generated_image_grid),
                        "generated_image_brightness": generated_image_brightness.item(),
                        "generated_image_std": generated_image_std.item(),
                        "generator_grad_norm": generator_grad_norm.item(),
                        "guidance_grad_norm": guidance_grad_norm.item()
                    } 

                    (
                        dmtrain_noisy_latents, dmtrain_pred_real_image, dmtrain_pred_fake_image, 
                        dmtrain_grad, dmtrain_gradient_norm
                    ) = (
                        log_dict['dmtrain_noisy_latents'], log_dict['dmtrain_pred_real_image'], log_dict['dmtrain_pred_fake_image'], 
                        log_dict['dmtrain_grad'], log_dict['dmtrain_gradient_norm']
                    )

                    gradient_brightness = dmtrain_grad.mean()
                    gradient_std = dmtrain_grad.std(dim=[1, 2, 3]).mean()

                    dmtrain_pred_real_image_mean = (dmtrain_pred_real_image*0.5+0.5).clamp(0, 1).mean()
                    dmtrain_pred_fake_image_mean = (dmtrain_pred_fake_image*0.5+0.5).clamp(0, 1).mean()

                    dmtrain_pred_read_image_std = (dmtrain_pred_real_image*0.5+0.5).clamp(0, 1).std()
                    dmtrain_pred_fake_image_std = (dmtrain_pred_fake_image*0.5+0.5).clamp(0, 1).std()

                    dmtrain_noisy_latents_grid = prepare_images_for_saving(dmtrain_noisy_latents, resolution=self.resolution)
                    dmtrain_pred_real_image_grid = prepare_images_for_saving(dmtrain_pred_real_image, resolution=self.resolution)
                    dmtrain_pred_fake_image_grid = prepare_images_for_saving(dmtrain_pred_fake_image, resolution=self.resolution)

                    gradient = dmtrain_grad 
                    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
                    gradient = (gradient - 0.5)/0.5
                    gradient = prepare_images_for_saving(gradient, resolution=self.resolution)

                    gradient_scale_grid = draw_valued_array(
                        dmtrain_grad.abs().mean(dim=[1, 2, 3]).cpu().numpy(), 
                        output_dir=self.wandb_folder
                    )

                    difference_scale_grid = draw_valued_array(
                        (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(dim=[1, 2, 3]).cpu().numpy(), 
                        output_dir=self.wandb_folder
                    )

                    difference = (dmtrain_pred_fake_image-dmtrain_pred_real_image)
                    
                    difference_brightness = difference.mean()
                    
                    difference = (difference - difference.min()) / (difference.max() - difference.min())
                    difference = (difference - 0.5)/0.5
                    difference = prepare_images_for_saving(difference, resolution=self.resolution)

                    dmtrain_timesteps_grid = draw_valued_array(
                        log_dict['dmtrain_timesteps'].squeeze().cpu().numpy(),
                        output_dir=self.wandb_folder
                    )

                    data_dict.update(
                        {
                            "dmtrain_noisy_latents_grid": wandb.Image(dmtrain_noisy_latents_grid),
                            "dmtrain_pred_real_image_grid": wandb.Image(dmtrain_pred_real_image_grid),
                            "dmtrain_pred_fake_image_grid": wandb.Image(dmtrain_pred_fake_image_grid),
                            "loss_dm": loss_dict['loss_dm'].item(),
                            "loss_fake_mean": loss_dict['loss_fake_mean'].item(),
                            "dmtrain_gradient_norm": dmtrain_gradient_norm,
                            "gradient": wandb.Image(gradient),
                            "difference": wandb.Image(difference),
                            "gradient_scale_grid": wandb.Image(gradient_scale_grid),
                            "difference_norm_grid": wandb.Image(difference_scale_grid),
                            "dmtrain_timesteps_grid": wandb.Image(dmtrain_timesteps_grid),
                            "gradient_brightness": gradient_brightness.item(),
                            "difference_brightness": difference_brightness.item(),
                            "gradient_std": gradient_std.item(),
                            "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                            "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                            "dmtrain_pred_read_image_std": dmtrain_pred_read_image_std.item(),
                            "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item()
                        }
                    )

                    (
                        faketrain_latents, faketrain_noisy_latents, faketrain_x0_pred
                    ) = (
                        log_dict['faketrain_latents'], log_dict['faketrain_noisy_latents'], 
                        log_dict['faketrain_x0_pred']
                    )

                    faketrain_latents_grid = prepare_images_for_saving(faketrain_latents, resolution=self.resolution)
                    faketrain_noisy_latents_grid = prepare_images_for_saving(faketrain_noisy_latents, resolution=self.resolution)
                    faketrain_x0_pred_grid = prepare_images_for_saving(faketrain_x0_pred, resolution=self.resolution)

                    data_dict.update({
                        "faketrain_latents": wandb.Image(faketrain_latents_grid),
                        "faketrain_noisy_latents": wandb.Image(faketrain_noisy_latents_grid),
                        "faketrain_x0_pred": wandb.Image(faketrain_x0_pred_grid)
                    })

                    if self.gan_classifier:
                        # Always log losses if present
                        if 'guidance_cls_loss' in loss_dict:
                            data_dict['guidance_cls_loss'] = float(loss_dict['guidance_cls_loss'])
                        if 'gen_cls_loss' in loss_dict:
                            data_dict['gen_cls_loss'] = float(loss_dict['gen_cls_loss'])

                        # Hinge/BCE path: we have "pred_realism_*"
                        if ("pred_realism_on_fake" in log_dict) and ("pred_realism_on_real" in log_dict):
                            pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                            pred_realism_on_real = log_dict["pred_realism_on_real"]
                            hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.detach().cpu().numpy())
                            hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.detach().cpu().numpy())
                            data_dict.update({
                                "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                                "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real),
                                "pred_realism_on_fake_mean": float(pred_realism_on_fake.mean()),
                                "pred_realism_on_real_mean": float(pred_realism_on_real.mean()),
                            })

                        # WGAN path: we have "critic_*" (no pred_realism_* keys)
                        elif ("critic_fake" in log_dict) and ("critic_real" in log_dict):
                            critic_fake = log_dict["critic_fake"]
                            critic_real = log_dict["critic_real"]
                            data_dict.update({
                                "critic_fake_mean": float(critic_fake.mean()),
                                "critic_real_mean": float(critic_real.mean()),
                            })
                            if "wgan_gp" in log_dict:
                                # could be a Python float or tensor
                                wgan_gp = log_dict["wgan_gp"]
                                wgan_gp = float(wgan_gp) if not torch.is_tensor(wgan_gp) else float(wgan_gp.detach().cpu().item())
                                data_dict["wgan_gp"] = wgan_gp

                            # Optional: visualize critic scores (not probabilities; use sigmoid only for plotting)
                            cf_sig = torch.sigmoid(critic_fake.detach())
                            cr_sig = torch.sigmoid(critic_real.detach())
                            hist_cf = draw_probability_histogram(cf_sig.cpu().numpy())
                            hist_cr = draw_probability_histogram(cr_sig.cpu().numpy())
                            data_dict.update({
                                "hist_critic_fake_sigmoid": wandb.Image(hist_cf),
                                "hist_critic_real_sigmoid": wandb.Image(hist_cr),
                            })

                    wandb.log(
                        data_dict,
                        step=self.global_step
                    )

        self.accelerator.wait_for_everyone()


    def train(self):
        accum = self.accelerator.gradient_accumulation_steps
        for _ in range(self.step, self.train_iters):
            self.train_one_step()

            # We just finished one micro-step; did we close an accumulation window?
            did_sync = ((self.step + 1) % max(1, accum) == 0)

            if did_sync and self.accelerator.is_main_process:
                global_step = (self.step + 1) // max(1, accum)

                if (not self.no_save) and global_step % self.log_iters == 0:
                    # train_one_step() already computed/used self.global_step, but ensure consistency:
                    self.global_step = global_step
                    self.save()

                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=global_step)
                    self.previous_time = current_time

            self.accelerator.wait_for_everyone()
            self.step += 1  # micro-step counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--label_dim", type=int, default=0)
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)

    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--dataset_name", type=str, default='ffhq')
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--delete_ckpts", action="store_true")
    parser.add_argument("--max_checkpoint", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)

    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--gan_classifier", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--generator_ckpt_path", type=str)

    # ---------------------- my additions -----------------------
    parser.add_argument("--dmd_loss_weight", type=float, default=1, 
        help="DMD loss weight, 0 means no DMD loss")
    parser.add_argument('--gan_multihead', action='store_true',
        help='Enable multi-scale discriminator heads (Sushko §3.2).')
    parser.add_argument('--gan_head_type', default='patch', choices=['patch','global'],
        help='patch = 1x1 conv → HxW map; global = 1x1 conv + GAP → scalar.')
    parser.add_argument('--gan_head_layers', type=str, default='all',
        help='Comma-separated layer names/indices to attach heads, or "all".')
    parser.add_argument('--gan_adv_loss', default='hinge', choices=['hinge','bce','wgan'],
        help='Adversarial loss form for each head.')
    parser.add_argument('--wgan_gp_lambda', type=float, default=10.0,
        help='>0 enables WGAN-GP with this lambda (e.g., 10.0).')
    parser.add_argument("--grad_accum_steps", type=int, default=1,
        help="Gradient accumulation steps for larger effective batch sizes.")
    # -----------------------------------------------------------

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args 

if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()