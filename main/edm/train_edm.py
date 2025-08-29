import matplotlib
matplotlib.use('Agg')

from main.utils import prepare_images_for_saving, draw_valued_array, cycle, draw_probability_histogram
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from main.data.lmdb_dataset import LMDBDataset
from main.edm.edm_unified_model import EDMUniModel
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse 
import shutil
import wandb 
import torch 
import time 
import os

from torchvision import transforms as T

from main.edm.utils_anchor import AnchorPairs  # imported for sampling anchor pairs (regression loss)

class Trainer:
    def __init__(self, args):

        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        accelerator_project_config = ProjectConfiguration(logging_dir=args.output_path)

        ddp = DistributedDataParallelKwargs(find_unused_parameters=True) # add this for discriminator head only training
        accelerator = Accelerator(
            gradient_accumulation_steps=1, # no accumulation
            mixed_precision="bf16",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp],   # <— add this for discriminator head only training
        )
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)
            self.output_path = output_path

            if args.cache_dir != "":
                self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
                os.makedirs(self.cache_dir, exist_ok=False)

        self.model = EDMUniModel(args, accelerator)
        self.dataset_name = args.dataset_name
        self.real_image_path = args.real_image_path

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.num_train_timesteps = args.num_train_timesteps 

        self.cls_loss_weight = args.cls_loss_weight 

        self.gan_classifier = args.gan_classifier 
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        self.dmd_loss_weight = args.dmd_loss_weight # Added by me
        self.no_save = args.no_save
        self.previous_time = None 
        self.step = 0 
        self.cache_checkpoints = (args.cache_dir != "")
        self.max_checkpoint = args.max_checkpoint

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading checkpoints without optimizer states from {args.ckpt_only_path}")
            state_dict = torch.load(args.ckpt_only_path, map_location="cpu")
            print(self.model.load_state_dict(state_dict, strict=False))
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
        hflip = T.RandomHorizontalFlip(p=0.5)  # add --hflip_p to argparse if not already
        real_dataset = LMDBDataset(args.real_image_path, transform=hflip)

        # -------------------------------------- TEMPORARY HACK FOR REPLICATING POKEMON DATASET ------------------------------------
        from torch.utils.data import Dataset

        class RepeatDataset(Dataset):
            def __init__(self, base, length):
                self.base, self.length = base, length
            def __len__(self): return self.length
            def __getitem__(self, i): return self.base[i % len(self.base)]

        # Make length a multiple of (world_size * batch_size * steps_per_epoch)
        effective_len = args.batch_size * accelerator.num_processes

        real_dataset = RepeatDataset(real_dataset, effective_len)
        # -------------------------------------- END OF TEMPORARY HACK ------------------------------------------------------------

        real_image_dataloader = torch.utils.data.DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True, 
            drop_last=True, num_workers=args.num_workers
        )
        real_image_dataloader = accelerator.prepare(real_image_dataloader)

        # Unwrap the DDP-wrapped generator for rank-0-only previews
        self.gen_eval = accelerator.unwrap_model(self.model.feedforward_model)

        self.real_image_dataloader = cycle(real_image_dataloader)

        # regression loss
        self.anchor_prob = args.anchor_prob
        self.anchor_radius = args.anchor_radius
        self.lambda_regression = args.lambda_regression

        self.anchors = None
        if args.anchor_pairs_path and self.anchor_prob > 0.0:
            if accelerator.is_main_process:
                print(f"[anchors] loading {args.anchor_pairs_path}")
            self.anchors = AnchorPairs(args.anchor_pairs_path, device=accelerator.device)
            
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
        self.FIXED_LABEL_INDEX = 0  # the only label we ever use
        # a [1, L] one-hot you can expand later
        fixed = torch.zeros(1, self.label_dim, device=accelerator.device, dtype=torch.float32)
        fixed[:, self.FIXED_LABEL_INDEX] = 1.0
        self.fixed_label_vec = fixed

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
        # Please note that, after loading the checkpoints, all random seed, learning rate, etc.. will be reset to align with the checkpoint.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print("loading a previous checkpoints including optimizer and random seed")
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.accelerator.is_main_process:
            print(f"start saving checkpoint to {output_path}")

        # IMPORTANT: call on ALL ranks. Accelerate will ensure only main writes.
        self.accelerator.save_state(output_path)

        if self.accelerator.is_main_process:
            # remove previous checkpoints if asked
            if self.delete_ckpts:
                for folder in os.listdir(self.output_path):
                    if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                        shutil.rmtree(os.path.join(self.output_path, folder), ignore_errors=True)

            # optional cache mirroring
            if self.cache_checkpoints:
                dst = os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(output_path, dst)

                checkpoints = sorted([f for f in os.listdir(self.cache_dir) if f.startswith("checkpoint_model")])
                if len(checkpoints) > self.max_checkpoint:
                    for f in checkpoints[:-self.max_checkpoint]:
                        shutil.rmtree(os.path.join(self.cache_dir, f), ignore_errors=True)

            print("done saving")


    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator
            
        # For conditional generation, randomly generate labels.
        labels = torch.randint(
            low=0, high=self.label_dim, size=(self.batch_size,), 
            device=accelerator.device, dtype=torch.long
        )
        # Convert these labels to one-hot encoding.
        labels = self.eye_matrix[labels]

        # Retrieve a batch of real images from the dataloader.
        real_dict = next(self.real_image_dataloader)

        # Extract the images from the dictionary and normalize them.
        # scaled from [0,1] to [-1,1].
        real_image = real_dict["images"] * 2.0 - 1.0  
        # TODO This is only the case for the unconditional pokemon dataset, change this in case you are adapting to class conditional datasets.
        real_label = labels # self.eye_matrix[real_dict["class_labels"].squeeze(dim=1)]

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

        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        # if we have anchors, sample them # regression loss
        # --- Anchor sampling (per-sample Bernoulli gate) ---
        if self.anchors is not None and self.anchor_prob > 0.0:
            B = scaled_noise.shape[0]
            mask = (torch.rand(B, device=self.accelerator.device) < self.anchor_prob)  # [B]
            num = int(mask.sum().item())
            if num > 0:
                idx = self.anchors.sample_indices(num, balanced=True)   # [num]
                z0, x_tgt, y_tgt = self.anchors.get(idx)                # [num,3,H,W], [num,3,H,W], [num,C]
                z = z0 + self.anchor_radius * torch.randn_like(z0)
                # swap in anchor latents + labels for these positions
                scaled_noise[mask] = z
                # stash targets for reg loss later
                reg_targets = (mask, x_tgt)
            else:
                reg_targets = None
        else:
            reg_targets = None
        # ---------------------------------------------------

        # generate images and optionaly compute the generator gradient
        generator_loss_dict, generator_log_dict = self.model(
            scaled_noise, timestep_sigma, labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True,
            guidance_turn=False
        )

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0 

        if COMPUTE_GENERATOR_GRADIENT:
            generator_loss += self.dmd_loss_weight * generator_loss_dict["loss_dm"] # DMD loss weight added by me

            if self.gan_classifier:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            # regression loss
            # --- Anchor regression loss (only on masked positions) ---
            if reg_targets is not None and self.lambda_regression > 0.0:
                mask, x_tgt = reg_targets
                if mask.any():
                    # Use images produced by the generator call above (keeps graph alive)
                    x_hat_full = generator_log_dict['generated_image_undetached']   # [B,3,H,W], requires_grad=True

                    x_hat = x_hat_full[mask]

                    import torchvision.utils as vutils
                    if self.accelerator.is_main_process and self.step % self.wandb_iters == 0 and mask.any():
                        grid_hat = vutils.make_grid((x_hat*0.5+0.5).clamp(0,1), nrow=min(8, x_hat.shape[0]))
                        grid_tgt = vutils.make_grid((x_tgt*0.5+0.5).clamp(0,1), nrow=min(8, x_tgt.shape[0]))
                        wandb.log({"reg/x_hat": wandb.Image(grid_hat), "reg/x_tgt": wandb.Image(grid_tgt)}, step=self.step)

                        grid_hat_full = vutils.make_grid((x_hat_full*0.5+0.5).clamp(0,1), nrow=min(8, x_hat_full.shape[0]))
                        wandb.log({"all_xhats": wandb.Image(grid_hat_full)}, step=self.step)
                    
                    # Robust regression
                    # reg_l1 = torch.nn.functional.l1_loss(x_hat, x_tgt)
                    reg_l2 = torch.nn.functional.mse_loss(x_hat, x_tgt)
                    reg_loss = reg_l2 # reg_l1 + 0.1 * reg_l2
                    generator_loss = generator_loss + self.lambda_regression * reg_loss

                    if self.accelerator.is_main_process and self.step % self.wandb_iters == 0:
                        wandb.log({
                            #"loss/reg_l1": reg_l1.item(),
                            "loss/reg_l2": reg_l2.item(),
                            "loss/reg_total": reg_loss.item(),
                            "anchor/used_frac_step": float(mask.float().mean().item())
                        }, step=self.step)
            # ----------------------------------------------------------

            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
            self.optimizer_generator.step()

            # if we also compute gan loss, the classifier also received gradient 
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.zero_grad() 
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()

        # update the guidance model (dfake and classifier)
        guidance_loss_dict, guidance_log_dict = self.model(
            scaled_noise, timestep_sigma, labels, 
            real_train_dict=real_train_dict,
            compute_generator_gradient=False,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )

        guidance_loss = 0 

        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.gan_classifier:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.cls_loss_weight

        self.accelerator.backward(guidance_loss)

        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.scheduler_guidance.step()
        self.optimizer_generator.zero_grad()


        # combine the two dictionaries 
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        # regression loss logging
        if self.step % self.wandb_iters == 0 and reg_targets is not None and accelerator.is_main_process:
            m = float(reg_targets[0].float().mean().item())
            print(f"[reg] used_frac={m:.3f}")

        if self.step % self.wandb_iters == 0:
            log_dict['generated_image'] = accelerator.gather(log_dict['generated_image'])
            log_dict['dmtrain_grad'] = accelerator.gather(log_dict['dmtrain_grad'])
            log_dict['dmtrain_timesteps'] = accelerator.gather(log_dict['dmtrain_timesteps'])
            log_dict['dmtrain_pred_real_image'] = accelerator.gather(log_dict['dmtrain_pred_real_image'])
            log_dict['dmtrain_pred_fake_image'] = accelerator.gather(log_dict['dmtrain_pred_fake_image'])

        if accelerator.is_main_process and self.step % self.wandb_iters == 0:
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

                eps = 1e-8  # small constant
                gmin, gmax = dmtrain_grad.min(), dmtrain_grad.max()
                den = (gmax - gmin).abs() + eps
                gradient = (dmtrain_grad - gmin) / den
                gradient = (gradient - 0.5) / 0.5  # [-1,1] for your viz
                gradient = prepare_images_for_saving(gradient, resolution=self.resolution, range_type="neg1pos1")

                gradient_scale_grid = draw_valued_array(
                    dmtrain_grad.abs().mean(dim=[1, 2, 3]).cpu().numpy(), 
                    output_dir=self.wandb_folder
                )

                difference_scale_grid = draw_valued_array(
                    (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(dim=[1, 2, 3]).cpu().numpy(), 
                    output_dir=self.wandb_folder
                )

                difference = dmtrain_pred_fake_image - dmtrain_pred_real_image
                dmin, dmax = difference.min(), difference.max()
                den = (dmax - dmin).abs() + 1e-8
                difference = (difference - dmin) / den
                difference = (difference - 0.5) / 0.5
                difference = prepare_images_for_saving(difference, resolution=self.resolution, range_type="neg1pos1")

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
                    data_dict['guidance_cls_loss'] = loss_dict['guidance_cls_loss'].item()
                    data_dict['gen_cls_loss'] = loss_dict['gen_cls_loss'].item()

                    pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                    pred_realism_on_real = log_dict["pred_realism_on_real"]

                    hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                    hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                    data_dict.update(
                        {
                            "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                            "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real)
                        }
                    )


                # _______________________________________________________
                # Log generated images and other data to wandb
                if self.accelerator.is_main_process:
                    import torchvision.utils as vutils
                    gen = self.gen_eval
                    gen.eval()

                    with torch.no_grad():
                        B_vis = 10
                        H = W = self.resolution
                        dev = self.accelerator.device

                        # ---- Random (no anchor bias) ----
                        z_rand = torch.randn(B_vis, 3, H, W, device=dev) * self.conditioning_sigma
                        y_rand = self.fixed_label_vec.expand(B_vis, -1).contiguous()   # [B_vis, label_dim]
                        t_rand = torch.full((B_vis,), self.conditioning_sigma, device=dev)

                        imgs_rand = gen(z_rand, t_rand, y_rand)                        # [-1,1]
                        img_rand  = (imgs_rand * 0.5 + 0.5).clamp(0, 1)                # [0,1]
                        grid_rand = vutils.make_grid(img_rand, nrow=4)
                        data_dict["grid/random"] = wandb.Image(grid_rand)

                        # ---- Near-Anchor (in regression region) ----
                        if self.anchors is not None:
                            idx = self.anchors.sample_indices(B_vis, balanced=True)
                            z0, _x_tgt, _y_tgt = self.anchors.get(idx)                 # ignore saved y
                            z0 = z0.to(dev)
                            z_anchor = z0 + self.anchor_radius * torch.randn_like(z0)
                            t_anchor = torch.full((B_vis,), self.conditioning_sigma, device=dev)
                            y0 = self.fixed_label_vec.expand(B_vis, -1).contiguous()

                            imgs_anchor = gen(z_anchor, t_anchor, y0)                  # [-1,1]
                            img_anchor  = (imgs_anchor * 0.5 + 0.5).clamp(0, 1)
                            grid_anchor = vutils.make_grid(img_anchor, nrow=4)
                            data_dict["grid/near_anchor"] = wandb.Image(grid_anchor)
                            data_dict["anchor/radius_frac"] = float(self.anchor_radius / self.conditioning_sigma)
                # _______________________________________________________




                wandb.log(
                    data_dict,
                    step=self.step
                )

        self.accelerator.wait_for_everyone()

    def train(self):
        for index in range(self.step, self.train_iters):
            self.train_one_step()

            if (not self.no_save) and self.step % self.log_iters == 0:
                # everyone lines up, everyone calls save_state (write happens only on main)
                self.accelerator.wait_for_everyone()
                self.save()
                self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
            self.step += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--label_dim", type=int, default=10)
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
    parser.add_argument("--dataset_name", type=str, default='imagenet')
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
    
    # --------------------- my additions -----------------------
    parser.add_argument("--dmd_loss_weight", type=float, default=1, help="DMD loss weight, 0 means no DMD loss")

    # Discriminator head only training
    parser.add_argument("--d_cls_head_only", action="store_true", help="if set, the discriminator backbone is frozen and only the head is trained") # I added this one
    # DMD with fraction of sigmas
    parser.add_argument("--dmd_keep_frac",type=float, default=1.0, help="Apply DMD only to the top fraction of sigmas (e.g., 0.35 keeps the highest 35%)") # I added this one

    parser.add_argument("--anchor_pairs_path", type=str, default=None,
        help="Path to offline anchor pairs .pt (created by make_anchor_pairs.py).")
    parser.add_argument("--anchor_prob", type=float, default=0.0,
        help="Probability per sample to use an anchor (and add regression).")
    parser.add_argument("--anchor_radius", type=float, default=0.2,
        help="Stddev of Gaussian jitter around anchor latent.")
    parser.add_argument("--lambda_regression", type=float, default=0.05,
        help="Weight for anchor regression (LPIPS/L1).")
    # ----------------------------------------------------------

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--generator_ckpt_path", type=str)

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