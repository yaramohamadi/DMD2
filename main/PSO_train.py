#!/usr/bin/env python
import os, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig

logger = get_logger(__name__, log_level="INFO")


# ----------------------------
# Sampling helpers (1-step)
# ----------------------------
@torch.no_grad()
def sample_sd15_1step_pil(
    unet,
    vae,
    tokenizer,
    text_encoder,
    prompt: str,
    device,
    seed: int,
    resolution: int = 512,
    num_train_timesteps: int = 1000,
    pred_eps: bool = False,
):
    latent_resolution = resolution // 8

    ids = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    text_emb = text_encoder(ids)[0]

    t = torch.full((1,), num_train_timesteps - 1, device=device, dtype=torch.long)

    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, 4, latent_resolution, latent_resolution, generator=g, device=device, dtype=torch.float32)

    out = unet(z, t, text_emb).sample  # eps or x0 depending on how your distilled model was trained

    if pred_eps:
        # uses your fixed alpha_cumprod(t=999) approximation
        alpha_prod_t = torch.tensor(0.0047, device=device, dtype=torch.float32).view(1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t
        x0 = (z - beta_prod_t.sqrt() * out) / alpha_prod_t.sqrt()
    else:
        x0 = out

    # decode
    sf = vae.config.scaling_factor  # 0.18215
    img = vae.decode(x0 / sf).sample.float()
    img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)[0]
    img = img.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(img)


def build_validation_prompts(unique_token: str, class_token: str, is_object: bool):
    if is_object:
        return [
            f"a {unique_token} {class_token} in the jungle",
            f"a {unique_token} {class_token} in the snow",
            f"a {unique_token} {class_token} on the beach",
            f"a {unique_token} {class_token} on a cobblestone street",
            f"a {unique_token} {class_token} on top of pink fabric",
            f"a {unique_token} {class_token} on top of a wooden floor",
            f"a {unique_token} {class_token} with a city in the background",
            f"a {unique_token} {class_token} with a mountain in the background",
            f"a {unique_token} {class_token} with a blue house in the background",
            f"a {unique_token} {class_token} on top of a purple rug in a forest",
            f"a {unique_token} {class_token} with a wheat field in the background",
            f"a {unique_token} {class_token} with a tree and autumn leaves in the background",
            f"a {unique_token} {class_token} with the Eiffel Tower in the background",
            f"a {unique_token} {class_token} floating on top of water",
            f"a {unique_token} {class_token} floating in an ocean of milk",
            f"a {unique_token} {class_token} on top of green grass with sunflowers around it",
            f"a {unique_token} {class_token} on top of a mirror",
            f"a {unique_token} {class_token} on top of the sidewalk in a crowded street",
            f"a {unique_token} {class_token} on top of a dirt road",
            f"a {unique_token} {class_token} on top of a white rug",
            f"a red {unique_token} {class_token}",
            f"a purple {unique_token} {class_token}",
            f"a shiny {unique_token} {class_token}",
            f"a wet {unique_token} {class_token}",
            f"a cube shaped {unique_token} {class_token}",
        ]
    else:
        return [
            f"a {unique_token} {class_token} in the jungle",
            f"a {unique_token} {class_token} in the snow",
            f"a {unique_token} {class_token} on the beach",
            f"a {unique_token} {class_token} on a cobblestone street",
            f"a {unique_token} {class_token} on top of pink fabric",
            f"a {unique_token} {class_token} on top of a wooden floor",
            f"a {unique_token} {class_token} with a city in the background",
            f"a {unique_token} {class_token} with a mountain in the background",
            f"a {unique_token} {class_token} with a blue house in the background",
            f"a {unique_token} {class_token} on top of a purple rug in a forest",
            f"a {unique_token} {class_token} wearing a red hat",
            f"a {unique_token} {class_token} wearing a santa hat",
            f"a {unique_token} {class_token} wearing a rainbow scarf",
            f"a {unique_token} {class_token} wearing a black top hat and a monocle",
            f"a {unique_token} {class_token} in a chef outfit",
            f"a {unique_token} {class_token} in a firefighter outfit",
            f"a {unique_token} {class_token} in a police outfit",
            f"a {unique_token} {class_token} wearing pink glasses",
            f"a {unique_token} {class_token} wearing a yellow shirt",
            f"a {unique_token} {class_token} in a purple wizard outfit",
            f"a red {unique_token} {class_token}",
            f"a purple {unique_token} {class_token}",
            f"a shiny {unique_token} {class_token}",
            f"a wet {unique_token} {class_token}",
            f"a cube shaped {unique_token} {class_token}",
        ]


def infer_is_object_from_instance_dir(instance_dir: str) -> bool:
    name = os.path.basename(instance_dir.rstrip("/"))
    return not ("cat" in name or "dog" in name)


def get_x0_from_noise(sample, model_output, timestep_like):
    # fixed alpha_prod for t=999 (SD1.5)
    alpha_prod_t = (torch.ones_like(timestep_like).float() * 0.0047).reshape(-1, 1, 1, 1).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t
    return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()


# ----------------------------
# Data
# ----------------------------
class InstanceDataset(Dataset):
    def __init__(self, instance_dir: str, instance_prompt: str, resolution: int = 512):
        self.paths = sorted(
            [p for p in Path(instance_dir).iterdir()
             if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        )
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {instance_dir}")

        self.instance_prompt = instance_prompt
        self.tfm = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {"pixel_values": self.tfm(img), "prompt": self.instance_prompt}


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompts, device):
    ids = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(ids)[0]


def add_unet_lora(unet, rank: int = 16, use_dora: bool = False):
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        use_dora=use_dora,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(cfg)
    return unet


@torch.no_grad()
def negative_images_to_latent_pool(args, unet, vae, tokenizer, text_encoder, accelerator):
    # sample negative PIL images (1-step) then encode to latents (x0)
    neg_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    neg_imgs = []
    base_seed = args.seed if args.seed is not None else 0
    for i in range(args.num_negatives):
        neg_imgs.append(
            sample_sd15_1step_pil(
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=args.class_prompt,
                device=accelerator.device,
                seed=base_seed + i,
                resolution=args.resolution,
                num_train_timesteps=args.num_train_timesteps,
                pred_eps=args.pred_eps,
            )
        )

    neg_px = torch.stack([neg_tfm(im) for im in neg_imgs]).to(accelerator.device, dtype=torch.float32)
    latents = vae.encode(neg_px).latent_dist.sample()
    latents = latents * vae.config.scaling_factor  # x0 in latent space
    return latents.detach().cpu()


@torch.no_grad()
def log_validation_sd15_1step(args, unet, vae, tokenizer, text_encoder, accelerator, global_step: int):
    validation_prompts = build_validation_prompts(args.unique_token, args.class_name, args.is_object)

    if accelerator.is_main_process:
        img_save_dir = os.path.join(args.output_dir, f"step_{global_step}")
        os.makedirs(img_save_dir, exist_ok=True)
        logger.info(f"Saving validation images to {img_save_dir}")

    base_seed = args.seed if args.seed is not None else 0
    img_counter = 0
    for p in validation_prompts:
        for _ in range(args.num_validation_images):
            im = sample_sd15_1step_pil(
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=p,
                device=accelerator.device,
                seed=base_seed + 100000 + img_counter,
                resolution=args.resolution,
                num_train_timesteps=args.num_train_timesteps,
                pred_eps=args.pred_eps,
            )
            if accelerator.is_main_process:
                im.save(os.path.join(args.output_dir, f"step_{global_step}", f"{img_counter}.png"))
            img_counter += 1


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dmd2_unet_ckpt", type=str, required=True, help="path to pytorch_model.bin (full UNet weights)")
    p.add_argument("--instance_dir", type=str, required=True)
    p.add_argument("--instance_prompt", type=str, required=True)
    p.add_argument("--class_prompt", type=str, required=True)

    # tokens for validation prompts
    p.add_argument("--unique_token", type=str, required=True)
    p.add_argument("--class_name", type=str, required=True)

    # core
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=600)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # output
    p.add_argument("--output_dir", type=str, default="./pso_lora_out")
    p.add_argument("--save_steps", type=int, default=200)

    # validation
    p.add_argument("--validation_steps", type=int, default=200)
    p.add_argument("--num_validation_images", type=int, default=1)

    # PSO knobs
    p.add_argument("--loss_type", type=str, default="pso_db", choices=["pso_db", "pso"])
    p.add_argument("--beta_pso", type=float, default=5.0)
    p.add_argument("--neg_defactor", type=float, default=0.1)
    p.add_argument("--prior_loss_weight", type=float, default=0.5)

    # negatives
    p.add_argument("--num_negatives", type=int, default=20)
    p.add_argument("--neg_refresh_steps", type=int, default=0, help="0=never refresh; else refresh every N steps")

    p.add_argument("--pred_eps", action="store_true")

    # LoRA
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--use_dora", action="store_true")

    args = p.parse_args()
    args.is_object = infer_is_object_from_instance_dir(args.instance_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    set_seed(args.seed)

    model_id = "runwayml/stable-diffusion-v1-5"

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch.float32)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=torch.float32)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Load UNet base and DMD2 weights
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=torch.float32)
    state_dict = torch.load(args.dmd2_unet_ckpt, map_location="cpu")
    unet.load_state_dict(state_dict, strict=True)

    # Freeze base weights, attach LoRA
    unet.requires_grad_(False)
    unet = add_unet_lora(unet, rank=args.rank, use_dora=args.use_dora)

    trainable_params = [pp for pp in unet.parameters() if pp.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    ds = InstanceDataset(args.instance_dir, args.instance_prompt, resolution=args.resolution)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True)

    unet, optimizer, dl = accelerator.prepare(unet, optimizer, dl)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # negative latent pool from images
    neg_pool = negative_images_to_latent_pool(
        args=args,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        accelerator=accelerator,
    )

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dl:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                prompts = batch["prompt"]  # list[str]
                bsz = pixel_values.shape[0]

                # refresh negative pool if requested
                if args.neg_refresh_steps and global_step > 0 and (global_step % args.neg_refresh_steps == 0):
                    neg_pool = negative_images_to_latent_pool(
                        args=args,
                        unet=accelerator.unwrap_model(unet),
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        accelerator=accelerator,
                    )

                # x0_pos latents (instance images)
                with torch.no_grad():
                    x0_pos = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # x0_neg latents (sample from pool)
                idx = torch.randint(0, args.num_negatives, (bsz,), device="cpu")
                x0_neg = neg_pool[idx].to(device=device, dtype=x0_pos.dtype)

                # shared z and t
                z = torch.randn_like(x0_pos)
                t = torch.full((bsz,), args.num_train_timesteps - 1, device=device, dtype=torch.long)

                # embeds for winner (instance) and loser (class)
                with torch.no_grad():
                    emb_w = encode_prompt(tokenizer, text_encoder, prompts, device)                     # (B,seq,dim)
                    emb_l = encode_prompt(tokenizer, text_encoder, [args.class_prompt] * bsz, device)   # (B,seq,dim)

                # single forward by concatenation (like SDXL does)
                z2 = torch.cat([z, z], dim=0)
                t2 = torch.cat([t, t], dim=0)
                emb2 = torch.cat([emb_w, emb_l], dim=0)

                pred2 = unet(z2, t2, emb2).sample
                if args.pred_eps:
                    ones = torch.ones(2 * bsz, device=device, dtype=torch.long)
                    pred2 = get_x0_from_noise(z2, pred2, ones)
                x0_pred_w, x0_pred_l = pred2.chunk(2, dim=0)

                # losses
                loss_w = ((x0_pred_w - x0_pos) ** 2).flatten(1).mean(1)  # (B,)
                loss_l = ((x0_pred_l - x0_neg) ** 2).flatten(1).mean(1)  # (B,)

                model_diff = loss_w - args.neg_defactor * loss_l
                logits = -model_diff

                if args.loss_type == "pso_db":
                    pref_loss = torch.relu(1.0 - args.beta_pso * logits).mean()
                else:
                    pref_loss = (-F.logsigmoid(args.beta_pso * logits)).mean()

                # prior loss on the loser branch (now meaningful)
                loss = pref_loss + args.prior_loss_weight * loss_l.mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and (global_step % 50 == 0 or global_step == 1):
                    implicit_acc = (logits > 0).float().mean().item()
                    print(
                        f"step {global_step:5d} | loss {loss.item():.4f} | pref {pref_loss.item():.4f} "
                        f"| lw {loss_w.mean().item():.4f} | ll {loss_l.mean().item():.4f} | acc {implicit_acc:.3f}"
                    )

                if accelerator.is_main_process and (global_step % args.save_steps == 0 or global_step == args.max_train_steps):
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    accelerator.unwrap_model(unet).save_attn_procs(save_dir)
                    print("Saved LoRA attn_procs:", save_dir)

                if accelerator.is_main_process and (global_step % args.validation_steps == 0):
                    log_validation_sd15_1step(
                        args=args,
                        unet=accelerator.unwrap_model(unet),
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        accelerator=accelerator,
                        global_step=global_step,
                    )

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    main()
