from main.dhariwal.dhariwal_network import get_edm_network, load_pt_with_logs
import torch.nn.functional as F
import torch.nn as nn
import dnnlib 
import pickle 
import torch
import copy 


# utils
def _avg_spatial(x):
    return x.mean(dim=(2,3), keepdim=False) if x.ndim == 4 else x  # [B,1,H,W]→[B,1]

def _gan_losses(logits_real, logits_fake, mode='hinge'):
    # Flatten so it works for [B,1], [B,HW], etc.
    logits_fake = logits_fake.view(logits_fake.size(0), -1)
    logits_real_flat = None if logits_real is None else logits_real.view(logits_real.size(0), -1)

    if mode == 'wgan':
        # Critic scores: higher for real than fake.
        d_loss_real = 0.0 if logits_real_flat is None else -logits_real_flat.mean()
        d_loss_fake = logits_fake.mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = -logits_fake.mean()  # generator tries to increase critic score
        return d_loss, g_loss

    if mode == 'hinge':
        d_loss_real = 0.0 if logits_real_flat is None else torch.relu(1.0 - logits_real_flat).mean()
        d_loss_fake = torch.relu(1.0 + logits_fake).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = (-logits_fake).mean()
        return d_loss, g_loss

    # 'bce'
    bce = nn.BCEWithLogitsLoss()
    zeros_f = torch.zeros_like(logits_fake)
    ones_f  = torch.ones_like(logits_fake)

    if logits_real_flat is None:
        d_loss = bce(logits_fake, zeros_f)
    else:
        ones_r = torch.ones_like(logits_real_flat)
        d_loss = bce(logits_real_flat, ones_r) + bce(logits_fake, zeros_f)

    g_loss = bce(logits_fake, ones_f)
    return d_loss, g_loss


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
#------------------------------------------------------------


class dhariwalGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args 
        self.accelerator = accelerator 

        # GAN multi-head options
        self.gan_multihead = getattr(args, 'gan_multihead', False)
        self.gan_head_type = getattr(args, 'gan_head_type', 'patch')
        self.gan_head_layers = getattr(args, 'gan_head_layers', 'all')
        self.gan_adv_loss = getattr(args, 'gan_adv_loss', 'hinge')
        self.wgan_gp_lambda = getattr(args, 'wgan_gp_lambda', 10.0)  # set >0 to enable GP (e.g., 10.0)

        # with dnnlib.util.open_url(args.model_id) as f:
        #    temp_edm = pickle.load(f)['ema']

        # initialize the real unet 
        self.real_unet = get_edm_network(args).to(accelerator.device)
        self.real_unet = load_pt_with_logs(self.real_unet, args.model_id)  # load the .pt file here
        # self.real_unet.load_state_dict(temp_edm.state_dict(), strict=True)
        self.real_unet.requires_grad_(False)
        del self.real_unet.model.map_augment
        self.real_unet.model.map_augment = None

        # initialize the fake unet 
        self.fake_unet = copy.deepcopy(self.real_unet)
        self.fake_unet = self.fake_unet.to(accelerator.device)          
        self.fake_unet.requires_grad_(True)

        # some training hyper-parameters 
        self.sigma_data = args.sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        self.gan_classifier = args.gan_classifier
        self.diffusion_gan = args.diffusion_gan 
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep
        
        # Figure out bottleneck channels dynamically (works for 256x256 ADM)
        with torch.no_grad():
            dummy_x = torch.zeros(1, 3, args.resolution, args.resolution, device=accelerator.device)
            dummy_sigma = torch.ones(1, device=accelerator.device) * self.sigma_min  # any valid sigma
            # unconditional => label None; conditional => pass a 1-hot of shape [1, label_dim]
            dummy_label = None if args.label_dim == 0 else torch.zeros(1, args.label_dim, device=accelerator.device)

            if accelerator.mixed_precision == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feat = self.fake_unet(dummy_x, dummy_sigma, dummy_label, return_bottleneck=True)
            else:
                feat = self.fake_unet(dummy_x, dummy_sigma, dummy_label, return_bottleneck=True)

            bottleneck_c = feat.shape[1]

        # Initialize head for single-head GAN (Dhariwal & Nichol)
        if self.gan_classifier and not self.gan_multihead:
            # ----- ORIGINAL single bottleneck head (unchanged) -----
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=bottleneck_c, out_channels=bottleneck_c, stride=2, padding=1),  # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=bottleneck_c),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=bottleneck_c, out_channels=bottleneck_c, stride=4, padding=0),  # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=bottleneck_c),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(kernel_size=1, in_channels=bottleneck_c, out_channels=1, stride=1, padding=0),
            )
            self.cls_pred_branch.requires_grad_(True)

        # Initialize heads for multi-head GAN (Sushko §3.2)
        elif self.gan_classifier and self.gan_multihead:
            # ----- MULTI-HEAD (shallow) path: 1x1 conv per tapped block -----
            with torch.no_grad():
                dummy_x = torch.zeros(1, 3, args.resolution, args.resolution, device=accelerator.device)
                dummy_sigma = torch.ones(1, device=accelerator.device) * self.sigma_min
                # Map EDM → DDPM state for UNet hook pass
                cfac = 1.0 / torch.sqrt(1.0 + dummy_sigma.view(1,1,1,1)**2)
                x_t = cfac * dummy_x
                t0 = torch.zeros(1, dtype=torch.long, device=accelerator.device)  # any valid timestep
                y0 = None if args.label_dim == 0 else torch.zeros(1, args.label_dim, device=accelerator.device)
                feats = self.fake_unet.extract_multi_scale_features(x_t, t0, y0, self.gan_head_layers)

            heads = nn.ModuleDict()
            for name, feat in feats.items():
                ch = feat.shape[1]
                if self.gan_head_type == 'patch':
                    # produce H×W logits (will be avg pooled later)
                    heads[name] = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)
                else:  # 'global'
                    # produce scalar logit per sample via GAP
                    heads[name] = nn.Sequential(
                        nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0),
                        nn.AdaptiveAvgPool2d(1),
                    )
            self.multi_heads = heads.to(accelerator.device)
            self.multi_heads.requires_grad_(True)
            # -------------------------------------------------------

        self.num_train_timesteps = args.num_train_timesteps  
        # small sigma first, large sigma later
        karras_sigmas = torch.flip(
            get_sigmas_karras(self.num_train_timesteps, sigma_max=self.sigma_max, sigma_min=self.sigma_min, 
                rho=self.rho
            ),
            dims=[0]
        )    
        self.register_buffer("karras_sigmas", karras_sigmas)

        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)
        # del temp_edm

    
    def compute_distribution_matching_loss(
        self, 
        latents,
        labels
    ):
        original_latents = latents 
        batch_size = latents.shape[0]

        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step, 
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size, 1, 1, 1], 
                device=latents.device,
                dtype=torch.long
            )

            noise = torch.randn_like(latents)

            timestep_sigma = self.karras_sigmas[timesteps]
            
            noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

            pred_real_image = self.real_unet(noisy_latents, timestep_sigma, labels)

            pred_fake_image = self.fake_unet(
                noisy_latents, timestep_sigma, labels
            )

            p_real = (latents - pred_real_image) 
            p_fake = (latents - pred_fake_image) 

            weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)    
            grad = (p_real - p_fake) / weight_factor
                
            grad = torch.nan_to_num(grad) 

        # this loss gives the grad as gradient through autodiff, following https://github.com/ashawkey/stable-dreamfusion 
        loss = 0.5 * F.mse_loss(original_latents, (original_latents-grad).detach(), reduction="mean")         

        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_pred_real_image": pred_real_image.detach(),
            "dmtrain_pred_fake_image": pred_fake_image.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
        }
        return loss_dict, dm_log_dict

    def compute_loss_fake(
        self,
        latents,
        labels,
    ):
        batch_size = latents.shape[0]

        latents = latents.detach() # no gradient to generator 
    
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1], 
            device=latents.device,
            dtype=torch.long
        )
        timestep_sigma = self.karras_sigmas[timesteps]
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

        fake_x0_pred = self.fake_unet(
            noisy_latents, timestep_sigma, labels
        )

        snrs = timestep_sigma**-2

        # weight_schedule karras 
        weights = snrs + 1.0 / self.sigma_data**2

        target = latents 

        loss_fake = torch.mean(
            weights * (fake_x0_pred - target)**2
        )

        loss_dict = {
            "loss_fake_mean": loss_fake
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach()
        }
        return loss_dict, fake_log_dict

    def compute_cls_logits(self, image, label):
        if self.diffusion_gan:
            timesteps = torch.randint(
                0, self.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long
            )
            timestep_sigma = self.karras_sigmas[timesteps]
            image = image + timestep_sigma.reshape(-1, 1, 1, 1) * torch.randn_like(image)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)
            timestep_sigma = self.karras_sigmas[timesteps]

        rep = self.fake_unet(
            image, timestep_sigma, label, return_bottleneck=True
        ).float() 

        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits

    # ---------- feature extraction hooks (for multi-head GAN) ----------
    def _extract_head_features(self, image, label):
        # optional diffusion noise for GAN
        if self.diffusion_gan:
            timesteps = torch.randint(0, self.diffusion_gan_max_timestep, [image.shape[0]],
                                    device=image.device, dtype=torch.long)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)
        timestep_sigma = self.karras_sigmas[timesteps]
        x = image + timestep_sigma.view(-1,1,1,1) * torch.randn_like(image)

        # EDM → DDPM mapping for UNet hooks
        from main.dhariwal.dhariwal_network import _map_sigma_to_t, _onehot_to_class_index
        cfac = 1.0 / torch.sqrt(1.0 + timestep_sigma.view(-1,1,1,1)**2)
        x_t = cfac * x
        t = _map_sigma_to_t(timestep_sigma, self.fake_unet.alphas_cumprod)
        y = _onehot_to_class_index(label)
        feats = self.fake_unet.extract_multi_scale_features(x_t, t, y, self.gan_head_layers)
        return feats

    
    def _critic_score(self, image, label):
        """Unified scalar critic score per sample, works for single-head and multi-head."""
        if self.gan_classifier and self.gan_multihead and getattr(self, 'multi_heads', None) is not None:
            feats = self._extract_head_features(image, label)
            outs = []
            for name, head in self.multi_heads.items():
                out = head(feats[name])      # [B,1,H,W] or [B,1,1,1]
                out = _avg_spatial(out)      # [B,1]
                outs.append(out)
            score = torch.stack(outs, dim=0).mean(dim=0)  # [B,1]
            return score.view(score.size(0))              # [B]
        else:
            score = self.compute_cls_logits(image, label)  # [B,1]
            return score.view(score.size(0))               # [B]


    def _wgan_gradient_penalty(self, real_img, fake_img, label):
        """WGAN-GP (Gulrajani et al.) on interpolates between real and fake."""
        if self.wgan_gp_lambda <= 0.0:
            return torch.tensor(0.0, device=real_img.device, dtype=real_img.dtype)

        batch_size = real_img.size(0)
        eps = torch.rand(batch_size, 1, 1, 1, device=real_img.device, dtype=real_img.dtype)
        interp = eps * real_img + (1 - eps) * fake_img
        interp.requires_grad_(True)

        # critic score per sample
        scores = self._critic_score(interp, label)  # [B]
        grad_outputs = torch.ones_like(scores)

        grads = torch.autograd.grad(
            outputs=scores, inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # [B, C, H, W]
        grads = grads.view(grads.size(0), -1)
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean() * self.wgan_gp_lambda
        return gp


    def compute_generator_clean_cls_loss(self, fake_image, fake_labels):
        # Get unified critic score for fake
        scores_fake = self._critic_score(fake_image, fake_labels)  # [B]

        if self.gan_adv_loss == 'wgan':
            g_loss = (-scores_fake).mean()
            return {"gen_cls_loss": g_loss}

        if self.gan_adv_loss == 'hinge':
            g_loss = (-scores_fake).mean()
            return {"gen_cls_loss": g_loss}

        # 'bce'
        g_loss = F.binary_cross_entropy_with_logits(
            scores_fake.view(-1, 1), torch.ones_like(scores_fake.view(-1, 1))
        )
        return {"gen_cls_loss": g_loss}


    def compute_guidance_clean_cls_loss(self, real_image, fake_image, real_label, fake_label):
        # Get unified critic scores
        scores_real = self._critic_score(real_image.detach(), real_label)  # [B]
        scores_fake = self._critic_score(fake_image.detach(), fake_label)  # [B]

        if self.gan_adv_loss == 'wgan':
            d_loss = (scores_fake - scores_real).mean()
            gp = self._wgan_gradient_penalty(real_image.detach(), fake_image.detach(), real_label)
            d_loss = d_loss + gp
            log_dict = {
                "critic_real": scores_real.detach(),
                "critic_fake": scores_fake.detach(),
                "wgan_gp": torch.as_tensor(gp).detach()
            }
            return {"guidance_cls_loss": d_loss}, log_dict

        # hinge / bce use the generic helper on raw scores
        d_loss, _ = _gan_losses(scores_real.view(-1, 1), scores_fake.view(-1, 1), mode=self.gan_adv_loss)
        log_dict = {
            # for monitoring, you can keep sigmoid’d versions if you like
            "pred_realism_on_real": torch.sigmoid(scores_real.view(-1, 1)).squeeze(1).detach(),
            "pred_realism_on_fake": torch.sigmoid(scores_fake.view(-1, 1)).squeeze(1).detach(),
        }
        return {"guidance_cls_loss": d_loss}, log_dict



    def generator_forward(
        self,
        image,
        labels
    ):
        loss_dict = {} 
        log_dict = {}

        # image.requires_grad_(True)
        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(image, labels)

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        if self.gan_classifier:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(image, labels)
            loss_dict.update(clean_cls_loss_dict)

        # loss_dm = loss_dict["loss_dm"]
        # gen_cls_loss = loss_dict["gen_cls_loss"]

        # grad_dm = torch.autograd.grad(loss_dm, image, retain_graph=True)[0]
        # grad_cls = torch.autograd.grad(gen_cls_loss, image, retain_graph=True)[0]

        # print(f"dm {grad_dm.abs().mean()} cls {grad_cls.abs().mean()}")

        return loss_dict, log_dict 

    def guidance_forward(
        self,
        image,
        labels,
        real_train_dict=None
    ):
        fake_dict, fake_log_dict = self.compute_loss_fake(
            image, labels
        )

        loss_dict = fake_dict 
        log_dict = fake_log_dict

        if self.gan_classifier:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['real_image'], 
                fake_image=image,
                real_label=real_train_dict['real_label'],
                fake_label=labels
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)
        return loss_dict, log_dict 

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None
    ):          
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict['image'],
                labels=generator_data_dict['label']
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict['image'],
                labels=guidance_data_dict['label'],
                real_train_dict=guidance_data_dict['real_train_dict']
            ) 
        else:
            raise NotImplementedError 

        return loss_dict, log_dict 