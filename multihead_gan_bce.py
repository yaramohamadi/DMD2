# multihead_gan_bce.py
# Minimal multi-head GAN critic with global heads + BCE losses.
# Assumes a guided-diffusion style UNet with .input_blocks (list) and .middle_block (module).

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGlobalBCEGan(nn.Module):
    """
    Multi-head GAN critic for diffusion UNet features.
    - Attaches one tiny 'global' head per selected feature: Conv1x1(C->1) + GAP -> scalar logit per head.
    - Aggregates by averaging across heads -> single scalar logit per image.
    - Provides BCE losses for D and G.
    - Hooks: all encoder input blocks + middle block ("all").
    """

    def __init__(self, unet: nn.Module, label_conditioned: bool = True, label_dim: Optional[int] = None):
        """
        Args:
            unet: your diffusion UNet (must expose .input_blocks (list) and .middle_block)
            label_conditioned: if True, forward expects 'y' labels; if False, y=None
            label_dim: unused here; kept for compatibility if you need to embed labels elsewhere
        """
        super().__init__()
        self.unet = unet
        self.label_conditioned = label_conditioned
        self.label_dim = label_dim

        # Heads are created lazily after a first feature pass so we know channels.
        self.multi_heads = nn.ModuleDict()
        self._hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._feat_buf: Dict[str, torch.Tensor] = {}

        # BCE criterion (with logits). If you want label smoothing, change targets below.
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    # -------- Feature hooks (ALL encoders + MID) --------

    def _resolve_feature_layers(self) -> Dict[str, nn.Module]:
        """
        Collect all encoder input blocks and the middle block.
        Returns: Ordered dict-like mapping names -> modules (e.g., {"in0": mod, ..., "mid": mod})
        """
        layers: Dict[str, nn.Module] = {}
        assert hasattr(self.unet, "input_blocks"), "UNet must have .input_blocks (list of encoder modules)"
        assert hasattr(self.unet, "middle_block"), "UNet must have .middle_block"
        for i, m in enumerate(self.unet.input_blocks):
            layers[f"in{i}"] = m
        layers["mid"] = self.unet.middle_block
        return layers

    def _register_hooks(self, modules: Dict[str, nn.Module]) -> None:
        self._feat_buf = {}
        self._hook_handles = {}

        def make_hook(name: str):
            def _hook(_m, _in, out):
                # Ensure tensor (some blocks return tuples); keep as [B,C,H,W]
                if isinstance(out, (list, tuple)):
                    out = out[0]
                self._feat_buf[name] = out
            return _hook

        for name, mod in modules.items():
            self._hook_handles[name] = mod.register_forward_hook(make_hook(name))

    def _remove_hooks(self) -> None:
        for h in self._hook_handles.values():
            h.remove()
        self._hook_handles.clear()

    @torch.no_grad()
    def _probe_and_build_heads(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> None:
        """
        One forward pass with hooks to infer channels and build global heads.
        """
        modules = self._resolve_feature_layers()
        self._register_hooks(modules)

        # Forward through UNet to populate self._feat_buf
        self._forward_unet_for_hooks(x, t, y)

        self._remove_hooks()

        # Build heads for each captured feature
        heads = {}
        for name, feat in self._feat_buf.items():
            c = feat.shape[1]
            heads[name] = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1, bias=True),
                nn.AdaptiveAvgPool2d(1)  # -> [B,1,1,1]
            )
        self.multi_heads = nn.ModuleDict(heads)

    def _forward_unet_for_hooks(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]):
        """
        Calls your UNet in the standard guided-diffusion signature.
        If your UNet uses a different signature, adapt this function.
        """
        try:
            if self.label_conditioned:
                _ = self.unet(x, t, y)
            else:
                _ = self.unet(x, t)
        except TypeError:
            # Fallbacks for slightly different signatures
            try:
                _ = self.unet(x, t, context=y)
            except TypeError:
                _ = self.unet(x, t)

    # -------- Critic score (global heads -> mean over heads) --------

    def critic_score(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Returns a single scalar logit per sample: [B]
        Steps:
          1) Hook all encoder inputs + mid, run UNet to capture features
          2) For each feature, apply global head (Conv1×1 + GAP) -> [B,1,1,1]
          3) Squeeze & average across heads -> [B]
        """
        # Lazily build heads the first time
        if len(self.multi_heads) == 0:
            with torch.no_grad():
                self._probe_and_build_heads(x, t, y)
                # Move heads to same device/dtype as UNet first params
                p = next(self.unet.parameters())
                self.multi_heads.to(device=p.device, dtype=p.dtype)

        # 1) capture features this pass
        modules = self._resolve_feature_layers()
        self._register_hooks(modules)
        self._forward_unet_for_hooks(x, t, y)
        self._remove_hooks()

        # 2) apply heads
        logits_per_head: List[torch.Tensor] = []
        for name, feat in self._feat_buf.items():
            head = self.multi_heads[name]
            logit = head(feat)          # [B,1,1,1]
            logit = logit.view(logit.size(0))  # [B]
            logits_per_head.append(logit)

        # 3) aggregate
        stacked = torch.stack(logits_per_head, dim=0)  # [H,B]
        score = stacked.mean(dim=0)                    # [B]
        return score

    # -------- BCE losses (no options) --------

    def d_loss(self,
               real_scores: torch.Tensor,
               fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Discriminator BCE-with-logits loss:
          - real -> 1.0
          - fake -> 0.0
        Inputs are raw logits [B].
        """
        target_real = torch.ones_like(real_scores)
        target_fake = torch.zeros_like(fake_scores)
        loss_real = self.bce_logits(real_scores, target_real)
        loss_fake = self.bce_logits(fake_scores, target_fake)
        return loss_real + loss_fake

    def g_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Generator BCE-with-logits loss:
          - fake -> 1.0  (i.e., fool D)
        Input is raw logits [B].
        """
        target = torch.ones_like(fake_scores)
        return self.bce_logits(fake_scores, target)

    # -------- Convenience helpers for training steps --------

    def forward_d(self,
                  real_x: torch.Tensor, fake_x: torch.Tensor,
                  t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute D loss given real/fake batches.
        NOTE: Detach fake_x upstream if G and D share graph in your loop.
        """
        real_scores = self.critic_score(real_x, t, y)  # [B]
        fake_scores = self.critic_score(fake_x, t, y)  # [B]
        return self.d_loss(real_scores, fake_scores)

    def forward_g(self,
                  fake_x: torch.Tensor,
                  t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute G loss given freshly generated fake_x.
        """
        fake_scores = self.critic_score(fake_x, t, y)  # [B]
        return self.g_loss(fake_scores)
