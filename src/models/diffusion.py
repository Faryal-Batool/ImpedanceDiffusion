# Module: DDPM mask-planning model and sampling logic.

import torch
from torch import nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from src.models.backbones.unet import ConditionalUnet1D   # 2D UNet over (B,C,H,W)
from src.utils.configs import DataDict, DiffusionModelType

# DiPPeR-style conditioning blocks
from src.models.perception import DiPPERImageEncoder, DiPPERStartGoalEncoder


# Class: DDPM trajectory-mask planner with training and sampling routines.
class Diffusion(nn.Module):
    """
    DDPM over 2D masks (start / goal / trajectory) with traversability-based conditioning.

    - Main visual input: traversability map `trav` of shape (B,1,H,W)
    - Diffused variable:  mask `mask_gt` of shape (B,3,H,W)
        ch0 = start mask
        ch1 = goal mask
        ch2 = trajectory mask

    Conditioning:
        trav + start_px + end_px → global_cond (zd) via image + coordinate encoders.

    Training (prediction_type="sample"):
        - x_0 = mask_gt (pixel-space trajectory mask)
        - Add noise with scheduler.add_noise(x_0, ε, t)
        - UNet predicts x_0 (clean mask) directly
        - Loss compares x0_pred vs mask_gt in pixel space

    Sampling:
        - Start/goal pixels are treated as known and inpainted at each step.
        - The sampled output is a mask in pixel space (B,3,H,W).
    """

    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, cfg, activation_func=nn.Softsign):
        super(Diffusion, self).__init__()

        self.model_type = cfg.model_type
        self.use_all_paths = cfg.use_all_paths
        self.sample_times = cfg.sample_times
        self.num_train_timesteps = 100

        # ------------------------------------------------------------------
        # DDPM scheduler: x0-prediction ("sample")
        # ------------------------------------------------------------------
        self.noise_scheduler = DDPMScheduler(
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            prediction_type="sample",       # <--- model predicts x_0 (clean mask)
            num_train_timesteps=self.num_train_timesteps,
            clip_sample_range=cfg.clip_sample_range,
            clip_sample=cfg.clip_sample,
            beta_schedule=cfg.beta_schedule,
            variance_type=cfg.variance_type,
        )
        self.time_steps = self.num_train_timesteps

        self.use_traversability = cfg.use_traversability
        self.estimate_traversability = cfg.estimate_traversability
        self.traversable_steps = cfg.traversable_steps

        self.use_goal = True
        self.add_heatmaps = False

        img_feat = getattr(cfg, "img_feat", 64)
        sg_dim   = getattr(cfg, "sg_dim", 128)

        self.zd = cfg.diffusion_zd      # global condition dimension
        self.mask_channels = 3          # start / goal / traj

        # ------------------------------------------------------------
        # 1) Traversability encoder → ResNet-18-like (DiPPeR style)
        # ------------------------------------------------------------
        self.img_enc = DiPPERImageEncoder(out_dim=img_feat)

        # ------------------------------------------------------------
        # 2) Start/Goal encoders (we normalize to [0,1] inside)
        # ------------------------------------------------------------
        self.start_enc = DiPPERStartGoalEncoder(in_dim=2, out_dim=sg_dim)
        self.goal_enc  = DiPPERStartGoalEncoder(in_dim=2, out_dim=sg_dim)

        # ------------------------------------------------------------
        # 3) Fuse [img_feat + sg_dim(start) + sg_dim(goal)] → zd
        # ------------------------------------------------------------
        Act = (lambda: activation_func()) if activation_func is not None else (lambda: nn.LeakyReLU(0.2))

        self.encoder = nn.Sequential(
            nn.Linear(img_feat + 2 * sg_dim, 1024), Act(),
            nn.Linear(1024, 2048),                  Act(),
            nn.Linear(2048, 512),                   Act(),
            nn.Linear(512, self.zd),                Act(),
        )
        self.trajectory_condition = nn.Linear(self.zd, self.zd)

        # ------------------------------------------------------------
        # 4) Diffusion backbone = 2D UNet over masks
        # ------------------------------------------------------------
        if self.model_type == DiffusionModelType.unet:
            self.diff_model = ConditionalUnet1D(
                input_dim=self.mask_channels,       # C = 3 (start/goal/traj)
                global_cond_dim=self.zd,            # global condition vector
                diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                down_dims=cfg.down_dims,
                kernel_size=cfg.kernel_size,
                cond_predict_scale=cfg.cond_predict_scale,
                n_groups=cfg.n_groups,
            )
        elif self.model_type == DiffusionModelType.crnn:
            raise NotImplementedError(
                "RNNDiffusion (crnn) is not supported in the new mask-based setup. "
                "Use DiffusionModelType.unet instead."
            )
        else:
            raise Exception("the diffusion model type is not defined")

    # ------------------------------------------------------------------
    # Helper: pixels → normalized [0,1] for conditioning MLP
    # ------------------------------------------------------------------
    @staticmethod
    # Function: Convert pixel coordinates into normalized [0, 1] coordinates.
    def _px_to_norm(xy_px: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert pixel coordinates (x_px,y_px) in [0,W-1]x[0,H-1]
        to normalized coordinates (x_norm,y_norm) in [0,1]^2.

        xy_px: (B,2) or (2,)
        NOTE: This is ONLY for conditioning encoders.
            The mask itself stays on the pixel grid.
        """
        if xy_px.dim() == 1:
            xy_px = xy_px[None, :]  # (1,2)

        x_px = xy_px[..., 0].float()
        y_px = xy_px[..., 1].float()

        x_norm = x_px / float(max(W - 1, 1))
        y_norm = y_px / float(max(H - 1, 1))

        xy_norm = torch.stack([x_norm, y_norm], dim=-1)
        return torch.clamp(xy_norm, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Conditioning builder: TRAV + start_px + end_px → global condition
    # ------------------------------------------------------------------
    # Function: Build the global DDPM conditioning vector from image and endpoint coordinates.
    def _build_condition_px(self, rgb, start_px, end_px):
        """
        Global conditioning from:
            - trav      : [B,1,H,W]
            - start_px  : [B,2] (x_px,y_px)
            - end_px    : [B,2] (x_px,y_px)

        We internally normalize start_px/end_px to [0,1]^2 for the MLPs,
        but the *mask* remains in pixel grid.

        Returns:
            h_condition: [B, zd] global conditioning vector.
        """
        B, _, H, W = rgb.shape

        # Ensure coords are batch-aligned and on same device
        start_px = start_px.to(rgb.device).float()
        end_px   = end_px.to(rgb.device).float()

        if start_px.dim() == 1:
            start_px = start_px.unsqueeze(0)
        if end_px.dim() == 1:
            end_px = end_px.unsqueeze(0)

        if start_px.size(0) != B or end_px.size(0) != B:
            raise ValueError(
                f"_build_condition_px: batch mismatch, trav B={B}, "
                f"start_px B={start_px.size(0)}, end_px B={end_px.size(0)}"
            )

        # 1) Traversability embedding
        img_g = self.img_enc(rgb)                      # [B, img_feat]

        # 2) Start & goal embeddings (in normalized [0,1]^2 space)
        start_norm = self._px_to_norm(start_px, H, W)   # (B,2)
        end_norm   = self._px_to_norm(end_px,   H, W)   # (B,2)

        start_g = self.start_enc(start_norm)            # [B, sg_dim]
        goal_g  = self.goal_enc(end_norm)               # [B, sg_dim]

        # 3) Fuse
        fused = torch.cat([img_g, start_g, goal_g], dim=1)  # [B, img_feat + 2*sg_dim]
        h = self.encoder(fused)                             # [B, zd]
        h_condition = self.trajectory_condition(h)          # [B, zd]

        return h_condition

    # ------------------------------------------------------------------
    # Noise helpers for masks
    # ------------------------------------------------------------------
    # Function: Sample Gaussian noise matching the DDPM mask tensor shape.
    def _add_mask_noise(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Sample Gaussian noise ε for a mask x_0.

        mask: (B,3,H,W)
        returns: noise with same shape, ~ N(0, I)
        """
        return torch.randn(mask.shape, device=mask.device)

    # Function: Sample diffusion time steps for a training batch.
    def _sample_timesteps(self, x: torch.Tensor, traversable_steps=None) -> torch.Tensor:
        """
        Sample diffusion time steps for each batch element.

        If use_traversability and traversable_steps is provided, we restrict
        time steps to that range. Otherwise we sample from [0, T).
        """
        if self.use_traversability and traversable_steps is not None:
            time_steps = traversable_steps
        else:
            time_steps = self.time_steps

        time_step = torch.randint(
            0, time_steps, (x.shape[0],),
            device=x.device
        ).long()
        return time_step

    # Function: Add scheduler noise to clean ground-truth masks.
    def add_mask_step_noise(self, x_0: torch.Tensor, traversable_step=None):
        """
        Adds DDPM noise to the GT mask (and optionally a second branch).

        Input:
            x_0:              (B,3,H,W) ground-truth mask (x_0 in pixel space)
            traversable_step: int or None

        Returns:
            noisy_x : (B or 2B,3,H,W)
            noise   : (B or 2B,3,H,W)
            time_step: (B or 2B,)
        """
        noise = self._add_mask_noise(x_0)
        time_step = self._sample_timesteps(x_0)

        noisy_x = self.noise_scheduler.add_noise(
            original_samples=x_0,
            noise=noise,
            timesteps=time_step
        )

        if self.use_traversability:
            x_0b = x_0.clone()
            noise_b = self._add_mask_noise(x_0b)

            if traversable_step is None:
                traversable_step = self.traversable_steps

            time_step_b = self._sample_timesteps(x_0b, traversable_steps=traversable_step)

            noisy_x_b = self.noise_scheduler.add_noise(
                original_samples=x_0b,
                noise=noise_b,
                timesteps=time_step_b
            )

            noise = torch.cat((noise, noise_b), dim=0)
            time_step = torch.cat((time_step, time_step_b), dim=0)
            noisy_x = torch.cat((noisy_x, noisy_x_b), dim=0)

        return noisy_x, noise, time_step

    # ------------------------------------------------------------------
    # TRAINING FORWARD: trav + start_px + end_px → x0 (mask) prediction
    # ------------------------------------------------------------------
    # Function: Run the module forward pass for training or encoding.
    def forward(self,
                rgb,
                # trav,
                mask_gt,
                start_px,
                end_px,
                traversable_step=None,
                occ_map=None,
                start_norm=None,
                end_norm=None):
        """
        Training forward.

        Inputs:
            trav    : [B,1,H,W] traversability in [0,1], 1=free
            mask_gt : [B,3,H,W] ground-truth mask in pixel space (x_0)
            start_px: [B,2] or (2,), pixel coords (x_px,y_px)
            end_px  : [B,2] or (2,), pixel coords (x_px,y_px)

        Returns:
            {
            DataDict.prediction: x0_pred  (B or 2B,3,H,W)  # model's x_0 estimate
            DataDict.noise     : noise    (B or 2B,3,H,W)  # ε used to build x_t
            DataDict.time_steps: time_step(B or 2B,)       # t for each sample
            }
        """
        assert mask_gt is not None, "Diffusion.forward: mask_gt is required."
        # assert trav is not None,    "Diffusion.forward: trav is required."
        assert start_px is not None and end_px is not None, \
            "Diffusion.forward: start_px and end_px are required."
        rgb = rgb.to(mask_gt.device)
        # trav = trav.to(mask_gt.device)
        start_px = start_px.to(mask_gt.device)
        end_px   = end_px.to(mask_gt.device)

        B, C, H, W = mask_gt.shape
        assert C == self.mask_channels, f"mask_gt should have {self.mask_channels} channels, got {C}"

        # 1) Build global conditioning from trav + start_px + end_px (coords normalized inside)
        h_condition = self._build_condition_px(rgb, start_px, end_px)  # [B, zd]

        # 2) Add noise to GT mask x_0
        noisy_mask, noise, time_step = self.add_mask_step_noise(
            x_0=mask_gt,
            traversable_step=traversable_step
        )

        # 3) If using traversability (2B case), duplicate condition to match batch
        global_cond = h_condition
        if self.use_traversability:
            global_cond = torch.cat([h_condition, h_condition], dim=0)

        # 4) Diffusion UNet predicts x_0 (clean mask) directly (prediction_type="sample")
        x0_pred = self.diff_model(
            noisy_mask,
            time_step,
            local_cond=None,
            global_cond=global_cond
        )

        # ---- SAFETY: ensure prediction spatial size matches GT mask ----
        if x0_pred.shape[-2:] != mask_gt.shape[-2:]:
            # Resize predicted mask to GT resolution (e.g. 128x128 -> 64x64)
            x0_pred = F.interpolate(
                x0_pred,
                size=mask_gt.shape[-2:],   # (H, W) from GT
                mode="bilinear",
                align_corners=False)

        return {
            DataDict.prediction: x0_pred,       # x_0 prediction, (B or 2B,3,H,W)
            DataDict.noise: noise,              # Gaussian ε used to build x_t (optional for loss)
            DataDict.time_steps: time_step,
        }

    # ------------------------------------------------------------------
    # SAMPLING: trav + start_px + end_px → sampled mask (with inpainting)
    # ------------------------------------------------------------------
    @torch.no_grad()
    # Function: Run DDPM reverse sampling to produce a trajectory mask.
    def sample(self,
            rgb,
            # trav,
            start_px,
            end_px,
            occ_map=None,
            start_norm=None,
            end_norm=None):
        """
        Inference / sampling in mask space with inpainting of start/goal pixels.

        Inputs:
            trav    : [B,1,H,W]
            start_px: [B,2] or (2,), pixel coords (x_px,y_px)
            end_px  : [B,2] or (2,), pixel coords (x_px,y_px)

        Returns:
            {
            DataDict.prediction     : mask_sample   (B,3,H,W) sampled mask in pixel space
            DataDict.all_trajectories: list of x_0 predictions (if use_all_paths=True)
            }
        """
        assert rgb is not None, "Diffusion.sample: 'trav' is required."
        assert start_px is not None and end_px is not None, \
            "Diffusion.sample: start_px and end_px are required."

        B, _, H, W = rgb.shape
        device = rgb.device

        start_px = start_px.to(device)
        end_px   = end_px.to(device)

        # ------------------------------------------------------------------
        # 1) Build global conditioning once (coords normalized inside)
        # ------------------------------------------------------------------
        h_condition = self._build_condition_px(rgb, start_px, end_px)  # [B, zd]

        # ------------------------------------------------------------------
        # 2) Initial noisy mask ~ N(0, I)
        # ------------------------------------------------------------------
        mask_sample = torch.randn(
            size=(B, self.mask_channels, H, W),
            dtype=h_condition.dtype,
            device=device,
            generator=None,
        )

        # ------------------------------------------------------------------
        # 3) Build "known" mask and inpainting mask for start/goal pixels
        # ------------------------------------------------------------------
        known_mask_x0 = torch.zeros_like(mask_sample)  # (B,3,H,W)

        # Helper: pixel coords -> clamped indices
        # Function: Clamp pixel coordinates into valid tensor indices.
        def px_to_indices(xy_px):
            if xy_px.dim() == 1:
                xy_px = xy_px[None, :]
            x = xy_px[..., 0].long()
            y = xy_px[..., 1].long()
            x = torch.clamp(x, 0, W - 1)
            y = torch.clamp(y, 0, H - 1)
            return x, y

        sx, sy = px_to_indices(start_px)   # (B,), (B,)
        gx, gy = px_to_indices(end_px)     # (B,), (B,)

        # Fix x_0 values for start and goal pixels in ch0 and ch1
        for b in range(B):
            known_mask_x0[b, 0, sy[b], sx[b]] = 1.0  # start channel
            known_mask_x0[b, 1, gy[b], gx[b]] = 1.0  # goal channel

        # Inpainting mask: 1 = unknown, 0 = fixed (known pixels)
        inpaint_mask = torch.ones_like(mask_sample)
        for b in range(B):
            inpaint_mask[b, 0, sy[b], sx[b]] = 0.0   # fix start pixel
            inpaint_mask[b, 1, gy[b], gx[b]] = 0.0   # fix goal pixel
        # Traj channel fully unknown → remains all ones

        # noise for known endpoints (for q(x_t | x_0_known))
        noise_known = torch.randn_like(mask_sample)

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.time_steps)
        scheduler.timesteps = scheduler.timesteps.to(device)
        alphas_cumprod = scheduler.alphas_cumprod.to(device)

        all_trajectories = []

        # ------------------------------------------------------------------
        # 4) Reverse diffusion loop with inpainting
        # ------------------------------------------------------------------
        for t in scheduler.timesteps:
            if (self.sample_times >= 0) and (
                t.item() < self.time_steps - 1 - self.sample_times
            ):
                break

            # UNet prediction: x_0 estimate (since prediction_type="sample")
            model_output = self.diff_model(
                mask_sample,
                t.expand(B),
                local_cond=None,
                global_cond=h_condition,
            )
            
            # ---- NEW: ensure UNet output matches mask_sample spatial size ----
            if model_output.shape[-2:] != mask_sample.shape[-2:]:
                model_output = F.interpolate(
                    model_output,
                    size=mask_sample.shape[-2:],   # (H,W) from current noisy sample
                    mode="bilinear",
                    align_corners=False,
                )

            # DDPM scheduler update: x_{t-1} from x_t and x_0
            step = scheduler.step(
                model_output,
                t,
                mask_sample,
                generator=None
            )
            mask_sample = step.prev_sample.contiguous()

            # Inpainting: overwrite known pixels with q(x_t | x_0_known)
            t_idx = int(t.item())
            alpha_bar = alphas_cumprod[t_idx]
            sqrt_alpha_bar = alpha_bar.sqrt().view(1, 1, 1, 1)
            sqrt_one_minus = (1.0 - alpha_bar).sqrt().view(1, 1, 1, 1)

            # q(x_t | x_0_known) = sqrt(alpha_bar)*x_0_known + sqrt(1-alpha_bar)*ε
            known_noisy = sqrt_alpha_bar * known_mask_x0 + sqrt_one_minus * noise_known
            mask_sample = inpaint_mask * mask_sample + (1.0 - inpaint_mask) * known_noisy

            if self.use_all_paths:
                all_trajectories.append(model_output.detach().cpu().numpy())

            if mask_sample.shape[-2:] != rgb.shape[-2:]:
                mask_sample = F.interpolate(
                    mask_sample,
                    size=rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )


        output = {
            DataDict.prediction: mask_sample,         # (B,3,H,W) pixel-space mask
            DataDict.all_trajectories: all_trajectories,
        }
        return output
