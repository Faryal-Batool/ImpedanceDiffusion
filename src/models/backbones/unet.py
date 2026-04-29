# Module: Conditional UNet backbone blocks used by the diffusion model.

import math
from typing import Union
import torch
import torch.nn as nn


# -----------------------------------------------------------
# Time embedding
# -----------------------------------------------------------

# Class: Sinusoidal embedding block for diffusion time steps.
class SinusoidalPosEmb(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    # Function: Run the module forward pass for training or encoding.
    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------
# 2D blocks
# -----------------------------------------------------------

# Class: Convolution, normalization, and activation block for the UNet.
class Conv2dBlock(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, inp_channels: int, out_channels: int,
                 kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    # Function: Run the module forward pass for training or encoding.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Class: Spatial downsampling layer for UNet encoder stages.
class Downsample2d(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    # Function: Run the module forward pass for training or encoding.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Class: Spatial upsampling layer for UNet decoder stages.
class Upsample2d(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    # Function: Run the module forward pass for training or encoding.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Class: Residual convolution block modulated by global conditioning.
class ConditionalResidualBlock2D(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_dim: int,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 cond_predict_scale: bool = False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else
            nn.Identity()
        )

        # Function: Initialize convolution weights inside the residual block.
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.cond_encoder.apply(_init_weights)

    # Function: Run the module forward pass for training or encoding.
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            embed = embed.view(embed.shape[0], self.out_channels, 1, 1)
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# -----------------------------------------------------------
# 2D Conditional UNet for masks (name kept for compatibility)
# -----------------------------------------------------------

# Class: Conditional 2D UNet backbone used for DDPM mask prediction.
class ConditionalUnet1D(nn.Module):
    """
    2D Conditional UNet over (B,C,H,W) masks.

    - Input:  sample      (B, C_in, H, W)
    - Output: prediction  (B, C_in, H, W)
    """

    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self,
                 input_dim: int,
                 local_cond_dim: int = None,
                 global_cond_dim: int = None,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=None,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 cond_predict_scale: bool = False):
        super().__init__()

        if down_dims is None:
            down_dims = [64, 128, 256, 512]

        self.input_dim = input_dim
        self.down_dims = list(down_dims)

        # Encoder channels: [C_in, d0, d1, d2, d3]
        self.enc_channels = [input_dim] + self.down_dims
        # Decoder channels: reverse of down_dims
        self.dec_channels = self.down_dims[::-1]

        # --- timestep encoder ---
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        self.local_cond_encoder = None  # not used in 2D version

        # --- middle blocks ---
        mid_dim = self.dec_channels[0]  # deepest encoder dim
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # --- downsampling path ---
        # We downsample at every level → H / 2 per level
        self.down_modules = nn.ModuleList([])
        num_down = len(self.down_dims)
        for i in range(num_down):
            dim_in = self.enc_channels[i]
            dim_out = self.enc_channels[i + 1]

            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock2D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Downsample2d(dim_out),   # ALWAYS downsample
            ]))

        # --- upsampling path ---
        # We upsample as many times as we downsampled.
        self.up_modules = nn.ModuleList([])
        num_up = len(self.dec_channels)
        for i in range(num_up):
            inC = self.dec_channels[i]
            outC = self.dec_channels[i] if i == num_up - 1 else self.dec_channels[i + 1]

            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    in_channels=inC * 2,
                    out_channels=outC,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock2D(
                    in_channels=outC,
                    out_channels=outC,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Upsample2d(outC),   # ALWAYS upsample
            ]))

        # After final upsample, channels = down_dims[0]
        start_dim = self.down_dims[0]
        self.final_conv = nn.Sequential(
            Conv2dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv2d(start_dim, input_dim, kernel_size=1),
        )

    # Function: Run the module forward pass for training or encoding.
    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond: torch.Tensor = None,
                global_cond: torch.Tensor = None) -> torch.Tensor:
        B = sample.shape[0]

        # --- timestep + global cond ---
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(sample.device)

        timestep = timestep.expand(B)
        global_feature = self.diffusion_step_encoder(timestep)

        if global_cond is not None:
            global_feature = torch.cat((global_feature, global_cond), dim=-1)

        # --- down path ---
        x = sample
        skips = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        # --- middle ---
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # --- up path ---
        for resnet1, resnet2, upsample in self.up_modules:
            skip = skips.pop()
            # safety resize if shapes drift
            if skip.shape[2:] != x.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat((x, skip), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # --- final projection ---
        x = self.final_conv(x)   # (B, C_in, H, W)
        return x
