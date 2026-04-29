# Module: Image and start-goal conditioning encoders.

import time
import warnings
import torch
from torch import nn
from src.utils.configs import DataDict
import torch.nn as nn
import torchvision.models as models

# Class: Legacy image/lidar encoder used by the older perception branch.
class LidarImageModel(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, input_channel=3, lidar_out_dim=512, norm_layer=True):
        super(LidarImageModel, self).__init__()
        if norm_layer:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([8, 12, 910]),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([16, 8, 453]),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([32, 6, 226]),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 2)), nn.ELU(), nn.LayerNorm([32, 2, 112]),
                nn.Flatten(), nn.Linear(7168, 2048), nn.LeakyReLU(0.2), nn.LayerNorm([2048]),
                nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.LayerNorm([1024]),
                nn.Linear(1024, lidar_out_dim), nn.ELU(), nn.LayerNorm([lidar_out_dim]),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 2)), nn.ELU(),
                nn.Flatten(), nn.Linear(7168, 2048), nn.LeakyReLU(0.2),
                nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, lidar_out_dim), nn.ELU()
            )

    # Function: Run the module forward pass for training or encoding.
    def forward(self, image):
        output = self.conv(image)
        return output


# Class: Legacy perception fusion module for lidar, velocity, and target features.
class Perception(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, cfg):
        super(Perception, self).__init__()
        self.cfg = cfg

        self.lidar_model = LidarImageModel(input_channel=self.cfg.lidar_num, lidar_out_dim=self.cfg.lidar_out,
                                           norm_layer=self.cfg.lidar_norm_layer)

        self.vel_model = nn.Sequential(
            nn.Linear(self.cfg.vel_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, self.cfg.vel_out), nn.LeakyReLU(0.2)
        )

        combo_input_dim = self.cfg.vel_out + self.cfg.lidar_out + 2
        self.combo_layers = nn.Sequential(
            nn.Linear(combo_input_dim, 2 * combo_input_dim), nn.ELU(),
            nn.Linear(2 * combo_input_dim, 2 * combo_input_dim), nn.ELU(),
            nn.Linear(2 * combo_input_dim, combo_input_dim), nn.LeakyReLU(0.2)
        )

    # Function: Run the module forward pass for training or encoding.
    def forward(self, lidar, vel, target):
        lidar_fts = self.lidar_model(lidar)  # B x 512

        VB, VN, VD = vel.size()
        vel_fts = self.vel_model(vel.view(VB, -1))  # B x 256

        observation = torch.concat((lidar_fts, vel_fts, target), dim=1)  # B x 770

        perception = self.combo_layers(observation)
        return perception
    


# -----------------------------------------------------------
# 1) RGB IMAGE ENCODER  (DiPPeR / DiPPeST Style)
# -----------------------------------------------------------

# Function: Create an RGB ResNet-18 backbone for image feature extraction.
def resnet18_rgb(pretrained: bool = True):
    """
    Build a standard ResNet-18 backbone that accepts **3-channel RGB** input.

    If pretrained=True, we use ImageNet weights (recommended).
    """
    try:
        # Newer torchvision API
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older versions
        model = models.resnet18(pretrained=pretrained)
    return model


# Class: RGB image encoder that produces global conditioning features.
class DiPPERImageEncoder(nn.Module):
    """
    Encodes an RGB image (img: [B,3,H,W]) into a global feature vector.

    Default uses an RGB ResNet-18 and then flattens the final
    global pooled representation to a vector of size `out_dim`.
    """
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()

        backbone = resnet18_rgb(pretrained=pretrained)
        # Remove the final FC layer, keep conv+pool -> (B,512,1,1)
        layers = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*layers)  # final: [B,512,1,1]

        self.out_dim = 512
        self.proj = nn.Identity() if out_dim == 512 else nn.Linear(512, out_dim)

    # Function: Run the module forward pass for training or encoding.
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W) RGB image in [0,1] (normalized outside if needed)
        """
        f = self.encoder(img)            # (B,512,1,1)
        g = f.view(f.size(0), -1)        # (B,512)
        return self.proj(g)              # (B,out_dim)


# -----------------------------------------------------------
# 2) START/GOAL EMBEDDING  (DiPPeR Style)
# -----------------------------------------------------------

# Class: MLP encoder for normalized start or goal coordinates.
class DiPPERStartGoalEncoder(nn.Module):
    """
    Embeds (x, y) 2D coordinates into a feature vector.

    INPUT to forward():
        xy: [B,2] with (x_norm, y_norm) in [0,1]^2

    OUTPUT:
        feat: [B,out_dim]
    """
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, in_dim: int = 2, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    # Function: Run the module forward pass for training or encoding.
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: [B,2], (x_norm, y_norm) in [0,1]^2
        """
        return self.mlp(xy)


# -----------------------------------------------------------
# 3) FULL DiPPeR CONDITIONING MODULE (RGB + START + GOAL)
# -----------------------------------------------------------

# Class: Combined RGB and endpoint conditioning module.
class DiPPERConditioning(nn.Module):
    """
    Combines:
        - RGB image embedding             from img:      [B,3,H,W]
        - Start embedding                 from start_px: [B,2] pixel (x_px, y_px)
        - Goal embedding                  from goal_px:  [B,2] pixel (x_px, y_px)

    GLOBAL DESIGN (PIXEL-SPACE):

      - Dataset & model pass start/end as **pixel coordinates** (x_px, y_px).
      - This module internally normalizes them to [0,1]^2 for the MLP,
        using the spatial size of the image.

    Typical usage in Diffusion:
        cond_vec = DiPPERConditioning(img, start_px, end_px)
        eps_pred = unet(sample=x_t, timestep=t, global_cond=cond_vec)
    """
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self,
                 img_dim: int = 512,
                 sg_dim: int = 128,
                 final_dim: int = 512,
                 pretrained_backbone: bool = True):
        super().__init__()
        # Encodes RGB image (B,3,H,W) -> (B,img_dim)
        self.img_enc = DiPPERImageEncoder(out_dim=img_dim,
                                          pretrained=pretrained_backbone)
        # Embeds normalized (x_norm,y_norm) -> (B,sg_dim)
        self.sg_enc  = DiPPERStartGoalEncoder(in_dim=2, out_dim=sg_dim)
        # Final fusion to a single conditioning vector
        self.final_fc = nn.Linear(img_dim + 2 * sg_dim, final_dim)

    # Function: Normalize pixel coordinates using the spatial size of the image tensor.
    def _pixels_to_norm(self,
                        xy_px: torch.Tensor,
                        img: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel coordinates (x_px, y_px) to normalized coords (x_norm, y_norm)
        using the spatial size of 'img'.

        INPUT:
          xy_px: (B,2) with (x_px, y_px) in pixel indices
          img  : (B,C,H,W) to infer (H,W)

        OUTPUT:
          xy_norm: (B,2) with values in approximately [0,1].

        NOTE:
          - x_px is divided by (W - 1)
          - y_px is divided by (H - 1)
          - We clamp to [0,1] to be safe.
        """
        assert xy_px.dim() == 2 and xy_px.size(1) == 2, \
            f"DiPPERConditioning._pixels_to_norm: expected (B,2), got {tuple(xy_px.shape)}"
        B, _, H, W = img.shape

        if xy_px.size(0) != B:
            raise ValueError(
                f"DiPPERConditioning: batch size mismatch between coords ({xy_px.size(0)}) "
                f"and img ({B})"
            )

        device = xy_px.device
        dtype = xy_px.dtype

        scale = torch.tensor([W - 1, H - 1], device=device, dtype=dtype)
        xy_norm = xy_px / scale  # broadcast divide
        xy_norm = torch.clamp(xy_norm, 0.0, 1.0)
        return xy_norm

    # Function: Run the module forward pass for training or encoding.
    def forward(self,
                img: torch.Tensor,
                start_px: torch.Tensor,
                goal_px: torch.Tensor) -> torch.Tensor:
        """
        img     : [B,3,H,W] RGB image
        start_px: [B,2] (x_px, y_px) pixel coordinates
        goal_px : [B,2] (x_px, y_px) pixel coordinates

        Returns:
            cond: [B,final_dim] conditioning vector for the UNet.
        """
        # 1) Global RGB feature
        img_feat = self.img_enc(img)  # [B,img_dim]

        # 2) Convert pixel coords -> normalized [0,1]^2 for MLP
        start_norm = self._pixels_to_norm(start_px, img)  # [B,2]
        goal_norm  = self._pixels_to_norm(goal_px, img)   # [B,2]

        # 3) Encode start / goal normalized coords
        start_feat = self.sg_enc(start_norm)  # [B,sg_dim]
        goal_feat  = self.sg_enc(goal_norm)   # [B,sg_dim]

        # 4) Concatenate all and project to final_dim
        cond = torch.cat([img_feat, start_feat, goal_feat], dim=-1)
        return self.final_fc(cond)           # [B,final_dim]
