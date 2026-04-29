# Module: Standalone inference and evaluation utilities for the DDPM planner.

import os
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.arguments import get_configuration
from src.models.model import get_model
from src.data_loader.dataset import TrajDataset
from src.utils.configs import DataDict


# --------------------------------------------------
# 1. Load config
# --------------------------------------------------
cfgs = get_configuration()

# TODO: set your test dataset root (folder with sample_xxxxxx)
# cfgs.data.root = r"/home/isr-lab3/Faryal_Batool/Testing_samples_dog_and_drone"
cfgs.data.root = r"/home/isr-lab3/Faryal_Batool/Testing_samples_top"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# --------------------------------------------------
# 2. Build model and load checkpoint
# --------------------------------------------------
model = get_model(cfgs.model, device=device)
model.to(device)

# TODO: set your trained checkpoint path
ckpt_path = r"/home/isr-lab3/Faryal_Batool/DTG-main_top/results/models/hnav_29.pth"
print(f"[INFO] Loading checkpoint from: {ckpt_path}")
state = torch.load(ckpt_path, map_location=device)

if "state_dict" in state:
    model.load_state_dict(state["state_dict"], strict=False)
else:
    model.load_state_dict(state, strict=False)

model.eval()

# --------------------------------------------------
# 3. Build test dataset (64×64 masks/trav)
# --------------------------------------------------
n_points = getattr(cfgs.data, "n_points", 128)  # still used by TrajDataset for resampling, but not by diffusion
test_dataset = TrajDataset(root=cfgs.data.root, n_points=n_points)

print(f"[INFO] Test samples: {len(test_dataset)}")

# --------------------------------------------------
# 4. Output folders
# --------------------------------------------------
output_root = Path("test_results_masks_64")
overlay_rgb_dir = output_root / "overlays_rgb"
overlay_trav_dir = output_root / "overlays_trav"
output_root.mkdir(parents=True, exist_ok=True)
overlay_rgb_dir.mkdir(parents=True, exist_ok=True)
overlay_trav_dir.mkdir(parents=True, exist_ok=True)

metrics_csv_path = output_root / "metrics_masks.csv"

# --------------------------------------------------
# 5. Simple mask metrics: per-channel MSE & IoU
# --------------------------------------------------
# Function: Compute mean-squared error between two mask tensors.
def compute_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a, b: (H,W) or (1,H,W) tensors in [0,1].
    """
    return ((a - b) ** 2).mean().item()


# Function: Compute thresholded intersection-over-union between two mask tensors.
def compute_iou(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Intersection-over-Union for binary masks.

    a, b: (H,W) or (1,H,W) tensors (can be soft in [0,1]).
    """
    if a.dim() == 3:
        a = a[0]
    if b.dim() == 3:
        b = b[0]

    a_bin = (a > threshold).float()
    b_bin = (b > threshold).float()

    intersection = (a_bin * b_bin).sum()
    union = (a_bin + b_bin).clamp(max=1.0).sum()

    if union.item() == 0.0:
        return 1.0 if intersection.item() == 0.0 else 0.0
    return (intersection / union).item()


# --------------------------------------------------
# 6. Evaluation loop (mask-based)
# --------------------------------------------------
all_mse_start = []
all_mse_goal = []
all_mse_traj = []

all_iou_start = []
all_iou_goal = []
all_iou_traj = []

with open(metrics_csv_path, mode="w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow([
        "idx",
        "sample_id",
        "mse_start",
        "mse_goal",
        "mse_traj",
        "iou_start",
        "iou_goal",
        "iou_traj",
    ])

    for idx in range(len(test_dataset)):
        # -------- Load sample from dataset --------
        sample_dir: Path = test_dataset.samples[idx]
        sample_id = sample_dir.name  # e.g., "sample_000001"

        sample = test_dataset[idx]

        rgb = sample["rgb"].unsqueeze(0).to(device)         # (1,3,H_rgb,W_rgb) - for visualization only
        trav = sample["trav"].unsqueeze(0).to(device)       # (1,1,H,W)         - main model input
        mask_gt = sample["mask_gt"].unsqueeze(0).to(device) # (1,3,H,W)

        start_px = sample["start_px"].unsqueeze(0).to(device)  # (1,2) (x_px,y_px)
        end_px   = sample["end_px"].unsqueeze(0).to(device)    # (1,2)

        _, _, Hm, Wm = mask_gt.shape  # GT mask resolution (should be 64×64)

        # -------- Build model input dict (trav + start_px + end_px) --------
        input_dict = {
            DataDict.camera: rgb,
            "rgb": rgb,          # (1,3,H,W), optional, not used by diffusion branch
            "trav": trav,          # (1,1,H,W)
            "start_px": start_px,  # (1,2)
            "end_px": end_px,      # (1,2)
            # "rgb" can be added but diffusion branch doesn't use it; only for visualization
        }

        with torch.no_grad():
            out = model(input_dict, sample=True)

        # Predicted mask from diffusion
        mask_pred = out[DataDict.prediction]  # (1,3,H,W) in pixel space
        if mask_pred.shape[0] > 1:
            # If for some reason you returned a bigger batch, use the first.
            mask_pred = mask_pred[:1]

        # Make sure spatial resolution matches GT (safety, should already be Hm×Wm)
        if mask_pred.shape[-2:] != (Hm, Wm):
            mask_pred = F.interpolate(
                mask_pred,
                size=(Hm, Wm),
                mode="bilinear",
                align_corners=False,
            )

        # Clamp to [0,1] range for metric stability
        mask_pred = torch.clamp(mask_pred, 0.0, 1.0)

        # -------------------------------------------------
        # 7. Compute per-channel MSE & IoU
        # -------------------------------------------------
        pred_start = mask_pred[:, 0:1, ...]  # (1,1,H,W)
        pred_goal  = mask_pred[:, 1:2, ...]
        pred_traj  = mask_pred[:, 2:3, ...]

        gt_start   = mask_gt[:, 0:1, ...]
        gt_goal    = mask_gt[:, 1:2, ...]
        gt_traj    = mask_gt[:, 2:3, ...]

        mse_start = compute_mse(pred_start, gt_start)
        mse_goal  = compute_mse(pred_goal,  gt_goal)
        mse_traj  = compute_mse(pred_traj,  gt_traj)

        iou_start = compute_iou(pred_start, gt_start)
        iou_goal  = compute_iou(pred_goal,  gt_goal)
        iou_traj  = compute_iou(pred_traj,  gt_traj)

        all_mse_start.append(mse_start)
        all_mse_goal.append(mse_goal)
        all_mse_traj.append(mse_traj)

        all_iou_start.append(iou_start)
        all_iou_goal.append(iou_goal)
        all_iou_traj.append(iou_traj)

        writer.writerow([
            idx,
            sample_id,
            f"{mse_start:.6f}",
            f"{mse_goal:.6f}",
            f"{mse_traj:.6f}",
            f"{iou_start:.6f}",
            f"{iou_goal:.6f}",
            f"{iou_traj:.6f}",
        ])

        # -------------------------------------------------
        # 8. Visualization: overlay masks on RGB & trav_map
        # -------------------------------------------------
        # Target visualization resolution
        VIS_H, VIS_W = 512, 512

        # ---- RGB base image ----
        # rgb: (1, 3, H_rgb, W_rgb) in [0,1]
        rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()  # (H_rgb, W_rgb, 3)

        # If RGB is not 512×512 for some reason, resize it for visualization
        if rgb_np.shape[0] != VIS_H or rgb_np.shape[1] != VIS_W:
            rgb_img = (rgb_np * 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb_img).resize((VIS_W, VIS_H), Image.BILINEAR)
            rgb_vis = np.array(rgb_img).astype(np.float32) / 255.0
        else:
            rgb_vis = rgb_np  # already 512×512

        # ---- Traversability map base image (from dataset tensor) ----
        # trav: (1,1,Hm,Wm) typically 64×64
        trav_vis = F.interpolate(
            trav, size=(VIS_H, VIS_W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()  # (VIS_H, VIS_W)

        # ---- Upsample predicted & GT masks to 512×512 for visualization ----
        mask_pred_vis = F.interpolate(
            mask_pred,
            size=(VIS_H, VIS_W),
            mode="bilinear",
            align_corners=False,
        )[0].cpu().numpy()  # (3, VIS_H, VIS_W)

        mask_gt_vis = F.interpolate(
            mask_gt,
            size=(VIS_H, VIS_W),
            mode="nearest",  # GT is binary -> nearest is better
            # align_corners=False,
        )[0].cpu().numpy()  # (3, VIS_H, VIS_W)

        # Trajectory channel for visualization (512×512)
        traj_pred_vis = mask_pred_vis[2]  # (VIS_H, VIS_W)
        traj_gt_vis   = mask_gt_vis[2]

        # 1) RGB overlay (512×512)
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb_vis)
        plt.imshow(traj_gt_vis, cmap="Greens", alpha=0.5, vmin=0.0, vmax=1.0)
        plt.imshow(traj_pred_vis, cmap="Reds",   alpha=0.5, vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.title(
            f"{sample_id} | Traj mask (512×512)\n"
            f"MSE_traj={mse_traj:.4f}, IoU_traj={iou_traj:.4f}"
        )
        rgb_out_path = overlay_rgb_dir / f"{sample_id}_rgb.png"
        plt.tight_layout()
        plt.savefig(rgb_out_path, dpi=150)
        plt.close()

        # 2) Traversability map overlay (512×512)
        plt.figure(figsize=(6, 6))
        plt.imshow(trav_vis, cmap="gray")
        plt.imshow(traj_gt_vis, cmap="Greens", alpha=0.5, vmin=0.0, vmax=1.0)
        plt.imshow(traj_pred_vis, cmap="Reds",   alpha=0.5, vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.title(
            f"{sample_id} | Traj on trav_map (512×512)\n"
            f"MSE_traj={mse_traj:.4f}, IoU_traj={iou_traj:.4f}"
        )
        trav_out_path = overlay_trav_dir / f"{sample_id}_trav.png"
        plt.tight_layout()
        plt.savefig(trav_out_path, dpi=150)
        plt.close()

# --------------------------------------------------
# 7. Aggregate metrics & save summary plots
# --------------------------------------------------
all_mse_start = np.array(all_mse_start, dtype=float)
all_mse_goal  = np.array(all_mse_goal,  dtype=float)
all_mse_traj  = np.array(all_mse_traj,  dtype=float)

all_iou_start = np.array(all_iou_start, dtype=float)
all_iou_goal  = np.array(all_iou_goal,  dtype=float)
all_iou_traj  = np.array(all_iou_traj,  dtype=float)

print("\n=== Overall test performance (mask-based, 64×64) ===")
print(f"Mean MSE  (start) : {all_mse_start.mean():.6f}")
print(f"Mean MSE  (goal)  : {all_mse_goal.mean():.6f}")
print(f"Mean MSE  (traj)  : {all_mse_traj.mean():.6f}")
print(f"Mean IoU  (start) : {all_iou_start.mean():.6f}")
print(f"Mean IoU  (goal)  : {all_iou_goal.mean():.6f}")
print(f"Mean IoU  (traj)  : {all_iou_traj.mean():.6f}")

# Simple per-sample index plots for trajectory channel
idxs = np.arange(len(all_mse_traj))

plt.figure(figsize=(8, 4))
plt.plot(idxs, all_mse_traj, marker="o", linewidth=1)
plt.xlabel("Sample index")
plt.ylabel("MSE (traj channel)")
plt.title("Trajectory Mask MSE per Sample")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_root / "metric_traj_mse.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(idxs, all_iou_traj, marker="o", linewidth=1)
plt.xlabel("Sample index")
plt.ylabel("IoU (traj channel)")
plt.title("Trajectory Mask IoU per Sample")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_root / "metric_traj_iou.png", dpi=150)
plt.close()
