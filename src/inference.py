# Module: Standalone inference and evaluation utilities for the DDPM planner.

import os
import csv
from pathlib import Path

import torch
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
train_poses = False

# TODO: set your test dataset root (folder with sample_xxxxxx)
cfgs.data.root = r"/home/isr-lab3/Faryal_Batool/Testing_samples"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 2. Build model and load checkpoint
# --------------------------------------------------
model = get_model(cfgs.model, device=device)
model.to(device)

# TODO: set your trained checkpoint path
ckpt_path = r"/home/isr-lab3/Faryal_Batool/DTG-main/1/models/hnav_29.pth"
state = torch.load(ckpt_path, map_location=device)

if "state_dict" in state:
    model.load_state_dict(state["state_dict"], strict=False)
else:
    model.load_state_dict(state, strict=False)

model.eval()

# --------------------------------------------------
# 3. Build test dataset
# --------------------------------------------------
n_points = getattr(cfgs.data, "n_points", 128)
test_dataset = TrajDataset(root=cfgs.data.root, n_points=n_points)

# --------------------------------------------------
# 4. Output folders
# --------------------------------------------------
output_root = Path("test_results")
overlay_rgb_dir = output_root / "overlays_rgb"
overlay_occ_dir = output_root / "overlays_occ"
output_root.mkdir(parents=True, exist_ok=True)
overlay_rgb_dir.mkdir(parents=True, exist_ok=True)
overlay_occ_dir.mkdir(parents=True, exist_ok=True)

metrics_csv_path = output_root / "metrics.csv"


# --------------------------------------------------
# 5. Helper: increments -> absolute trajectory
# --------------------------------------------------
# Function: Convert predicted trajectory increments into absolute normalized coordinates.
def increments_to_absolute(start_xy: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
    """
    start_xy: (2,) in [0,1]
    inc:      (N,2) increments in normalized space
    returns:  (N,2) absolute positions in [0,1]
              p[0] = start + inc[0], p[i] = p[i-1] + inc[i]
    """
    abs_traj = torch.zeros_like(inc)
    abs_traj[0] = start_xy + inc[0]
    for i in range(1, inc.size(0)):
        abs_traj[i] = abs_traj[i - 1] + inc[i]
    return abs_traj.clamp(0.0, 1.0)


# --------------------------------------------------
# 6. Evaluation loop
# --------------------------------------------------
all_traj_mse = []
all_endpoint_err = []
all_endpoint_head_err = []

train_poses = getattr(cfgs.loss, "train_poses", True)

with open(metrics_csv_path, mode="w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow([
        "sample_idx",
        "sample_id",
        "traj_mse",
        "endpoint_l2",
        "endpoint_head_l2"  # may be empty if no end_hat
    ])

    for idx in range(len(test_dataset)):
        # -------- Load sample from dataset --------
        sample = test_dataset[idx]
        sample_dir: Path = test_dataset.samples[idx]
        sample_id = sample_dir.name  # e.g., "sample_000001"

        rgb_tensor = sample["rgb"].unsqueeze(0).to(device)   # (1,3,H,W)
        start      = sample["start"].unsqueeze(0).to(device) # (1,2)
        end_gt     = sample["end"].unsqueeze(0).to(device)   # (1,2)
        traj_gt    = sample["traj"].unsqueeze(0).to(device)  # (1,N,2)

        B, C, H, W = rgb_tensor.shape

        # Load original rgb.png and occ_map.png from disk for overlay
        rgb_img_pil = Image.open(sample_dir / "rgb.png").convert("RGB")
        rgb_img = np.asarray(rgb_img_pil, dtype=np.float32) / 255.0  # (H,W,3)

        occ_img_pil = Image.open(sample_dir / "occ_map.png").convert("L")
        # normalize to [0,1]; this is your original traversability/occupancy map
        occ_img = np.asarray(occ_img_pil, dtype=np.float32) / 255.0  # (H,W)

        # -------- Build model input dict (RGB + start only) --------
        input_dict = {
            DataDict.camera: rgb_tensor,
            "start": start,
            # If your model conditions on 'end', uncomment:
            # "end": end_gt,
        }

        with torch.no_grad():
            out = model(input_dict, sample=True)

        traj_pred_raw = out[DataDict.prediction][0]  # (N,2)

        # -------------------------------------------------
        # Handle train_poses vs increments
        # -------------------------------------------------
        if train_poses:
            # Model outputs absolute positions in [0,1]
            traj_pred_norm = traj_pred_raw.clamp(0.0, 1.0)
        else:
            # Model outputs increments
            traj_pred_norm = increments_to_absolute(start[0], traj_pred_raw)

        # -------------------------------------------------
        # Metrics in normalized space
        # -------------------------------------------------
        traj_mse = ((traj_pred_norm - traj_gt[0]) ** 2).mean().item()
        end_pred = traj_pred_norm[-1]  # (2,)
        endpoint_err = torch.norm(end_pred - end_gt[0]).item()

        all_traj_mse.append(traj_mse)
        all_endpoint_err.append(endpoint_err)

        endpoint_head_err_str = ""
        if "end_hat" in out:
            end_hat = out["end_hat"][0]
            endpoint_head_err_val = torch.norm(end_hat - end_gt[0]).item()
            all_endpoint_head_err.append(endpoint_head_err_val)
            endpoint_head_err_str = f"{endpoint_head_err_val:.6f}"

        # Write row to CSV
        writer.writerow([
            idx,
            sample_id,
            f"{traj_mse:.6f}",
            f"{endpoint_err:.6f}",
            endpoint_head_err_str
        ])

        # -------------------------------------------------
        # Convert to pixel coordinates for visualization
        # -------------------------------------------------
        traj_gt_px = traj_gt[0].cpu().numpy().copy()
        traj_gt_px[:, 0] *= (W-1)
        traj_gt_px[:, 1] *= (H-1)

        traj_pred_px = traj_pred_norm.cpu().numpy().copy()
        traj_pred_px[:, 0] *= (W-1)
        traj_pred_px[:, 1] *= (H-1)

        # -------------------------------------------------
        # Save overlays (no plt.show())
        # -------------------------------------------------
        # 1) RGB overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb_img)
        plt.plot(traj_gt_px[:, 0],  traj_gt_px[:, 1],  label="GT",   linewidth=2)
        plt.plot(traj_pred_px[:, 0], traj_pred_px[:, 1], "--",
                 label="Pred", linewidth=2)
        plt.ylim(H,0)
        plt.xlim(0,W)
        plt.legend()
        plt.title(f"Traj on RGB | {sample_id}\nMSE={traj_mse:.4f}, EndErr={endpoint_err:.4f}")
        # plt.gca().invert_yaxis()
        plt.tight_layout()
        rgb_out_path = overlay_rgb_dir / f"{sample_id}_rgb.png"
        plt.savefig(rgb_out_path, dpi=150)
        plt.close()

        # 2) occ_map overlay (original occ_map.png)
        plt.figure(figsize=(6, 6))
        plt.imshow(occ_img, cmap="gray")
        plt.plot(traj_gt_px[:, 0],  traj_gt_px[:, 1],  label="GT",   linewidth=2)
        plt.plot(traj_pred_px[:, 0], traj_pred_px[:, 1], "--",
                 label="Pred", linewidth=2)
        plt.ylim(H,0)
        plt.xlim(0,W)
        plt.legend()
        plt.title(f"Traj on occ_map | {sample_id}\nMSE={traj_mse:.4f}, EndErr={endpoint_err:.4f}")
        # plt.gca().invert_yaxis()
        plt.tight_layout()
        occ_out_path = overlay_occ_dir / f"{sample_id}_occ.png"
        plt.savefig(occ_out_path, dpi=150)
        plt.close()

# --------------------------------------------------
# 7. Aggregate metrics & save metric plots
# --------------------------------------------------
all_traj_mse = np.array(all_traj_mse, dtype=float)
all_endpoint_err = np.array(all_endpoint_err, dtype=float)

print("\n=== Overall test performance (over all samples) ===")
print(f"Mean Trajectory MSE    : {all_traj_mse.mean():.6f}")
print(f"Mean Endpoint L2 error : {all_endpoint_err.mean():.6f}")
if len(all_endpoint_head_err) > 0:
    all_endpoint_head_err = np.array(all_endpoint_head_err, dtype=float)
    print(f"Mean Endpoint-head L2  : {all_endpoint_head_err.mean():.6f}")

# x-axis = sample index (one per image), y-axis = metric value
idxs = np.arange(len(all_traj_mse))

# Trajectory MSE plot
plt.figure(figsize=(8, 4))
plt.plot(idxs, all_traj_mse, marker="o", linewidth=1)
plt.xlabel("Sample index (image)")
plt.ylabel("Trajectory MSE (normalized space)")
plt.title("Trajectory MSE per Image")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_root / "metric_traj_mse.png", dpi=150)
plt.close()

# Endpoint error plot
plt.figure(figsize=(8, 4))
plt.plot(idxs, all_endpoint_err, marker="o", linewidth=1)
plt.xlabel("Sample index (image)")
plt.ylabel("Endpoint L2 error (normalized space)")
plt.title("Endpoint Error per Image")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_root / "metric_endpoint_err.png", dpi=150)
plt.close()

# Endpoint-head plot (if available)
if len(all_endpoint_head_err) > 0:
    idxs2 = np.arange(len(all_endpoint_head_err))
    plt.figure(figsize=(8, 4))
    plt.plot(idxs2, all_endpoint_head_err, marker="o", linewidth=1)
    plt.xlabel("Sample index (image with end_hat)")
    plt.ylabel("Endpoint-head L2 error")
    plt.title("Endpoint-head Error per Image")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_root / "metric_endpoint_head_err.png", dpi=150)
    plt.close()
