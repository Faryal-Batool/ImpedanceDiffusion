# Module: Debug visualization helpers for checking dataset samples.

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TrajDataset   # <-- import your existing dataset

# ----------------------------
# USER SETTINGS
# ----------------------------
DATASET_ROOT = r"/home/isr-lab3/Faryal_Batool/Training samples"
BATCH_SIZE = 10
OUT_DIR = Path("debug_overlays")
OUT_DIR.mkdir(exist_ok=True)
# ----------------------------

# Function: Save a debug image showing RGB, trajectory, start, and goal overlays.
def plot_overlay(rgb, traj, start, end, out_path):
    """
    rgb: (3, H, W)
    traj: (N,2) in pixel coords
    start, end: (2,) in pixel coords
    """
    rgb = rgb.permute(1,2,0).cpu().numpy()   # (H,W,3)
    H, W = rgb.shape[:2]

    plt.figure(figsize=(6,6))
    plt.imshow(rgb)

    plt.plot(traj[:,0], traj[:,1], linewidth=2, c='blue')
    plt.scatter(start[0], start[1], s=60, c='green')
    plt.scatter(end[0],   end[1],   s=60, c='red', marker='X')

    # Keep correct image coord orientation
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.axis("off")

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Function: Script entry point that assembles configuration and launches the requested workflow.
def main():
    dataset = TrajDataset(DATASET_ROOT, n_points=128)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    batch_id = 0
    for batch_idx, batch in enumerate(loader):
        batch_folder = OUT_DIR / f"batch_{batch_id:03d}"
        batch_folder.mkdir(exist_ok=True)

        rgb = batch["rgb"]          # (B,3,H,W)
        traj = batch["traj"]        # (B,N,2) normalized
        start = batch["start"]      # (B,2) normalized
        end = batch["end"]          # (B,2) normalized

        H = rgb.shape[-2]
        W = rgb.shape[-1]

        for i in range(rgb.shape[0]):
            sample_id = batch_idx * BATCH_SIZE + i
            out_file = batch_folder / f"sample_{sample_id:06d}.png"

            # Convert to pixel coords
            traj_px = traj[i].cpu().numpy().copy()
            start_px = start[i].cpu().numpy().copy()
            end_px   = end[i].cpu().numpy().copy()

            traj_px[:,0] *= (W-1)
            traj_px[:,1] *= (H-1)
            start_px *= np.array([W-1, H-1])
            end_px   *= np.array([W-1, H-1])

            # Plot and save
            plot_overlay(rgb[i], traj_px, start_px, end_px, out_file)

        print(f"[Batch {batch_id}] Saved {len(rgb)} samples.")

        batch_id += 1

    print("\nAll batches processed! Check folder:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()