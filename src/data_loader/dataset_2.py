# Module: Dataset and rasterization utilities for RGB-conditioned mask training.

# dataset.py
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# ------------------------------
# Helpers
# ------------------------------

# Function: Load an RGB image as a numpy array.
def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

# Function: Load an occupancy image as a grayscale numpy array.
def load_occ(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)

# Function: Load a traversability map from disk and squeeze singleton dimensions.
def load_trav(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.squeeze(arr).astype(np.float32)

# Function: Resize a numpy image to the model square resolution.
def resize_np(img: np.ndarray, size: int, is_gray: bool = False) -> np.ndarray:
    """Resize numpy image to (size,size)."""
    if img.ndim == 2 or is_gray:
        pil = Image.fromarray(img)
        pil = pil.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(pil)
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(pil)

# Function: Load one normalized point from a JSON file.
def load_point01(json_path: Path):
    with open(json_path, "r") as f:
        pt = json.load(f)
    return float(pt["x"]), float(pt["y"])

# Function: Clip normalized coordinates into the [0, 1] range.
def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

# Function: Normalize trajectory coordinate ordering and optional y-axis flipping.
def traj_to_norm01(traj: np.ndarray, order: str = "yx", flip_y: bool = False) -> np.ndarray:
    """
    traj: (N,2) normalized [0,1], stored either as (x,y) or (y,x)
    order:
      - "xy": traj[:,0]=x, traj[:,1]=y
      - "yx": traj[:,0]=y, traj[:,1]=x  (your planner case)
    flip_y:
      - True: y := 1 - y
    returns: (N,2) normalized (x,y) clipped to [0,1]
    """
    t = np.asarray(traj, dtype=np.float32)
    if t.ndim != 2 or t.shape[1] != 2:
        raise ValueError("traj must be (N,2)")

    # Expect normalized
    if np.nanmax(np.abs(t)) > 1.5:
        raise ValueError("traj looks like pixels; this function expects normalized [0,1].")

    if order == "xy":
        xy = t.copy()
    elif order == "yx":
        xy = t[:, [1, 0]].copy()  # swap (y,x)->(x,y)
    else:
        raise ValueError("order must be 'xy' or 'yx'")

    xy = np.clip(xy, 0.0, 1.0)

    if flip_y:
        xy[:, 1] = 1.0 - xy[:, 1]

    return np.clip(xy, 0.0, 1.0)


# Function: Resample a polyline at approximately uniform arclength intervals.
def resample_by_arclength(traj01: np.ndarray, n: int) -> np.ndarray:
    """
    Resample polyline in normalized space to n points, roughly equal arc-length.
    Keeps it float (no rounding).
    """
    t = np.asarray(traj01, dtype=np.float32)
    if len(t) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(t) == 1:
        return np.repeat(t, n, axis=0)

    diffs = np.diff(t, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglens)])
    total = s[-1]

    if total < 1e-8:
        return np.repeat(t[:1], n, axis=0)

    s_new = np.linspace(0.0, total, n, dtype=np.float32)
    x_new = np.interp(s_new, s, t[:, 0]).astype(np.float32)
    y_new = np.interp(s_new, s, t[:, 1]).astype(np.float32)
    out = np.stack([x_new, y_new], axis=1)
    return clamp01(out)

# ------------------------------
# Rasterization: line drawing
# ------------------------------

# Function: Paint a soft Gaussian blob into a single-channel mask.
def splat_gaussian(mask2d: np.ndarray, cx: float, cy: float, sigma: float = 0.9, cutoff: float = 3.0):
    """Soft blob centered at (cx,cy) in pixel coords on a single-channel mask."""
    H, W = mask2d.shape
    rad = max(1, int(np.ceil(cutoff * sigma)))
    x0 = max(0, int(np.floor(cx)) - rad)
    x1 = min(W - 1, int(np.floor(cx)) + rad)
    y0 = max(0, int(np.floor(cy)) - rad)
    y1 = min(H - 1, int(np.floor(cy)) + rad)

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    d2 = (X - cx) ** 2 + (Y - cy) ** 2
    g = np.exp(-d2 / (2.0 * sigma * sigma))

    patch = mask2d[y0:y1 + 1, x0:x1 + 1]
    np.maximum(patch, g, out=patch)
    mask2d[y0:y1 + 1, x0:x1 + 1] = patch


# Function: Paint a small hard endpoint marker into a single-channel mask.
def paint_hard_disk(mask2d: np.ndarray, cx: float, cy: float, r: int = 1):
    """Hard disk / square-ish stamp for start/goal."""
    H, W = mask2d.shape
    xi = int(round(cx))
    yi = int(round(cy))
    x0 = max(0, xi - r)
    x1 = min(W - 1, xi + r)
    y0 = max(0, yi - r)
    y1 = min(H - 1, yi + r)
    mask2d[y0:y1 + 1, x0:x1 + 1] = 1.0

# Function: Rasterize a hard line segment into a mask.
def draw_line_hard(mask: np.ndarray, x0, y0, x1, y1, thickness: int = 1):
    """
    Draw a connected hard line segment onto mask using simple sampling.
    mask: HxW float32 or uint8
    x0,y0,x1,y1: float pixel coords
    """
    H, W = mask.shape
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy))) + 1
    if steps <= 1:
        steps = 2
    xs = np.linspace(x0, x1, steps)
    ys = np.linspace(y0, y1, steps)

    r = max(int(thickness), 1)
    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        x_min = max(0, xi - r)
        x_max = min(W - 1, xi + r)
        y_min = max(0, yi - r)
        y_max = min(H - 1, yi + r)
        mask[y_min:y_max+1, x_min:x_max+1] = 1.0


# Function: Rasterize a soft anti-aliased line segment into a mask.
def draw_line_soft(mask: np.ndarray, x0, y0, x1, y1, sigma: float = 0.75, cutoff: float = 3.0):
    """
    Draw an anti-aliased (soft) line by splatting Gaussians along the segment.
    sigma in pixels; cutoff controls window size ~ cutoff*sigma.
    """
    H, W = mask.shape
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy))) + 1
    if steps <= 1:
        steps = 2
    xs = np.linspace(x0, x1, steps)
    ys = np.linspace(y0, y1, steps)

    rad = max(1, int(np.ceil(cutoff * sigma)))
    two_sigma2 = 2.0 * (sigma ** 2)

    for x, y in zip(xs, ys):
        cx = float(x)
        cy = float(y)
        x0i = max(0, int(np.floor(cx)) - rad)
        x1i = min(W - 1, int(np.floor(cx)) + rad)
        y0i = max(0, int(np.floor(cy)) - rad)
        y1i = min(H - 1, int(np.floor(cy)) + rad)

        xs_grid = np.arange(x0i, x1i + 1, dtype=np.float32)
        ys_grid = np.arange(y0i, y1i + 1, dtype=np.float32)
        X, Y = np.meshgrid(xs_grid, ys_grid)

        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        g = np.exp(-d2 / two_sigma2)

        # max-composite keeps strongest response (nice for lines)
        patch = mask[y0i:y1i + 1, x0i:x1i + 1]
        np.maximum(patch, g, out=patch)
        mask[y0i:y1i + 1, x0i:x1i + 1] = patch


# Function: Convert a normalized trajectory polyline into a mask channel.
def rasterize_traj_mask(traj01: np.ndarray, size: int, mode: str = "soft",
                        thickness: int = 1, sigma: float = 0.75) -> np.ndarray:
    """
    traj01: (N,2) normalized [0,1] floats
    returns mask: (size,size) float32 in [0,1]
    """
    H = W = size
    mask = np.zeros((H, W), dtype=np.float32)

    # float pixel coords (no rounding!)
    xs = traj01[:, 0] * (W - 1)
    ys = traj01[:, 1] * (H - 1)

    for i in range(len(traj01) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        if mode == "hard":
            draw_line_hard(mask, x0, y0, x1, y1, thickness=thickness)
        elif mode == "soft":
            draw_line_soft(mask, x0, y0, x1, y1, sigma=sigma)
        else:
            raise ValueError("mode must be 'hard' or 'soft'")

    # Ensure endpoints are visible
    if len(traj01) >= 1:
        if mode == "hard":
            draw_line_hard(mask, xs[0], ys[0], xs[0], ys[0], thickness=max(1, thickness))
            draw_line_hard(mask, xs[-1], ys[-1], xs[-1], ys[-1], thickness=max(1, thickness))
        else:
            draw_line_soft(mask, xs[0], ys[0], xs[0], ys[0], sigma=sigma)
            draw_line_soft(mask, xs[-1], ys[-1], xs[-1], ys[-1], sigma=sigma)

    return np.clip(mask, 0.0, 1.0)


# ------------------------------
# Dataset
# ------------------------------

# Class: Dataset that loads RGB samples, endpoints, trajectories, and DDPM mask targets.
class TrajMaskDataset(Dataset):
    """
    Outputs are consistent with typical structure:
      - rgb:   (3, S, S) float32 in [0,1]
      - occ:   (1, S, S) float32 in [0,1]  (if exists)
      - trav:  (1, S, S) float32           (if exists)
      - start_px, end_px: int64 [2] pixel coords in SxS space (x,y)
      - start_01, end_01: float32 [2] normalized coords
      - traj_01: (N,2) float32 normalized polyline
      - traj_mask: (1,S,S) float32 in [0,1] (soft or hard)
    """

    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(
        self,
        root_dir: str,
        img_size: int = 64,
        n_points: int = 128,
        traj_mask_mode: str = "hard",   # "soft" or "hard"
        line_thickness: int = 2,        # for hard mode
        soft_sigma: float = 0.75,       # for soft mode
        use_occ: bool = True,
        use_trav: bool = True,
        traj_order: str = "yx", 
        traj_flip_y: bool = False,
    ):
        self.root = Path(root_dir)
        self.img_size = int(img_size)
        self.n_points = int(n_points)
        self.traj_mask_mode = traj_mask_mode
        self.line_thickness = int(line_thickness)
        self.soft_sigma = float(soft_sigma)
        self.use_occ = use_occ
        self.use_trav = use_trav
        self.traj_order = traj_order
        self.traj_flip_y = traj_flip_y

        self.soft_start_goal = True
        self.start_goal_sigma = 0.9
        # or for hard:
        self.start_goal_radius = 1


        self.samples = sorted([p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("sample_")])
        if len(self.samples) == 0:
            raise RuntimeError(f"No sample_XXXXX folders found in: {self.root}")

    # Function: Return the number of available dataset samples.
    def __len__(self):
        return len(self.samples)

    # Function: Load one dataset sample and prepare tensors for the model.
    def __getitem__(self, idx: int):
        sp = self.samples[idx]

        # ---- Load RGB (512) then resize to SxS
        rgb = load_rgb(sp / "rgb.png")
        rgb_s = resize_np(rgb, self.img_size, is_gray=False)
        rgb_s = np.ascontiguousarray(rgb_s).copy()
        rgb_t = torch.from_numpy(rgb_s).float().permute(2, 0, 1) / 255.0

        # rgb_t = torch.from_numpy(rgb_s).float().permute(2, 0, 1) / 255.0  # (3,S,S)

        # ---- Optionals
        occ_t = None
        if self.use_occ and (sp / "occ_map.png").exists():
            occ = load_occ(sp / "occ_map.png")
            occ_s = resize_np(occ, self.img_size, is_gray=True)
            occ_t = torch.from_numpy(occ_s).float().unsqueeze(0) / 255.0  # (1,S,S)

        trav_t = None
        if self.use_trav and (sp / "trav_map.npy").exists():
            trav = load_trav(sp / "trav_map.npy")
            # trav might be float already; resize as float
            trav_s = resize_np(trav.astype(np.float32), self.img_size, is_gray=True).astype(np.float32)
            trav_t = torch.from_numpy(trav_s).float().unsqueeze(0)  # (1,S,S)

        # ---- Load start/end in normalized space (this is your source of truth)
        sx01, sy01 = load_point01(sp / "start_xy.json")
        gx01, gy01 = load_point01(sp / "end_xy.json")
        # start_01 = torch.tensor([sx01, sy01], dtype=torch.float32)
        # end_01   = torch.tensor([gx01, gy01], dtype=torch.float32)

        # Convert to pixel coords in SxS (consistent with your existing style)
        W = H = self.img_size
        start_px = torch.tensor([int(round(sx01 * (W - 1))), int(round(sy01 * (H - 1)))], dtype=torch.int64)
        end_px   = torch.tensor([int(round(gx01 * (W - 1))), int(round(gy01 * (H - 1)))], dtype=torch.int64)

        # ---- Load trajectory (normalized floats) and resample in normalized space
        traj = np.load(sp / "traj_xy.npy")  # expected normalized [0,1]
        # traj01 = traj_to_norm01(traj)       # (N,2) float in [0,1]
        traj01 = traj_to_norm01(traj, order=self.traj_order, flip_y=self.traj_flip_y)
        traj01 = resample_by_arclength(traj01, self.n_points)  # (n_points,2) float
        traj_01_t = torch.from_numpy(traj01).float()  # (N,2)

                # ---- Build (N,2) pixel trajectory in SxS for returning (int64) like your old code
        S = self.img_size
        traj_px_float = np.stack([traj01[:, 0] * (S - 1), traj01[:, 1] * (S - 1)], axis=1)
        traj_px_int = np.rint(np.clip(traj_px_float, 0, S - 1)).astype(np.int64)  # (N,2) (x,y)

        traj_px_t = torch.from_numpy(traj_px_int).long()

        # ---- Rasterize trajectory channel (float mask in [0,1])
        traj_mask = rasterize_traj_mask(
            traj01=traj01,
            size=S,
            mode=self.traj_mask_mode,
            thickness=self.line_thickness,
            sigma=self.soft_sigma
        )  # (S,S) float32

        # ---- Build mask_gt: (3,S,S)
        mask_gt = np.zeros((3, S, S), dtype=np.float32)

        # start/goal centers in pixel coords (float)
        sx_f = float(sx01) * (S - 1)
        sy_f = float(sy01) * (S - 1)
        gx_f = float(gx01) * (S - 1)
        gy_f = float(gy01) * (S - 1)

        # Start/Goal channels: choose hard or soft blobs
        # If you want start/goal also soft, use splat_gaussian; otherwise paint_hard_disk.
        if getattr(self, "soft_start_goal", True):
            sg_sigma = float(getattr(self, "start_goal_sigma", 0.9))
            splat_gaussian(mask_gt[0], sx_f, sy_f, sigma=sg_sigma)
            splat_gaussian(mask_gt[1], gx_f, gy_f, sigma=sg_sigma)
        else:
            r = int(getattr(self, "start_goal_radius", 1))
            paint_hard_disk(mask_gt[0], sx_f, sy_f, r=r)
            paint_hard_disk(mask_gt[1], gx_f, gy_f, r=r)

        # Trajectory channel
        mask_gt[2] = traj_mask

        mask_gt_t = torch.from_numpy(mask_gt).float()  # (3,S,S)

        out = {
            "rgb": rgb_t,              # (3,S,S)
            "start_px": start_px,      # (2,) int64 (x,y)
            "end_px": end_px,          # (2,) int64 (x,y)
            "traj_px": traj_px_t,      # (N,2) int64 (x,y)
            "mask_gt": mask_gt_t,      # (3,S,S) float32 in [0,1] (soft) or {0,1} (hard-ish)
        }

        # Keep your other conditioning inputs consistent (trav/occ)
        if trav_t is not None:
            out["trav"] = trav_t
        if occ_t is not None:
            out["occ"] = occ_t

        # Optional: return float traj too for debugging
        out["traj_01"] = traj_01_t    # (N,2)

        return out


        # # ---- Rasterize trajectory into a mask in SxS
        # traj_mask = rasterize_traj_mask(
        #     traj01=traj01,
        #     size=self.img_size,
        #     mode=self.traj_mask_mode,
        #     thickness=self.line_thickness,
        #     sigma=self.soft_sigma
        # )
        # traj_mask_t = torch.from_numpy(traj_mask).float().unsqueeze(0)  # (1,S,S)

        # out = {
        #     "rgb": rgb_t,
        #     "mask_gt": traj_mask_t,
        #     "traj_01": traj_01_t,
        #     "start_px": start_px,
        #     "end_px": end_px,
        #     "start_01": start_01,
        #     "end_01": end_01,
        #     "sample_dir": str(sp),
        # }
        # if occ_t is not None:
        #     out["occ"] = occ_t
        # if trav_t is not None:
        #     out["trav"] = trav_t

        # return out



