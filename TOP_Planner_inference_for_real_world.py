# Module: Top-view DDPM inference script with world-coordinate export.

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import csv

from indoor_utils import indoor_VLA_utils

from src.utils.arguments import get_configuration
from src.models.model import get_model
from src.utils.configs import DataDict
import heapq

# ============================================================
# Camera -> world conversion parameters
# ============================================================
CAMERA_HEIGHT_M = 4.4   # meters (camera height above ground)
DIAGONAL_FOV_DEG = 78.0 # degrees
WORLD_Z_M = 1.0         # constant z for exported world coordinates

# ============================================================
# Utility: scale pixel coordinate from (H0,W0) → (H,W)
# ============================================================
# Function: Scale one pixel coordinate from an original image size into a target image size.
def scale_point_to_target(x_px, y_px, orig_hw, target_hw):
    H0, W0 = orig_hw
    Ht, Wt = target_hw
    sx = Wt / float(W0)
    sy = Ht / float(H0)
    x2 = int(np.clip(round(x_px * sx), 0, Wt - 1))
    y2 = int(np.clip(round(y_px * sy), 0, Ht - 1))
    return x2, y2
# ============================================================
# Coordinate convention conversion (Image 01 -> Image 02)
# Image 01 (your DDPM pixels): origin at TOP-LEFT, +x to RIGHT, +y DOWN.
# Image 02 (desired):          origin at BOTTOM-RIGHT, +x UP, +y LEFT.
#
# For an image of size (W,H) in pixels:
#   x2 = (H-1) - y1   (upwards from bottom)
#   y2 = (W-1) - x1   (leftwards from right)
# ============================================================
# Function: Convert top-left-origin image pixels to the project bottom-right-origin convention.
def convert_pixels_conv1_to_conv2(x1_px, y1_px, W, H):
    """Convert pixel coordinates from convention-1 to convention-2.

    Accepts floats (e.g., smoothed polylines). Returns floats.
    """
    x1 = float(x1_px)
    y1 = float(y1_px)
    W = float(W)
    H = float(H)
    x2 = (H - 1.0) - y1
    y2 = (W - 1.0) - x1
    return x2, y2


# ============================================================
# Inference wrapper: run DDPM once and return traj logits/prob
# ============================================================
@torch.no_grad()
# Function: Run one DDPM sampling call for a start-goal pair and return the trajectory probability map.
def run_ddpm_once(model, rgb_t, start_xy_64, goal_xy_64, device):
    """
    start_xy_64, goal_xy_64: (x,y) ints in 64x64
    Returns:
      pred_traj_logits: (1,1,64,64) torch
      pred_traj_prob:   (64,64) numpy in [0,1]
      infer_ms: float
    """
    start_px = torch.tensor([[start_xy_64[0], start_xy_64[1]]],
                            device=device, dtype=torch.long)
    end_px   = torch.tensor([[goal_xy_64[0],  goal_xy_64[1]]],
                            device=device, dtype=torch.long)

    input_dict = {
        DataDict.camera: rgb_t,
        "rgb": rgb_t,
        "start_px": start_px,
        "end_px": end_px,
    }

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_dict, sample=True)   # ✅ ONE INFERENCE CALL
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    pred = out[DataDict.prediction]              # (1,3,64,64)
    pred_traj = pred[:, 2:3, :, :]               # (1,1,64,64)
    prob = torch.sigmoid(pred_traj)[0, 0].detach().cpu().numpy()  # (64,64)

    return pred_traj, prob, (t1 - t0) * 1000.0


# ============================================================
# Dijkstra shortest path on probability
# ============================================================
# Function: Extract a path from a probability map using graph search.
def dijkstra_path_from_prob(prob64: np.ndarray,
                            start_xy: tuple,
                            goal_xy: tuple,
                            connectivity: int = 8,
                            eps: float = 1e-6,
                            prob_floor: float | None = None):
    """
    Dijkstra on cost = -log(prob + eps).
    Returns list[(x,y)] from start to goal, or None if unreachable.
    """
    H, W = prob64.shape
    sx, sy = map(int, start_xy)
    gx, gy = map(int, goal_xy)

    p = np.clip(prob64, 0.0, 1.0)
    cost_map = -np.log(p + eps)

    if prob_floor is not None:
        blocked = p < prob_floor
    else:
        blocked = np.zeros((H, W), dtype=bool)

    blocked[sy, sx] = False
    blocked[gy, gx] = False

    if connectivity == 8:
        nbrs = [(-1,-1), (0,-1), (1,-1),
                (-1, 0),         (1, 0),
                (-1, 1), (0, 1), (1, 1)]
    elif connectivity == 4:
        nbrs = [(0,-1), (-1,0), (1,0), (0,1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    INF = 1e18
    dist = np.full((H, W), INF, dtype=np.float64)
    prev = np.full((H, W, 2), -1, dtype=np.int16)

    dist[sy, sx] = 0.0
    pq = [(0.0, sx, sy)]

    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != dist[y, x]:
            continue
        if (x, y) == (gx, gy):
            break

        for dx, dy in nbrs:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if blocked[ny, nx]:
                continue

            step_len = np.hypot(dx, dy)
            step_cost = step_len * (cost_map[ny, nx])

            nd = d + step_cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx, 0] = x
                prev[ny, nx, 1] = y
                heapq.heappush(pq, (nd, nx, ny))

    if dist[gy, gx] >= INF/2:
        return None

    path = []
    cx, cy = gx, gy
    path.append((cx, cy))
    while not (cx == sx and cy == sy):
        px = int(prev[cy, cx, 0])
        py = int(prev[cy, cx, 1])
        if px < 0 or py < 0:
            return None
        cx, cy = px, py
        path.append((cx, cy))

    path.reverse()
    return path


# Function: Simplify a path by removing near-collinear interior points.
def remove_collinear_points(path, angle_tol_deg=1.0):
    if path is None or len(path) <= 2:
        return path

    pts = np.array(path, dtype=np.float32)
    keep = [0]
    tol = np.deg2rad(angle_tol_deg)

    for i in range(1, len(pts) - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue

        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        ang = np.arccos(cosang)

        if ang > tol:
            keep.append(i)

    keep.append(len(pts) - 1)
    pts2 = pts[keep]
    return [(float(x), float(y)) for x, y in pts2]


# Function: Smooth a path with Chaikin corner cutting.
def chaikin_smooth(path, n_iters=3, keep_ends=True):
    if path is None or len(path) < 3:
        return path

    pts = np.array(path, dtype=np.float32)

    for _ in range(n_iters):
        new_pts = []
        if keep_ends:
            new_pts.append(pts[0])

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_pts.append(Q)
            new_pts.append(R)

        if keep_ends:
            new_pts.append(pts[-1])

        pts = np.vstack(new_pts)

    return [(float(x), float(y)) for x, y in pts]


# Function: Clamp path points to valid image bounds.
def clip_path_to_bounds(path, W, H):
    if path is None:
        return None
    out = []
    for x, y in path:
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))
        out.append((x, y))
    return out

# Function: Map a polyline from the 64x64 model grid back to original image pixels.
def upscale_poly64_to_orig(poly64, orig_w, orig_h):
    """
    poly64: list[(x64,y64)] in 0..63 (floats allowed).
    Returns list[(x_orig,y_orig)] in original pixel coords (floats).
    """
    if poly64 is None or len(poly64) == 0:
        return []

    W64 = 64.0
    H64 = 64.0
    ow = float(orig_w)
    oh = float(orig_h)

    poly_orig = []
    for (x64, y64) in poly64:
        x64 = float(x64)
        y64 = float(y64)

        # Map 0..63 -> 0..(orig_w-1) and 0..(orig_h-1)
        x_orig = (x64 / (W64 - 1.0)) * (ow - 1.0)
        y_orig = (y64 / (H64 - 1.0)) * (oh - 1.0)
        poly_orig.append((x_orig, y_orig))
    return poly_orig


# Function: Resample a path to a fixed number of points.
def resample_polyline(points, n=200):
    """Resample polyline to n points uniformly by arc-length. points: list[(x,y)] floats."""
    if points is None or len(points) < 2:
        return points
    pts = np.asarray(points, dtype=np.float64)
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-9:
        return [(float(x), float(y)) for x, y in pts]
    t = np.linspace(0.0, s[-1], n)
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    return list(map(tuple, np.stack([x, y], axis=1)))

# Function: Smooth a path with exponential moving average.
def smooth_ema(points, alpha=0.25, keep_ends=True):
    """Exponential moving average smoothing. alpha small => smoother (0.15–0.35 typical)."""
    if points is None or len(points) < 3:
        return points
    pts = np.asarray(points, dtype=np.float64)
    out = pts.copy()
    start_i = 1 if keep_ends else 0
    end_i = len(pts) - 1 if keep_ends else len(pts)
    for i in range(start_i, end_i):
        out[i] = alpha * pts[i] + (1.0 - alpha) * out[i - 1]
    if keep_ends:
        out[0] = pts[0]
        out[-1] = pts[-1]
    return [(float(x), float(y)) for x, y in out]

# ============================================================
# FIXED: Convert 64x64 pixel path -> (x,y,z) in meters using indoor_utils
# ============================================================
# Function: Convert a 64x64 model-grid path into world-space x, y, z coordinates.
def path64_to_world_xyz(poly64, image_path: str,
                        camera_height_m: float = CAMERA_HEIGHT_M,
                        diagonal_fov_deg: float = DIAGONAL_FOV_DEG,
                        z_m: float = WORLD_Z_M):
    """
    poly64: list[(x64,y64)] in 0..63 (may be float after smoothing).

    Pipeline:
      1) Use the REAL image at image_path (e.g., 1920x1080) to get image metrics.
      2) Upscale 64x64 -> original pixels in **Convention-1** (Image 01):
            origin TOP-LEFT, +x RIGHT, +y DOWN.
      3) Convert those pixels to **Convention-2** (Image 02):
            origin BOTTOM-RIGHT, +x UP, +y LEFT.
      4) Convert convention-2 pixels to percent [0..100] (x spans HEIGHT, y spans WIDTH).
      5) indoor_utils.process_coordinates -> (x_m, y_m)
    """
    if poly64 is None or len(poly64) == 0:
        return []

    # Metrics based on the real image on disk
    image_metrics = indoor_VLA_utils.calculate_real_world_size(
        image_path, camera_height_m, diagonal_fov_deg
    )
    orig_w = float(image_metrics["image_width_px"])   # width in px
    orig_h = float(image_metrics["image_height_px"])  # height in px

    # 64x64 grid extents
    W64 = 64.0
    H64 = 64.0

    # Denominators for percent conversion (avoid divide-by-zero)
    denom_w = max(1.0, orig_w - 1.0)  # width denom
    denom_h = max(1.0, orig_h - 1.0)  # height denom

    optimized_coordinates = {}

    for i, (x64, y64) in enumerate(poly64):
        x64 = float(x64)
        y64 = float(y64)

        # Upscale 64 -> original pixels (Convention-1)
        x_orig = (x64 / (W64 - 1.0)) * (orig_w - 1.0)
        y_orig = (y64 / (H64 - 1.0)) * (orig_h - 1.0)

        # Convert Convention-1 -> Convention-2
        x_c2, y_c2 = convert_pixels_conv1_to_conv2(x_orig, y_orig, W=orig_w, H=orig_h)

        # Convention-2 pixels -> percent [0..100]
        # Note: in convention-2, x spans image HEIGHT, y spans image WIDTH.
        x_pct_c2 = (x_c2 / denom_h) * 100.0
        y_pct_c2 = (y_c2 / denom_w) * 100.0

        # Keep your existing indoor_utils axis expectation:
        # feed [y_pct, x_pct] (but now both are in the NEW convention).
        optimized_coordinates[f"p{i:04d}"] = {"coordinates": [y_pct_c2, x_pct_c2]}

    result_coordinates = indoor_VLA_utils.process_coordinates(
        optimized_coordinates, diagonal_fov_deg, camera_height_m, image_metrics
    )

    world_xyz = []
    for i in range(len(poly64)):
        key = f"p{i:04d}"
        xy = result_coordinates[key]["coordinates"]
        # x_m = round(float(xy[0]), 2)
        # y_m = round(float(xy[1]), 2)
        x_m = float(xy[0])
        y_m = float(xy[1])
        world_xyz.append((x_m, y_m, float(z_m)))
        

    return world_xyz


# ============================================================
# MAIN (single-run inference per image)
# ============================================================
cfgs = get_configuration()

test_image_root = Path("/home/isr-lab3/Faryal_Batool/DTG-main_top/Experiment_annotations")
image_paths = sorted([p for p in test_image_root.iterdir()
                      if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
print(f"[INFO] Total test images found: {len(image_paths)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

model = get_model(cfgs.model, device=device).to(device)
ckpt_path = r"/home/isr-lab3/Faryal_Batool/DTG-main_top/results/models/hnav_29.pth"
print(f"[INFO] Loading checkpoint from: {ckpt_path}")
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
model.eval()

output_root = Path("Experiments_for_ICUAS")
overlay_dir = output_root / "overlays_single_64"
overlay_dir_1080 = output_root / "overlays_single_1920x1080"
pixel_csv_dir = output_root / "pixel_csv_1920x1080"
pixel_csv_dir.mkdir(parents=True, exist_ok=True)
overlay_dir_1080.mkdir(parents=True, exist_ok=True)
output_root.mkdir(parents=True, exist_ok=True)
overlay_dir.mkdir(parents=True, exist_ok=True)

world_csv_dir = output_root / "world_csv"
world_csv_dir.mkdir(parents=True, exist_ok=True)

# HARDCODED_POINTS = [
#     ((550, 1041), (1398, 16)),  # s10_hard_obstacle
#     ((549, 1042), (1396, 16)),  # s10_soft_obstacle
#     ((545, 1045), (1394, 16)),  # s1_hard_obstacle
#     ((546, 1042), (1396, 18)),  # s1_soft_obstacle
#     ((545, 1042), (1400, 18)),  # s2_hard_obstacle
#     ((546, 1041), (1396, 16)),  # s2_soft_obstacle
#     ((546, 1044), (1398, 16)),  # s3_hard_obstacle
#     ((546, 1044), (1398, 16)),  # s3_soft_obstacle
#     ((549, 1042), (1398, 15)),  # s4_hard_obstacle
#     ((546, 1041), (1396, 18)),  # s4_soft_obstacle
#     ((546, 1041), (1398, 18)),  # s5_hard_obstacle
#     ((641, 1023), (1398, 15)),  # s5_soft_obstacle
#     ((549, 1044), (1398, 13)),  # s6_hard_obstacle
#     ((555, 1049), (1394, 15)),  # s6_soft_obstacle
#     ((548, 1041), (1396, 16)),  # s7_hard_obstacle
#     ((548, 1041), (1399, 19)),  # s7_soft_obstacle
#     ((558, 1042), (1395, 20)),  # s8_hard_obstacle
#     ((559, 1041), (1398, 18)),  # s8_soft_obstacle
#     ((554, 1041), (1394, 18)),  # s9_hard_obstacle
#     ((556, 1046), (1395, 20)),  # s9_soft_obstacle
# ]

HARDCODED_POINTS = [
    # ((526, 1003), (1567, 8)),  # exp_1
    # ((527, 1003), (1583, 8)),  # exp_2a
    # ((511, 1005), (1555, 9)),  # exp_2b
    # ((545, 1071), (1512, 13)),  # exp_3
    # ((500, 996), (1517, 19)),  # exp_4_dynamic
    # ((502, 1071), (1522, 12)),  # exp_5
    ((502, 1071), (1522, 12)),  # exp_6
    # ((550, 1071), (1560, 14)),  # exp_7
    # ((549, 1069), (1552, 11)),  # exp_8
    # ((551, 1068), (1550, 11)),  # exp_9


]

assert len(HARDCODED_POINTS) == len(image_paths), "❌ hardcoded points must match images!"

for idx, rgb_path in enumerate(image_paths):
    sample_id = rgb_path.stem
    print(f"\n[INFO] Processing {sample_id}")

    # Load RGB (you are providing real 1920x1080 images here)
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
    ORIG_H, ORIG_W = rgb_np.shape[0], rgb_np.shape[1]

    # resize RGB -> 64×64 for model
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
    rgb_t = F.interpolate(rgb_t, size=(64, 64), mode="bilinear", align_corners=False).to(device)

    # Start/goal are in the same resolution as the input image (1920x1080)
    (sx_raw, sy_raw), (gx_raw, gy_raw) = HARDCODED_POINTS[idx]
    sx_64, sy_64 = scale_point_to_target(sx_raw, sy_raw, (ORIG_H, ORIG_W), (64, 64))
    gx_64, gy_64 = scale_point_to_target(gx_raw, gy_raw, (ORIG_H, ORIG_W), (64, 64))

    # ✅ RUN DDPM EXACTLY ONCE
    pred_traj, prob64, ms = run_ddpm_once(
        model=model,
        rgb_t=rgb_t,
        start_xy_64=(sx_64, sy_64),
        goal_xy_64=(gx_64, gy_64),
        device=device
    )
    print(f"[INFO] Single inference: {ms:.1f} ms | prob_max={prob64.max():.3f} prob_mean={prob64.mean():.3f}")

    # Connected path via Dijkstra on prob map
    poly = dijkstra_path_from_prob(
        prob64,
        start_xy=(sx_64, sy_64),
        goal_xy=(gx_64, gy_64),
        connectivity=8,
        eps=1e-6,
        prob_floor=None
    )

    if poly is None:
        poly = [(sx_64, sy_64), (gx_64, gy_64)]

    # Smooth the same polyline that will be plotted in blue
    poly = remove_collinear_points(poly, angle_tol_deg=2.0)
    poly = chaikin_smooth(poly, n_iters=5, keep_ends=True)
    poly = resample_polyline(poly, n=160)        # 120–220 is a good range
    poly = smooth_ema(poly, alpha=0.22, keep_ends=True)
    poly = clip_path_to_bounds(poly, W=64, H=64)
    
    # ============================================================
    # NEW: Full-resolution (1920×1080) overlay on the original image
    # ============================================================
    poly_orig = upscale_poly64_to_orig(poly, ORIG_W, ORIG_H)
    
    
    # ============================================================
    # NEW: Save 1920×1080 pixel trajectory to CSV (same as overlay)
    # ============================================================

    pixel_csv_path = pixel_csv_dir / f"{sample_id}_traj_pixels_1920x1080_conv2.csv"
    with open(pixel_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Convention-2 pixels: origin bottom-right, +x up, +y left
        # x_px spans [0..ORIG_H-1], y_px spans [0..ORIG_W-1]
        w.writerow(["idx", "x_px_conv2", "y_px_conv2"])

        for i, (x_o, y_o) in enumerate(poly_orig):
            # Convert the SAME plotted path (convention-1 pixels) into convention-2 pixels
            x2_f, y2_f = convert_pixels_conv1_to_conv2(x_o, y_o, W=ORIG_W, H=ORIG_H)

            # Round to nearest integer pixel and clamp to valid ranges
            x_i = int(np.clip(round(x2_f), 0, ORIG_H - 1))
            y_i = int(np.clip(round(y2_f), 0, ORIG_W - 1))
            w.writerow([i, x_i, y_i])

        # ============================================================
        # NEW: ALSO save a debug CSV with BOTH conventions in one file
        # (conv1: origin top-left, +x right, +y down)
        # (conv2: origin bottom-right, +x up, +y left)
        # ============================================================
        pixel_csv_path_both = pixel_csv_dir / f"{sample_id}_traj_pixels_1920x1080_both.csv"
        with open(pixel_csv_path_both, "w", newline="") as f2:
            w2 = csv.writer(f2)
            w2.writerow(["idx", "x_px_conv1", "y_px_conv1", "x_px_conv2", "y_px_conv2"])
            for i, (x_o, y_o) in enumerate(poly_orig):
                # conv1 integer pixels (x spans [0..ORIG_W-1], y spans [0..ORIG_H-1])
                x1_i = int(np.clip(round(float(x_o)), 0, ORIG_W - 1))
                y1_i = int(np.clip(round(float(y_o)), 0, ORIG_H - 1))

                # conv2 integer pixels (x spans [0..ORIG_H-1], y spans [0..ORIG_W-1])
                x2_f, y2_f = convert_pixels_conv1_to_conv2(x1_i, y1_i, W=ORIG_W, H=ORIG_H)
                x2_i = int(np.clip(round(float(x2_f)), 0, ORIG_H - 1))
                y2_i = int(np.clip(round(float(y2_f)), 0, ORIG_W - 1))

                w2.writerow([i, x1_i, y1_i, x2_i, y2_i])

        plt.figure(figsize=(ORIG_W / 200.0, ORIG_H / 200.0))
        plt.imshow(rgb_np, interpolation="nearest")
        plt.axis("off")

        # Draw the same smoothed trajectory (upscaled to original pixels)
        xs_o = [p[0] for p in poly_orig]
        ys_o = [p[1] for p in poly_orig]
        plt.plot(xs_o, ys_o, linewidth=4, color="blue")

        # Start/goal markers in ORIGINAL pixel space (you already have them in HARDCODED_POINTS)
        plt.scatter([sx_raw], [sy_raw], marker="+", s=600, linewidths=6, color="#39FF14")
        plt.scatter([gx_raw], [gy_raw], marker="+", s=600, linewidths=6, color="red")

        out_path_1080 = overlay_dir_1080 / f"{sample_id}_single_1920x1080.png"
        plt.tight_layout(pad=0)
        plt.savefig(out_path_1080, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

        # --- Export world coordinates (x,y,z) to CSV ---
        x_plot = []
        y_plot = []
        world_xyz = path64_to_world_xyz(poly, image_path=str(rgb_path), z_m=WORLD_Z_M)
        world_xyz = np.array(world_xyz, dtype=float) 
        world_xyz[:, 0] *= -1   # x flip
        world_xyz[:, 1] *= -1   # y flip
        
        
        xy = world_xyz[:, :2]
        xy_s = np.zeros_like(xy)
        xy_s[0] = xy[0]
        alpha = 0.25  # 0.15 smoother, 0.35 less smooth
        for i in range(1, len(xy)):
            xy_s[i] = alpha * xy[i] + (1.0 - alpha) * xy_s[i-1]

        world_xyz[:, 0] = xy_s[:, 0]
        world_xyz[:, 1] = xy_s[:, 1]
        # world_xyz = np.array(world_xyz) # list of (x_m, y_m, z_m) -> numpy array for easy saving/plotting
        # print(world_xyz)
        # print('--------------------------------------------')
        # world_xyz = np.flip(world_xyz, axis=0) # reverse order to match original path direction (start->goal)
        
        csv_path = world_csv_dir / f"{sample_id}_traj_world.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "x_m", "y_m", "z_m"])
            for i, (x_m, y_m, z_m) in enumerate(world_xyz):
                w.writerow([i, x_m, y_m, z_m])
                x_plot.append(x_m)
                y_plot.append(y_m)
        
        # print(world_xyz)
        # --- Plot world coordinates (x,y) in meters ---
        plt.figure(figsize=(4, 4))
        plt.plot(x_plot, y_plot, marker="o", color="blue")
        plt.scatter(x_plot[0], y_plot[0], marker="+", s=200, linewidths=5, color="#39FF14", label="Start")
        plt.scatter(x_plot[-1], y_plot[-1], marker="+", s=200, linewidths=5, color="red", label="Goal")
        plt.title(f"{sample_id} - World Coordinates (Z={WORLD_Z_M}m)")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.grid()
        world_plot_path = world_csv_dir / f"{sample_id}_traj_world_plot.png"
        plt.savefig(world_plot_path, dpi=150)
        plt.close()

        # Visualize (original behavior: 64x64 overlay)
        rgb_vis = rgb_t[0].detach().cpu().permute(1, 2, 0).numpy()
        rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

        plt.figure(figsize=(3, 3))
        plt.imshow(rgb_vis, interpolation="nearest")
        plt.axis("off")

        plt.imshow(prob64, alpha=0.35, interpolation="nearest")

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        plt.plot(xs, ys, linewidth=2, color="blue")

        plt.scatter([sx_64], [sy_64], marker="+", s=200, linewidths=5, color="#39FF14")
        plt.scatter([gx_64], [gy_64], marker="+", s=200, linewidths=5, color="red")

        out_path = overlay_dir / f"{sample_id}_single.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

print(f"\n[DONE] Saved single-run overlays to: {overlay_dir}")