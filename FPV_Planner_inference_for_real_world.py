#!/usr/bin/env python3
"""
Two-segment DDPM top-view trajectory stitching (hardcoded A->B->C):

- Run #1: A -> B (hardcoded intermediate)
- Run #2: B -> C (hardcoded final)
- Then stitch the two extracted polylines and overlay on the image.

You only need to change:
- test_image_root
- ckpt_path
- HARDCODED_POINTS_ABC (must match #images) in 512x512 coords (or whatever your image size is;
  we rescale from the actual loaded image size to 64x64).
"""

import time
import heapq
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import csv

from indoor_utils import (
    convert_pixels_conv1_to_conv2,
    ensure_dir,
    indoor_VLA_utils,
    scale_point_to_target,
    upscale_poly64_to_orig,
)

from src.utils.arguments import get_configuration
from src.models.model import get_model
from src.utils.configs import DataDict

# ============================================================
# Camera -> world conversion parameters
# ============================================================
CAMERA_HEIGHT_M = 4.4   # meters (camera height above ground)
DIAGONAL_FOV_DEG = 78.0 # degrees
WORLD_Z_M = 1.0         # constant z for exported world coordinates

# Function: Find the nearest true pixel in a boolean mask around a requested coordinate.
def nearest_true_pixel(mask: np.ndarray, xy, max_r=6):
    """Find nearest pixel (within max_r) where mask==True. Return xy unchanged if none found."""
    H, W = mask.shape
    x0, y0 = int(xy[0]), int(xy[1])
    best = None
    best_d = 1e9
    for r in range(max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    d = dx*dx + dy*dy
                    if d < best_d:
                        best_d = d
                        best = (x, y)
        if best is not None:
            return best
    return (x0, y0)


# Function: Run Dijkstra over the full probability grid when thresholded paths fail.
def dijkstra_full_grid(prob64: np.ndarray,
                       start_xy, goal_xy,
                       eps=1e-6,
                       diag_cost=1.414,
                       lam_turn=0.2,
                       low_prob_penalty=3.0):
    """
    Dijkstra over *all* pixels (no threshold/CC).
    Cost encourages high-prob pixels but still allows passing through low-prob areas.
    """
    H, W = prob64.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    INF = 1e18
    dist = np.full((H, W), INF, dtype=np.float64)
    prev = np.full((H, W, 2), -1, dtype=np.int32)
    prev_dir = np.zeros((H, W, 2), dtype=np.int32)

    pq = []
    dist[sy, sx] = 0.0
    heapq.heappush(pq, (0.0, sx, sy))

    nbrs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),         (1, 0),
            (-1, 1), (0, 1), (1, 1)]

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != dist[y, x]:
            continue
        if (x, y) == (gx, gy):
            break

        pdx, pdy = prev_dir[y, x]

        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue

            step = diag_cost if (dx != 0 and dy != 0) else 1.0
            p = float(prob64[ny, nx])

            # main term: prefer higher prob
            c_prob = -np.log(p + eps)

            # extra penalty when prob is extremely low (prevents silly wandering)
            c_low = low_prob_penalty * max(0.0, 0.10 - p)  # tweak 0.10 as you like

            c_turn = 0.0
            if lam_turn > 0.0 and (pdx != 0 or pdy != 0):
                c_turn = lam_turn * (0.0 if (pdx, pdy) == (dx, dy) else 1.0)

            nd = d + step + c_prob + c_low + c_turn
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx] = (x, y)
                prev_dir[ny, nx] = (dx, dy)
                heapq.heappush(pq, (nd, nx, ny))

    if (sx, sy) != (gx, gy) and prev[gy, gx, 0] < 0:
        return None

    path = [(gx, gy)]
    x, y = gx, gy
    while (x, y) != (sx, sy):
        px, py = prev[y, x]
        if px < 0:
            return None
        x, y = int(px), int(py)
        path.append((x, y))
    path.reverse()
    return path

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

    # Denominators for percent conversion (avoid divide-by-zero)
    denom_w = max(1.0, orig_w - 1.0)  # width denom
    denom_h = max(1.0, orig_h - 1.0)  # height denom

    optimized_coordinates = {}

    for i, (x_orig, y_orig) in enumerate(upscale_poly64_to_orig(poly64, orig_w, orig_h)):
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
        x_m = round(float(xy[0]), 2)
        y_m = round(float(xy[1]), 2)
        world_xyz.append((x_m, y_m, float(z_m)))
        

    return world_xyz

# ============================================================
# Inference wrapper: one DDPM call
# ============================================================
@torch.no_grad()
# Function: Run one DDPM sampling call for a start-goal pair and return the trajectory probability map.
def run_ddpm_once(model, rgb_t, start_xy_64, goal_xy_64, device):
    start_px = torch.tensor([[start_xy_64[0], start_xy_64[1]]], device=device, dtype=torch.long)
    end_px   = torch.tensor([[goal_xy_64[0],  goal_xy_64[1]]],  device=device, dtype=torch.long)

    input_dict = {
        DataDict.camera: rgb_t,
        "rgb": rgb_t,
        "start_px": start_px,
        "end_px": end_px,
    }

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_dict, sample=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    pred = out[DataDict.prediction]        # (1,3,64,64)
    pred_traj = pred[:, 2:3, :, :]         # (1,1,64,64)
    prob64 = torch.sigmoid(pred_traj)[0, 0].detach().cpu().numpy()  # (64,64)

    return pred_traj, prob64, (t1 - t0) * 1000.0


# ============================================================
# Robust segment extraction: CC from start + Dijkstra on -log(prob)
# ============================================================
# Function: Return the connected mask component reachable from the start pixel.
def connected_component_from_start(mask: np.ndarray, start_xy):
    """Keep only 8-connected component that contains start."""
    H, W = mask.shape
    sx, sy = start_xy
    if not (0 <= sx < W and 0 <= sy < H) or (not mask[sy, sx]):
        return np.zeros_like(mask, dtype=bool)

    visited = np.zeros_like(mask, dtype=bool)
    stack = [(sx, sy)]
    visited[sy, sx] = True

    nbrs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),         (1, 0),
            (-1, 1), (0, 1), (1, 1)]

    while stack:
        x, y = stack.pop()
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and (not visited[ny, nx]) and mask[ny, nx]:
                visited[ny, nx] = True
                stack.append((nx, ny))

    return visited


# Function: Run Dijkstra on a thresholded high-probability component.
def dijkstra_path_on_prob(prob64: np.ndarray,
                          start_xy,
                          goal_xy,
                          top_percentile=95.0,
                          eps=1e-6,
                          diag_cost=1.414,
                          lam_turn=0.2):
    """
    Build threshold mask from prob, keep CC from start, then Dijkstra to goal.

    Cost = step_length + (-log(prob+eps)) + lam_turn * (turn_indicator)

    Returns:
      list[(x,y)] or None
    """
    H, W = prob64.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    pct_try = [top_percentile, 92.0, 90.0, 88.0, 85.0]
    cc = None
    for pct in pct_try:
        th = np.percentile(prob64, pct)
        mask = prob64 >= th
        cc = connected_component_from_start(mask, start_xy)
        if 0 <= gx < W and 0 <= gy < H and cc[gy, gx]:
            break

    if cc is None or not (0 <= gx < W and 0 <= gy < H and cc[gy, gx]):
        return None

    INF = 1e18
    dist = np.full((H, W), INF, dtype=np.float64)
    prev = np.full((H, W, 2), -1, dtype=np.int32)
    prev_dir = np.zeros((H, W, 2), dtype=np.int32)

    pq = []
    dist[sy, sx] = 0.0
    heapq.heappush(pq, (0.0, sx, sy))

    nbrs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),         (1, 0),
            (-1, 1), (0, 1), (1, 1)]

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != dist[y, x]:
            continue
        if (x, y) == (gx, gy):
            break

        pdx, pdy = prev_dir[y, x]

        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not cc[ny, nx]:
                continue

            step = diag_cost if (dx != 0 and dy != 0) else 1.0
            p = float(prob64[ny, nx])
            c_prob = -np.log(p + eps)

            c_turn = 0.0
            if lam_turn > 0.0 and (pdx != 0 or pdy != 0):
                c_turn = lam_turn * (0.0 if (pdx, pdy) == (dx, dy) else 1.0)

            nd = d + step + c_prob + c_turn
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx] = (x, y)
                prev_dir[ny, nx] = (dx, dy)
                heapq.heappush(pq, (nd, nx, ny))

    if (sx, sy) != (gx, gy) and prev[gy, gx, 0] < 0:
        return None

    path = [(gx, gy)]
    x, y = gx, gy
    while (x, y) != (sx, sy):
        px, py = prev[y, x]
        if px < 0:
            return None
        x, y = int(px), int(py)
        path.append((x, y))
    path.reverse()
    return path


# ============================================================
# Post-processing (optional)
# ============================================================
# Function: Remove points that create implausibly large jumps in a path.
def clean_path_big_jumps(points, max_jump=8.0):
    if points is None or len(points) < 2:
        return points
    cleaned = [points[0]]
    for p in points[1:]:
        prev = cleaned[-1]
        if np.hypot(p[0] - prev[0], p[1] - prev[1]) <= max_jump:
            cleaned.append(p)
    return cleaned if len(cleaned) >= 2 else points


# Function: Resample a polyline at approximately uniform arclength intervals.
def resample_by_arclength(points, n_samples=200):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 2:
        return points
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return points
    s_new = np.linspace(0, s[-1], n_samples)
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    return list(zip(x_new.tolist(), y_new.tolist()))


# Function: Smooth a path with a spline when SciPy is available, otherwise use Chaikin smoothing.
def smooth_path_spline(points, n_samples=200, smoothness=2.0):
    if points is None or len(points) < 4:
        return points

    points_rs = resample_by_arclength(points, n_samples=max(60, n_samples))
    pts = np.array(points_rs, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]

    try:
        from scipy.interpolate import splprep, splev

        d = np.hypot(np.diff(x), np.diff(y))
        u = np.concatenate([[0.0], np.cumsum(d)])
        u = u / (u[-1] + 1e-8)

        tck, _ = splprep([x, y], u=u, s=float(smoothness), k=3)
        u_new = np.linspace(0.0, 1.0, n_samples)
        x_s, y_s = splev(u_new, tck)
        return list(zip(x_s, y_s))

    except ImportError:
        # Function: Apply Chaikin corner cutting as a spline-free smoothing fallback.
        def chaikin(pts_in, iters=3):
            pts_in = np.array(pts_in, dtype=np.float32)
            for _ in range(iters):
                new_pts = [pts_in[0]]
                for i in range(len(pts_in) - 1):
                    p = pts_in[i]
                    q = pts_in[i + 1]
                    new_pts.append(0.75 * p + 0.25 * q)
                    new_pts.append(0.25 * p + 0.75 * q)
                new_pts.append(pts_in[-1])
                pts_in = np.array(new_pts, dtype=np.float32)
            return pts_in

        out = chaikin(points_rs, iters=3)
        return resample_by_arclength(out.tolist(), n_samples=n_samples)


# ============================================================
# New: hardcoded 2-run inference (A->B, B->C) + stitch
# ============================================================
# Function: Run one DDPM segment and extract a robust start-to-goal polyline.
def ddpm_segment_polyline(model, rgb_t, start_xy_64, goal_xy_64, device,
                          poly_top_percentile=95.0,
                          lam_turn=0.2,
                          snap_radius=6,
                          fullgrid_fallback=True):
    """
    One DDPM inference for (start->goal), then extract a robust polyline.

    Strategy:
      A) Try CC+threshold Dijkstra (fast, clean)
      B) If start/goal not in CC, snap them to nearest CC pixel
      C) If still fails, fall back to full-grid Dijkstra (always continuous)
    """
    _, prob64, ms = run_ddpm_once(model, rgb_t, start_xy_64, goal_xy_64, device)

    start_xy_64 = (int(start_xy_64[0]), int(start_xy_64[1]))
    goal_xy_64  = (int(goal_xy_64[0]),  int(goal_xy_64[1]))

    seg = None
    used_mode = "cc"

    # --- Try CC/threshold path first (with adaptive percentiles inside) ---
    seg = dijkstra_path_on_prob(
        prob64,
        start_xy=start_xy_64,
        goal_xy=goal_xy_64,
        top_percentile=poly_top_percentile,
        lam_turn=lam_turn
    )

    # If failed, try snapping endpoints into the reachable CC and retry
    if seg is None:
        # build a mask similar to your CC logic (use the same percentile ladder)
        pct_try = [poly_top_percentile, 92.0, 90.0, 88.0, 85.0, 80.0]
        for pct in pct_try:
            th = np.percentile(prob64, pct)
            mask = (prob64 >= th)
            cc = connected_component_from_start(mask, start_xy_64)
            if cc.any():
                s2 = nearest_true_pixel(cc, start_xy_64, max_r=snap_radius)
                g2 = nearest_true_pixel(cc, goal_xy_64,  max_r=snap_radius)
                if cc[g2[1], g2[0]]:  # goal snapped into reachable CC
                    seg = dijkstra_path_on_prob(
                        prob64,
                        start_xy=s2,
                        goal_xy=g2,
                        top_percentile=pct,
                        lam_turn=lam_turn
                    )
                    if seg is not None:
                        # force exact endpoints back
                        seg[0] = start_xy_64
                        seg[-1] = goal_xy_64
                        break

    # --- Full-grid fallback (never threshold) ---
    if (seg is None) and fullgrid_fallback:
        used_mode = "fullgrid"
        seg = dijkstra_full_grid(
            prob64,
            start_xy=start_xy_64,
            goal_xy=goal_xy_64,
            lam_turn=lam_turn
        )

    if seg is None or len(seg) < 2:
        seg = [start_xy_64, goal_xy_64]
        used_mode = "line"

    debug = {
        "start": tuple(start_xy_64),
        "goal": tuple(goal_xy_64),
        "infer_ms": ms,
        "prob_max": float(prob64.max()),
        "prob_mean": float(prob64.mean()),
        "mode": used_mode
    }
    return seg, debug


# Function: Concatenate two paths while avoiding a duplicated joint endpoint.
def stitch_paths(path1, path2):
    """
    Concatenate two polylines without duplicating the joint point.
    """
    if not path1:
        return path2
    if not path2:
        return path1
    if path1[-1] == path2[0]:
        return path1 + path2[1:]
    return path1 + path2


# Function: Run DDPM for A-to-B and B-to-C and stitch the two segments together.
def run_two_goal_ddpm_path(model, rgb_t, points_abc_64, device,
                           poly_top_percentile=95.0,
                           lam_turn=0.2,
                           do_smooth=True,
                           jump_max=8.0,
                           spline_samples=220,
                           spline_smoothness=2.0):
    """
    Run DDPM planning for A->B and B->C, then return the stitched path.
    """
    A, B, C = points_abc_64

    seg1, dbg1 = ddpm_segment_polyline(
        model=model, rgb_t=rgb_t,
        start_xy_64=A, goal_xy_64=B,
        device=device,
        poly_top_percentile=poly_top_percentile,
        lam_turn=lam_turn
    )
    seg2, dbg2 = ddpm_segment_polyline(
        model=model, rgb_t=rgb_t,
        start_xy_64=B, goal_xy_64=C,
        device=device,
        poly_top_percentile=poly_top_percentile,
        lam_turn=lam_turn
    )

    stitched_path = stitch_paths(seg1, seg2)
    if do_smooth:
        stitched_path = clean_path_big_jumps(stitched_path, max_jump=jump_max)
        stitched_path = smooth_path_spline(
            stitched_path,
            n_samples=spline_samples,
            smoothness=spline_smoothness
        )

    return stitched_path, (dbg1, dbg2)


# Function: Build the model, load checkpoint weights, and switch it to evaluation mode.
def load_model(cfgs, ckpt_path, device):
    model = get_model(cfgs.model, device=device).to(device)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
    model.eval()
    return model


# Function: Load an RGB image and resize it into the model input tensor shape.
def load_rgb_tensor(rgb_path, device, size=64):
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
    rgb_t = F.interpolate(rgb_t, size=(size, size), mode="bilinear", align_corners=False).to(device)
    return rgb_np, rgb_t


# Function: Save trajectory pixels in project coordinate conventions for downstream tools.
def save_pixel_csvs(poly_orig, orig_w, orig_h, sample_id, pixel_csv_dir):
    conv2_path = pixel_csv_dir / f"{sample_id}_traj_pixels_1920x1080_conv2.csv"
    with open(conv2_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "x_px_conv2", "y_px_conv2"])
        for i, (x_o, y_o) in enumerate(poly_orig):
            x2_f, y2_f = convert_pixels_conv1_to_conv2(x_o, y_o, W=orig_w, H=orig_h)
            x_i = int(np.clip(round(x2_f), 0, orig_h - 1))
            y_i = int(np.clip(round(y2_f), 0, orig_w - 1))
            writer.writerow([i, x_i, y_i])

    both_path = pixel_csv_dir / f"{sample_id}_traj_pixels_1920x1080_both.csv"
    with open(both_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "x_px_conv1", "y_px_conv1", "x_px_conv2", "y_px_conv2"])
        for i, (x_o, y_o) in enumerate(poly_orig):
            x1_i = int(np.clip(round(float(x_o)), 0, orig_w - 1))
            y1_i = int(np.clip(round(float(y_o)), 0, orig_h - 1))
            x2_f, y2_f = convert_pixels_conv1_to_conv2(x1_i, y1_i, W=orig_w, H=orig_h)
            x2_i = int(np.clip(round(float(x2_f)), 0, orig_h - 1))
            y2_i = int(np.clip(round(float(y2_f)), 0, orig_w - 1))
            writer.writerow([i, x1_i, y1_i, x2_i, y2_i])


# Function: Save a full-resolution overlay of the predicted path on the source image.
def save_original_overlay(rgb_np, poly_orig, start_raw, goal_raw, sample_id, overlay_dir):
    orig_h, orig_w = rgb_np.shape[:2]
    plt.figure(figsize=(orig_w / 200.0, orig_h / 200.0))
    plt.imshow(rgb_np, interpolation="nearest")
    plt.axis("off")
    plt.plot([p[0] for p in poly_orig], [p[1] for p in poly_orig], linewidth=4, color="blue")
    plt.scatter([start_raw[0]], [start_raw[1]], marker="+", s=600, linewidths=6, color="#39FF14")
    plt.scatter([goal_raw[0]], [goal_raw[1]], marker="+", s=600, linewidths=6, color="red")
    plt.tight_layout(pad=0)
    plt.savefig(overlay_dir / f"{sample_id}_single_1920x1080.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


# Function: Export world-coordinate CSVs and plots for a predicted trajectory.
def save_world_outputs(stitched_path, rgb_path, sample_id, world_csv_dir):
    world_xyz = np.array(path64_to_world_xyz(stitched_path, image_path=str(rgb_path), z_m=WORLD_Z_M), dtype=float)
    if world_xyz.size == 0:
        return

    world_xyz[:, 0] *= -1
    world_xyz[:, 1] *= -1

    csv_path = world_csv_dir / f"{sample_id}_traj_world.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "x_m", "y_m", "z_m"])
        for i, (x_m, y_m, z_m) in enumerate(world_xyz):
            writer.writerow([i, x_m, y_m, z_m])

    plt.figure(figsize=(4, 4))
    plt.plot(world_xyz[:, 0], world_xyz[:, 1], marker="o", color="blue")
    plt.scatter(world_xyz[0, 0], world_xyz[0, 1], marker="+", s=200, linewidths=5, color="#39FF14", label="Start")
    plt.scatter(world_xyz[-1, 0], world_xyz[-1, 1], marker="+", s=200, linewidths=5, color="red", label="Goal")
    plt.title(f"{sample_id} - World Coordinates (Z={WORLD_Z_M}m)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid()
    plt.savefig(world_csv_dir / f"{sample_id}_traj_world_plot.png", dpi=150)
    plt.close()


# Function: Save a compact 64x64 visualization with A, B, and C markers.
def save_64_overlay(rgb_t, stitched_path, points_abc_64, sample_id, overlay_dir):
    A, B, C = points_abc_64
    rgb_vis = rgb_t[0].detach().cpu().permute(1, 2, 0).numpy()
    rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

    plt.figure(figsize=(3, 3))
    plt.imshow(rgb_vis, interpolation="nearest")
    plt.axis("off")
    plt.plot([p[0] for p in stitched_path], [p[1] for p in stitched_path], linewidth=2, color="blue")
    plt.scatter([A[0]], [A[1]], marker="+", s=200, linewidths=5, color="#39FF14")
    plt.scatter([B[0]], [B[1]], marker="+", s=200, linewidths=5, color="cyan")
    plt.scatter([C[0]], [C[1]], marker="+", s=200, linewidths=5, color="red")
    plt.tight_layout()
    plt.savefig(overlay_dir / f"{sample_id}_A_B_C.png", dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================
# Function: Script entry point that assembles configuration and launches the requested workflow.
def main():
    cfgs = get_configuration()

    # -----------------------------
    # EDIT THESE PATHS
    # -----------------------------
    test_image_root = Path("/home/isr-lab3/Faryal_Batool/DTG-main_top/Experiment_annotations")
    ckpt_path = Path("/home/isr-lab3/Faryal_Batool/DTG-main_v6/results/models/hnav_29.pth")

    image_paths = sorted([p for p in test_image_root.iterdir()
                          if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    print(f"[INFO] Total test images found: {len(image_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = load_model(cfgs, ckpt_path, device)

    # Output
    output_root = Path("Experiments_ICUAS_fpv")
    ensure_dir(output_root)
    overlay_dir = ensure_dir(output_root / "overlays_64")
    overlay_dir_1080 = ensure_dir(output_root / "overlays_single_1920x1080")
    pixel_csv_dir = ensure_dir(output_root / "pixel_csv_1920x1080")
    world_csv_dir = ensure_dir(output_root / "world_csv")


    # ------------------------------------------------------------
    # IMPORTANT: one (A,B,C) triple per image, given in 512x512 coords
    # (we rescale from the actual loaded image size to 64x64)
    # ------------------------------------------------------------
    # HARDCODED_POINTS_ABC = [
    #     # Example (replace with your real triples, one per image):
    #     # ((Ax,Ay), (Bx,By), (Cx,Cy)),
    #     ((550, 1041),((1122, 548)), (1398, 16)),  # s10_hard_obstacle
    #     ((549, 1042),((1122, 548)), (1396, 16)),  # s10_soft_obstacle
    #     ((545, 1045),((1122, 548)), (1394, 16)),  # s1_hard_obstacle
    #     ((546, 1042),((1122, 548)), (1396, 18)),  # s1_soft_obstacle
    #     ((545, 1042),((1122, 548)), (1400, 18)),  # s2_hard_obstacle
    #     ((546, 1041), ((1122, 548)),(1396, 16)),  # s2_soft_obstacle
    #     ((546, 1044),((1122, 548)), (1398, 16)),  # s3_hard_obstacle
    #     ((546, 1044),((1122, 548)), (1398, 16)),  # s3_soft_obstacle
    #     ((549, 1042),((1122, 548)), (1398, 15)),  # s4_hard_obstacle
    #     ((546, 1041),((1122, 548)), (1396, 18)),  # s4_soft_obstacle
    #     ((546, 1041), ((1122, 548)),(1398, 18)),  # s5_hard_obstacle
    #     ((641, 1023),((1122, 548)), (1398, 15)),  # s5_soft_obstacle
    #     ((549, 1044), ((1122, 548)),(1398, 13)),  # s6_hard_obstacle
    #     ((555, 1049),((1122, 548)), (1394, 15)),  # s6_soft_obstacle
    #     ((548, 1041), ((1122, 548)),(1396, 16)),  # s7_hard_obstacle
    #     ((548, 1041), ((1122, 548)),(1399, 19)),  # s7_soft_obstacle
    #     ((558, 1042), ((1122, 548)),(1395, 20)),  # s8_hard_obstacle
    #     ((559, 1041),((1122, 548)), (1398, 18)),  # s8_soft_obstacle
    #     ((554, 1041),((1122, 548)), (1394, 18)),  # s9_hard_obstacle
    #     ((556, 1046),((1122, 548)), (1395, 20)),  # s9_soft_obstacl
    # ]
    HARDCODED_POINTS_ABC= [
    # ((526, 1003), (1102, 417),(1567, 8)),  # exp_1
    # ((527, 1003), (1000, 400), (1583, 8)),  # exp_2a
    # ((511, 1005), (1098, 401), (1555, 9)),  # exp_2b
    # ((527, 1003),(1000, 450), (1583, 8)),  # exp_3
    # ((545, 1071),(1098, 401), (1512, 13)),  # exp_3
    # ((500, 996), (969, 467), (1517, 19)),  # exp_4_dynamic
    # ((502, 1071),(900, 250), (1522, 12)),  # exp_5
    ((502, 1071),(900, 250), (1522, 12)),  # exp_6
    # ((550, 1071),(1001, 483), (1560, 14)),  # exp_7
    #  ((549, 1069),(1001, 483), (1552, 11)),  # exp_8
    # ((551, 1068),(1001, 483), (1550, 11)),  # exp_9



]



    assert len(HARDCODED_POINTS_ABC) == len(image_paths), \
        f"HARDCODED_POINTS_ABC({len(HARDCODED_POINTS_ABC)}) must match images({len(image_paths)})"

    # Path extraction params
    POLY_TOP_PCT = 95.0
    LAM_TURN = 0.2

    # Optional final smoothing
    DO_FINAL_SMOOTH = True
    JUMP_MAX = 8.0
    SPLINE_SAMPLES = 220
    SPLINE_SMOOTHNESS = 2.0

    for idx, rgb_path in enumerate(image_paths):
        sample_id = rgb_path.stem
        print(f"\n[INFO] Processing {sample_id}")

        rgb_np, rgb_t = load_rgb_tensor(rgb_path, device, size=64)
        ORIG_H, ORIG_W = rgb_np.shape[0], rgb_np.shape[1]

        # Scale A,B,C from original image space -> 64x64
        (ax_raw, ay_raw), (bx_raw, by_raw), (cx_raw, cy_raw) = HARDCODED_POINTS_ABC[idx]
        ax_64, ay_64 = scale_point_to_target(ax_raw, ay_raw, (ORIG_H, ORIG_W), (64, 64))
        bx_64, by_64 = scale_point_to_target(bx_raw, by_raw, (ORIG_H, ORIG_W), (64, 64))
        cx_64, cy_64 = scale_point_to_target(cx_raw, cy_raw, (ORIG_H, ORIG_W), (64, 64))

        A = (ax_64, ay_64)
        B = (bx_64, by_64)
        C = (cx_64, cy_64)

        stitched_path, (dbg1, dbg2) = run_two_goal_ddpm_path(
            model=model,
            rgb_t=rgb_t,
            points_abc_64=(A, B, C),
            device=device,
            poly_top_percentile=POLY_TOP_PCT,
            lam_turn=LAM_TURN,
            do_smooth=DO_FINAL_SMOOTH,
            jump_max=JUMP_MAX,
            spline_samples=SPLINE_SAMPLES,
            spline_smoothness=SPLINE_SMOOTHNESS
        )

        print(f"[INFO] A={A} B={B} C={C}")
        print(f"  [Run1] start={dbg1['start']} goal={dbg1['goal']} infer={dbg1['infer_ms']:.1f}ms "
              f"prob_max={dbg1['prob_max']:.3f}")
        print(f"  [Run2] start={dbg2['start']} goal={dbg2['goal']} infer={dbg2['infer_ms']:.1f}ms "
              f"prob_max={dbg2['prob_max']:.3f}")
        
        poly_orig = upscale_poly64_to_orig(stitched_path, ORIG_W, ORIG_H)
        save_pixel_csvs(poly_orig, ORIG_W, ORIG_H, sample_id, pixel_csv_dir)
        save_original_overlay(rgb_np, poly_orig, (ax_raw, ay_raw), (cx_raw, cy_raw), sample_id, overlay_dir_1080)
        save_world_outputs(stitched_path, rgb_path, sample_id, world_csv_dir)
        save_64_overlay(rgb_t, stitched_path, (A, B, C), sample_id, overlay_dir)

    print(f"\n[DONE] Saved overlays to: {overlay_dir}")


if __name__ == "__main__":
    main()
