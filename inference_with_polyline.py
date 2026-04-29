# Module: Top-view DDPM inference script for single start-goal planning.

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.arguments import get_configuration
from src.models.model import get_model
from src.utils.configs import DataDict
import heapq



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
# Optional: extract a simple polyline from probability map
# (greedy stepping toward goal, staying in high-prob areas)
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

    Args:
      prob64: (H,W) probability map in [0,1]
      start_xy, goal_xy: (x,y)
      connectivity: 4 or 8
      eps: avoids log(0)
      prob_floor: if not None, treat cells with prob < prob_floor as blocked
                  (useful if you want to force path through high-prob corridor)

    Returns:
      path: list[(x,y)] from start to goal, or None if unreachable
    """
    H, W = prob64.shape
    sx, sy = map(int, start_xy)
    gx, gy = map(int, goal_xy)

    # Clamp and convert prob -> cost
    p = np.clip(prob64, 0.0, 1.0)
    cost_map = -np.log(p + eps)  # lower cost where prob is high

    # Optional: block very low-prob cells
    if prob_floor is not None:
        blocked = p < prob_floor
    else:
        blocked = np.zeros((H, W), dtype=bool)

    # Start/goal must be unblocked
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

    # Dijkstra structures
    INF = 1e18
    dist = np.full((H, W), INF, dtype=np.float64)
    prev = np.full((H, W, 2), -1, dtype=np.int16)  # store previous (x,y)

    dist[sy, sx] = 0.0
    pq = [(0.0, sx, sy)]  # (dist, x, y)

    # Early exit if start==goal
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    while pq:
        d, x, y = heapq.heappop(pq)

        # stale entry
        if d != dist[y, x]:
            continue

        # reached goal
        if (x, y) == (gx, gy):
            break

        for dx, dy in nbrs:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if blocked[ny, nx]:
                continue

            # movement cost: encourage shorter paths + follow high-prob regions
            step_len = np.hypot(dx, dy)  # 1 or sqrt(2)
            step_cost = step_len * (cost_map[ny, nx])

            nd = d + step_cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx, 0] = x
                prev[ny, nx, 1] = y
                heapq.heappush(pq, (nd, nx, ny))

    # If goal was never reached
    if dist[gy, gx] >= INF/2:
        return None

    # Reconstruct path by backtracking
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
    """
    Removes near-collinear middle points to reduce "staircase" artifacts.
    Keeps endpoints.
    """
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

        # angle between segments
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        ang = np.arccos(cosang)

        # If angle is very small (almost straight), drop point
        if ang > tol:
            keep.append(i)

    keep.append(len(pts) - 1)
    pts2 = pts[keep]
    return [(float(x), float(y)) for x, y in pts2]


# Function: Smooth a path with Chaikin corner cutting.
def chaikin_smooth(path, n_iters=3, keep_ends=True):
    """
    Chaikin corner cutting: generates a smooth curve from a polyline.
    Keeps endpoints if keep_ends=True.
    """
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


# ============================================================
# MAIN (single-run inference per image)
# ============================================================
cfgs = get_configuration()

test_image_root = Path("/home/isr-lab3/Faryal_Batool/DTG-main_top/resized_dataset_images_2")
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

output_root = Path("test_results_SINGLE_ddpm")
overlay_dir = output_root / "overlays_single_64"
output_root.mkdir(parents=True, exist_ok=True)
overlay_dir.mkdir(parents=True, exist_ok=True)

HARDCODED_POINTS = [
    ((147, 494), (372, 10)),  # s10_hard
    ((147, 494), (372, 10)),  # s10_soft
    ((146, 500), (372, 15)),  # s1_hard
    ((146, 493), (371, 8)),   # s1_soft
    ((144, 494), (374, 10)),  # s2_hard
    ((146, 497), (372, 7)),   # s2_soft
    ((146, 495), (373, 8)),   # s3_hard
    ((145, 494), (374, 6)),   # s3_soft
    ((145, 495), (372, 10)),  # s4_hard
    ((146, 495), (372, 9)),   # s4_soft
    ((146, 493), (373, 6)),   # s5_hard
    ((200, 498), (373, 10)),  # s5_soft
    ((145, 494), (373, 6)),   # s6_hard
    ((149, 498), (374, 10)),  # s6_soft
    ((146, 493), (372, 9)),   # s7_hard
    ((151, 495), (372, 6)),   # s7_soft
    ((147, 499), (373, 10)),  # s8_hard
    ((152, 498), (373, 11)),  # s8_soft
    ((148, 497), (373, 10)),  # s9_hard
    ((149, 498), (372, 10)) # s9_soft
    
]
assert len(HARDCODED_POINTS) == len(image_paths), "❌ hardcoded points must match images!"

for idx, rgb_path in enumerate(image_paths):
    sample_id = rgb_path.stem
    print(f"\n[INFO] Processing {sample_id}")

    # Load RGB
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
    ORIG_H, ORIG_W = rgb_np.shape[0], rgb_np.shape[1]

    # resize RGB -> 64×64 for model
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
    rgb_t = F.interpolate(rgb_t, size=(64, 64), mode="bilinear", align_corners=False).to(device)

    # Scale (512-space) start/goal to this image size -> then to 64×64
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

    # Optional polyline extraction (so you still see a connected path)
    poly = dijkstra_path_from_prob(
    prob64,
    start_xy=(sx_64, sy_64),
    goal_xy=(gx_64, gy_64),
    connectivity=8,
    eps=1e-6,
    prob_floor=None  # or try 0.05 / 0.1 to force staying in high-prob corridor
)

    if poly is None:
        # fallback: straight line if Dijkstra can't connect
        poly = [(sx_64, sy_64), (gx_64, gy_64)]
    # --- Smooth the Dijkstra polyline ---
    
    poly = remove_collinear_points(poly, angle_tol_deg=2.0)  # try 1~5 deg
    poly = chaikin_smooth(poly, n_iters=5, keep_ends=True)   # try 2~5 iters
    poly = clip_path_to_bounds(poly, W=64, H=64)             # stay inside image

    # Visualize
    rgb_vis = rgb_t[0].detach().cpu().permute(1, 2, 0).numpy()
    rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

    plt.figure(figsize=(3, 3))
    plt.imshow(rgb_vis, interpolation="nearest")
    plt.axis("off")

    # draw probability heatmap lightly
    plt.imshow(prob64, alpha=0.35, interpolation="nearest")

    # draw polyline
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    plt.plot(xs, ys, linewidth=2, color="blue")

    # start/goal markers
    plt.scatter([sx_64], [sy_64], marker="+", s=200, linewidths=5, color="#39FF14")
    plt.scatter([gx_64], [gy_64], marker="+", s=200, linewidths=5, color="red")

    out_path = overlay_dir / f"{sample_id}_single.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

print(f"\n[DONE] Saved single-run overlays to: {overlay_dir}")



# import os
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from PIL import Image

# from src.utils.arguments import get_configuration
# from src.models.model import get_model
# from src.utils.configs import DataDict


# # ============================================================
# # Utility: scale pixel coordinate from (H0,W0) → (H,W)
# # ============================================================
# def scale_point_to_target(x_px, y_px, orig_hw, target_hw):
#     H0, W0 = orig_hw
#     Ht, Wt = target_hw
#     sx = Wt / float(W0)
#     sy = Ht / float(H0)
#     x2 = int(np.clip(round(x_px * sx), 0, Wt - 1))
#     y2 = int(np.clip(round(y_px * sy), 0, Ht - 1))
#     return x2, y2


# # ============================================================
# # Inference wrapper: run DDPM once and return traj logits/prob
# # ============================================================
# @torch.no_grad()
# def run_ddpm_once(model, rgb_t, start_xy_64, goal_xy_64, device):
#     """
#     start_xy_64, goal_xy_64: (x,y) ints in 64x64
#     Returns:
#       pred_traj_logits: (1,1,64,64) torch
#       pred_traj_prob:   (64,64) numpy in [0,1]
#     """
#     start_px = torch.tensor([[start_xy_64[0], start_xy_64[1]]], device=device, dtype=torch.long)
#     end_px   = torch.tensor([[goal_xy_64[0],  goal_xy_64[1]]],  device=device, dtype=torch.long)

#     input_dict = {
#         DataDict.camera: rgb_t,
#         "rgb": rgb_t,
#         "start_px": start_px,
#         "end_px": end_px,
#     }

#     if device == "cuda":
#         torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     out = model(input_dict, sample=True)
#     if device == "cuda":
#         torch.cuda.synchronize()
#     t1 = time.perf_counter()

#     pred = out[DataDict.prediction]              # (1,3,64,64)
#     pred_traj = pred[:, 2:3, :, :]               # (1,1,64,64) logits-ish
#     prob = torch.sigmoid(pred_traj)[0, 0].detach().cpu().numpy()  # (64,64)

#     return pred_traj, prob, (t1 - t0) * 1000.0


# # ============================================================
# # Choose intermediate point x on predicted traj "between" start & goal
# # ============================================================
# def choose_intermediate_point(traj_prob_64: np.ndarray,
#                               start_xy: tuple,
#                               goal_xy: tuple,
#                               top_percentile: float = 97.0,
#                               min_step_px: float = 6.0,
#                               goal_tol_px: float = 3.0):
#     """
#     Picks a waypoint x from the predicted trajectory probability map.

#     Strategy:
#       - take pixels in the top (100 - top_percentile)% probability mass
#       - among those, choose the one with maximum progress along the start->goal direction
#       - require that it is at least min_step_px away from start
#       - stop if start is already within goal_tol_px of goal

#     Returns:
#       next_xy (x,y) or None if can't find a useful intermediate.
#     """
#     sx, sy = start_xy
#     gx, gy = goal_xy

#     # If already close enough, no need for intermediate
#     if np.hypot(gx - sx, gy - sy) <= goal_tol_px:
#         return None

#     H, W = traj_prob_64.shape

#     # Threshold by percentile to get candidate pixels
#     th = np.percentile(traj_prob_64, top_percentile)
#     cand_y, cand_x = np.nonzero(traj_prob_64 >= th)
#     if len(cand_x) == 0:
#         return None

#     # Direction from start to goal
#     dx = gx - sx
#     dy = gy - sy
#     norm = np.hypot(dx, dy)
#     if norm < 1e-6:
#         return None
#     ux, uy = dx / norm, dy / norm

#     # Evaluate candidates by progress along direction
#     best = None
#     best_progress = -1e18

#     for x, y in zip(cand_x, cand_y):
#         # must be away from start
#         d_start = np.hypot(x - sx, y - sy)
#         if d_start < min_step_px:
#             continue

#         # progress along start->goal (projection)
#         progress = (x - sx) * ux + (y - sy) * uy

#         # We only want points that move toward the goal
#         if progress <= 0:
#             continue

#         # prefer points that are also somewhat "on the way" (optional)
#         # slight preference toward higher probability
#         score = progress + 0.5 * traj_prob_64[y, x]

#         if score > best_progress:
#             best_progress = score
#             best = (int(x), int(y))

#     return best


# # ============================================================
# # Iterative replanning: run DDPM multiple times to reach goal
# # ============================================================
# def iterative_ddpm_waypoints(model,
#                              rgb_t,
#                              start_xy_64,
#                              goal_xy_64,
#                              device,
#                              max_iters: int = 3,
#                              top_percentile: float = 97.0,
#                              min_step_px: float = 6.0,
#                              goal_tol_px: float = 3.0):
#     """
#     Re-runs DDPM multiple times:
#       start0 -> x1 -> x2 -> ... -> goal

#     Returns:
#       waypoints: list of (x,y) including start and goal
#       per_iter_info: list of dicts with timing/prob stats
#     """
#     waypoints = [tuple(map(int, start_xy_64))]
#     per_iter_info = []

#     cur = tuple(map(int, start_xy_64))
#     goal = tuple(map(int, goal_xy_64))

#     for k in range(max_iters):
#         # If close to goal, finish
#         if np.hypot(goal[0] - cur[0], goal[1] - cur[1]) <= goal_tol_px:
#             break

#         pred_traj, prob64, ms = run_ddpm_once(model, rgb_t, cur, goal, device)

#         # pick intermediate
#         nxt = choose_intermediate_point(
#             prob64, cur, goal,
#             top_percentile=top_percentile,
#             min_step_px=min_step_px,
#             goal_tol_px=goal_tol_px
#         )

#         per_iter_info.append({
#             "iter": k,
#             "start": cur,
#             "goal": goal,
#             "chosen": nxt,
#             "infer_ms": ms,
#             "prob_max": float(prob64.max()),
#             "prob_mean": float(prob64.mean()),
#         })

#         # If DDPM cannot propose a meaningful intermediate, stop
#         if nxt is None:
#             break

#         waypoints.append(nxt)
#         cur = nxt

#     # always append the final goal
#     if waypoints[-1] != goal:
#         waypoints.append(goal)

#     return waypoints, per_iter_info


# # ============================================================
# # MAIN (your inference loop + iterative logic)
# # ============================================================
# cfgs = get_configuration()

# test_image_root = Path("/home/isr-lab3/Faryal_Batool/DTG-main_top/resized_dataset_images_2")
# image_paths = sorted([p for p in test_image_root.iterdir()
#                       if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
# print(f"[INFO] Total test images found: {len(image_paths)}")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[INFO] Using device: {device}")

# model = get_model(cfgs.model, device=device).to(device)
# ckpt_path = r"/home/isr-lab3/Faryal_Batool/DTG-main_top/results/models/hnav_29.pth"
# print(f"[INFO] Loading checkpoint from: {ckpt_path}")
# state = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
# model.eval()

# output_root = Path("test_results_impedance_top_view")
# overlay_dir = output_root / "overlays_iterative_64"
# output_root.mkdir(parents=True, exist_ok=True)
# overlay_dir.mkdir(parents=True, exist_ok=True)

# # ------------------------------------------------------------
# # Your hardcoded points in 512×512 space.
# # IMPORTANT: must match len(image_paths)
# # ------------------------------------------------------------
# HARDCODED_POINTS = [
#     ((147, 494), (372, 10)),  # s10_hard
#     ((147, 494), (372, 10)),  # s10_soft
#     ((146, 500), (372, 15)),  # s1_hard
#     ((146, 493), (371, 8)),   # s1_soft
#     ((144, 494), (374, 10)),  # s2_hard
#     ((146, 497), (372, 7)),   # s2_soft
#     ((146, 495), (373, 8)),   # s3_hard
#     ((145, 494), (374, 6)),   # s3_soft
#     ((145, 495), (372, 10)),  # s4_hard
#     ((146, 495), (372, 9)),   # s4_soft
#     ((146, 493), (373, 6)),   # s5_hard
#     ((200, 498), (373, 10)),  # s5_soft
#     ((145, 494), (373, 6)),   # s6_hard
#     ((149, 498), (374, 10)),  # s6_soft
#     ((146, 493), (372, 9)),   # s7_hard
#     ((151, 495), (372, 6)),   # s7_soft
#     ((147, 499), (373, 10)),  # s8_hard
#     ((152, 498), (373, 11)),  # s8_soft
#     ((148, 497), (373, 10)),  # s9_hard
#     ((149, 498), (372, 10)) # s9_soft
    
# ]
# assert len(HARDCODED_POINTS) == len(image_paths), "❌ hardcoded points must match images!"

# # ------------------------------------------------------------
# # Hyperparameters for iterative chaining
# # ------------------------------------------------------------
# MAX_CHAIN_ITERS = 1          # run DDPM up to 3 times per image
# TOP_PCT = 97.0               # pick waypoint from top 3% heat
# MIN_STEP = 6.0               # must move at least this many pixels per hop (64x64 grid)
# GOAL_TOL = 3.0               # if within this, consider goal reached


# for idx, rgb_path in enumerate(image_paths):
#     sample_id = rgb_path.stem
#     print(f"\n[INFO] Processing {sample_id}")

#     # Load RGB
#     rgb_img = Image.open(rgb_path).convert("RGB")
#     rgb_np = np.array(rgb_img).astype(np.float32) / 255.0
#     ORIG_H, ORIG_W = rgb_np.shape[0], rgb_np.shape[1]

#     # resize RGB -> 64×64 for model
#     rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
#     rgb_t = F.interpolate(rgb_t, size=(64, 64), mode="bilinear", align_corners=False).to(device)

#     # Scale (512-space) start/goal to this image size -> then to 64×64
#     (sx_raw, sy_raw), (gx_raw, gy_raw) = HARDCODED_POINTS[idx]

#     sx_64, sy_64 = scale_point_to_target(sx_raw, sy_raw, (ORIG_H, ORIG_W), (64, 64))
#     gx_64, gy_64 = scale_point_to_target(gx_raw, gy_raw, (ORIG_H, ORIG_W), (64, 64))

#     # Run iterative chaining
#     waypoints, info = iterative_ddpm_waypoints(
#         model=model,
#         rgb_t=rgb_t,
#         start_xy_64=(sx_64, sy_64),
#         goal_xy_64=(gx_64, gy_64),
#         device=device,
#         max_iters=MAX_CHAIN_ITERS,
#         top_percentile=TOP_PCT,
#         min_step_px=MIN_STEP,
#         goal_tol_px=GOAL_TOL
#     )

#     print("[INFO] Waypoints:", waypoints)
#     for it in info:
#         print(f"  iter={it['iter']} start={it['start']} chosen={it['chosen']} infer={it['infer_ms']:.1f}ms prob_max={it['prob_max']:.3f}")

#     # Visualize: connect waypoints with polyline
#     rgb_vis = rgb_t[0].detach().cpu().permute(1, 2, 0).numpy()
#     rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

#     plt.figure(figsize=(3, 3))
#     plt.imshow(rgb_vis, interpolation="nearest")
#     plt.axis("off")

#     xs = [p[0] for p in waypoints]
#     ys = [p[1] for p in waypoints]
#     plt.plot(xs, ys, linewidth=2, color="blue")

#     # start/goal markers
#     plt.scatter([sx_64], [sy_64], marker="+", s=200, linewidths=5, color="#39FF14")
#     plt.scatter([gx_64], [gy_64], marker="+", s=200, linewidths=5, color="red")

#     # show intermediate waypoints (optional)
#     if len(waypoints) > 2:
#         plt.scatter(xs[1:-1], ys[1:-1], s=40, color="cyan")

#     out_path = overlay_dir / f"{sample_id}_iterative_chain.png"
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()

# print(f"\n[DONE] Saved iterative chaining overlays to: {overlay_dir}")
