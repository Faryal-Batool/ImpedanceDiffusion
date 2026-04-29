# Module: Loss and evaluation metrics for DDPM mask reconstruction.

import copy
import math
import os
import pickle
import shutil
from os.path import join, exists

import cv2
import imageio
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from src.models.diff_hausdorf import HausdorffLoss
from src.utils.configs import GeneratorType, DataDict, Hausdorff, LossNames


# Class: Loss module for mask-based DDPM training and evaluation.
class Loss(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, cfg):
        super(Loss, self).__init__()

        self.generator_type = cfg.generator_type
        self.use_traversability = cfg.use_traversability
        self.collision_distance = 20  # kept from old code, not used now

        # === OLD GEOMETRIC STUFF (kept for CVAE branch / backward compat) ===
        # These are not used in the new mask-based diffusion path.
        self.target_dis = nn.MSELoss(reduction="mean")
        self.distance = HausdorffLoss(mode=cfg.distance_type)

        # === NEW: mask reconstruction criterion for diffusion ===
        # prediction is x0 (clean mask), target is mask_gt ∈ {0,1}
        self.mask_mse = nn.MSELoss(reduction="mean")

        # Optional: different weights per channel (start, goal, traj)
        self.mask_start_weight = getattr(cfg, "mask_start_weight", 1.0)
        self.mask_goal_weight  = getattr(cfg, "mask_goal_weight", 2.0)
        self.mask_traj_weight  = getattr(cfg, "mask_traj_weight", 2.0)

        # Legacy config fields (used only in CVAE path or for weighting)
        self.train_poses = cfg.train_poses
        self.distance_type = cfg.distance_type
        self.scale_waypoints = cfg.scale_waypoints
        self.last_ratio = cfg.last_ratio
        self.distance_ratio = cfg.distance_ratio
        self.vae_kld_ratio = cfg.vae_kld_ratio
        self.traversability_ratio = cfg.traversability_ratio
        self.endpoint_ratio = getattr(cfg, "endpoint_ratio", 1.0)

        self.map_resolution = cfg.map_resolution
        self.map_range = cfg.map_range
        self.output_dir = cfg.output_dir
        if self.output_dir:
            if not exists(self.output_dir):
                os.makedirs(self.output_dir)

    # ----------------------------------------------------------------------
    # DIFFUSION LOSS (MASK-BASED, NOT TRAJECTORY-BASED ANYMORE)
    # ----------------------------------------------------------------------
    # Function: Compute the diffusion branch mask loss.
    def forward_diffusion(self, input_dict):
        """
        Diffusion loss for traversability-conditioned MASK generation.

        New representation:
        -------------------
        - mask_gt: ground-truth mask, (B,3,H,W)
              ch0 = start mask  (1 at start pixel(s), 0 elsewhere)
              ch1 = goal mask   (1 at goal pixel(s),  0 elsewhere)
              ch2 = trajectory  (1 along path pixels, 0 elsewhere)

        - mask_pred_full: predicted clean mask x0 from DDPM, (B,3,H,W) or (2B,3,H,W)
              If use_traversability=True and the model duplicates the batch to 2B,
              we simply replicate the target so losses still align.

        Loss design:
        ------------
        - path_dis (LossNames.path_dis):
              MSE on trajectory channel (ch2) between mask_pred and mask_gt.
        - last_dis (LossNames.last_dis):
              MSE on start/goal channels (ch0 & ch1).
        - traversability (LossNames.traversability), optional:
              if occ_map is present (1=obstacle), penalize predicted trajectory
              probability/mask on obstacle pixels.

        Total loss:
            all_loss = distance_ratio       * path_dis
                     + last_ratio           * last_dis
                     + traversability_ratio * traversability_loss (if used)
        """

        # ---- Ground truth & predictions ----
        mask_gt = input_dict["mask_gt"]                  # (B,3,H,W)
        mask_pred_full = input_dict[DataDict.prediction] # (B,3,H,W) or (2B,3,H,W)

        B_gt   = mask_gt.shape[0]
        B_pred = mask_pred_full.shape[0]
        device = mask_gt.device

        # ---- Handle DTG-style 2B duplication (if present) ----
        if self.use_traversability and (B_pred == 2 * B_gt):
            # Model predicted 2B masks; target is still B.
            # We replicate mask_gt to match shape so every prediction has a target.
            mask_gt_expanded = torch.cat([mask_gt, mask_gt], dim=0)  # (2B,3,H,W)
            mask_pred = mask_pred_full
        else:
            mask_gt_expanded = mask_gt
            mask_pred = mask_pred_full

        # Make sure shapes match
        assert mask_pred.shape == mask_gt_expanded.shape, \
            f"mask_pred {mask_pred.shape} and mask_gt {mask_gt_expanded.shape} must match."

        # ------------------------------------------------------------------
        # 1) Channel-wise mask reconstruction losses
        # ------------------------------------------------------------------
        # Channels:
        #   [0] start, [1] goal, [2] trajectory
        pred_start = mask_pred[:, 0:1, ...]
        pred_goal  = mask_pred[:, 1:2, ...]
        pred_traj  = mask_pred[:, 2:3, ...]

        gt_start   = mask_gt_expanded[:, 0:1, ...]
        gt_goal    = mask_gt_expanded[:, 1:2, ...]
        gt_traj    = mask_gt_expanded[:, 2:3, ...]

        # MSE per channel
        start_mse = self.mask_mse(pred_start, gt_start)
        goal_mse  = self.mask_mse(pred_goal,  gt_goal)
        traj_mse  = self.mask_mse(pred_traj,  gt_traj)

        # Weighted combination per channel (weights kept for future use)
        start_mse = self.mask_start_weight * start_mse
        goal_mse  = self.mask_goal_weight  * goal_mse
        traj_mse  = self.mask_traj_weight  * traj_mse

        # reinterpretation:
        #   path_dis  <- trajectory channel reconstruction (traj_mse)
        #   last_dis  <- endpoints reconstruction (average of start+goal)
        endpoint_mse = 0.5 * (start_mse + goal_mse)

        path_dis      = traj_mse
        last_pose_dis = endpoint_mse

        output = {}
        output[LossNames.path_dis] = path_dis
        output[LossNames.last_dis] = last_pose_dis

        # Base loss = mask reconstruction (geometry-style)
        all_loss = (
            self.distance_ratio * path_dis
            + self.last_ratio * last_pose_dis
        )

        # ------------------------------------------------------------------
        # 2) TRAVERSABILITY LOSS: penalize trajectory on obstacles, if available
        # ------------------------------------------------------------------
        # Expect occ_map: (B,1,H,W) with 1=obstacle, 0=free (as in dataset.py)
        occ_map = input_dict.get("occ_map", None)
        if self.use_traversability and occ_map is not None:
            occ = occ_map.to(device)
            if occ.dim() == 3:
                occ = occ.unsqueeze(1)           # (B,H,W) -> (B,1,H,W)
            elif occ.dim() == 4 and occ.size(1) != 1:
                occ = occ[:, :1, ...]           # pick first channel

            # Align occ with mask_pred batch size if we have 2B:
            if occ.shape[0] != mask_pred.shape[0]:
                # Just replicate occ for the second half
                factor = mask_pred.shape[0] // occ.shape[0]
                occ = occ.repeat(factor, 1, 1, 1)

            # We treat mask_pred values as "trajectory intensity" in [~0,1].
            # If you plan to use logits, you could apply torch.sigmoid here.
            traj_pred_on_obstacles = pred_traj * occ  # (B or 2B,1,H,W), only where obstacles
            trav_loss = traj_pred_on_obstacles.mean()

            all_loss = all_loss + self.traversability_ratio * trav_loss

            output[LossNames.traversability] = trav_loss
            # Optional debug metric: mean predicted traj on obstacles could be logged

        # ------------------------------------------------------------------
        # 3) Final combined loss
        # ------------------------------------------------------------------
        output[LossNames.loss] = all_loss
        return output

    # ----------------------------------------------------------------------
    # GENERAL FORWARD: dispatch CVAE vs Diffusion
    # ----------------------------------------------------------------------
    # Function: Run the module forward pass for training or encoding.
    def forward(self, input_dict):
        if self.generator_type == GeneratorType.cvae:
            return self.forward_cvae(input_dict=input_dict)
        elif self.generator_type == GeneratorType.diffusion:
            return self.forward_diffusion(input_dict=input_dict)
        else:
            raise ValueError(f"Unknown generator_type: {self.generator_type}")

    # ----------------------------------------------------------------------
    # (OPTIONAL / LEGACY) CVAE LOSS
    # ----------------------------------------------------------------------
    # Function: Legacy CVAE loss placeholder that blocks unsupported use.
    def forward_cvae(self, input_dict):
        """
        CVAE loss path is not adapted to the new mask-based pipeline.

        If you still need CVAE training, you should re-implement this method
        explicitly for your current representation. For now, we raise
        NotImplementedError to avoid silent misuse.
        """
        raise NotImplementedError(
            "CVAE loss is not implemented in the new mask-based setup. "
            "Set generator_type=diffusion to use the DDPM mask loss."
        )

    # ----------------------------------------------------------------------
    # (UNCHANGED) UTILITIES FOR OLD PATH-BASED VISUALIZATION
    # ----------------------------------------------------------------------
    # Function: Convert metric path coordinates into local-map pixel coordinates.
    def convert_path_pixel(self, trajectory):
        return np.clip(
            np.around(trajectory / self.map_resolution)[:, :2] + self.map_range,
            0,
            np.inf,
        )

    # Function: Render a trajectory over a local map for debugging.
    def show_path_local_map(self, trajectory, gt_path, local_map, idx=0, indices=0):
        return write_png(
            local_map=local_map,
            center=np.array(
                [local_map.shape[0] / 2, local_map.shape[1] / 2]
            ),
            file=join(
                self.output_dir,
                "local_map_trajectory_{}.png".format(indices + idx),
            ),
            paths=[self.convert_path_pixel(trajectory=trajectory)],
            others=self.convert_path_pixel(trajectory=gt_path),
        )

    @torch.no_grad()
    # Function: Compute evaluation metrics using the same mask loss surface.
    def evaluate(self, input_dict, indices=0):
        """
        Evaluation in the new mask-based setting.

        We mirror the training losses but do NOT backprop:
            - path_dis  := trajectory channel MSE
            - last_dis  := start+goal channels MSE

        This replaces the old trajectory-based Hausdorff evaluation.
        """
        mask_gt = input_dict["mask_gt"]             # (B,3,H,W)
        mask_pred = input_dict[DataDict.prediction] # (B,3,H,W) from inference

        # Ensure shapes compatible
        B_gt = mask_gt.shape[0]
        B_pred = mask_pred.shape[0]
        if B_pred != B_gt:
            # For evaluation we assume no 2B duplication; if there is, use first B.
            mask_pred = mask_pred[:B_gt]

        # Channels
        pred_start = mask_pred[:, 0:1, ...]
        pred_goal  = mask_pred[:, 1:2, ...]
        pred_traj  = mask_pred[:, 2:3, ...]

        gt_start   = mask_gt[:, 0:1, ...]
        gt_goal    = mask_gt[:, 1:2, ...]
        gt_traj    = mask_gt[:, 2:3, ...]

        start_mse = self.mask_mse(pred_start, gt_start)
        goal_mse  = self.mask_mse(pred_goal,  gt_goal)
        traj_mse  = self.mask_mse(pred_traj,  gt_traj)

        endpoint_mse = 0.5 * (start_mse + goal_mse)

        path_dis      = traj_mse
        last_pose_dis = endpoint_mse

        output = {
            LossNames.evaluate_path_dis: path_dis,
            LossNames.evaluate_last_dis: last_pose_dis,
        }

        # Evaluation "loss" scalar for logging: same weighting as training
        eval_loss = (
            self.distance_ratio * path_dis
            + self.last_ratio * last_pose_dis
        )
        output[LossNames.loss] = eval_loss

        # NOTE: old path-based PNG visualization is not used here because we
        # now operate directly in mask space. You can add mask overlays separately
        # if you want visual debugging.
        return output



# Function: Render trajectory and point overlays into an image file.
def write_png(local_map=None, rgb_local_map=None, center=None, targets=None, paths=None, paths_color=None, path=None,
              crop_edge=None, others=None, file=None):
    dis = 2
    x_range = [local_map.shape[0], 0]
    y_range = [local_map.shape[1], 0]
    if rgb_local_map is not None:
        local_map_fig = rgb_local_map
    else:
        local_map_fig = np.repeat(local_map[:, :, np.newaxis], 3, axis=2) * 255
    if center is not None:
        assert center.shape[0] == 2 and len(center.shape) == 1, "path should be 2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(center + np.array([x, y]))
        all_points = np.stack(all_points).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if targets is not None and len(targets) > 0:
        xs, ys = targets[:, 0], targets[:, 1]
        xs = np.clip(xs, dis, local_map_fig.shape[0] - dis)
        ys = np.clip(ys, dis, local_map_fig.shape[1] - dis)
        clipped_targets = np.stack((xs, ys), axis=-1)

        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(clipped_targets + np.array([x, y]))
        if len(clipped_targets.shape) == 2:
            all_points = np.concatenate(all_points, axis=0).astype(int)
        else:
            all_points = np.stack(all_points, axis=0).astype(int)

        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if others is not None:
        assert others.shape[1] == 2 and len(others.shape) == 2, "path should be Nx2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(others + np.array([x, y]))
        all_points = np.concatenate(all_points, axis=0).astype(int)

        xs, ys = all_points[:, 0], all_points[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 255
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 0

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if path is not None:
        assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
        all_pts = path
        all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                  all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                  all_pts), axis=0)
        xs, ys = all_pts[:, 0], all_pts[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 0
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 255

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if paths is not None:
        for p_idx in range(len(paths)):
            path = paths[p_idx]
            if len(path) == 1 or np.any(path[0] == np.inf):
                continue
            path = np.asarray(path, dtype=int)
            assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
            all_pts = path
            all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                      all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                      all_pts), axis=0)
            xs, ys = all_pts[:, 0], all_pts[:, 1]
            xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
            ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
            if paths_color is not None:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 0
                local_map_fig[xs, ys, 2] = paths_color[p_idx]
            else:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 255
                local_map_fig[xs, ys, 2] = 255

            if x_range[0] > min(all_pts[:, 0]):
                x_range[0] = min(all_pts[:, 0])
            if x_range[1] < max(all_pts[:, 0]):
                x_range[1] = max(all_pts[:, 0])
            if y_range[0] > min(all_pts[:, 1]):
                y_range[0] = min(all_pts[:, 1])
            if y_range[1] < max(all_pts[:, 1]):
                y_range[1] = max(all_pts[:, 1])
    if crop_edge:
        local_map_fig = local_map_fig[
                        max(0, x_range[0] - crop_edge):min(x_range[1] + crop_edge, local_map_fig.shape[0]),
                        max(0, y_range[0] - crop_edge):min(y_range[1] + crop_edge, local_map_fig.shape[1])]
    if file is not None:
        cv2.imwrite(file, local_map_fig)
    return local_map_fig




# def forward_diffusion(self, input_dict):
#     """
#     Diffusion loss for start+end+RGB-conditioned trajectory generation.

#     - ygt: ground-truth trajectory (B, N, 2) in [0,1]  (absolute positions)
#     - y_hat_full: predicted trajectory (B, N, 2) or (2B, N, 2) if use_traversability=True

#       * First  B rows -> geometry loss (Hausdorff + endpoint)
#       * Second B rows -> traversability loss (if duplicated)

#     Assumptions:
#       - When self.train_poses = False, the model predicts INCREMENTS.
#       - When self.train_poses = True, the model predicts ABSOLUTE positions.
#     """
#     # ---- Ground truth & predictions ----
#     ygt        = input_dict[DataDict.path]       # (B, N, 2) in [0,1], ABSOLUTE
#     y_hat_full = input_dict[DataDict.prediction] # (B, N, 2) or (2B, N, 2)

#     B_gt   = ygt.shape[0]
#     B_pred = y_hat_full.shape[0]
#     device = ygt.device

#     # Optional hyperparameters (safe defaults if not present)
#     smooth_ratio    = getattr(self, "smooth_ratio", 0.0)
#     endpoint_ratio  = getattr(self, "endpoint_ratio", 0.0)

#     # ---- Handle DTG "2B" duplication pattern if present ----
#     if self.use_traversability and (B_pred == 2 * B_gt):
#         # First half: geometry (Hausdorff + last waypoint)
#         y_hat_geom = y_hat_full[:B_gt]          # (B, N, 2)
#         # Second half: traversability-only path
#         y_hat_trav = y_hat_full[B_gt:]          # (B, N, 2)
#     else:
#         # No duplication: same predictions for both
#         y_hat_geom = y_hat_full                 # (B, N, 2)
#         y_hat_trav = y_hat_full                 # (B, N, 2)

#     output = {}

#     # ------------------------------------------------------------------
#     # 1) GEOMETRIC LOSSES: Hausdorff + last waypoint (on y_hat_geom)
#     # ------------------------------------------------------------------
#     # start and goal from input dict (absolute in [0,1])
#     start_xy = input_dict.get(DataDict.start_xy, None)   # (B,2)
#     goal_xy  = input_dict.get(DataDict.goal_xy, None)    # (B,2) – optional

#     # Scale to geometry space (e.g., pixels) if desired
#     # In DTG-style setups, typically:
#     #   - train_poses=True  -> scale_waypoints=20.0
#     #   - train_poses=False -> scale_waypoints=1.0
#     if self.train_poses:
#         # Model predicts absolute positions directly
#         # Both GT and predictions are scaled the same way
#         ygt_geom    = ygt * self.scale_waypoints                     # (B,N,2)
#         y_hat_poses = y_hat_geom * self.scale_waypoints              # (B,N,2)
#     else:
#         # Model predicts increments -> we must reconstruct absolute positions
#         # Correct reconstruction:
#         #   p_i = start_xy + sum_{j<=i} Δp_j
#         assert start_xy is not None, "start_xy is required when train_poses=False"
#         start_xy = start_xy.to(device=device, dtype=y_hat_geom.dtype)        # (B,2)

#         # Cumulative increments: (B,N,2)
#         y_hat_cumsum = torch.cumsum(y_hat_geom, dim=1)
#         # Anchor with start position
#         y_hat_abs = y_hat_cumsum + start_xy.unsqueeze(1)                     # (B,N,2)

#         # Scale both GT and predicted to the same geometry space
#         ygt_geom    = ygt * self.scale_waypoints                             # (B,N,2)
#         y_hat_poses = y_hat_abs * self.scale_waypoints                       # (B,N,2)

#     # Hausdorff (or DTG-style) distance over the whole path
#     path_dis      = self.distance(ygt_geom, y_hat_poses).mean()
#     # L2 on last waypoint in the same geometry space
#     last_pose_dis = self.target_dis(ygt_geom[:, -1, :], y_hat_poses[:, -1, :])

#     output[LossNames.path_dis] = path_dis
#     output[LossNames.last_dis] = last_pose_dis

#     # Base loss: geometry only
#     all_loss = (
#         self.distance_ratio * path_dis
#         + self.last_ratio * last_pose_dis
#     )

#     # ------------------------------------------------------------------
#     # 1b) Optional smoothness regularization on increments
#     # ------------------------------------------------------------------
#     if (not self.train_poses) and (smooth_ratio > 0.0):
#         # Penalize large changes in increments to encourage smooth motions
#         # y_hat_geom: (B,N,2) are increments
#         incr_diff = y_hat_geom[:, 1:, :] - y_hat_geom[:, :-1, :]            # (B,N-1,2)
#         smooth_loss = torch.mean(incr_diff ** 2)

#         all_loss = all_loss + smooth_ratio * smooth_loss
#         output["smoothness"] = smooth_loss

#     # ------------------------------------------------------------------
#     # 1c) Optional endpoint consistency in increment space
#     # ------------------------------------------------------------------
#     if (not self.train_poses) and (endpoint_ratio > 0.0) and (start_xy is not None):
#         # Encourage the SUM of increments to match (goal - start)
#         # This is complementary (in increment space) to last_pose_dis (in absolute space)
#         if goal_xy is not None:
#             goal_xy = goal_xy.to(device=device, dtype=y_hat_geom.dtype)
#             target_delta = goal_xy - start_xy                        # (B,2)
#         else:
#             # Fallback: compute from ground-truth trajectory
#             target_delta = ygt[:, -1, :] - ygt[:, 0, :]              # (B,2)

#         sum_increments = torch.sum(y_hat_geom, dim=1)                # (B,2)
#         endpoint_cons = self.target_dis(sum_increments, target_delta)

#         all_loss = all_loss + endpoint_ratio * endpoint_cons
#         output["endpoint_consistency"] = endpoint_cons

#     # ------------------------------------------------------------------
#     # 2) TRAVERSABILITY LOSS: use y_hat_trav + trav/local_map if available
#     # ------------------------------------------------------------------
#     if self.use_traversability and (DataDict.local_map in input_dict):
#         trav_map = input_dict[DataDict.local_map]   # could be (B,1,H,W) or (B,H,W)

#         # Normalize to (B,1,H,W)
#         if trav_map.dim() == 3:
#             trav_map = trav_map.unsqueeze(1)        # (B,H,W) -> (B,1,H,W)
#         elif trav_map.dim() == 4 and trav_map.size(1) != 1:
#             trav_map = trav_map[:, :1, ...]         # (B,C,H,W) -> (B,1,H,W)

#         # traj_norm in [0,1] for traversability penalty
#         if self.train_poses:
#             # y_hat_trav ~ absolute coords in [0,1] -> clamp
#             traj_norm = torch.clamp(y_hat_trav, 0.0, 1.0)
#         else:
#             # increments -> reconstruct absolute, then clamp
#             assert start_xy is not None, "start_xy is required for traversability with increments"
#             y_hat_trav_cum = torch.cumsum(y_hat_trav, dim=1)        # (B,N,2)
#             y_hat_trav_abs = y_hat_trav_cum + start_xy.unsqueeze(1)
#             traj_norm = torch.clamp(y_hat_trav_abs, 0.0, 1.0)

#         # traversability-aware penalty along the path
#         trav_loss_batch, mean_vals = self._local_collision_traversability(
#             traj_norm, trav_map
#         )                                           # both (B,)

#         traversability_loss_mean = trav_loss_batch.mean()
#         all_loss = all_loss + self.traversability_ratio * traversability_loss_mean

#         output[LossNames.traversability] = traversability_loss_mean
#         output["mean_obstacle_distance"] = mean_vals.mean()

#     # ------------------------------------------------------------------
#     # 3) Final combined loss
#     # ------------------------------------------------------------------
#     output[LossNames.loss] = all_loss
#     return output
