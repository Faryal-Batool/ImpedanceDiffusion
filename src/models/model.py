# Module: High-level model wrapper around DDPM planning backends.

import torch
from torch import nn

from src.models.perception import Perception, LidarImageModel
from src.models.vae import CVAE
from src.models.diffusion import Diffusion

from src.utils.configs import DataDict, GeneratorType


# Class: Navigation model wrapper that connects inputs to the configured planner.
class HNav(nn.Module):
    """
    High-level navigation model wrapper.

    Two possible generator backends:
      1) CVAE  (legacy path-based generator with Perception encoder)
      2) Diffusion (NEW: mask-based DDPM operating on pixel-space masks)

    For the diffusion case, the core design is:

      Inputs (from dataset / dataloader):
        - trav    : (B,1,H,W)  traversability map in [0,1], 1 = free
        - mask_gt : (B,3,H,W)  ground-truth trajectory mask in pixel space
                              channel 0 -> start pixels
                              channel 1 -> goal  pixels
                              channel 2 -> trajectory/path pixels
        - start_px: (B,2) or (2,) optional conditioning, pixel (x_px, y_px)
        - end_px  : (B,2) or (2,) optional conditioning, pixel (x_px, y_px)

      Diffusion operates on:
        x0 = mask_gt   # in pixel space
        x_t ~ q(x_t | x0, t)
        eps_hat = UNet(x_t, t, trav, start_px, end_px, ...)

      Sampling / inpainting is handled inside Diffusion.sample().
    """

    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, config, device):
        super(HNav, self).__init__()
        self.config = config
        self.device = device

        self.generator_type = config.generator_type

        # ------------------------------------------------------------------
        # 1) CVAE branch (legacy)
        # ------------------------------------------------------------------
        if self.generator_type == GeneratorType.cvae:
            # Old CVAE path: perception + CVAE generator
            self.perception = Perception(self.config.perception)
            self.generator = CVAE(self.config.cvae)

        # ------------------------------------------------------------------
        # 2) Diffusion branch (NEW: mask-based DDPM on traversability)
        # ------------------------------------------------------------------
        elif self.generator_type == GeneratorType.diffusion:
            # For diffusion, perception is handled INSIDE Diffusion using:
            #   - trav: (B,1,H,W) as the main conditioning "image"
            #   - optional start_px / end_px coordinate conditioning
            self.perception = None
            self.generator = Diffusion(self.config.diffusion)

        else:
            raise ValueError("The generator type is not defined")

    # ----------------------------------------------------------------------
    # Helper: standardize optional pixel coordinate tensors
    # ----------------------------------------------------------------------
    @staticmethod
    # Function: Ensure an optional coordinate tensor has a batch dimension.
    def _ensure_batched_coords(tensor_or_none):
        """
        Ensure coordinate tensor is in batched shape (B,2) if present.

        Accepts:
          - None
          - Tensor/array of shape (2,)
          - Tensor of shape (B,2)

        Returns:
          - None if input is None
          - Tensor of shape (B,2) otherwise

        NOTE:
          We keep this conservative: if something already has batch dim,
          we leave it unchanged. This avoids surprising reshapes.
        """
        if tensor_or_none is None:
            return None
        if not torch.is_tensor(tensor_or_none):
            # assume dataloader already converted; if not, convert here
            tensor_or_none = torch.as_tensor(tensor_or_none)

        if tensor_or_none.dim() == 1 and tensor_or_none.numel() == 2:
            # (2,) -> (1,2)
            return tensor_or_none.unsqueeze(0)
        # assume already (B,2) or higher; leave as-is
        return tensor_or_none

    # ----------------------------------------------------------------------
    # Forward (TRAINING)
    # ----------------------------------------------------------------------
    # Function: Run the module forward pass for training or encoding.
    def forward(self, input_dict, sample=False):
        """
        Forward pass.

        For GeneratorType.diffusion (mask-based DDPM):

          Expected keys in input_dict (from dataset.py):
            - "trav"    : (B,1,H,W)  traversability map, float32 in [0,1], 1=free
            - "mask_gt" : (B,3,H,W)  ground-truth mask, float32 in {0,1}
                            ch0 = start pixels
                            ch1 = goal  pixels
                            ch2 = trajectory pixels
            - "start_px": (B,2) or (2,), optional, pixel (x_px, y_px)
            - "end_px"  : (B,2) or (2,), optional, pixel (x_px, y_px)

          Output:
            - generator output dict, e.g. containing:
                "eps_hat", "eps", "t", "mask_pred", etc. (defined in Diffusion)
            - plus pass-through fields such as "trav", "mask_gt", "start_px",
              "end_px" so that Loss / logging code can access them.
        """
        if sample:
            return self.sample(input_dict=input_dict)

        output = {}

        # Keep old fields for backward compatibility if present
        if DataDict.path in input_dict:
            output[DataDict.path] = input_dict[DataDict.path]
        if DataDict.local_map in input_dict:
            output[DataDict.local_map] = input_dict[DataDict.local_map]

        # ------------------------------------------------------------------
        # 1) CVAE branch (unchanged)
        # ------------------------------------------------------------------
        if self.generator_type == GeneratorType.cvae:
            observation = self.perception(
                lidar=input_dict[DataDict.lidar],
                vel=input_dict[DataDict.vel],
                target=input_dict[DataDict.target]
            )
            gen_out = self.generator(
                observation=observation,
                gt_path=input_dict[DataDict.path]
            )

        # ------------------------------------------------------------------
        # 2) Diffusion branch (mask-based DDPM)
        # ------------------------------------------------------------------
        elif self.generator_type == GeneratorType.diffusion:
            # ---- Standardized inputs from dataset ----
            # Main conditioning image: traversability map
            rgb = input_dict.get("rgb", None)  # (B,3,H,W), optional, not used here
            # Traversability map (main conditioning image for diffusion)
            #   - (B,1,H,W) float32 in [0,1],
            # trav    = input_dict.get("trav", None)        # (B,1,H,W), float32 in [0,1]
            mask_gt = input_dict.get("mask_gt", None)     # (B,3,H,W), float32 in {0,1}
            # Optional occupancy map (not required for core DDPM)
            occ_map = input_dict.get("occ_map", None)     # (B,1,H,W), optional

            # Optional pixel-space conditioning:
            # start_px / end_px are (x_px, y_px) pixel coordinates.
            start_px = self._ensure_batched_coords(input_dict.get("start_px", None))
            end_px   = self._ensure_batched_coords(input_dict.get("end_px", None))

            # Old field that may still exist in configs (not essential for masks)
            trav_step = input_dict.get(DataDict.traversable_step, None)

            # ---- Safety checks (fail fast for obvious issues) ----
            # assert trav is not None, "HNav.forward: missing 'trav' (B,1,H,W) for diffusion"
            assert mask_gt is not None, "HNav.forward: missing 'mask_gt' (B,3,H,W) for DDPM supervision"

            # assert trav.dim() == 4 and trav.size(1) == 1, \
                # f"HNav.forward: 'trav' must be (B,1,H,W), got {tuple(trav.shape)}"
            assert mask_gt.dim() == 4 and mask_gt.size(1) == 3, \
                f"HNav.forward: 'mask_gt' must be (B,3,H,W), got {tuple(mask_gt.shape)}"

            # ---- Call diffusion generator ----
            # NOTE:
            #   - We do NOT use rgb here; conditioning is based on trav + coords.
            #   - start_px / end_px are kept in pixel space and interpreted as such
            #     inside Diffusion for inpainting masks, etc.
            gen_out = self.generator(
                # trav=trav,
                rgb= rgb,
                mask_gt=mask_gt,
                start_px=start_px,
                end_px=end_px,
                # occ_map=occ_map,
                traversable_step=trav_step,
            )

        else:
            raise ValueError("Invalid generator type")

        # Merge generator outputs
        output.update(gen_out)

        # ------------------------------------------------------------------
        # Pass-through useful fields so Loss / logging can access them
        # ------------------------------------------------------------------
        if "rgb" in input_dict:
            output["rgb"] = input_dict["rgb"]             # (B,3,H,W), optional
        # if "trav" in input_dict:
        #     output["trav"] = input_dict["trav"]           # (B,1,H,W)
        # if "occ_map" in input_dict:
        #     output["occ_map"] = input_dict["occ_map"]     # (B,1,H,W)

        # Pixel-space coordinates (if present). Convention: (x_px, y_px).
        if "start_px" in input_dict:
            output["start_px"] = input_dict["start_px"]   # (B,2) or (2,)
        if "end_px" in input_dict:
            output["end_px"] = input_dict["end_px"]       # (B,2) or (2,)

        # Ground-truth mask (pixel-space)
        if "mask_gt" in input_dict:
            output["mask_gt"] = input_dict["mask_gt"]     # (B,3,H,W)

        # Normalized coordinates are no longer required for the mask-based DDPM,
        # but if they exist in some older pipelines, we forward them untouched.
        if "start_norm" in input_dict:
            output["start_norm"] = input_dict["start_norm"]
        if "end_norm" in input_dict:
            output["end_norm"] = input_dict["end_norm"]

        return output

    # ----------------------------------------------------------------------
    # Sampling / inference
    # ----------------------------------------------------------------------
    # Function: Run DDPM reverse sampling to produce a trajectory mask.
    def sample(self, input_dict):
        """
        Inference / sampling mode.

        For diffusion (mask-based):

          Required:
            - "trav": (B,1,H,W) traversability map, float32 in [0,1], 1=free

          Optional:
            - "start_px": (B,2) or (2,), pixel (x_px, y_px) for inpainting constraints
            - "end_px"  : (B,2) or (2,), pixel (x_px, y_px) for inpainting constraints

          Returns:
            gen_out from Diffusion.sample(), e.g. containing:
              - "mask_sample": (B,3,H,W) sampled mask in pixel space
              - possibly derived trajectories, etc. (as implemented there)
        """
        output = {}

        # Backward-compatibility pass-through fields
        if DataDict.path in input_dict:
            output[DataDict.path] = input_dict[DataDict.path]
        if DataDict.local_map in input_dict:
            output[DataDict.local_map] = input_dict[DataDict.local_map]

        # ------------------------------------------------------------------
        # 1) CVAE sampling
        # ------------------------------------------------------------------
        if self.generator_type == GeneratorType.cvae:
            observation = self.perception(
                lidar=input_dict[DataDict.lidar],
                vel=input_dict[DataDict.vel],
                target=input_dict[DataDict.target]
            )
            gen_out = self.generator.sample(observation=observation)

        # ------------------------------------------------------------------
        # 2) Diffusion sampling (mask-based)
        # ------------------------------------------------------------------
        elif self.generator_type == GeneratorType.diffusion:
            rgb = input_dict.get("rgb", None)  # (B,3,H,W), optional, not used here
            # Traversability map (main conditioning image for diffusion)
            #   - (B,1,H,W) float32 in [0,1],
            #   - 1 = free, 0 = occupied
            #   - Used for diffusion sampling
            #   - NOTE: this is the main conditioning image for diffusion
            # trav = input_dict.get("trav", None)       # (B,1,H,W)
            # occ_map = input_dict.get("occ_map", None)  # Optional, unused here

            start_px = self._ensure_batched_coords(input_dict.get("start_px", None))
            end_px   = self._ensure_batched_coords(input_dict.get("end_px", None))

            # assert trav is not None, "HNav.sample: need 'trav' (B,1,H,W) for diffusion sampling"

            gen_out = self.generator.sample(
                rgb=rgb,
                start_px=start_px,
                end_px=end_px,
                # occ_map=occ_map,
            )

        else:
            raise ValueError("Invalid generator type")

        output.update(gen_out)

        # Pass-through useful fields for visualization / logging
        if "trav" in input_dict:
            output["trav"] = input_dict["trav"]
        # if "occ_map" in input_dict:
        #     output["occ_map"] = input_dict["occ_map"]
        if "start_px" in input_dict:
            output["start_px"] = input_dict["start_px"]
        if "end_px" in input_dict:
            output["end_px"] = input_dict["end_px"]
        if "rgb" in input_dict:
            # RGB is still useful for visualization overlays if available
            output["rgb"] = input_dict["rgb"]

        return output


# Function: Factory function for constructing the navigation model wrapper.
def get_model(config, device):
    """
    Factory to build the HNav model.

    NOTE:
      - 'device' is stored but device placement is typically handled
        by the training code via model.to(device) and batch.to(device).
    """
    return HNav(config=config, device=device)
