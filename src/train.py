# Module: Python module for the DDPM trajectory-planning project.

import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess

from warnings import warn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import datetime, timedelta

from src.utils.configs import (
    TrainingConfig,
    ScheduleMethods,
    LossNames,
    LogNames,
    LogTypes,
    DataDict,
    GeneratorType,
)
from src.loss import Loss
    # Loss.forward dispatches CVAE vs Diffusion.
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader
import io
import numpy as np
import matplotlib.pyplot as plt


# Class: Training orchestration class for model, data, optimizer, logging, and evaluation.
class Trainer:
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, cfgs: TrainingConfig):
        """
        Main training driver.

        With the new diffusion setup (GeneratorType.diffusion):

          - Dataset returns per sample:
                "trav"    : (1,H,W)  traversability map in [0,1], 1 = free
                "rgb"     : (3,H,W)  RGB image (for visualization only)
                "mask_gt" : (3,H,W)  GT mask channels:
                                  ch0 = start pixel(s)
                                  ch1 = goal  pixel(s)
                                  ch2 = trajectory pixels
                "start_px": (2,)     (x_px, y_px)
                "end_px"  : (2,)     (x_px, y_px)

          - DataLoader batches give:
                "trav"    : (B,1,H,W)
                "mask_gt" : (B,3,H,W)
                "start_px": (B,2)
                "end_px"  : (B,2)

          - HNav/Diffusion take these and perform DDPM over masks in pixel space.
          - Loss is mask-based (x0 reconstruction + optional traversability).

        CVAE branch is kept for backward compatibility and uses the old path-based setup.
        """
        self.name = cfgs.name
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.train_time_steps = cfgs.train_time_steps
        self.train_poses = cfgs.loss.train_poses

        self.iteration = 0
        self.epoch = 0
        self.training = False

        self.global_step = 0
        # set up gpus
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            print("world size: ", int(os.environ['WORLD_SIZE']))
            self.distributed = cfgs.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
        else:
            print("world size: ", 0)
            self.distributed = cfgs.data.distributed = False

        # ----------------- Model -----------------
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot
        if self.snapshot:
            state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)

        # ----------------- Logging: TensorBoard -----------------
        self.output_dir = cfgs.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        log_dir = os.path.join(self.output_dir, "tensorboard")
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        print(f"[INFO] TensorBoard logging to: {log_dir}")

        # ----------------- Optimizer & Scheduler -----------------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfgs.lr,
            weight_decay=cfgs.weight_decay
        )
        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, cfgs.lr_decay_steps, gamma=cfgs.lr_decay
            )
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                eta_min=cfgs.lr_min,
                T_0=cfgs.lr_t0,
                T_mult=cfgs.lr_tm
            )
        else:
            raise ValueError("the current scheduler is not defined")

        if self.snapshot and not cfgs.only_model:
            self.load_learning_parameters(state_dict)

        # ----------------- Loss -----------------
        if self.device == "cuda":
            self.loss_func = Loss(cfg=cfgs.loss).cuda()
        else:
            self.loss_func = Loss(cfg=cfgs.loss).to(self.device)

        # ----------------- Datasets -----------------
        self.training_data_loader = train_data_loader(cfg=cfgs.data)
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data)

        self.use_traversability = cfgs.loss.use_traversability
        self.generator_type = cfgs.model.generator_type
        self.time_step_loss_buffer = []
        self.time_step_number = cfgs.model.diffusion.traversable_steps
        self.traversability_threshold = cfgs.traversability_threshold

    # Function: Configure single-process or distributed GPU execution.
    def _set_model_gpus(self, cfg):
        if self.distributed:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print("os world size: {}, local_rank: {}, rank: {}".format(world_size, local_rank, rank))

            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(seconds=5000)
            )
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            print(
                'Training in distributed mode with multiple processes, 1 GPU per process. '
                'Process %d, total %d.' % (self.current_rank, world_size)
            )
            dist.barrier()
        else:
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        if self.distributed:
            self.model.cuda()
        else:
            self.model.to(self.device)

        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers while sync-bn enabled.'
                )

        if self.distributed:
            if cfg.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
            self.model = DDP(
                self.model,
                device_ids=[cfg.local_rank],
                broadcast_buffers=not cfg.no_ddp_bb,
                find_unused_parameters=True
            )

    # Function: Load model weights from a snapshot and report key mismatches.
    def load_snapshot(self, snapshot):
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            warn('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn('Unexpected keys: {}'.format(unexpected_keys))
        print('Model has been loaded.')
        return state_dict

    # Function: Restore epoch, iteration, optimizer, and scheduler states.
    def load_learning_parameters(self, state_dict):
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            print('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                print('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                print('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    # Function: Save model, optimizer, scheduler, and progress state to disk.
    def save_snapshot(self, filename):
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)

        # save snapshot with optimizer & scheduler
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(self.output_dir, str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)

    # Function: Close writers and distributed process groups after training.
    def cleanup(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.distributed:
            dist.destroy_process_group()

    # Function: Switch the trainer into training mode with gradients enabled.
    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    # Function: Switch the trainer into evaluation mode with gradients disabled.
    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    # Function: Apply one optimizer update and clear gradients.
    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Function: Run one train or evaluation batch through model, loss, and logging outputs.
    def step(self, data_dict, train=True) -> dict:
        """
        One train/eval step.

        In the NEW diffusion setup:
        ---------------------------
        Dataset / DataLoader provide:
          - "trav":     (B,1,H,W) traversability map in [0,1]
          - "rgb":      (B,3,H,W) RGB image (only for visualization / logging)
          - "mask_gt":  (B,3,H,W) DDPM x0 target mask (start / goal / traj)
          - "start_px": (B,2)     (x_px,y_px)
          - "end_px":   (B,2)     (x_px,y_px)
          - "occ_map":  (B,1,H,W) OPTIONAL, 1=obstacle, 0=free  (if you enable it)

        Diffusion branch:
          - HNav → Diffusion receives trav, mask_gt, start_px, end_px.
          - Loss is purely mask-based; it expects DataDict.prediction and "mask_gt".
        CVAE branch:
          - Old path-based behaviour is kept under GeneratorType.cvae.
        """
        # move to device
        data_dict = to_device(data_dict, device=self.device)

        # (legacy) map "rgb" → DataDict.camera if present (not used in new diffusion)
        if "rgb" in data_dict:
            data_dict[DataDict.camera] = data_dict["rgb"]

        # For backwards compatibility: use "trav" also as DataDict.local_map if needed
        if "trav" in data_dict:
            trav = data_dict["trav"]  # (B,1,H,W)
            data_dict[DataDict.local_map] = trav  # continuous [0,1] traversability

        # Trajectory GT is no longer primary for diffusion, but may exist for CVAE
        gt_path = data_dict.get(DataDict.path, None)

        if train:
            # ----------------- FORWARD PASS -----------------
            output_dict = self.model(data_dict, sample=False)

            # ----------------------------------------------------------
            # BRANCH: CVAE (old trajectory-based code)
            # ----------------------------------------------------------
            if self.generator_type == GeneratorType.cvae:
                # scale GT if train_poses=True to match loss
                if getattr(self.loss_func, "train_poses", False) and gt_path is not None:
                    scale = float(self.loss_func.scale_waypoints)
                    output_dict[DataDict.path] = gt_path * scale
                else:
                    output_dict[DataDict.path] = gt_path

                if DataDict.local_map in data_dict:
                    lm = data_dict[DataDict.local_map]
                    if lm.dim() == 3:
                        lm = lm.unsqueeze(1)
                        data_dict[DataDict.local_map] = lm
                    output_dict[DataDict.local_map] = data_dict[DataDict.local_map]

            # ----------------------------------------------------------
            # BRANCH: DIFFUSION (new mask-based pipeline)
            # ----------------------------------------------------------
            elif self.generator_type == GeneratorType.diffusion:
                # Ensure GT mask and occupancy are passed to the loss
                if "mask_gt" in data_dict:
                    output_dict["mask_gt"] = data_dict["mask_gt"]  # (B,3,H,W)
                if "occ_map" in data_dict:
                    output_dict["occ_map"] = data_dict["occ_map"]  # (B,1,H,W)

            # ----------------- LOSS COMPUTATION -----------------
            # (Loss.forward will dispatch based on generator_type)
            loss_dict = self.loss_func(output_dict)
            output_dict.update(loss_dict)

        else:
            # ====================== INFERENCE / EVAL ======================
            output_dict = self.model(data_dict, sample=True)

            if self.generator_type == GeneratorType.cvae:
                # For evaluation we still want GT path in original [0,1]
                output_dict["gt_path"] = gt_path

                if getattr(self.loss_func, "train_poses", False) and gt_path is not None:
                    scale = float(self.loss_func.scale_waypoints)
                    output_dict[DataDict.path] = gt_path * scale
                else:
                    output_dict[DataDict.path] = gt_path

                if DataDict.local_map in data_dict:
                    lm = data_dict[DataDict.local_map]
                    if lm.dim() == 3:
                        lm = lm.unsqueeze(1)
                        data_dict[DataDict.local_map] = lm
                    output_dict[DataDict.local_map] = data_dict[DataDict.local_map]

            elif self.generator_type == GeneratorType.diffusion:
                # attach GT mask & occ_map for evaluation in mask space
                if "mask_gt" in data_dict:
                    output_dict["mask_gt"] = data_dict["mask_gt"]
                if "occ_map" in data_dict:
                    output_dict["occ_map"] = data_dict["occ_map"]

            # Evaluate (no grad) via Loss.evaluate (mask-based in diffusion)
            eval_dict = self.loss_func.evaluate(output_dict)
            if eval_dict:
                output_dict.update(eval_dict)

        # For diffusion, "gt_path" is not essential anymore, but we keep it for
        # compatibility / potential visualization.
        if gt_path is not None:
            output_dict["gt_path"] = gt_path

        return output_dict

    # Function: Write scalar metrics to TensorBoard.
    def update_log(self, results, timestep=None, log_name=None):
        """
        Log metrics to TensorBoard.
        log_name: "train" or "evaluation" (see LogTypes).
        """
        if self.tb_writer is None:
            return

        step = self.iteration

        if timestep is not None:
            self.tb_writer.add_scalar(f"{log_name}/step_time", timestep, step)

        prefix = "" if log_name is None else f"{log_name}/"

        for key, value in results.items():
            try:
                scalar = value.item() if hasattr(value, "item") else float(value)
            except Exception:
                continue  # skip non-scalars

            self.tb_writer.add_scalar(prefix + key, scalar, step)

    # Function: Run one full training epoch.
    def run_epoch(self):
        """
        Run one training epoch over the training_data_loader.

        NOTE:
          - For diffusion, we call self.step(...) `train_time_steps` times per
            batch, as in the original DTG code (this effectively multiplies
            the number of gradient updates per data batch).
        """
        self.optimizer.zero_grad()

        last_time = time.time()
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            self.iteration += 1

            # traversable_step used inside diffusion for time-step sampling
            # (2-branch trick). It is a scalar upper bound on timesteps.
            data_dict[DataDict.traversable_step] = self.time_step_number

            for step_iteration in range(self.train_time_steps):
                output_dict = self.step(data_dict=data_dict, train=True)
                torch.cuda.empty_cache()

                output_dict[LossNames.loss].backward()
                self.optimizer_step()
                optimize_time = time.time()

                # Log losses and step time to TensorBoard
                output_dict = release_cuda(output_dict)
                self.update_log(
                    results=output_dict,
                    timestep=optimize_time - last_time,
                    log_name=LogTypes.train
                )
                last_time = time.time()
        self.scheduler.step()

        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth'.format(self.output_dir, self.name, self.epoch))

    # Function: Run periodic evaluation over the validation DataLoader.
    def inference_epoch(self):
        """
        Periodic evaluation epoch (no gradient) over evaluation_data_loader.

        For diffusion:
          - Uses Loss.evaluate in mask space.
          - Logs per-batch metrics and a mean epoch loss.
        """
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0) and (self.epoch != 0):
            for iteration, data_dict in enumerate(
                    tqdm(self.evaluation_data_loader,
                         desc="Evaluation Losses Epoch {}".format(self.epoch))):
                sum_loss = 0.0
                count = 0

                start_time = time.time()
                output_dict = self.step(data_dict, train=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_time = time.time()

                # assume Loss.evaluate added "loss" into output_dict
                if "loss" in output_dict:
                    sum_loss += output_dict["loss"].item()
                    count += 1

                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                self.update_log(
                    results=output_dict,
                    timestep=step_time - start_time,
                    log_name=LogTypes.others
                )
            if count > 0:
                mean_eval_loss = sum_loss / count
                self.tb_writer.add_scalar("evaluation/mean_loss", mean_eval_loss, self.epoch)

    # Function: Execute the full train/evaluate loop.
    def run(self):
        """
        Top-level training loop.
        """
        torch.autograd.set_detect_anomaly(True)
        for self.epoch in range(self.epoch, self.max_epoch, 1):
            # ---- eval BEFORE training each epoch (as in your original code) ----
            self.set_eval_mode()
            self.inference_epoch()

            # ---- training ----
            self.set_train_mode()
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
                if self.evaluation_freq > 0:
                    self.evaluation_data_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()
        self.cleanup()
