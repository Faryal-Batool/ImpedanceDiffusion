# Module: DataLoader factory and collation helpers for DDPM training/evaluation.

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

# NEW: import your rewritten dataset
from src.data_loader.dataset_2 import TrajMaskDataset   # <- update to your actual path


# Function: Seed each DataLoader worker for reproducible numpy and random behavior.
def reset_seed_worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# Function: Collate dataset samples while converting numpy arrays to tensors.
def safe_collate(batch):
    """
    Leave torch.Tensors as-is.
    Convert numpy arrays to tensors.
    Everything else: default collate.
    """
    # Function: Convert numpy arrays to tensors during collation while leaving tensors untouched.
    def to_tensor_if_needed(x):
        import numpy as _np
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x
        if isinstance(x, _np.ndarray):
            return _torch.from_numpy(x)
        return x

    mapped = []
    for item in batch:
        mapped.append({k: to_tensor_if_needed(v) for k, v in item.items()})
    return default_collate(mapped)


# Function: Construct the trajectory-mask DataLoader for training or evaluation.
def get_data_loader(cfg, train=True):
    """
    If cfg has train_root / val_root, use them.
    Otherwise fall back to cfg.root.
    """
    # choose root folder
    if hasattr(cfg, "train_root") and hasattr(cfg, "val_root"):
        root = cfg.train_root if train else cfg.val_root
    else:
        root = cfg.root

    # --------- NEW: dataset options with safe defaults ---------
    img_size = getattr(cfg, "img_size", 64)
    n_points = getattr(cfg, "n_points", 128)

    # mask style
    traj_mask_mode = getattr(cfg, "traj_mask_mode", "soft")  # "soft" or "hard"
    soft_sigma = float(getattr(cfg, "soft_sigma", 0.75))
    line_thickness = int(getattr(cfg, "line_thickness", 1))

    # your trajectory storage convention
    traj_order = getattr(cfg, "traj_order", "yx")            # <-- important for your data
    traj_flip_y = bool(getattr(cfg, "traj_flip_y", False))   # set True if you notice vertical flip

    use_occ = bool(getattr(cfg, "use_occ", False))
    use_trav = bool(getattr(cfg, "use_trav", False))
    # ----------------------------------------------------------

    dataset = TrajMaskDataset(
        root_dir=root,
        img_size=img_size,
        n_points=n_points,
        traj_mask_mode=traj_mask_mode,
        line_thickness=line_thickness,
        soft_sigma=soft_sigma,
        traj_order=traj_order,
        traj_flip_y=traj_flip_y,
        use_occ=use_occ,
        use_trav=use_trav,
    )

    # sampler: shuffle only for training
    if getattr(cfg, "distributed", False):
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = None

    # If sampler is provided, DataLoader.shuffle must be False
    shuffle_flag = (sampler is None and getattr(cfg, "shuffle", train))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=getattr(cfg, "batch_size", 32),
        num_workers=getattr(cfg, "num_workers", 4),
        shuffle=shuffle_flag,
        sampler=sampler,
        collate_fn=safe_collate,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=True,
        persistent_workers=(getattr(cfg, "num_workers", 0) > 0),
        drop_last=False,
    )
    return data_loader


# Function: Build the training DataLoader from a copied config.
def train_data_loader(cfg):
    cfg_local = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfg_local, train=True)


# Function: Build the evaluation DataLoader from a copied config.
def evaluation_data_loader(cfg):
    cfg_local = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfg_local, train=False)


# import copy
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.utils.data._utils.collate import default_collate

# # swap to your RGB+traj dataset
# from src.data_loader.dataset import TrajDataset   # <-- was TrainData


# def reset_seed_worker_init_fn(worker_id):
#     seed = torch.initial_seed() % (2 ** 32)
#     np.random.seed(seed)
#     random.seed(seed)


# def safe_collate(batch):
#     """
#     Leave torch.Tensors as-is (fast path).
#     Convert numpy arrays to tensors.
#     Everything else: let default collate handle (lists/dicts of tensors).
#     """
    

#     def to_tensor_if_needed(x):
#         import numpy as _np
#         import torch as _torch
#         if isinstance(x, _torch.Tensor):
#             return x
#         if isinstance(x, _np.ndarray):
#             # keep dtype if float/bool/int; model expects float32 generally
#             return _torch.from_numpy(x)
#         return x

#     # map each dict’s values
#     mapped = []
#     for item in batch:
#         mapped.append({k: to_tensor_if_needed(v) for k, v in item.items()})
#     return default_collate(mapped)


# def get_data_loader(cfg, train=True):
#     """
#     If cfg has train_root / val_root, use them.
#     Otherwise fall back to cfg.root.
#     """
#     # choose root folder
#     if hasattr(cfg, "train_root") and hasattr(cfg, "val_root"):
#         root = cfg.train_root if train else cfg.val_root
#     else:
#         root = cfg.root

#     dataset = TrajDataset(root=root, n_points=getattr(cfg, "n_points", 128))

#     # sampler: shuffle only for training
#     if cfg.distributed:
#         sampler = DistributedSampler(dataset, shuffle=train)
#     else:
#         sampler = None

#     # If sampler is provided, DataLoader.shuffle must be False
#     shuffle_flag = (sampler is None and getattr(cfg, "shuffle", train))

#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=cfg.batch_size,
#         num_workers=cfg.num_workers,
#         shuffle=shuffle_flag,
#         sampler=sampler,
#         collate_fn=safe_collate,
#         worker_init_fn=reset_seed_worker_init_fn,
#         pin_memory=True,
#         persistent_workers=(cfg.num_workers > 0),
#         drop_last=False,
#     )
#     return data_loader


# def train_data_loader(cfg):
#     cfg_local = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfg_local, train=True)


# def evaluation_data_loader(cfg):
#     cfg_local = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfg_local, train=False)

# import copy
# import os
# import pickle
# import random
# import warnings
# import cv2
# from functools import partial
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, DistributedSampler

# from src.data_loader.dataset import TrainData


# def reset_seed_worker_init_fn(worker_id):
#     r"""Reset seed for data loader worker."""
#     seed = torch.initial_seed() % (2 ** 32)
#     # print(worker_id, seed)
#     np.random.seed(seed)
#     random.seed(seed)


# def registration_collate_fn_stack_mode(data_dicts):
#     r"""Collate function for registration in stack mode.
#     Args:
#         data_dicts (List[Dict])
#     Returns:
#         collated_dict (Dict)
#     """
#     # merge data with the same key from different samples into a list
#     collated_dict = {}
#     for data_dict in data_dicts:
#         for key, value in data_dict.items():
#             value = torch.from_numpy(np.asarray(value)).to(torch.float)
#             if key not in collated_dict:
#                 collated_dict[key] = []
#             collated_dict[key].append(value)
#     for key, value in collated_dict.items():
#         collated_dict[key] = torch.stack(value, dim=0)
#     return collated_dict


# def get_data_loader(cfg, train=True):
#     dataset = TrainData(cfg=cfg, train=train)
#     sampler = DistributedSampler(dataset) if cfg.distributed else None
#     data_loader = DataLoader(
#         dataset=dataset,
#         batch_size=cfg.batch_size,
#         num_workers=cfg.num_workers,
#         shuffle=cfg.shuffle,
#         sampler=sampler,
#         collate_fn=partial(registration_collate_fn_stack_mode),
#         worker_init_fn=reset_seed_worker_init_fn,
#         pin_memory=False,
#         drop_last=False,
#     )
#     return data_loader


# def train_data_loader(cfg):
#     """
#     This function is to create a training dataloader with pytorch interface
#     Args:
#         cfg: The configuration of the dataset
#     Returns:
#         a dataloader in pytorch format
#     """
#     cfgs = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfgs, train=True)


# def evaluation_data_loader(cfg):
#     """
#     This function is to create a evaluation dataloader with pytorch interface
#     Args:
#         cfg: The configuration of the dataset
#     Returns:
#         a dataloader in pytorch format
#     """
#     cfgs = copy.deepcopy(cfg)
#     return get_data_loader(cfg=cfgs, train=False)