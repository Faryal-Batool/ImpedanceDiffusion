# Module: Entry point for loading configuration and starting DDPM training.

import random, numpy as np, torch
import multiprocessing as mp
from src.train import Trainer
from src.utils.arguments import get_configuration

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cfgs = get_configuration()
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    trainer = Trainer(cfgs=cfgs)
    trainer.run()


# import torch

# from src.train import Trainer
# from src.utils.arguments import get_configuration


# if __name__ == "__main__":
#     cfgs = get_configuration()
#     trainer = Trainer(cfgs=cfgs)
#     torch.autograd.set_detect_anomaly(True)
#     trainer.run()
#     torch.autograd.set_detect_anomaly(False)
