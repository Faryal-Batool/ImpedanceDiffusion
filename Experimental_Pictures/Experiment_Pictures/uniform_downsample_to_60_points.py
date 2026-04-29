# Module: Utility script for uniformly downsampling trajectory points.

import pandas as pd
import numpy as np

INPUT_CSV = r"//home/isr-lab3/Faryal_Batool/DTG-main_top/Experiments_for_ICUAS/world_csv/exp_6_static_traj_world.csv"
# INPUT_CSV = r"/home/isr-lab3/Faryal_Batool/DTG-main_v6/Experiments_ICUAS_fpv/world_csv/exp_6_static_traj_world.csv"
OUTPUT_CSV = r"/home/isr-lab3/Faryal_Batool/DTG-main_top/Experiments_for_ICUAS/world_csv/exp_6_global_path_60.csv"
N = 60

df = pd.read_csv(INPUT_CSV)

# Select evenly spaced indices
idxs = np.linspace(0, len(df) - 1, N, dtype=int)

df_down = df.iloc[idxs].reset_index(drop=True)

# Fix idx column
df_down["idx"] = range(len(df_down))

df_down.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV, "with", len(df_down), "points")
