from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class GlobalPath:
    def __init__(self, csv_path: Path, z_takeoff: float):
        self.csv_path = Path(csv_path)
        self.z_takeoff = float(z_takeoff)
        self.points = self._load_csv_xyz()
        self.points[:, 2] = self.z_takeoff

    def _load_csv_xyz(self) -> np.ndarray:
        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            xyz = [
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
                for row in reader
            ]
        xyz = np.array(xyz, dtype=float)
        return np.round(xyz, 2)

    def __len__(self):
        return len(self.points)

    def waypoint(self, index: int) -> np.ndarray:
        point = self.points[index].copy()
        point[2] = self.z_takeoff
        return point

    @property
    def final_xy(self) -> np.ndarray:
        return self.points[-1, :2]


def load_csv_xyz_simple(csv_path: str) -> np.ndarray:
    return GlobalPath(Path(csv_path), z_takeoff=1.0).points
