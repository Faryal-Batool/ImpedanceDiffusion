# Variable Impedance Conrtol Experiment

This folder is portable. It does not require machine-specific absolute paths.

## Run

```bash
python3 run_swarm_experiment.py --csv-path /path/to/global_path.csv
```

The CSV must contain `x_m`, `y_m`, and `z_m` columns.

## Useful Arguments

```bash
python3 run_swarm_experiment.py \
  --csv-path /path/to/global_path.csv \
  --output-dir /path/to/output_results \
  --experiment Exp_9_dynamic \
  --z-takeoff 1.0 \
  --center 1.8,-1.5,1.0 \
  --obstacles human_1,trolley_1 \
  --pose-topic /poses
```

Run `python3 run_swarm_experiment.py --help` for the full list.

## Environment Variables

Instead of passing paths every time, you can set:

```bash
export SWARM_GLOBAL_PATH_CSV=/path/to/global_path.csv
export SWARM_OUTPUT_DIR=/path/to/output_results
```

If `--csv-path` is omitted, the runner also checks for common relative locations
such as `./global_path_60.csv`, `./Global_paths/exp_9_global_path_60.csv`, and
the old development layout beside this refactored folder.

## Outputs

Each run writes:

- `logs/drone_poses_<experiment>.txt`
- `logs/obstacles_<experiment>.txt`
- `logs/drone_poses_<experiment>.csv`
- `logs/obstacles_<experiment>.csv`
- `graphs/trajectories_exp.png`

