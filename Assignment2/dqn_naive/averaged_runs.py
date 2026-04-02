import pandas as pd
import numpy as np
import glob

PATTERN = "Assignment2/dqn_naive/dqn_naive_results_*.csv"
OUTPUT  = "Assignment2/dqn_naive/average_dqn_naive_results.csv"
N_POINTS = 500

files = sorted(glob.glob(PATTERN))
print(f"Found {len(files)} files: {files}")

# find the step range across all runs
all_min, all_max = [], []
for f in files:
    df = pd.read_csv(f)
    all_min.append(df["env_step"].min())
    all_max.append(df["env_step"].max())

# use common step grid from the smallest shared range
common_steps = np.linspace(max(all_min), min(all_max), N_POINTS)

# interpolate each run onto the common step grid
raw_aligned      = []
smoothed_aligned = []
for f in files:
    df = pd.read_csv(f).sort_values("env_step")

    raw_interp      = np.interp(common_steps, df["env_step"], df["Episode_Return"])
    smoothed_interp = np.interp(common_steps, df["env_step"], df["Episode_Return_smooth"])

    raw_aligned.append(raw_interp)
    smoothed_aligned.append(smoothed_interp)

raw_aligned      = np.array(raw_aligned)       # shape: (n_runs, N_POINTS)
smoothed_aligned = np.array(smoothed_aligned)

# output has same columns as input, just averaged over runs
summary = pd.DataFrame({
    "Episode_Return":        raw_aligned.mean(axis=0),
    "Episode_Return_smooth": smoothed_aligned.mean(axis=0),
    "env_step":              common_steps,
})

summary.to_csv(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")
print(summary.head())