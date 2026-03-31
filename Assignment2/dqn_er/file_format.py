import pandas as pd
import numpy as np
import glob

def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

files = sorted(glob.glob("Assignment2/dqn_er/dqn_er_seed*.csv"))
print(f"Found {len(files)} files")

for f in files:
    df = pd.read_csv(f)
    
    smoothed = moving_average(df["Episode_Return"].values)
    trim = len(df) - len(smoothed)
    
    df_out = pd.DataFrame({
        "Episode_Return": df["Episode_Return"].values[trim:],
        "Episode_Return_smooth": smoothed,
        "env_step": df["env_step"].values[trim:]
    })
    
    df_out.to_csv(f, index=False)  # overwrite the original file
    print(f"Updated {f}")

print("Done — all files now have Episode_Return_smooth column")