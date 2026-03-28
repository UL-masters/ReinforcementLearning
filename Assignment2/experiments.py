import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

baseline = pd.read_csv("Assignment2/BaselineDataCartPole.csv")
dqn_naive = pd.read_csv("Assignment2/dqn_results.csv")
dqn_experience_replay = pd.read_csv("Assignment2/dqn_experience_replay_results.csv")

# plot 
plt.plot(baseline["env_step"], baseline["Episode_Return_smooth"], label="Baseline")
plt.plot(dqn_naive["env_step"], dqn_naive["Episode_Return_smooth"], label="DQN Naive")
plt.plot(dqn_experience_replay["env_step"], dqn_experience_replay["Episode_Return_smooth"], label="DQN Experience Replay")

plt.xlabel("Environment Steps")
plt.ylabel("Episode Return (smoothed)")
plt.title("Performance Comparison")
plt.legend()

# save plot
plt.savefig("Assignment2/performance_comparison.png")

# ---------- METRICS --------

# final performance

baseline_final = baseline["Episode_Return"].tail(50).mean()
dqn_naive_final = dqn_naive["Episode_Return"].tail(50).mean()
dqn_experience_replay_final = dqn_experience_replay["Episode_Return"].tail(50).mean()

print("Baseline final:", baseline_final)
print("DQN final:", dqn_naive_final)
print("DQN Experience Replay final:", dqn_experience_replay_final)

# stability / variance (std of last 50 episodes)

baseline_std = baseline["Episode_Return"].tail(50).std()
dqn_naive_std = dqn_naive["Episode_Return"].tail(50).std()
dqn_experience_replay_std = dqn_experience_replay["Episode_Return"].tail(50).std()

print("Baseline std:", baseline_std)
print("DQN std:", dqn_naive_std)
print("DQN Experience Replay std:", dqn_experience_replay_std)

# sample efficiency (steps to reach return of 195)

threshold = 195  

def steps_to_threshold(df):
    for i in range(len(df)):
        if df["Episode_Return_smooth"].iloc[i] >= threshold:
            return df["env_step"].iloc[i]
    return None

print("Baseline steps:", steps_to_threshold(baseline))
print("DQN steps:", steps_to_threshold(dqn_naive))
print("DQN Experience Replay steps:", steps_to_threshold(dqn_experience_replay))

# AUC (area under curve)

baseline_auc = np.trapezoid(baseline["Episode_Return_smooth"], baseline["env_step"])
dqn_auc = np.trapezoid(dqn_naive["Episode_Return_smooth"], dqn_naive["env_step"])
dqn_experience_replay_auc = np.trapezoid(dqn_experience_replay["Episode_Return_smooth"], dqn_experience_replay["env_step"])

print("Baseline AUC:", baseline_auc)
print("DQN AUC:", dqn_auc)
print("DQN Experience Replay AUC:", dqn_experience_replay_auc)

# save metrics to a text file
with open("Assignment2/metrics.txt", "w") as f:
    f.write(f"Final Performance (mean return of last 50 episodes):\n")
    f.write(f"Baseline: {baseline_final:.2f}\n")
    f.write(f"DQN Naive: {dqn_naive_final:.2f}\n")
    f.write(f"DQN Experience Replay: {dqn_experience_replay_final:.2f}\n\n")

    f.write(f"Stability (std of return of last 50 episodes):\n")
    f.write(f"Baseline: {baseline_std:.2f}\n")
    f.write(f"DQN Naive: {dqn_naive_std:.2f}\n")
    f.write(f"DQN Experience Replay: {dqn_experience_replay_std:.2f}\n\n")

    f.write(f"Sample Efficiency (steps to reach return of 195):\n")
    f.write(f"Baseline: {steps_to_threshold(baseline)} steps\n")
    f.write(f"DQN Naive: {steps_to_threshold(dqn_naive)} steps\n")
    f.write(f"DQN Experience Replay: {steps_to_threshold(dqn_experience_replay)} steps\n\n")

    f.write(f"AUC (area under curve of smoothed returns):\n")
    f.write(f"Baseline AUC: {baseline_auc:.2f}\n")
    f.write(f"DQN Naive AUC: {dqn_auc:.2f}\n")
    f.write(f"DQN Experience Replay AUC: {dqn_experience_replay_auc:.2f}\n")
    
    

