import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

baseline = pd.read_csv("Assignment2/BaselineDataCartPole.csv")
dqn_naive = pd.read_csv("Assignment2/dqn_results.csv")

# plot 
plt.plot(baseline["env_step"], baseline["Episode_Return_smooth"], label="Baseline")
plt.plot(dqn_naive["env_step"], dqn_naive["Episode_Return_smooth"], label="DQN Naive")

plt.xlabel("Environment Steps")
plt.ylabel("Episode Return (smoothed)")
plt.title("Performance Comparison")
plt.legend()
plt.show()

# ---------- METRICS --------

# final performance

baseline_final = baseline["Episode_Return"].tail(50).mean()
dqn_naive_final = dqn_naive["Episode_Return"].tail(50).mean()

print("Baseline final:", baseline_final)
print("DQN final:", dqn_naive_final)

# stability / variance (std of last 50 episodes)

baseline_std = baseline["Episode_Return"].tail(50).std()
dqn_naive_std = dqn_naive["Episode_Return"].tail(50).std()

print("Baseline std:", baseline_std)
print("DQN std:", dqn_naive_std)

# sample efficiency (steps to reach return of 195)

threshold = 195  

def steps_to_threshold(df):
    for i in range(len(df)):
        if df["Episode_Return_smooth"].iloc[i] >= threshold:
            return df["env_step"].iloc[i]
    return None

print("Baseline steps:", steps_to_threshold(baseline))
print("DQN steps:", steps_to_threshold(dqn_naive))

# AUC (area under curve)

baseline_auc = np.trapezoid(baseline["Episode_Return_smooth"], baseline["env_step"])
dqn_auc = np.trapezoid(dqn_naive["Episode_Return_smooth"], dqn_naive["env_step"])

print("Baseline AUC:", baseline_auc)
print("DQN AUC:", dqn_auc)

