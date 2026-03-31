import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIGS ---------------- #

configs = {
    "Naive":   "Assignment2/dqn_naive/average_dqn_naive_results.csv",
    "ER Only": "Assignment2/dqn_er/average_dqn_er_results.csv",
    "TN Only": "Assignment2/dqn_tn/average_dqn_target_network_results.csv",
    "TN + ER": "Assignment2/dqn_er_tn/average_dqn_er_tn_results.csv",
}

# load data
data = []
for label, path in configs.items():
    df = pd.read_csv(path)
    data.append((df, label))

# ---------------- PLOTTING ---------------- #

fig, ax = plt.subplots(figsize=(10, 5))

for df, label in data:
    ax.plot(df["env_step"], df["Episode_Return_smooth"], label=label)

ax.set_xlabel("Environment Steps")
ax.set_ylabel("Mean Return")
ax.set_title("DQN Variants Comparison on CartPole-v1")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("Assignment2/comparison_dqn_variants.png", dpi=150)
plt.show()

# ---------------- METRICS ---------------- #

threshold = 495  # CartPole-v1 solved threshold

def steps_to_threshold(df, col="Episode_Return_smooth"):
    for i in range(len(df)):
        if df[col].iloc[i] >= threshold:
            return df["env_step"].iloc[i]
    return None

print("\n--- Final Performance (mean of last 50 episodes) ---")
for df, label in data:
    mean = df["Episode_Return"].tail(50).mean()
    std = df["Episode_Return"].tail(50).std()
    print(f"{label}: {mean:.1f} ± {std:.1f}")

print("\n--- Sample Efficiency (steps to reach return of 495) ---")
for df, label in data:
    print(f"{label}: {steps_to_threshold(df)}")

print("\n--- AUC (area under smoothed return curve) ---")
for df, label in data:
    auc = np.trapezoid(df["Episode_Return_smooth"], df["env_step"])
    print(f"{label}: {auc:.0f}")

# ---------------- SAVE METRICS ---------------- #

with open("Assignment2/final_metrics.txt", "w") as f:
    f.write("Final Performance (mean return of last 50 episodes):\n")
    for df, label in data:
        mean = df["Episode_Return"].tail(50).mean()
        std  = df["Episode_Return"].tail(50).std()
        f.write(f"  {label}: {mean:.2f} ± {std:.2f}\n")

    f.write("\nSample Efficiency (steps to reach return of 495):\n")
    for df, label in data:
        f.write(f"  {label}: {steps_to_threshold(df)}\n")

    f.write("\nAUC (area under smoothed return curve):\n")
    for df, label in data:
        auc = np.trapezoid(df["Episode_Return_smooth"], df["env_step"])
        f.write(f"  {label}: {auc:.0f}\n")

print("\nMetrics saved to Assignment2/final_metrics.txt")