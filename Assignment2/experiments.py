import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# -------- LOAD DATA --------

baseline             = pd.read_csv("Assignment2/BaselineDataCartPole.csv")
dqn_naive            = pd.read_csv("Assignment2/dqn_results.csv")
dqn_er               = pd.read_csv("Assignment2/dqn_experience_replay_results.csv")
dqn_tn               = pd.read_csv("Assignment2/dqn_target_network_results.csv")
dqn_full             = pd.read_csv("Assignment2/dqn_full_results.csv")

# -------- FOUR-WAY COMPARISON PLOT --------

fig, ax = plt.subplots(figsize=(10, 5))

configs = [
    (dqn_naive, "Naive",      "#e07b54"),
    (dqn_tn,    "TN only",    "#5b8dd9"),
    (dqn_er,    "ER only",    "#57a773"),
    (dqn_full,  "TN + ER",    "#9b59b6"),
    (baseline,  "Baseline",   "#888888"),
]

for df, label, color in configs:
    ax.plot(df["env_step"], df["Episode_Return_smooth"],
            label=label, color=color, linewidth=1.8)

ax.axhline(y=500, color="black", linestyle="--", linewidth=1, alpha=0.4, label="Optimal (500)")
ax.set_xlabel("Environment Steps", fontsize=12)
ax.set_ylabel("Episode Return (smoothed)", fontsize=12)
ax.set_title("DQN Variants — Four-Way Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("Assignment2/four_way_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to Assignment2/four_way_comparison.png")

# -------- METRICS --------

threshold = 495  # CartPole-v1 is considered solved at 500, use 495 as threshold

def steps_to_threshold(df, col="Episode_Return_smooth"):
    for i in range(len(df)):
        if df[col].iloc[i] >= threshold:
            return df["env_step"].iloc[i]
    return None

print("\n--- Final Performance (mean of last 50 episodes) ---")
for df, label, _ in configs:
    print(f"{label}: {df['Episode_Return'].tail(50).mean():.1f} ± {df['Episode_Return'].tail(50).std():.1f}")

print("\n--- Sample Efficiency (steps to reach return of 495) ---")
for df, label, _ in configs:
    print(f"{label}: {steps_to_threshold(df)}")

print("\n--- AUC (area under smoothed return curve) ---")
for df, label, _ in configs:
    auc = np.trapezoid(df["Episode_Return_smooth"], df["env_step"])
    print(f"{label}: {auc:.0f}")

# -------- SAVE METRICS --------

with open("Assignment2/metrics.txt", "w") as f:
    f.write("Final Performance (mean return of last 50 episodes):\n")
    for df, label, _ in configs:
        mean = df["Episode_Return"].tail(50).mean()
        std  = df["Episode_Return"].tail(50).std()
        f.write(f"  {label}: {mean:.2f} ± {std:.2f}\n")

    f.write("\nSample Efficiency (steps to reach return of 495):\n")
    for df, label, _ in configs:
        f.write(f"  {label}: {steps_to_threshold(df)}\n")

    f.write("\nAUC (area under smoothed return curve):\n")
    for df, label, _ in configs:
        auc = np.trapezoid(df["Episode_Return_smooth"], df["env_step"])
        f.write(f"  {label}: {auc:.0f}\n")

print("\nMetrics saved to Assignment2/metrics.txt")