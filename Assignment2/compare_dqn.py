import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

configs = {
    "Naive":   "Assignment2/dqn_naive/average_dqn_naive_results.csv",
    "ER Only": "Assignment2/dqn_er/average_dqn_er_results.csv",
    # "TN Only": "Assignment2/dqn_tn/average_dqn_tn_results.csv",
    # "TN + ER": "Assignment2/dqn_tn_er/average_dqn_tn_er_results.csv",
}

fig, ax = plt.subplots(figsize=(10, 5))

for label, path in configs.items():
    df = pd.read_csv(path)
    ax.plot(df["env_step"], df["Episode_Return_smooth"], label=label)

ax.set_xlabel("Environment Steps")
ax.set_ylabel("Mean Return")
ax.set_title("DQN Variants Comparison on CartPole-v1")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("Assignment2/comparison_naive_eq.png", dpi=150)
plt.show()