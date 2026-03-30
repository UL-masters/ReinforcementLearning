import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# -------- LOAD SUMMARY --------

summary_df = pd.read_csv("Assignment2/experiments-ablation/summary.csv")

# all 5 parameters including train_freq
params = ["lr", "train_freq", "epsilon_decay_steps", "hidden_size", "gamma"]
titles = ["Learning Rate", "Update-to-Data Ratio", "Exploration (decay steps)", "Network Size", "Discount Factor (γ)"]

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()

# shared y-axis range across all subplots
all_means = summary_df["final_mean"]
all_stds  = summary_df["std_across_seeds"]
y_min = max(0, (all_means - all_stds).min() - 5)
y_max = (all_means + all_stds).max() + 5

for ax, param, title in zip(axs, params, titles):
    sub_df = summary_df[summary_df["param"] == param].sort_values("value")

    x   = sub_df["value"].values
    y   = sub_df["final_mean"].values
    std = sub_df["std_across_seeds"].values

    ax.plot(x, y, marker='o', markersize=7, linewidth=2, color="#3A7DC9", zorder=3)
    ax.fill_between(x, y - std, y + std, alpha=0.15, color="#3A7DC9", zorder=2)

    # mark best value
    best_idx = np.argmax(y)
    ax.scatter(x[best_idx], y[best_idx], color="#e07b54", zorder=4,
               s=80, label=f"Best: {x[best_idx]}")
    ax.legend(fontsize=8)

    # axis formatting
    if param == "lr":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    elif param in ("epsilon_decay_steps",):
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(param, fontsize=10)
    ax.set_ylabel("Mean Return (last 50 eps)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

# hide the unused 6th subplot
axs[-1].set_visible(False)

plt.suptitle("DQN Hyperparameter Ablation Study", fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("Assignment2/experiments-ablation/ablation_study.png", bbox_inches='tight', dpi=150)
plt.show()
print("Ablation plot saved to Assignment2/experiments-ablation/ablation_study.png")