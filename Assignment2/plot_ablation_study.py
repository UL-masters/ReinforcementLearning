import gymnasium as gym
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


#  -------- PLOTTING ABLATION STUDY RESULTS --------

 # nicer plots with seaborn
sns.set_theme(style="whitegrid", palette="muted")

summary_df = pd.read_csv("Assignment2/experiments/summary.csv")

params = ["lr", "epsilon_decay_steps", "hidden_size", "gamma"]
titles = ["Learning Rate", "Exploration", "Hidden Size", "Gamma"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# shared y-axis range across all subplots
all_means = summary_df["final_mean"]
all_stds = summary_df["std"]
y_min = (all_means - all_stds).min() - 2
y_max = (all_means + all_stds).max() + 2

for ax, param, title in zip(axs.flatten(), params, titles):
    sub_df = summary_df[summary_df["param"] == param].sort_values("value")

    x = sub_df["value"].values
    y = sub_df["final_mean"].values
    std = sub_df["std"].values

    # Shaded band instead of error bars
    ax.plot(x, y, marker='o', markersize=7, linewidth=2, color="#3A7DC9", zorder=3)
    ax.fill_between(x, y - std, y + std, alpha=0.15, color="#3A7DC9", zorder=2)

    if param == "lr":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    
    if param == "epsilon_decay_steps":
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel("Return", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("DQN Hyperparameter Ablation", fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("Assignment2/experiments/ablation_study.png", bbox_inches='tight', dpi=150)
plt.show()
    