import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# input csv files for each DQN variant
CONFIGS = {
    "Naive":   "Assignment2/dqn_naive/average_dqn_naive_results.csv",
    "ER Only": "Assignment2/dqn_er/average_dqn_er_results.csv",
    "TN Only": "Assignment2/dqn_tn/average_dqn_target_network_results.csv",
    "TN + ER": "Assignment2/dqn_er_tn/average_dqn_er_tn_results.csv",
}

PLOT_PATH = "Assignment2/comparison_dqn_variants.png"
METRICS_PATH = "Assignment2/final_metrics.txt"
THRESHOLD = 495  # CartPole-v1 solved threshold


# load all result files into (dataframe, label) tuples
def load_data(configs):
    data = []
    for label, path in configs.items():
        df = pd.read_csv(path)
        data.append((df, label))
    return data


# create and save the comparison plot
def plot_comparison(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    for df, label in data:
        ax.plot(df["env_step"], df["Episode_Return_smooth"], label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Return")
    ax.set_title("DQN Variants Comparison on CartPole-v1")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


# return the first step where performance crosses the target threshold
def steps_to_threshold(df, threshold, col="Episode_Return_smooth"):
    for i in range(len(df)):
        if df[col].iloc[i] >= threshold:
            return df["env_step"].iloc[i]
    return None


# compute summary metrics for all variants
def compute_metrics(data, threshold):
    metrics = []
    for df, label in data:
        final_mean = df["Episode_Return"].tail(50).mean()
        final_std = df["Episode_Return"].tail(50).std()
        sample_efficiency = steps_to_threshold(df, threshold)
        auc = np.trapezoid(df["Episode_Return_smooth"], df["env_step"])
        metrics.append({
            "label": label,
            "final_mean": final_mean,
            "final_std": final_std,
            "sample_efficiency": sample_efficiency,
            "auc": auc,
        })
    return metrics


# print metric blocks
def print_metrics(metrics, threshold):
    print("\n--- Final Performance (mean of last 50 episodes) ---")
    for item in metrics:
        print(f"{item['label']}: {item['final_mean']:.1f} +- {item['final_std']:.1f}")

    print(f"\n--- Sample Efficiency (steps to reach return of {threshold}) ---")
    for item in metrics:
        print(f"{item['label']}: {item['sample_efficiency']}")

    print("\n--- AUC (area under smoothed return curve) ---")
    for item in metrics:
        print(f"{item['label']}: {item['auc']:.0f}")


# write metrics to text file
def save_metrics(metrics, threshold, output_path):
    with open(output_path, "w") as f:
        f.write("Final Performance (mean return of last 50 episodes):\n")
        for item in metrics:
            f.write(f"  {item['label']}: {item['final_mean']:.2f} +- {item['final_std']:.2f}\n")

        f.write(f"\nSample Efficiency (steps to reach return of {threshold}):\n")
        for item in metrics:
            f.write(f"  {item['label']}: {item['sample_efficiency']}\n")

        f.write("\nAUC (area under smoothed return curve):\n")
        for item in metrics:
            f.write(f"  {item['label']}: {item['auc']:.0f}\n")


# run comparison workflow end-to-end
def main():
    data = load_data(CONFIGS)
    plot_comparison(data, PLOT_PATH)

    metrics = compute_metrics(data, THRESHOLD)
    print_metrics(metrics, THRESHOLD)
    save_metrics(metrics, THRESHOLD, METRICS_PATH)

    print(f"\nMetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()