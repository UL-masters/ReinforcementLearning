from datetime import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random
import torch
from DQN_naive import NaiveAgent
from main_Naive import moving_average

# run one training experiment for a given config and random seed
def run_experiment(config, total_steps=500_000, train_freq=4, seed=0):
    env = gym.make("CartPole-v1")
    agent = NaiveAgent(**config)

    random.seed(seed)
    torch.manual_seed(seed)
    state, _ = env.reset(seed=seed)

    returns = []
    steps_log = []
    env_step = 0

    episode_return = 0

    while env_step < total_steps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if env_step % train_freq == 0:
            agent.train_step(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_return += reward
        env_step += 1
        agent.decay_epsilon()

        if done:
            returns.append(episode_return)
            steps_log.append(env_step)
            state, _ = env.reset()
            episode_return = 0

    env.close()
    return returns, steps_log


# save returns and environment steps to csv
def save_results(returns, steps, filename):
    pd.DataFrame({
        "Episode_Return": returns,
        "env_step": steps
    }).to_csv(filename, index=False)


# summarize final performance using the last 50 episodes
def evaluate(returns):
    tail = returns[-50:] if len(returns) >= 50 else returns
    return {
        "final_mean": np.mean(tail),
        "std": np.std(tail)
    }


# interpolate runs onto a shared step axis before averaging
def interpolate_to_common_steps(all_returns, all_steps, n_points=500):
    common_steps = np.linspace(
        max(s[0] for s in all_steps),
        min(s[-1] for s in all_steps),
        n_points
    )
    interpolated = [np.interp(common_steps, steps, returns)
                    for returns, steps in zip(all_returns, all_steps)]
    return np.array(interpolated), common_steps


if __name__ == "__main__":
    start = datetime.now()
    print(f"Starting ablation study at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # directory for per-run and per-setting outputs
    os.makedirs("Assignment2/experiments", exist_ok=True)

    # baseline config - middle value of each parameter range
    baseline_config = {
        "lr": 1e-3,
        "epsilon_decay_steps": 500_000,
        "hidden_size": 64,
        "gamma": 0.9
    }

    # parameters to sweep - train_freq handled separately
    experiments = {
        "lr":                   [1e-4, 1e-3, 1e-2],
        "train_freq":           [1, 4, 16],
        "epsilon_decay_steps":  [100_000, 500_000, 1_000_000],
        "hidden_size":          [32, 64, 256],
        "gamma":                [0.8, 0.9, 0.99],
    }

    N_SEEDS = 5
    results_summary = []

    # sweep one parameter at a time around the baseline configuration
    for param, values in experiments.items():
        print(f"\n=== Testing {param} ===")

        for val in values:
            config = baseline_config.copy()
            train_freq = baseline_config.get("train_freq", 4)

            if param == "train_freq":
                train_freq = val
            else:
                config[param] = val

            all_runs, all_steps = [], []

            # run multiple seeds for each parameter value
            for seed in range(N_SEEDS):
                print(f"  {param}={val} | seed {seed}")
                returns, steps = run_experiment(config, train_freq=train_freq, seed=seed)
                all_runs.append(returns)
                all_steps.append(steps)
                save_results(returns, steps,
                             f"Assignment2/experiments/{param}_{val}_seed{seed}.csv")
            
            # average across seeds on common step axis
            interpolated, common_steps = interpolate_to_common_steps(all_runs, all_steps)
            mean_returns = np.mean(interpolated, axis=0)
            save_results(mean_returns, common_steps,
                         f"Assignment2/experiments/{param}_{val}.csv")

            metrics = evaluate(mean_returns)
            results_summary.append({
                "param": param,
                "value": val,
                "final_mean": metrics["final_mean"],
                "std": metrics["std"],
                "std_across_seeds": np.std([np.mean(r[-50:]) for r in all_runs])
            })

    # write aggregated summary across all ablation settings
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("Assignment2/experiments-ablation/summary.csv", index=False)

    end = datetime.now()
    print(f"\nAblation study completed at {end.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(duration: {end - start})")

    print("\n--- Best value per parameter ---")
    for param in experiments:
        subset = summary_df[summary_df["param"] == param]
        best = subset.loc[subset["final_mean"].idxmax()]
        print(f"  {param}: {best['value']} "
              f"(mean={best['final_mean']:.1f} ± {best['std_across_seeds']:.1f})")