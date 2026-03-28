import gymnasium as gym
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import random
import torch

from DQN_naive import NaiveAgent 

# # ablation study to test the impact of different hyperparameters on the naive DQN agent's performance on CartPole-v1. 
# # we will vary 
# # - learning rate
# # - update-to-data ratio (how many environment steps per update)
# # - network size
# # - exploration factor
# # - epsilon decay
# # - hidden layer size
# # - gamma
# # and save results for analysis


# moving average smoothing of returns
def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')



# run single experiment with given config and return episode returns and steps log
def run_experiment(config, total_steps=200_000, train_freq=1, seed=0):

    env = gym.make("CartPole-v1")
    agent = NaiveAgent(**config)

    returns = []
    steps_log = []

    env_step = 0
    state, _ = env.reset(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    episode_return = 0

    # main training loop
    while env_step < total_steps:
        action = agent.select_action(state)

        # take action and observe next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # train on this transition
        if env_step % train_freq == 0:
            agent.train_step(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_return += reward
        env_step += 1
        agent.decay_epsilon()

        # log results at end of episode
        if done:
            returns.append(episode_return)
            steps_log.append(env_step)

            state, _ = env.reset()
            episode_return = 0

    env.close()

    return returns, steps_log



# save results to CSV for analysis and plotting
def save_results(returns, steps, filename):
    df = pd.DataFrame({
        "Episode_Return": returns,
        "env_step": steps
    })

    df.to_csv(filename, index=False)

# evaluate final performance and stability of the agent based on returns
def evaluate(returns):
    return {
        "final_mean": np.mean(returns[-50:]),
        "std": np.std(returns[-50:])
    }

def interpolate_to_common_steps(all_returns, all_steps, n_points=500):
    """Interpolate runs to a common step axis for averaging across seeds."""
    common_steps = np.linspace(
        max(steps[0] for steps in all_steps),   # start where all runs have data
        min(steps[-1] for steps in all_steps),  # end where all runs still have data
        n_points
    )
    interpolated = []
    for returns, steps in zip(all_returns, all_steps):
        interp = np.interp(common_steps, steps, returns)
        interpolated.append(interp)
    return np.array(interpolated), common_steps


# main function to run all experiments and save summary results
if __name__ == "__main__":

    os.makedirs("Assignment2/experiments", exist_ok=True)

    # baseline config — middle value of each parameter range
    baseline_config = {
        "lr": 1e-3,
        # train_freq is handled separately since it's not a parameter of the agent
        "epsilon_decay_steps": 500_000,
        "hidden_size": 64,
        "gamma": 0.95 # TODO: experiment with 0.8, 0.95 and 0.99 because baseline is 0.95 for all experiments
    }

    # params to test — varied in orders of magnitude
    experiments = {
        "lr": [1e-4, 1e-3, 1e-2],                        # orders of magnitude: 0.0001, 0.001, 0.01
        "train_freq": [1, 4, 16],                          # orders of magnitude: 1, 4, 16
        "epsilon_decay_steps": [100_000, 500_000, 1_000_000],  # orders of magnitude: 100k, 500k, 1M
        "hidden_size": [32, 64, 256],                      # orders of magnitude: 32, 64, 256
        "gamma": [0.8, 0.9, 0.99],                         # orders of magnitude: reflects different time horizons
    }

    results_summary = []
    N_SEEDS = 3
    for param, values in experiments.items():
        print(f"\n=== Testing {param} ===")

        for val in values:
            all_runs = []
            all_steps = []
            config = baseline_config.copy()
            train_freq = 4
            if param == "train_freq":
                train_freq = val
            else:
                config[param] = val

            print(f"Running {param} = {val}")
            for seed_val in range(N_SEEDS):
                returns, steps = run_experiment(config, train_freq=train_freq, seed=seed_val)
                all_runs.append(returns)
                all_steps.append(steps)

            # average across seeds
            interpolated, common_steps = interpolate_to_common_steps(all_runs, all_steps)
            mean_returns = np.mean(interpolated, axis=0)

            # Save CSV
            filename = f"Assignment2/experiments/{param}_{val}.csv"
            save_results(mean_returns, common_steps, filename)

            # Evaluate
            metrics = evaluate(mean_returns)

            results_summary.append({
                "param": param,
                "value": val,
                "final_mean": metrics["final_mean"],
                "std": metrics["std"]
            })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("Assignment2/experiments/summary.csv", index=False)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

print("Ablation study completed. Results saved in Assignment2/experiments/")
for param in experiments.keys():
    subset = summary_df[summary_df["param"] == param]
    best = subset.loc[subset["final_mean"].idxmax()]
    print(f"Best {param}: {best['value']} (mean return: {best['final_mean']:.1f})")

# Best lr: 0.0001 (mean return: 17.9)
# Best train_freq: 16.0 (mean return: 17.7)
# Best epsilon_decay_steps: 1000000.0 (mean return: 22.5)
# Best hidden_size: 32.0 (mean return: 17.7)
# Best gamma: 0.9 (mean return: 55.6)

