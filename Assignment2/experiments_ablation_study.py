import gymnasium as gym
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from DQN_naive import NaiveAgent 

# # ablation study to test the impact of different hyperparameters on the naive DQN agent's performance on CartPole-v1. 
# # we will vary 
# # - learning rate
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
def run_experiment(config, total_steps=200_000):

    env = gym.make("CartPole-v1")
    agent = NaiveAgent(**config)

    returns = []
    steps_log = []

    env_step = 0
    state, _ = env.reset()
    episode_return = 0

    # main training loop
    while env_step < total_steps:
        action = agent.select_action(state)

        # take action and observe next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # train on this transition
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

    smoothed = moving_average(returns)
    trim = len(returns) - len(smoothed)

    df = pd.DataFrame({
        "Episode_Return": returns[trim:],
        "Episode_Return_smooth": smoothed,
        "env_step": steps[trim:]
    })

    df.to_csv(filename, index=False)

# evaluate final performance and stability of the agent based on returns
def evaluate(returns):
    return {
        "final_mean": np.mean(returns[-50:]),
        "std": np.std(returns[-50:])
    }


# main function to run all experiments and save summary results
if __name__ == "__main__":

    os.makedirs("Assignment2/experiments", exist_ok=True)

    # baseline config
    baseline_config = {
        "lr": 1e-4,
        "epsilon_decay_steps": 1_000_000,
        "hidden_size": 64,
        "gamma": 0.8
    }

    # params to test
    experiments = {
        "lr": [1e-5, 1e-4, 1e-3],                       
        "epsilon_decay_steps": [200_000, 800_000, 1_000_000],
        "hidden_size": [32, 64, 128],                
        "gamma": [0.9, 0.95, 0.99],                  
    }

    results_summary = []

    for param, values in experiments.items():
        print(f"\n=== Testing {param} ===")

        for val in values:
            config = baseline_config.copy()
            config[param] = val

            print(f"Running {param} = {val}")

            returns, steps = run_experiment(config)

            # Save CSV
            filename = f"Assignment2/experiments/{param}_{val}.csv"
            save_results(returns, steps, filename)

            # Evaluate
            metrics = evaluate(returns)

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

