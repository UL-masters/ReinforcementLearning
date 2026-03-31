import gymnasium as gym
import numpy as np
import pandas as pd
from DQN_TN import TargetNetworkAgent

N_SEEDS = 5
TOTAL_STEPS = 1_000_000

all_runs = []
all_steps = []

for seed in range(N_SEEDS):
    print(f"\n=== Seed {seed} ===")

    env = gym.make("CartPole-v1")
    agent = TargetNetworkAgent()

    import random, torch

    random.seed(seed)
    torch.manual_seed(seed)
    state, _ = env.reset(seed=seed)

    returns = []
    steps_log = []
    env_step = 0
    episode = 0
    episode_return = 0

    while env_step < TOTAL_STEPS:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.train_step(state, action, reward, next_state, float(terminated))
        state = next_state
        episode_return += reward
        env_step += 1
        agent.decay_epsilon()

        if done:
            returns.append(episode_return)
            steps_log.append(env_step)
            if episode % 50 == 0:
                print(
                    f"Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {agent.epsilon:.3f}")
            state, _ = env.reset()
            episode_return = 0
            episode += 1

    env.close()
    all_runs.append(returns)
    all_steps.append(steps_log)

# interpolate to common step axis and average
from experiments_ablation_study import interpolate_to_common_steps


def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


interpolated, common_steps = interpolate_to_common_steps(all_runs, all_steps)
mean_returns = np.mean(interpolated, axis=0)
smoothed = moving_average(mean_returns)
trim = len(mean_returns) - len(smoothed)

df = pd.DataFrame({
    "Episode_Return": mean_returns[trim:],
    "Episode_Return_smooth": smoothed,
    "env_step": common_steps[trim:]
})

df.to_csv("Assignment2/dqn_target_network_results.csv", index=False)
print(f"\nDone. Results saved.")