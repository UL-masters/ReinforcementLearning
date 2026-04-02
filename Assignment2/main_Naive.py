import gymnasium as gym
import numpy as np
import pandas as pd
import random
import torch
import os
from DQN_naive import NaiveAgent

os.makedirs("Assignment2/dqn_naive", exist_ok=True)

def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

# hyperparameters based on ablation study
# - learning rate: 0.0001
# - update-to-data ratio: 1 (train every 1 step)
# - exploration decay steps: 500_000
# - network size: 256 hidden units
# - discount factor: 0.90

TOTAL_STEPS = 1_000_000
TRAIN_FREQ = 1

for SEED in range(5):
    print(f"\n=== Seed {SEED+1}/5 ===")

    env = gym.make("CartPole-v1")
    naive_agent = NaiveAgent()

    random.seed(SEED)
    torch.manual_seed(SEED)
    state, _ = env.reset(seed=SEED)

    returns = []
    steps_log = []
    env_step = 0
    episode = 0
    episode_return = 0

    while env_step < TOTAL_STEPS:
        action = naive_agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if env_step % TRAIN_FREQ == 0:
            naive_agent.train_step(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_return += reward
        env_step += 1
        naive_agent.decay_epsilon()

        if done:
            returns.append(episode_return)
            steps_log.append(env_step)
            if episode % 50 == 0:
                print(f"Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {naive_agent.epsilon:.3f}")
            state, _ = env.reset()
            episode_return = 0
            episode += 1

    env.close()

    smoothed = moving_average(returns)
    trim = len(returns) - len(smoothed)

    df = pd.DataFrame({
        "Episode_Return":        returns[trim:],
        "Episode_Return_smooth": smoothed,
        "env_step":              steps_log[trim:]
    })

    df.to_csv(f"Assignment2/dqn_naive/dqn_naive_results_{SEED}.csv", index=False)
    print(f"Seed {SEED} done.")

print("\nAll seeds completed.")