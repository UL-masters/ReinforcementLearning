import gymnasium as gym
import numpy as np
import pandas as pd
import random
import torch
import os
from DQN_experience_replay import ExperienceReplayAgent

os.makedirs("Assignment2/dqn_er", exist_ok=True)

def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')

SEED = 4
TOTAL_STEPS = 1_000_000
TRAIN_FREQ = 1

env = gym.make("CartPole-v1")
agent = ExperienceReplayAgent(
    lr=1e-4,
    hidden_size=256,
    epsilon_decay_steps=500_000,
    gamma=0.9
)

random.seed(SEED)
torch.manual_seed(SEED)
state, _ = env.reset(seed=SEED)

returns = []
steps_log = []
env_step = 0
episode = 0
episode_return = 0

while env_step < TOTAL_STEPS:
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    agent.store_transition(state, action, reward, next_state, float(terminated))
    if env_step % TRAIN_FREQ == 0:
        agent.train_step()

    state = next_state
    episode_return += reward
    env_step += 1
    agent.decay_epsilon()

    if done:
        returns.append(episode_return)
        steps_log.append(env_step)
        if episode % 50 == 0:
            print(f"Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {agent.epsilon:.3f}")
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

df.to_csv(f"Assignment2/dqn_er/dqn_er_results_{SEED}.csv", index=False)
print(f"Done. Total episodes: {episode}, Total steps: {env_step}")