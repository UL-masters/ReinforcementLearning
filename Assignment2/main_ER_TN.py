import gymnasium as gym
import numpy as np
import pandas as pd
from DQN_ER_TN import FullDQNAgent

env = gym.make("CartPole-v1")
agent = FullDQNAgent()

TOTAL_STEPS = 1_000_000
TRAIN_FREQ = 1  # train every N steps

returns = []
steps_log = []

env_step = 0
episode = 0

state, _ = env.reset()
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

def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')

smoothed = moving_average(returns)
trim = len(returns) - len(smoothed)

df = pd.DataFrame({
    "Episode_Return": returns[trim:],
    "Episode_Return_smooth": smoothed,
    "env_step": steps_log[trim:]
})

df.to_csv("Assignment2/dqn_er_tn/dqn_er_tn_results_5.csv", index=False)
print(f"Done. Total episodes: {episode}, Total steps: {env_step}")