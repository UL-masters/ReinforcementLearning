import gymnasium as gym
import numpy as np
import pandas as pd
import random
import torch
from DQN_target_network import TargetNetworkAgent


def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


if __name__ == "__main__":
    seed = 0  # single run

    env = gym.make("CartPole-v1")
    agent = TargetNetworkAgent()

    random.seed(seed)
    torch.manual_seed(seed)
    state, _ = env.reset(seed=seed)

    TOTAL_STEPS = 1_000_000

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
                    f"Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {agent.epsilon:.3f}"
                )

            state, _ = env.reset()
            episode_return = 0
            episode += 1

    env.close()

    # smoothing (optional)
    smoothed = moving_average(returns)
    trim = len(returns) - len(smoothed)

    df = pd.DataFrame({
        "Episode_Return": returns[trim:],
        "Episode_Return_smooth": smoothed,
        "env_step": steps_log[trim:]
    })

    df.to_csv("Assignment2/dqn_tn/dqn_target_network_results_5.csv", index=False)

    print("\nDone. Single run results saved.")