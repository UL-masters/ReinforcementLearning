import gymnasium as gym
import numpy as np
import pandas as pd
import random
import torch
import os
from DQN_target_network import TargetNetworkAgent

os.makedirs("Assignment2/dqn_tn", exist_ok=True)

# smooth learning curves for easier visual comparison
def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')



# training configuration
TOTAL_STEPS = 1_000_000

if __name__ == "__main__":
    # run multiple seeds for robust evaluation
    for SEED in range(5):
        print(f"\n=== Seed {SEED+1}/5 ===")

        env = gym.make("CartPole-v1")
        agent = TargetNetworkAgent()

        random.seed(SEED)
        torch.manual_seed(SEED)
        state, _ = env.reset(seed=SEED)

        returns = []
        steps_log = []
        env_step = 0
        episode = 0
        episode_return = 0

        # interact with the environment until the step budget is exhausted
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
                    print(f"Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {agent.epsilon:.3f}")
                state, _ = env.reset()
                episode_return = 0
                episode += 1

        env.close()

        # align smoothed and raw series before saving results
        smoothed = moving_average(returns)
        trim = len(returns) - len(smoothed)

        df = pd.DataFrame({
            "Episode_Return":        returns[trim:],
            "Episode_Return_smooth": smoothed,
            "env_step":              steps_log[trim:]
        })

        # save per-seed learning curve data
        df.to_csv(f"Assignment2/dqn_tn/dqn_target_network_results_{SEED}.csv", index=False)
        print(f"Seed {SEED} done.")

    print("\nAll seeds completed.")