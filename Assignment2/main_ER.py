import gymnasium as gym
import numpy as np
import pandas as pd
import random
import torch
from DQN_experience_replay import ExperienceReplayAgent

def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

def interpolate_to_common_steps(all_returns, all_steps, n_points=500):
    common_steps = np.linspace(
        max(steps[0] for steps in all_steps),
        min(steps[-1] for steps in all_steps),
        n_points
    )
    interpolated = []
    for returns, steps in zip(all_returns, all_steps):
        interp = np.interp(common_steps, steps, returns)
        interpolated.append(interp)
    return np.array(interpolated), common_steps

def run_experiment(seed=0):
    env = gym.make("CartPole-v1")
    agent = ExperienceReplayAgent(
        lr=1e-4,
        hidden_size=256,
        epsilon_decay_steps=500_000,
        gamma=0.9
    )
    random.seed(seed)
    torch.manual_seed(seed)

    TOTAL_STEPS = 1_000_000
    TRAIN_FREQ = 1

    returns = []
    steps_log = []
    env_step = 0
    episode = 0
    state, _ = env.reset(seed=seed)
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
                print(f"[Seed {seed}] Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {agent.epsilon:.3f}")
            state, _ = env.reset()
            episode_return = 0
            episode += 1

    env.close()
    return returns, steps_log

if __name__ == "__main__":
    N_SEEDS = 5
    all_runs = []
    all_steps = []

    for seed in range(N_SEEDS):
        print(f"\n=== Running seed {seed+1}/{N_SEEDS} ===")
        returns, steps = run_experiment(seed=seed)
        all_runs.append(returns)
        all_steps.append(steps)
        # save individual seed result
        pd.DataFrame({"Episode_Return": returns, "env_step": steps}).to_csv(
            f"Assignment2/dqn_er_seed{seed}.csv", index=False
        )

    # average across seeds
    interpolated, common_steps = interpolate_to_common_steps(all_runs, all_steps)
    mean_returns = np.mean(interpolated, axis=0)
    std_returns = np.std(interpolated, axis=0)

    # save averaged results
    df = pd.DataFrame({
        "Episode_Return": mean_returns,
        "Episode_Return_std": std_returns,
        "env_step": common_steps
    })
    df.to_csv("Assignment2/dqn_experience_replay_results.csv", index=False)
    print(f"\nDone. Final mean return (last 50 points): {np.mean(mean_returns[-50:]):.1f} ± {np.mean(std_returns[-50:]):.1f}")