import gymnasium as gym
import numpy as np
import pandas as pd
from DQN_experience_replay import ExperienceReplayAgent

env = gym.make("CartPole-v1")
naive_agent = ExperienceReplayAgent()

TOTAL_STEPS = 1_000_000 # total environment steps to train for
TARGET_UPDATE_FREQ = 1000  # steps, not episodes
TRAIN_FREQ = 1   # train every N steps

returns = []
steps_log = []

env_step = 0
episode = 0

state, _ = env.reset()
episode_return = 0
done = False

while env_step < TOTAL_STEPS:
    action = naive_agent.select_action(state) # select action using epsilon-greedy policy
    next_state, reward, terminated, truncated, _ = env.step(action) # take action in the environment and observe next state and reward
    done = terminated or truncated

    # train using experience replay
    naive_agent.store_transition(state, action, reward, next_state, float(terminated))

    if env_step % TRAIN_FREQ == 0:
        naive_agent.train_step()
        
    # update
    state = next_state
    episode_return += reward
    env_step += 1
    naive_agent.decay_epsilon()

    if done:
        returns.append(episode_return)
        steps_log.append(env_step)

        # print progress every 50 episodes
        if episode % 50 == 0:
            print(f"Episode {episode} | Steps {env_step} | Return: {episode_return:.1f} | Epsilon: {naive_agent.epsilon:.3f}")

        # reset environment for next episode
        state, _ = env.reset()
        episode_return = 0
        episode += 1

env.close()

# save results in a csv file for analysis and plotting
def moving_average(data, window=10):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid') # moving average for smoothing returns

smoothed = moving_average(returns)
trim = len(returns) - len(smoothed)

df = pd.DataFrame({
    "Episode_Return": returns[trim:],
    "Episode_Return_smooth": smoothed,
    "env_step": steps_log[trim:]
})

df.to_csv("Assignment2/dqn_experience_replay_results.csv", index=False)
print(f"Done. Total episodes: {episode}, Total steps: {env_step}")