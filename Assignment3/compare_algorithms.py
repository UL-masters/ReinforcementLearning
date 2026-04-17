import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from reinforce import PolicyNetwork, compute_returns, collect_episode, reinforce_update
from AC import ValueNetwork, ac_update
from A2C import a2c_update


ENVIRONMENT_NAME = "CartPole-v1"
GAMMA = 0.99
LEARNING_RATE = 1e-3
N_EPISODES = 1000
HIDDEN_SIZE = 64
SEED = 42
ENTROPY_COEF = 0.01
LOG_INTERVAL = 50


# ----- REINFORCE ----- #

def train_reinforce():
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy    = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    returns_history = []

    print("Training REINFORCE...")
    for episode in range(1, N_EPISODES + 1):
        log_probs, rewards, _, ep_return = collect_episode(env, policy)
        reinforce_update(optimizer, log_probs, rewards, GAMMA)
        returns_history.append(ep_return)
        if episode % LOG_INTERVAL == 0:
            print(f"  Episode {episode}/{N_EPISODES} | "
                  f"Avg Return: {np.mean(returns_history[-LOG_INTERVAL:]):.2f}")
    env.close()
    return returns_history


# ----- AC (basic Actor-Critic) ----- #

def train_ac():
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    value_net  = ValueNetwork(state_dim, HIDDEN_SIZE)
    actor_opt  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    critic_opt = optim.Adam(value_net.parameters(),  lr=LEARNING_RATE)
    returns_history = []
    print("Training AC...")
    for episode in range(1, N_EPISODES + 1):
        log_probs, rewards, states, ep_return = collect_episode(env, policy_net)
        ac_update(actor_opt, critic_opt, log_probs, rewards, states, value_net, GAMMA)
        returns_history.append(ep_return)
        if episode % LOG_INTERVAL == 0:
            print(f"  Episode {episode}/{N_EPISODES} | "
                  f"Avg Return: {np.mean(returns_history[-LOG_INTERVAL:]):.2f}")
    env.close()
    return returns_history


# ----- A2C ----- #

def train_a2c():
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    value_net  = ValueNetwork(state_dim, HIDDEN_SIZE)
    actor_opt  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    critic_opt = optim.Adam(value_net.parameters(),  lr=LEARNING_RATE)
    returns_history = []

    print("Training A2C...")
    for episode in range(1, N_EPISODES + 1):
        log_probs, rewards, states, ep_return = collect_episode(env, policy_net)
        a2c_update(actor_opt, critic_opt, log_probs, rewards, states,
                   policy_net, value_net, GAMMA)
        returns_history.append(ep_return)
        if episode % LOG_INTERVAL == 0:
            print(f"  Episode {episode}/{N_EPISODES} | "
                  f"Avg Return: {np.mean(returns_history[-LOG_INTERVAL:]):.2f}")
    env.close()
    return returns_history


# ----- plotting ----- #

def smooth(returns, window=50):
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(returns) + 1)
    return x, smoothed

def plot_comparison(reinforce_returns, ac_returns, a2c_returns, window=50):
    fig, ax = plt.subplots(figsize=(11, 5))
    episodes = np.arange(1, N_EPISODES + 1)

    # REINFORCE
    rx, rs = smooth(reinforce_returns, window)
    ax.plot(episodes, reinforce_returns, alpha=0.15, color="#E87040")
    ax.plot(rx, rs, color="#E87040", linewidth=2, label="REINFORCE (rolling avg)")

    # AC
    acx, acs = smooth(ac_returns, window)
    ax.plot(episodes, ac_returns, alpha=0.15, color="#16A34A")
    ax.plot(acx, acs, color="#16A34A", linewidth=2, label="AC (rolling avg)")

    # A2C
    a2cx, a2cs = smooth(a2c_returns, window)
    ax.plot(episodes, a2c_returns, alpha=0.15, color="#2563EB")
    ax.plot(a2cx, a2cs, color="#2563EB", linewidth=2, label="A2C (rolling avg)")

    ax.axhline(500, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Max return (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("REINFORCE vs AC vs A2C on CartPole-v1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("comparison_learning_curves.png", dpi=150)
    plt.show()
    print("Saved comparison_learning_curves.png")


# ----- main ----- #

if __name__ == "__main__":
    reinforce_returns = train_reinforce()
    ac_returns        = train_ac()
    a2c_returns       = train_a2c()
    plot_comparison(reinforce_returns, ac_returns, a2c_returns)