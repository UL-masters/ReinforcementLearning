import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


ENVIRONMENT_NAME = "CartPole-v1"
GAMMA = 0.99
LEARNING_RATE = 1e-3
N_EPISODES = 1000
HIDDEN_SIZE = 64
SEED = 42
ENTROPY_COEF = 0.01
LOG_INTERVAL = 50


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
    def forward(self, x):
        return self.net(x)
    def get_distribution(self, state):
        return Categorical(logits=self.forward(state))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_returns(rewards, gamma):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def collect_episode(env, policy):
    log_probs, rewards, states = [], [], []
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        states.append(state_tensor)
        dist = policy.get_distribution(state_tensor)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)
    return log_probs, rewards, states, sum(rewards)


def reinforce_update(optimizer, log_probs, rewards, gamma):
    returns = compute_returns(rewards, gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss = -(torch.stack(log_probs) * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


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


def a2c_update(actor_opt, critic_opt, log_probs, rewards, states, policy, value_net, gamma):
    returns     = compute_returns(rewards, gamma)
    states_t    = torch.stack(states)
    values      = value_net(states_t)
    advantages  = returns - values.detach()
    advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    entropy     = torch.stack([policy.get_distribution(s).entropy() for s in states]).mean()
    actor_loss  = -(torch.stack(log_probs) * advantages).sum() - ENTROPY_COEF * entropy
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
    critic_loss = nn.functional.mse_loss(value_net(states_t), returns)
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
    return actor_loss.item(), critic_loss.item()


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


def smooth(returns, window=50):
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(returns) + 1)
    return x, smoothed


def plot_comparison(reinforce_returns, a2c_returns, window=50):
    fig, ax = plt.subplots(figsize=(11, 5))
    episodes = np.arange(1, N_EPISODES + 1)

    # REINFORCE
    rx, rs = smooth(reinforce_returns, window)
    ax.plot(episodes, reinforce_returns, alpha=0.15, color="#E87040")
    ax.plot(rx, rs, color="#E87040", linewidth=2, label=f"REINFORCE (rolling avg)")

    # A2C  — paste your logged returns here if you don't want to re-run
    ax2c_x, a2c_s = smooth(a2c_returns, window)
    ax.plot(episodes, a2c_returns, alpha=0.15, color="#2563EB")
    ax.plot(ax2c_x, a2c_s, color="#2563EB", linewidth=2, label=f"A2C (rolling avg)")

    ax.axhline(500, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Max return (500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("REINFORCE vs A2C on CartPole-v1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("comparison_learning_curves.png", dpi=150)
    plt.show()
    print("Saved comparison_learning_curves.png")


if __name__ == "__main__":
    reinforce_returns = train_reinforce()
    a2c_returns       = train_a2c()
    plot_comparison(reinforce_returns, a2c_returns)