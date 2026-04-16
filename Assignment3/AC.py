import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

ENVIRONMENT_NAME = "CartPole-v1"
GAMMA = 0.99            # discount factor
LEARNING_RATE = 1e-3    # learning rate for both actor and critic optimizers
N_EPISODES = 1000       # number of episodes to train on
HIDDEN_SIZE = 64        # number of neurons in the hidden layers
SEED = 42               # random seed for reproducibility

LOG_INTERVAL = 50


# ----- policy network (actor) ----- #
# pi_theta : S -> Delta(A)

class PolicyNetwork(nn.Module):

    # initialize the policy network with given state and action dimensions, and hidden layer size
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    # forward pass through the network to get action logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # get action distribution for given state by passing it through network
    def get_distribution(self, state: torch.Tensor) -> Categorical:
        logits = self.forward(state)
        return Categorical(logits=logits)


# ----- value network (critic) ----- #
# Q_phi : S -> R  (approximates Q^pi(s,a) via MC returns, so maps state -> scalar)

class ValueNetwork(nn.Module):

    # initialize the value network with given state dimension and hidden layer size
    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    # forward pass to get scalar Q-value estimate for a given state
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# compute discounted returns from list of rewards and discount factor gamma
def compute_returns(rewards: list, gamma: float) -> torch.Tensor:

    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    return returns


# run one episode and collect log probs, rewards, states, and total return
def collect_episode(env: gym.Env, policy: PolicyNetwork) -> tuple:

    log_probs = []
    rewards = []
    states = []

    state, _ = env.reset()

    done = False
    while not done:

        state_tensor = torch.tensor(state, dtype=torch.float32)
        states.append(state_tensor)

        # sample action from policy distribution
        dist = policy.get_distribution(state_tensor)
        action = dist.sample()

        # store log probability of chosen action
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)

        # step environment with chosen action
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        rewards.append(reward)
        state = next_state

    episode_return = sum(rewards)
    return log_probs, rewards, states, episode_return


def ac_update(
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    log_probs: list,
    rewards: list,
    states: list,
    value_net: ValueNetwork,
    gamma: float,
) -> tuple:

    # compute Monte-Carlo returns as Q estimate: G_t = r_t + gamma * r_{t+1} + ...
    returns = compute_returns(rewards, gamma)

    # stack states and compute Q(s) for each visited state
    states_tensor = torch.stack(states)

    log_probs_tensor = torch.stack(log_probs)

    # actor loss: use raw MC returns (Q-values) as the weight — no baseline subtracted
    # negative because we want gradient ASCENT on J(theta)
    actor_loss = -(log_probs_tensor * returns).sum()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # critic loss: MSE between predicted Q(s) and MC return G_t
    # re-forward after actor step so gradients are clean
    q_values_new = value_net(states_tensor)
    critic_loss = nn.functional.mse_loss(q_values_new, returns)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()


# train actor and critic using basic AC and return list of episode returns
def train() -> list:

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)

    # get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # initialize actor (policy) and critic (value) networks and their optimizers
    policy_net = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    value_net  = ValueNetwork(state_dim, HIDDEN_SIZE)

    actor_optimizer  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(value_net.parameters(),  lr=LEARNING_RATE)

    episode_returns = []

    # loop over episodes to collect data and update actor and critic
    for episode in range(1, N_EPISODES + 1):

        # collect log probabilities, rewards, states, and return for one episode
        log_probs, rewards, states, episode_return = collect_episode(env, policy_net)

        # perform AC update for both actor and critic using collected episode data
        actor_loss, critic_loss = ac_update(
            actor_optimizer, critic_optimizer,
            log_probs, rewards, states,
            value_net, GAMMA,
        )

        episode_returns.append(episode_return)

        # print progress every LOG_INTERVAL episodes
        if episode % LOG_INTERVAL == 0:
            avg_return = np.mean(episode_returns[-LOG_INTERVAL:])
            print(f"Episode {episode}/{N_EPISODES} | "
                  f"Average Return: {avg_return:.2f} | "
                  f"Actor Loss: {actor_loss:.4f} | "
                  f"Critic Loss: {critic_loss:.4f}")

    env.close()
    return episode_returns


def plot_returns(returns: list, window: int = 50):

    episodes = np.arange(1, len(returns) + 1)

    # rolling average for smoother curve
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    smooth_x = np.arange(window, len(returns) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, returns, alpha=0.3, color="steelblue", label="Return per episode")
    plt.plot(smooth_x, smoothed, color="steelblue", linewidth=2, label=f"Rolling avg (window={window})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("AC on CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ac_learning_curve.png", dpi=150)
    plt.show()
    print("Plot saved to ac_learning_curve.png")


if __name__ == "__main__":
    returns = train()
    plot_returns(returns)