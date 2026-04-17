import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from reinforce import PolicyNetwork, collect_episode, reinforce_update
from AC import ValueNetwork, ac_update
from A2C import a2c_update


ENVIRONMENT_NAME = "CartPole-v1"
GAMMA = 0.99
LEARNING_RATE = 1e-3
N_EPISODES = 1000
HIDDEN_SIZE = 64
SEED = 42
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
    steps_history   = []
    total_steps     = 0

    print("Training REINFORCE...")
    for episode in range(1, N_EPISODES + 1):

        log_probs, rewards, states, ep_return = collect_episode(env, policy)
        reinforce_update(optimizer, log_probs, rewards, GAMMA)

        ep_length = len(rewards)
        total_steps += ep_length

        returns_history.append(ep_return)
        steps_history.append(total_steps)

        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode} | Avg Return: {np.mean(returns_history[-LOG_INTERVAL:]):.2f}")

    env.close()
    return steps_history, returns_history


# ----- AC ----- #

def train_ac():
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    value_net = ValueNetwork(state_dim, HIDDEN_SIZE)

    actor_opt  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    critic_opt = optim.Adam(value_net.parameters(),  lr=LEARNING_RATE)

    returns_history = []
    steps_history = []
    total_steps = 0

    print("Training AC...")
    for episode in range(1, N_EPISODES + 1):

        log_probs, rewards, states, ep_return = collect_episode(env, policy_net)
        ac_update(actor_opt, critic_opt, log_probs, rewards, states, value_net, GAMMA)

        ep_length = len(rewards)
        total_steps += ep_length

        returns_history.append(ep_return)
        steps_history.append(total_steps)

        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode} | Avg Return: {np.mean(returns_history[-LOG_INTERVAL:]):.2f}")

    env.close()
    return steps_history, returns_history 


# ----- A2C ----- #

def train_a2c():
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    value_net = ValueNetwork(state_dim, HIDDEN_SIZE)

    actor_opt = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    critic_opt = optim.Adam(value_net.parameters(),  lr=LEARNING_RATE)

    returns_history = []
    steps_history = []
    total_steps = 0

    print("Training A2C...")
    for episode in range(1, N_EPISODES + 1):

        log_probs, rewards, states, ep_return = collect_episode(env, policy_net)
        a2c_update(actor_opt, critic_opt, log_probs, rewards, states,
                   policy_net, value_net, GAMMA)

        ep_length = len(rewards)
        total_steps += ep_length

        returns_history.append(ep_return)
        steps_history.append(total_steps)

        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode} | Avg Return: {np.mean(returns_history[-LOG_INTERVAL:]):.2f}")

    env.close()
    return steps_history, returns_history   


# ----- plotting ----- #

def smooth_steps(steps, returns, window=50):
    smoothed_returns = np.convolve(returns, np.ones(window)/window, mode="valid")
    smoothed_steps   = steps[window-1:]
    return smoothed_steps, smoothed_returns


def plot_comparison(reinforce_data, ac_data, a2c_data, window=50):
    fig, ax = plt.subplots(figsize=(11, 5))

    r_steps, r_returns = reinforce_data
    ac_steps, ac_returns = ac_data
    a2c_steps, a2c_returns = a2c_data

    # REINFORCE
    rx, rs = smooth_steps(r_steps, r_returns, window)
    ax.plot(r_steps, r_returns, alpha=0.15)
    ax.plot(rx, rs, linewidth=2, label="REINFORCE")

    # AC
    acx, acs = smooth_steps(ac_steps, ac_returns, window)
    ax.plot(ac_steps, ac_returns, alpha=0.15)
    ax.plot(acx, acs, linewidth=2, label="AC")

    # A2C
    a2cx, a2cs = smooth_steps(a2c_steps, a2c_returns, window)
    ax.plot(a2c_steps, a2c_returns, alpha=0.15)
    ax.plot(a2cx, a2cs, linewidth=2, label="A2C")

    ax.axhline(500, linestyle="--", alpha=0.6)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Return")
    ax.set_title("Learning Curves of REINFORCE, AC, and A2C on CartPole-v1")

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    fig.savefig("Assignment3/comparison_learning_curves.png", dpi=150)
    print("Comparison plot saved to Assignment3/comparison_learning_curves.png")


# ----- main ----- #

if __name__ == "__main__":
    reinforce_data = train_reinforce()
    ac_data = train_ac()
    a2c_data = train_a2c()

    plot_comparison(reinforce_data, ac_data, a2c_data)