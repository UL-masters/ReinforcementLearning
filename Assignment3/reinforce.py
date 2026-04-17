import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
 
 
# ----- hyperparameters ----- #
 
ENVIRONMENT_NAME = "CartPole-v1"
GAMMA = 0.99 # discount factor
LEARNING_RATE = 1e-3 # learning rate for the optimizer
N_EPISODES = 1000 # number of episodes to train on
HIDDEN_SIZE = 64 # number of neurons in the hidden layer of policy network
SEED = 42 # random seed for reproducibility

# how often to print progress in episodes
LOG_INTERVAL = 50

# ----- policy network ----- #
# pi_theta : S -> change(A)

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
        
    # forward pass through the network to get action probabilities
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    # get action distribution for given state by passing it through network and applying softmax
    def get_distribution(self, state: torch.Tensor) -> Categorical:
        logits = self.forward(state)
        return Categorical(logits=logits)
    
    
# ----- helper functions ----- #

# compute discounted returns from list of rewards anddiscount factor gamma
def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # normalize returns to reduce variance
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns

# run one episode and collect log probabilities of actions taken, rewards received, states, and total return
def collect_episode(env: gym.Env, policy: PolicyNetwork) -> tuple[list, list, list, float]:
    
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

# ----- REINFORCE update step ----- #

def reinforce_update(optimizer: optim.Optimizer, log_probs: list, rewards: list, gamma: float) -> float:
    
    returns = compute_returns(rewards, gamma)
    
    # stack log probabilities into a single tensor
    log_probs_tensor = torch.stack(log_probs)
    
    # policy loss: negative because we want gradient ASCENT on J(theta)
    loss = -(log_probs_tensor * returns).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


# ----- main training loop ----- #

# train the policy network using REINFORCE algorithm and return list of episode returns 
def train() -> list[float]:
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=SEED)
    
    # get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # initialize policy network and optimizer
    policy = PolicyNetwork(state_dim, action_dim, HIDDEN_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    episode_returns = []
    
    # loop over episodes to collect data and update policy
    for episode in range(1, N_EPISODES + 1):
        
        # collect log probabilities, rewards, and return for one episode
        log_probs, rewards, episode_return = collect_episode(env, policy)
        
        # perform REINFORCE update using collected data
        loss = reinforce_update(optimizer, log_probs, rewards, GAMMA)
        
        episode_returns.append(episode_return)
        
        # print progress every LOG_INTERVAL episodes
        if episode % LOG_INTERVAL == 0:
            avg_return = np.mean(episode_returns[-LOG_INTERVAL:])
            print(f"Episode {episode}/{N_EPISODES} |"
                  f"Average Return: {avg_return:.2f} |"
                  f" Loss: {loss:.4f}")
            
    env.close()
    return episode_returns

# ----- plotting ----- #

def plot_returns(returns: list[float], window: int = 50):
   
    episodes = np.arange(1, len(returns) + 1)
 
    # rolling average for smoother curve
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    smooth_x = np.arange(window, len(returns) + 1)
 
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, returns, alpha=0.3, color="steelblue", label="Return per episode")
    plt.plot(smooth_x, smoothed, color="steelblue", linewidth=2,label=f"Rolling avg (window={window})")
 
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("REINFORCE on CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reinforce_learning_curve.png", dpi=150)
    plt.show()
    print("Plot saved to reinforce_learning_curve.png")
    
# ----- main ----- #

if __name__ == "__main__":
    returns = train()
    plot_returns(returns)