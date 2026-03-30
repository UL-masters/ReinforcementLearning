import torch
import torch.nn as nn
import torch.optim as optim
import random

# hyperparameters based on ablation study:
# - learning rate: 0.0001
# - update-to-data ratio: 1 (train every 1 step)
# - exploration decay steps: 500 000
# - network size: 256 hidden units
# - discount factor: 0.90

# class representing a naive DQN agent without experience replay or target networks
class NaiveAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-4, hidden_size=256, epsilon_decay_steps=500_000, gamma=0.9):
        
        self.model = QNetwork(state_dim, action_dim, hidden_size) # Q-network to estimate action values
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # learning rate and Adam optimizer

        self.gamma = gamma
        self.action_dim = action_dim

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - 0.05) / epsilon_decay_steps # decay epsilon from 1.0 to 0.05 over epsilon_decay_steps

    # epsilon-greedy action selection
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        # convert state to tensor and get action with highest Q-value
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
        with torch.no_grad():
            return self.model(state).argmax().item()

    # perform a training step using the observed transition
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # get the Q-value for the taken action
        q_sa = self.model(state)[0, action]

        # compute the target Q-value using the reward and the max Q-value of the next state
        with torch.no_grad():
            next_q = self.model(next_state).max(1)[0]
            target = reward + self.gamma * next_q * (1 - done)

        # compute the MSE loss btwn Q-value and target
        loss = nn.MSELoss()(q_sa, target.squeeze())

        # backpropagate the loss and update the model parameters
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

    # decay epsilon after each step
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
   
   
# simple feedforward neural network to represent the Q-function     
class QNetwork(nn.Module):
    
    def __init__(self, state_dim=4, action_dim=2, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            # two hidden layers with 64 units each and ReLU activations
            nn.Linear(state_dim, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    # forward pass to compute Q-values for all actions given a state
    def forward(self, x):
        return self.net(x)
    
    
    