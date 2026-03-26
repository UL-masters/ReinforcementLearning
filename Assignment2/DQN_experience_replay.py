import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# class representing a DQN agent with experience replay
class ExperienceReplayAgent:
    def __init__(self, state_dim=4, action_dim=2):
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4) # learning rate and Adam optimizer

        self.gamma = 0.99
        self.action_dim = action_dim

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - 0.05) / (0.5 * 1_000_000) # decay epsilon from 1.0 to 0.05 over 500k steps

        self.replay_buffer = ReplayBuffer(100_000)
        self.batch_size = 64

    # epsilon-greedy action selection
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        # convert state to tensor and get action with highest Q-value
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
        with torch.no_grad():
            return self.model(state).argmax().item()
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    # perform a training step using replay buffer
    def train_step(self):
        # only train if enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q(s,a)
        q_values = self.model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # target
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_sa, target)

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
    
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            # two hidden layers with 64 units each and ReLU activations
            nn.Linear(state_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    # forward pass to compute Q-values for all actions given a state
    def forward(self, x):
        return self.net(x)
    

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    
    