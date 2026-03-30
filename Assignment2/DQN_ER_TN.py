import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque


class FullDQNAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-4, hidden_size=256,
                 epsilon_decay_steps=500_000, gamma=0.9, target_update_freq=1000):

        self.model = QNetwork(state_dim, action_dim, hidden_size)
        self.target_model = copy.deepcopy(self.model)  # frozen target network
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.action_dim = action_dim
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - 0.05) / epsilon_decay_steps

        self.replay_buffer = ReplayBuffer(100_000)
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # current Q-values from online network
        q_values = self.model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # target Q-values from frozen target network
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1

        # periodically sync target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)


class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

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