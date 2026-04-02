import torch
import torch.nn as nn
import torch.optim as optim
import random

# hyperparameters based on ablation study:
# - learning rate: 0.0001
# - update-to-data ratio: 1 (train every 1 step)
# - exploration decay steps: 500_000
# - network size: 256 hidden units
# - discount factor: 0.90

class NaiveAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-4, hidden_size=256,
                 epsilon_decay_steps=500_000, gamma=0.9):

        self.model = QNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.action_dim = action_dim

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = (1.0 - 0.05) / epsilon_decay_steps

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        q_sa = self.model(state)[0, action]

        with torch.no_grad():
            next_q = self.model(next_state).max(1)[0]
            target = reward + self.gamma * next_q * (1 - done)

        loss = nn.MSELoss()(q_sa, target.squeeze())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

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