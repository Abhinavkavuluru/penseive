"""
Double DQN Implementation for Adaptive Bitrate Video Streaming
Based on the Pensieve environment, replacing A3C with Double DQN algorithm

Double DQN addresses overestimation bias in standard DQN by:
- Using the primary network to SELECT the best action
- Using the target network to EVALUATE the value of that action
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os

# Environment constants (matching Pensieve)
S_INFO = 6  # State information dimensions
S_LEN = 8   # History length
A_DIM = 6   # Number of bitrate levels

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 100  # Update target network every N steps
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 5000
HIDDEN_SIZE = 128

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayMemory:
    """Experience Replay Buffer for Double DQN"""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Video Streaming ABR

    Input: State vector [S_INFO, S_LEN] containing:
        - Last bitrate (normalized)
        - Buffer size (normalized)
        - Throughput history
        - Download time history
        - Next chunk sizes for all bitrates
        - Remaining chunks (normalized)

    Output: Q-values for each of the 6 bitrate actions
    """

    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim  # (S_INFO, S_LEN) = (6, 8)
        self.action_dim = action_dim

        # Flatten input size
        input_size = state_dim[0] * state_dim[1]  # 6 * 8 = 48

        # Network architecture - similar to Pensieve but for Q-values
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_dim)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Flatten the state
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)

        return q_values


class DoubleDQNAgent:
    """
    Double DQN Agent for Adaptive Bitrate Streaming

    Key differences from standard DQN:
    - Uses two networks: primary (online) and target
    - Primary network selects actions
    - Target network evaluates actions (reduces overestimation)
    """

    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Networks
        self.primary_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.primary_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=LEARNING_RATE)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)

        # Training step counter
        self.steps = 0

        # Metrics
        self.losses = []
        self.rewards = []
        self.qoe_history = []

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        """
        # Calculate epsilon (linear decay)
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-self.steps / EPS_DECAY)

        if training and random.random() < epsilon:
            # Exploration: random action
            return random.randrange(self.action_dim)
        else:
            # Exploitation: best Q-value action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.primary_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step using Double DQN update

        Double DQN Update:
        Q(s,a) = r + γ * Q_target(s', argmax_a' Q_primary(s', a'))

        The key insight: use primary network to SELECT the best action,
        but use target network to EVALUATE its value.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        # Sample batch
        batch = self.memory.sample(BATCH_SIZE)

        # Unpack batch
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Current Q values
        current_q = self.primary_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use primary network to select action, target network to evaluate
        with torch.no_grad():
            # Primary network selects best action for next state
            next_actions = self.primary_network(next_states).argmax(dim=1)
            # Target network evaluates the selected action
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # TD target
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.primary_network.parameters(), 1.0)
        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Update target network periodically
        if self.steps % TARGET_UPDATE == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Copy weights from primary to target network"""
        self.target_network.load_state_dict(self.primary_network.state_dict())

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'primary_network': self.primary_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'losses': self.losses,
            'rewards': self.rewards,
            'qoe_history': self.qoe_history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.primary_network.load_state_dict(checkpoint['primary_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.losses = checkpoint.get('losses', [])
        self.rewards = checkpoint.get('rewards', [])
        self.qoe_history = checkpoint.get('qoe_history', [])
        print(f"Model loaded from {path}")


def compute_qoe(bitrate, rebuf, last_bitrate,
                video_bit_rate=[300, 750, 1200, 1850, 2850, 4300],
                rebuf_penalty=4.3, smooth_penalty=1.0):
    """
    Compute QoE (Quality of Experience) reward

    QoE = bitrate (Mbps) - rebuf_penalty * rebuf_time - smooth_penalty * |bitrate_change|

    This is the linear QoE metric used in Pensieve.
    """
    M_IN_K = 1000.0
    reward = video_bit_rate[bitrate] / M_IN_K \
             - rebuf_penalty * rebuf \
             - smooth_penalty * np.abs(video_bit_rate[bitrate] - video_bit_rate[last_bitrate]) / M_IN_K
    return reward
