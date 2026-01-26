"""
PPO (Proximal Policy Optimization) Implementation for Adaptive Bitrate Video Streaming
Based on the Pensieve environment

PPO is an on-policy algorithm that:
- Uses a clipped surrogate objective to prevent too large policy updates
- Is more sample efficient than vanilla policy gradient methods
- Provides more stable training than A3C
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os

# Environment constants (matching Pensieve)
S_INFO = 6  # State information dimensions
S_LEN = 8   # History length
A_DIM = 6   # Number of bitrate levels

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95  # GAE parameter
CLIP_EPSILON = 0.2  # PPO clip parameter
LEARNING_RATE = 0.0003
ENTROPY_COEF = 0.01  # Entropy bonus coefficient
VALUE_COEF = 0.5  # Value loss coefficient
MAX_GRAD_NORM = 0.5  # Gradient clipping
PPO_EPOCHS = 4  # Number of PPO update epochs per batch
BATCH_SIZE = 64
HIDDEN_SIZE = 128


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO

    Shared feature extractor with separate heads for:
    - Actor: outputs action probabilities (policy)
    - Critic: outputs state value estimate

    Input: State vector [S_INFO, S_LEN]
    Actor Output: Probability distribution over 6 bitrate actions
    Critic Output: Scalar value estimate V(s)
    """

    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(ActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Flatten input size
        input_size = state_dim[0] * state_dim[1]  # 6 * 8 = 48

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Flatten the state
        x = x.view(x.size(0), -1)

        # Shared features
        features = self.shared(x)

        # Actor output (action probabilities)
        action_probs = self.actor(features)

        # Critic output (value estimate)
        value = self.critic(features)

        return action_probs, value

    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, value.squeeze(-1), entropy

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer to store rollout experiences for PPO"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        """Compute returns and GAE advantages"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])

        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + np.array(self.values)

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size):
        """Generate batches for PPO update"""
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                self.advantages[batch_indices],
                self.returns[batch_indices]
            )

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None


class PPOAgent:
    """
    PPO Agent for Adaptive Bitrate Streaming

    Key features:
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE)
    - Actor-Critic architecture with shared features
    """

    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Actor-Critic network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training step counter
        self.steps = 0
        self.updates = 0

        # Metrics
        self.losses = []
        self.rewards = []
        self.qoe_history = []

    def select_action(self, state, training=True):
        """
        Select action using current policy

        Args:
            state: Current state
            training: If True, sample from distribution; if False, use argmax
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value, _ = self.network.get_action(
                state_tensor, deterministic=not training
            )

        return action.item(), log_prob.item(), value.item()

    def store_experience(self, state, action, reward, log_prob, value, done):
        """Store experience in rollout buffer"""
        self.buffer.add(state, action, reward, value, log_prob, done)

    def update(self, last_value):
        """
        Perform PPO update

        PPO Objective:
        L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - c1 * L_value + c2 * H[π]

        where r(θ) = π_new(a|s) / π_old(a|s) is the probability ratio
        """
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value)

        # PPO update epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(PPO_EPOCHS):
            for states, actions, old_log_probs, advantages, returns in \
                    self.buffer.get_batches(BATCH_SIZE):

                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy outputs
                new_log_probs, values, entropy = self.network.evaluate_actions(states, actions)

                # Compute probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Clear buffer
        self.buffer.clear()
        self.updates += 1

        if n_updates > 0:
            return {
                'loss': total_loss / n_updates,
                'policy_loss': total_policy_loss / n_updates,
                'value_loss': total_value_loss / n_updates,
                'entropy': total_entropy / n_updates
            }
        return None

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'updates': self.updates,
            'losses': self.losses,
            'rewards': self.rewards,
            'qoe_history': self.qoe_history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint.get('updates', 0)
        self.losses = checkpoint.get('losses', [])
        self.rewards = checkpoint.get('rewards', [])
        self.qoe_history = checkpoint.get('qoe_history', [])
        print(f"Model loaded from {path}")
