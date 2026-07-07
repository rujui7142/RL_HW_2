"""Deep Q-Network agent (HW2, Question 2).

Implements:
  - a 2-layer MLP q-hat(s; w) in R^{|A|} (single forward pass gives the
    Q-value of every action, eq. (2)/(4) in the assignment),
  - a FIFO replay buffer D,
  - a target network w^- synced to w every `target_update_freq` updates
    (eq. (4)/(5)),
  - an epsilon-greedy `sample_action(explore=True)` policy.
"""
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """A single-hidden-layer ("2-layer") MLP mapping s -> R^{|A|}."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(state))
        return self.layer2(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity FIFO buffer D storing (s, a, r, s') transitions."""

    def __init__(self, capacity: int, seed: int = None):
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = self._rng.sample(self._buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    """DQN agent with target network and epsilon-greedy exploration."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 20_000,
        batch_size: int = 64,
        min_buffer_size: int = 500,
        target_update_freq: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay_steps: int = 5_000,
        device: str = "cpu",
        seed: int = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.q_network = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, seed=seed)

        self._train_calls = 0

    def epsilon(self, step: int) -> float:
        """Linear anneal from eps_start to eps_end over eps_decay_steps."""
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def sample_action(self, state: np.ndarray, explore: bool = True, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection.

        :param state: current state as a flat array.
        :param explore: if False, always take the greedy action (test time).
        :param epsilon: exploration probability, used only if explore=True.
        """
        if explore and self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def can_train(self) -> bool:
        return len(self.replay_buffer) >= max(self.min_buffer_size, self.batch_size)

    def train_step(self) -> float:
        """One SGD update of w using eq. (4)/(5): the target y uses w^-."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        q_values = self.q_network(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_q_values

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_calls += 1
        if self._train_calls % self.target_update_freq == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self) -> None:
        """Copy w -> w^- (the target network)."""
        self.target_network.load_state_dict(self.q_network.state_dict())
