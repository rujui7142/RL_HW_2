"""DQN training/evaluation loop for the HW2 Question 2 CartPole tasks.

Protocol (as specified in the assignment):
  - train for a fixed number of environment steps,
  - every `eval_every_episodes` training episodes, run
    `n_eval_episodes` pure-exploitation (explore=False) evaluation
    episodes and record their mean/std return,
  - every episode (training or evaluation) starts from a random
    feasible state.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from agents.dqn import DQNAgent
from envs.cartpole_env import (
    action_to_dict,
    make_cartpole_env,
    obs_to_array,
    reset_to_random_state,
)


@dataclass
class TrainingResult:
    variant: str
    train_episode_returns: List[float] = field(default_factory=list)
    train_episode_end_steps: List[int] = field(default_factory=list)
    eval_checkpoint_episodes: List[int] = field(default_factory=list)
    eval_checkpoint_steps: List[int] = field(default_factory=list)
    eval_means: List[float] = field(default_factory=list)
    eval_stds: List[float] = field(default_factory=list)
    eval_all_returns: List[List[float]] = field(default_factory=list)


def run_episode(env, agent, rng, explore: bool, epsilon: float = 0.0,
                max_steps: int = 1000):
    """Roll out a single episode starting from a random feasible state."""
    obs = reset_to_random_state(env, rng)
    state = obs_to_array(obs)
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = agent.sample_action(state, explore=explore, epsilon=epsilon)
        obs, reward, terminated, truncated, _ = env.step(action_to_dict(action))
        done = terminated or truncated
        total_reward += reward
        state = obs_to_array(obs)
        steps += 1
    return total_reward, steps


def evaluate(env, agent, rng, n_episodes: int) -> List[float]:
    return [run_episode(env, agent, rng, explore=False)[0] for _ in range(n_episodes)]


def train_dqn(
    variant: str,
    total_steps: int = 10_000,
    eval_every_episodes: int = 10,
    n_eval_episodes: int = 10,
    seed: int = 0,
    agent_kwargs: dict = None,
) -> TrainingResult:
    agent_kwargs = dict(agent_kwargs or {})
    agent_kwargs.setdefault("eps_decay_steps", total_steps // 2)

    train_env = make_cartpole_env(variant=variant)
    eval_env = make_cartpole_env(variant=variant)

    train_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1)

    state_dim = len(train_env.observation_space.spaces)
    n_actions = train_env.action_space.spaces["force-side"].n

    agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, seed=seed, **agent_kwargs)

    result = TrainingResult(variant=variant)

    global_step = 0
    episode_idx = 0

    obs = reset_to_random_state(train_env, train_rng)
    state = obs_to_array(obs)
    episode_return = 0.0

    while global_step < total_steps:
        epsilon = agent.epsilon(global_step)
        action = agent.sample_action(state, explore=True, epsilon=epsilon)
        next_obs, reward, terminated, truncated, _ = train_env.step(action_to_dict(action))
        done = terminated or truncated
        next_state = obs_to_array(next_obs)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        global_step += 1

        if agent.can_train():
            agent.train_step()

        if done:
            episode_idx += 1
            result.train_episode_returns.append(episode_return)
            result.train_episode_end_steps.append(global_step)

            if episode_idx % eval_every_episodes == 0:
                eval_returns = evaluate(eval_env, agent, eval_rng, n_eval_episodes)
                result.eval_checkpoint_episodes.append(episode_idx)
                result.eval_checkpoint_steps.append(global_step)
                result.eval_means.append(float(np.mean(eval_returns)))
                result.eval_stds.append(float(np.std(eval_returns)))
                result.eval_all_returns.append(eval_returns)

            obs = reset_to_random_state(train_env, train_rng)
            state = obs_to_array(obs)
            episode_return = 0.0

    train_env.close()
    eval_env.close()

    return result, agent
