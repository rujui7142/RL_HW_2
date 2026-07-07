# RL HW2 — CartPole (RDDL) + DQN

BGU Planning and Reinforcement Learning course, Homework 2.

## Question 1 — Environment

Two CartPole RDDL environments, based on the pyRDDLGym-project `rddlrepository`
CartPole_Discrete domain, extended with a third "do nothing" action
(`0` = left, `1` = right, `2` = no-op):

- **sparse** (`src/rddl_files/cartpole/domain_v1.rddl`) — original +1 reward
  per surviving step.
- **quadratic** (`src/rddl_files/cartpole/domain_v2.rddl`) — dense reward
  penalizing the squared state values and control effort.

Built via `make_cartpole_env(variant="sparse" | "quadratic")` in
[`src/envs/cartpole_env.py`](src/envs/cartpole_env.py), with a pygame-based
visualizer attached for rendering rollouts.

## Question 2 — DQN

A DQN agent (2-layer MLP, replay buffer, target network, epsilon-greedy
exploration) in [`src/agents/dqn.py`](src/agents/dqn.py), trained via
[`src/training/train_dqn.py`](src/training/train_dqn.py). Every episode
(training and evaluation) starts from a random feasible state.

Written answers (math derivations + results discussion) are kept locally as
`HW2_Question2_Written.docx` (not tracked in this repo).

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

## Usage

```bash
# smoke-test both Question 1 environments
python scripts/test_env.py

# train DQN (Question 2) on one reward variant
python scripts/run_dqn.py --variant sparse --steps 10000
python scripts/run_dqn.py --variant quadratic --steps 10000
```

Results (reward curves, raw returns) are saved under `outputs/`.
