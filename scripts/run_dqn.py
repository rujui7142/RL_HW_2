"""Run DQN training (HW2, Question 2) for one CartPole reward variant.

Usage:
    python scripts/run_dqn.py --variant sparse --steps 10000
    python scripts/run_dqn.py --variant quadratic --steps 10000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from training.train_dqn import train_dqn  # noqa: E402
from visualization.plotting import plot_reward_curve  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["sparse", "quadratic"], required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = OUT_DIR / f"dqn_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training DQN on '{args.variant}' variant for {args.steps} steps...")
    t0 = time.time()
    result, agent = train_dqn(variant=args.variant, total_steps=args.steps, seed=args.seed)
    elapsed = time.time() - t0

    n_episodes = len(result.train_episode_returns)
    n_checkpoints = len(result.eval_checkpoint_episodes)
    n_eval_episodes = sum(len(r) for r in result.eval_all_returns)
    print(f"Done in {elapsed:.1f}s: {n_episodes} training episodes, "
          f"{n_checkpoints} evaluation checkpoints, "
          f"{n_eval_episodes} total evaluation episodes.")

    np.savez(
        out_dir / "results.npz",
        variant=result.variant,
        train_episode_returns=np.array(result.train_episode_returns),
        train_episode_end_steps=np.array(result.train_episode_end_steps),
        eval_checkpoint_episodes=np.array(result.eval_checkpoint_episodes),
        eval_checkpoint_steps=np.array(result.eval_checkpoint_steps),
        eval_means=np.array(result.eval_means),
        eval_stds=np.array(result.eval_stds),
    )

    plot_reward_curve(
        x=result.eval_checkpoint_episodes,
        means=result.eval_means,
        stds=result.eval_stds,
        title=f"DQN on CartPole ({args.variant} reward) — evaluation return",
        save_path=out_dir / "reward_curve.png",
    )
    print(f"Saved results and plot to {out_dir}")


if __name__ == "__main__":
    main()
