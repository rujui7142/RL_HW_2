"""Reward-curve plotting for DQN training runs (HW2, Question 2)."""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_reward_curve(
    x: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    title: str,
    xlabel: str = "Training episode",
    ylabel: str = "Evaluation episode return",
    save_path: Path = None,
):
    """Plot mean evaluation return vs. iteration with a +-1 std envelope."""
    x = np.asarray(x)
    means = np.asarray(means)
    stds = np.asarray(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, means, color="tab:blue", label="mean evaluation return")
    ax.fill_between(x, means - stds, means + stds, color="tab:blue", alpha=0.25,
                     label="+-1 std envelope")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig
