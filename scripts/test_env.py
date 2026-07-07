"""Smoke test for the Question 1 CartPole environments.

Runs a few random-action steps on both reward variants, prints the
action space / a transition, and saves one rendered frame per variant
so the visualizer can be checked without a display.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from envs import make_cartpole_env  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_variant(variant: str) -> None:
    env = make_cartpole_env(variant=variant)
    print(f"\n=== variant: {variant} ===")
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)

    obs, _ = env.reset(seed=0)
    print("initial obs:", obs)

    total_reward = 0.0
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"t={t} action={action} reward={reward:.4f} obs={obs}")
        if terminated or truncated:
            obs, _ = env.reset()

    OUT_DIR.mkdir(exist_ok=True)
    image = env.render(to_display=False)
    image.save(OUT_DIR / f"cartpole_{variant}.png")
    print(f"saved render to {OUT_DIR / f'cartpole_{variant}.png'}")
    print(f"total reward over 5 random steps: {total_reward:.4f}")

    env.close()


if __name__ == "__main__":
    run_variant("sparse")
    run_variant("quadratic")
