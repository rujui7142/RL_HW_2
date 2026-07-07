"""Factory for the modified CartPole RDDL environments (HW2, Question 1).

Both reward variants share the same augmented action space:
    0 -> push cart left   (force = -FORCE-MAG)
    1 -> push cart right  (force = +FORCE-MAG)
    2 -> do nothing        (force = 0)

Variants:
    "sparse"    -> unchanged domain: +1 reward per surviving step.
    "quadratic" -> dense reward penalizing the squared state values
                   and control effort (see domain_v2.rddl).
"""
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
from pyRDDLGym.core.env import RDDLEnv

from visualization.cartpole_viz import CartPoleVisualizer

CartPoleRewardVariant = Literal["sparse", "quadratic"]

# canonical order used everywhere a state dict is flattened to an array
STATE_KEYS = ("pos", "ang-pos", "vel", "ang-vel")

_RDDL_DIR = Path(__file__).resolve().parent.parent / "rddl_files" / "cartpole"

_VARIANT_FILES = {
    "sparse": ("domain_v1.rddl", "instance_v1.rddl"),
    "quadratic": ("domain_v2.rddl", "instance_v2.rddl"),
}


def make_cartpole_env(
    variant: CartPoleRewardVariant = "sparse",
    enforce_action_constraints: bool = True,
    vectorized: bool = False,
    **env_kwargs,
) -> RDDLEnv:
    """Build a CartPole RDDLEnv for the requested reward variant.

    :param variant: "sparse" (original +1/step reward) or "quadratic"
        (dense state/control-effort penalty reward).
    :param enforce_action_constraints: raise if an out-of-range action
        is passed to step().
    :param vectorized: forwarded to RDDLEnv (array vs. scalar obs/actions).
    :param env_kwargs: any extra keyword arguments forwarded to RDDLEnv.
    """
    if variant not in _VARIANT_FILES:
        raise ValueError(
            f"Unknown CartPole reward variant '{variant}', "
            f"expected one of {list(_VARIANT_FILES)}"
        )

    domain_file, instance_file = _VARIANT_FILES[variant]
    env = RDDLEnv(
        domain=str(_RDDL_DIR / domain_file),
        instance=str(_RDDL_DIR / instance_file),
        enforce_action_constraints=enforce_action_constraints,
        vectorized=vectorized,
        **env_kwargs,
    )
    env.set_visualizer(CartPoleVisualizer)
    return env


def reset_to_random_state(
    env: RDDLEnv,
    rng: np.random.Generator,
    low: float = -0.05,
    high: float = 0.05,
    seed: int = None,
) -> Dict[str, float]:
    """Reset an episode and overwrite the initial state with a random
    feasible state (small uniform perturbation around the equilibrium,
    the same convention used by the classic OpenAI Gym CartPole).

    The sampled range [-0.05, 0.05] is comfortably inside the domain's
    state-invariant bounds for every state variable (POS-LIMIT=2.4,
    ANG-LIMIT~0.209), so the sampled state is always feasible.

    :param env: environment produced by make_cartpole_env (non-vectorized).
    :param rng: numpy random Generator used to sample the state.
    :param low: lower bound of the uniform sampling range.
    :param high: upper bound of the uniform sampling range.
    :param seed: optional seed forwarded to env.reset().
    :return: the new observation dict.
    """
    env.reset(seed=seed)

    random_state = {key: float(rng.uniform(low, high)) for key in STATE_KEYS}

    sampler = env.sampler
    for key, value in random_state.items():
        sampler.subs[key] = value
        sampler.state[key] = value
    env.state = sampler.state

    return dict(sampler.state)


def obs_to_array(obs: Dict[str, Any]) -> np.ndarray:
    """Flatten an observation dict into a fixed-order float32 array."""
    return np.array([obs[key] for key in STATE_KEYS], dtype=np.float32)


def action_to_dict(action: int) -> Dict[str, int]:
    """Wrap a scalar action index into the RDDLEnv action dict format."""
    return {"force-side": int(action)}
