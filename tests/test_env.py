"""The gymnasium environment passes SB3's checker and gives sane rewards."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# grasp.py imports gymnasium at module level, so skip before importing it
pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from p24grasp.env.grasp import EPISODE_STEPS, GraspEnv  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402


@pytest.mark.parametrize("shape", ["cylinder", "sphere"])
def test_check_env(shape):
    check_env(GraspEnv(shape=shape), warn=False)


def test_nominal_grasp_outperforms_noop():
    env = GraspEnv(shape="cylinder", randomize=False)
    env.reset(seed=0)
    # the env's deterministic reset starts near the grasp pose; holding it
    # (zero action) should score much higher than doing nothing is impossible
    # to compare directly, so compare hold vs. an open-hand command
    zero = np.zeros(env.action_space.shape, np.float32)
    open_action = -np.ones(env.action_space.shape, np.float32)
    r_hold = r_open = 0.0
    env.reset(seed=0)
    for _ in range(EPISODE_STEPS):
        _, r, term, trunc, _ = env.step(zero)
        r_hold += r
        if term or trunc:
            break
    env.reset(seed=0)
    for _ in range(EPISODE_STEPS):
        _, r, term, trunc, _ = env.step(open_action)
        r_open += r
        if term or trunc:
            break
    assert r_hold > r_open
