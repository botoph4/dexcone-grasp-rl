"""FK consistency between HandModel, MuJoCo and (optionally) PyRoki."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
from p24grasp.kinematics.ik import chain_limits, fist_center, solve_angles
from p24grasp.model.urdf import HandModel
from p24grasp.paths import ensure_hand_xml, urdf_path


@pytest.fixture(scope="module")
def hand():
    return HandModel(str(urdf_path()))


def test_fk_matches_mujoco(hand):
    model = mujoco.MjModel.from_xml_path(str(ensure_hand_xml()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for chain in hand.chains:
        q = np.zeros(len(chain_limits(hand, chain)[0]))
        tip = hand.tip_point(chain, q)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chain.tip_link)
        err = np.linalg.norm(tip - data.xpos[bid]) * 1000
        assert err < 0.01, f"{chain.name} FK mismatch: {err:.4f} mm"


def test_solve_angles_returns_valid(hand):
    q = solve_angles(hand, 0.032, 0.85)
    assert q.shape == (16,)
    assert np.all(np.isfinite(q))
    # within joint limits
    lo = np.concatenate([chain_limits(hand, c)[0] for c in hand.chains])
    hi = np.concatenate([chain_limits(hand, c)[1] for c in hand.chains])
    assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)


def test_fist_center(hand):
    c = fist_center(hand)
    assert c.shape == (3,)
    assert np.all(np.isfinite(c))
