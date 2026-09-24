"""SO(3) rotation-spring helper used by the web viewer's drag-to-rotate."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.viewers.web import quat_angle, rotation_torque


def test_quat_angle():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_x90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    assert abs(quat_angle(q_id, q_x90) - np.pi / 2) < 1e-9
    assert quat_angle(q_id, q_id) == 0.0
    # sign-symmetric: quaternions q and -q are the same rotation
    assert quat_angle(q_x90, -q_x90) == 0.0


def test_torque_direction_and_clamp():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_x90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    tau = rotation_torque(q_id, q_x90)          # rotate id -> +x 90 deg
    assert tau[0] > 0 and abs(tau[1]) < 1e-12 and abs(tau[2]) < 1e-12
    tau_back = rotation_torque(q_x90, q_id)
    assert tau_back[0] < 0
    assert np.allclose(rotation_torque(q_id, q_id), 0.0)
    clamped = rotation_torque(q_id, q_x90, k=1000.0, max_tau=0.03)
    assert np.all(np.abs(clamped) <= 0.03 + 1e-12)
    assert np.allclose(clamped[0], 0.03)


def test_damping_opposes_angular_velocity():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_x90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    # small k so the undamped torque stays below the clamp
    base = rotation_torque(q_id, q_x90, k=0.01)
    damped = rotation_torque(q_id, q_x90, k=0.01, omega=np.array([1.0, 0.0, 0.0]))
    assert base[0] < 0.03
    assert damped[0] < base[0]
