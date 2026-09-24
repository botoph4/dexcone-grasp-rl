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


def test_tilts_from_quat_roundtrip():
    from p24grasp.viewers.web import tilts_from_quat

    def quat_of(tx, ty):
        ax, ay = tx / 2, ty / 2
        qx = np.array([np.cos(ax), np.sin(ax), 0, 0])
        qy = np.array([np.cos(ay), 0, np.sin(ay), 0])
        w1, x1, y1, z1 = qx
        w2, x2, y2, z2 = qy
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    rng = np.random.default_rng(0)
    for _ in range(20):
        tx = rng.uniform(-np.pi, np.pi)
        ty = rng.uniform(-np.pi, np.pi)
        tx2, ty2 = tilts_from_quat(quat_of(tx, ty))
        # same axis direction is the criterion (rotation about the axis is
        # invisible for a cylinder)
        axis1 = np.array([np.sin(ty), -np.sin(tx) * np.cos(ty), np.cos(tx) * np.cos(ty)])
        axis2 = np.array([np.sin(ty2), -np.sin(tx2) * np.cos(ty2), np.cos(tx2) * np.cos(ty2)])
        assert np.linalg.norm(axis1 - axis2) < 1e-9
