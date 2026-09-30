"""Human-angle -> P24-joint scaling: limits, coupling, monotonicity."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.angles import HandAngles  # noqa: E402
from p24grasp.teleop.retarget import (  # noqa: E402
    P24_JOINT_NAMES,
    P24_LIMITS,
    THUMB_FLEX_SCALE,
    DirectAngleScaling,
)


def _angles(flexion: np.ndarray, abduction: np.ndarray | None = None) -> HandAngles:
    return HandAngles(flexion=flexion, abduction=abduction if abduction is not None
                      else np.zeros(5),
                      flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                      presence=1.0)


def test_joint_names_match_urdf_order():
    assert len(P24_JOINT_NAMES) == 20
    assert P24_JOINT_NAMES[:4] == ("thumb_dh_joint_1", "thumb_dh_joint_2",
                                   "thumb_dh_joint_3", "thumb_dh_joint_4")
    assert P24_JOINT_NAMES[-1] == "little_dh_joint_4"


def test_open_hand_maps_to_zero_command():
    angles = _angles(np.zeros((5, 3)))
    q = DirectAngleScaling().retarget(angles)
    np.testing.assert_allclose(q, 0.0, atol=1e-9)


def test_command_respects_limits():
    angles = _angles(np.full((5, 3), 150.0), abduction=np.full(5, 60.0))
    q = DirectAngleScaling().retarget(angles)
    assert np.all(q >= P24_LIMITS[:, 0] - 1e-9)
    assert np.all(q <= P24_LIMITS[:, 1] + 1e-9)


def test_monotonic_in_pip():
    ret = DirectAngleScaling()
    q_lo = ret.retarget(_angles(np.full((5, 3), 10.0)))
    q_hi = ret.retarget(_angles(np.full((5, 3), 100.0)))
    assert np.all(q_hi >= q_lo - 1e-9)


def test_coupled_mode_slaves_dip_to_pip():
    angles = _angles(np.full((5, 3), 90.0))
    q = DirectAngleScaling(couple_dip_pip=True).retarget(angles)
    for offset in (4, 8, 12, 16):  # index..little DIP = j4 follows PIP = j3
        np.testing.assert_allclose(q[offset + 3], q[offset + 2])


def test_uncoupled_mode_uses_human_dip():
    angles = _angles(np.full((5, 3), 90.0))
    q = DirectAngleScaling(couple_dip_pip=False).retarget(angles)
    dip = q[7]  # index DIP: 90 deg human -> 80 deg robot
    np.testing.assert_allclose(dip, 80.0, atol=0.5)


def test_mcp_always_uses_human_mcp():
    angles = _angles(np.full((5, 3), 90.0))
    q = DirectAngleScaling().retarget(angles)
    np.testing.assert_allclose(q[4], 80.0, atol=0.5)  # human MCP -> robot j1


def test_thumb_baseline_mapping():
    # BEHAVIOR-style: three unsigned bends -> j1/j2/j4; j3 neutral
    flexion = np.zeros((5, 3))
    flexion[0, 0] = 30.0  # thumb CMC bend
    flexion[0, 1] = 50.0  # thumb MP bend
    flexion[0, 2] = 60.0  # thumb IP bend
    q = DirectAngleScaling().retarget(_angles(flexion))
    assert q[0] == 30.0 * THUMB_FLEX_SCALE[0]
    assert q[1] == 50.0
    assert q[2] == 0.0  # CMC abduction neutral in the baseline
    assert q[3] == 60.0


def test_coupled_distal_is_half_total_minus_zero_compensation():
    flexion = np.zeros((5, 3))
    flexion[1, 1] = 90.0  # index PIP
    flexion[1, 2] = 40.0  # index DIP
    q = DirectAngleScaling().retarget(_angles(flexion))
    expected = 0.5 * (90.0 + 40.0 - 10.0337)
    np.testing.assert_allclose(q[6], expected, atol=0.5)  # index PIP
    np.testing.assert_allclose(q[7], expected, atol=0.5)  # index DIP = PIP


def test_nan_angles_map_to_zero():
    angles = _angles(np.full((5, 3), np.nan))
    q = DirectAngleScaling().retarget(angles)
    np.testing.assert_allclose(q, 0.0, atol=1e-9)
