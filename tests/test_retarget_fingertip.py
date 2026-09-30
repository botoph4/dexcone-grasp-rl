"""Fingertip-position retargeting: recovery, pinch, mimic coupling, hold."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.paths import urdf_path  # noqa: E402
from p24grasp.teleop.angles import HandAngles  # noqa: E402
from p24grasp.teleop.retarget import FingertipRetargeter  # noqa: E402


def _retargeter() -> FingertipRetargeter:
    # exact-recovery tests: no smoothing bias, generous evaluation budget
    return FingertipRetargeter(smooth_weight=0.0, max_nfev=400)


def _tips_at(retargeter: FingertipRetargeter, q16: np.ndarray) -> np.ndarray:
    """Robot fingertip positions (5, 3) at the 16-DOF configuration."""
    tips = []
    for name in retargeter._chain_names:  # pylint: disable=protected-access
        start, end = retargeter.hand.chain_slices[name]
        tips.append(retargeter.hand.tip_point(
            retargeter._chain(name), q16[start:end]))  # pylint: disable=protected-access
    return np.stack(tips)


def test_solve_recovers_open_hand():
    ret = _retargeter()
    q_zero = np.zeros(ret.hand.n_active)
    targets = _tips_at(ret, q_zero)
    q20 = ret.solve(targets)
    assert q20.shape == (20,)
    np.testing.assert_allclose(q20, 0.0, atol=1.0)


def test_solve_matches_flexed_targets_and_pinch():
    ret = _retargeter()
    q_ref = np.zeros(ret.hand.n_active)
    # flex index (MCP, PIP) and thumb MP; chain order: index, little, middle, ring, thumb
    index_s, _ = ret.hand.chain_slices["index"]
    q_ref[index_s + 0] = 0.8  # index MCP
    q_ref[index_s + 2] = 1.0  # index PIP
    thumb_s, _ = ret.hand.chain_slices["thumb"]
    q_ref[thumb_s + 3] = 0.9  # thumb MP
    targets = _tips_at(ret, q_ref)
    q20 = ret.solve(targets)
    q_sol = ret._q20_to_active(q20)  # pylint: disable=protected-access
    tips_sol = _tips_at(ret, np.asarray(q_sol))
    np.testing.assert_allclose(tips_sol, targets, atol=3e-3)  # ~3 mm


def test_pinch_distance_preserved():
    ret = _retargeter()
    q_ref = np.zeros(ret.hand.n_active)
    index_s, _ = ret.hand.chain_slices["index"]
    q_ref[index_s + 2] = 1.0  # index PIP
    thumb_s, _ = ret.hand.chain_slices["thumb"]
    q_ref[thumb_s + 3] = 0.9  # thumb MP
    targets = _tips_at(ret, q_ref)
    pinch_ref = np.linalg.norm(targets[0] - targets[4])  # index(0) vs thumb(4)
    q20 = ret.solve(targets)
    q_sol = ret._q20_to_active(q20)  # pylint: disable=protected-access
    tips_sol = _tips_at(ret, np.asarray(q_sol))
    pinch_sol = np.linalg.norm(tips_sol[0] - tips_sol[4])
    assert abs(pinch_sol - pinch_ref) < 4e-3  # pinch gap ~4 mm


def test_output_couples_dip_to_pip():
    ret = _retargeter()
    q_ref = np.zeros(ret.hand.n_active)
    index_s, _ = ret.hand.chain_slices["index"]
    q_ref[index_s + 2] = 0.7  # index PIP
    targets = _tips_at(ret, q_ref)
    q20 = ret.solve(targets)
    for offset in (4, 8, 12, 16):  # index..little URDF starts
        np.testing.assert_allclose(q20[offset + 3], q20[offset + 2])  # DIP == PIP


def test_hold_on_missing_keypoints():
    ret = _retargeter()
    empty = HandAngles.zeros()
    out = ret.retarget(empty, keypoints3d=np.full((21, 3), np.nan))
    np.testing.assert_allclose(out, np.zeros(20))
    # after one solve, a missing-keypoint frame holds the last command
    q_ref = np.zeros(ret.hand.n_active)
    index_s, _ = ret.hand.chain_slices["index"]
    q_ref[index_s + 2] = 0.7
    ret.solve(_tips_at(ret, q_ref))
    held = ret.retarget(empty, keypoints3d=np.full((21, 3), np.nan))
    np.testing.assert_allclose(held, ret._q20_prev)  # pylint: disable=protected-access


def _realistic_open_keypoints(roll_deg: float = 0.0) -> np.ndarray:
    """Open right hand facing the camera: fingers away (+z), thumb to the
    image left (-x), middle finger along the palm ray."""
    kp = np.zeros((21, 3))
    kp[:, 2] = 0.30
    kp[0] = (0.0, 0.0, 0.30)  # wrist
    # middle finger straight ahead (+z); index/ring/little fan in -y/+y? no:
    # palm-plane fan in the camera x-y plane, thumb toward -x
    kp[9] = (0.0, 0.0, 0.35)
    kp[10] = (0.0, 0.0, 0.39)
    kp[11] = (0.0, 0.0, 0.43)
    kp[12] = (0.0, 0.0, 0.46)
    kp[5] = (-0.01, 0.0, 0.35)
    kp[6] = (-0.02, 0.0, 0.39)
    kp[7] = (-0.03, 0.0, 0.43)
    kp[8] = (-0.04, 0.0, 0.46)
    kp[13] = (0.01, 0.0, 0.35)
    kp[14] = (0.02, 0.0, 0.39)
    kp[15] = (0.03, 0.0, 0.43)
    kp[16] = (0.04, 0.0, 0.46)
    kp[17] = (0.02, 0.0, 0.35)
    kp[18] = (0.04, 0.0, 0.39)
    kp[19] = (0.06, 0.0, 0.43)
    kp[20] = (0.08, 0.0, 0.46)
    kp[1] = (-0.03, 0.0, 0.33)
    kp[2] = (-0.06, 0.0, 0.36)
    kp[3] = (-0.08, 0.0, 0.39)
    kp[4] = (-0.10, 0.0, 0.41)
    if roll_deg:
        angle = np.radians(roll_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
        kp = (rotation @ kp.T).T
    return kp


def test_targets_straight_hand_point_along_palm_x():
    ret = FingertipRetargeter(scale=1.0, smooth_weight=0.0, max_nfev=400)
    targets = ret.targets_from_keypoints(_realistic_open_keypoints())
    assert targets is not None
    # finger direction maps to palm +x: every target's x dominates
    assert np.all(targets[:, 0] > 0.0)
    assert np.all(np.abs(targets[:, 0]) > np.abs(targets[:, 1]))


def test_targets_are_invariant_to_camera_roll():
    ret = FingertipRetargeter(scale=1.0, smooth_weight=0.0, max_nfev=400)
    baseline = ret.targets_from_keypoints(_realistic_open_keypoints(roll_deg=0.0))
    # a fresh retargeter (no frame smoothing history) with a rolled camera
    ret2 = FingertipRetargeter(scale=1.0, smooth_weight=0.0, max_nfev=400)
    rolled = ret2.targets_from_keypoints(_realistic_open_keypoints(roll_deg=40.0))
    np.testing.assert_allclose(rolled, baseline, atol=0.02)


def test_thumb_lands_on_positive_palm_y():
    ret = FingertipRetargeter(scale=1.0, smooth_weight=0.0, max_nfev=400)
    targets = ret.targets_from_keypoints(_realistic_open_keypoints())
    # thumb chain is last; robot thumb lives at +y
    assert targets[4, 1] > 0.0


def test_mimic_chain_has_three_active_joints():
    hand = HandModel(urdf_path())
    index = next(c for c in hand.chains if c.name == "index")
    assert len(index.active_joint_names) == 3  # MCP, abd, PIP (DIP mimicked)
    thumb = next(c for c in hand.chains if c.name == "thumb")
    assert len(thumb.active_joint_names) == 4


def test_straight_human_hand_maps_to_open_robot_hand():
    ret = _retargeter()
    kp = np.full((21, 3), np.nan)
    kp[0] = (0.0, 0.0, 0.30)  # wrist
    for tip in (4, 8, 12, 16, 20):
        kp[tip] = kp[0] + (0.0, 0.0, 0.16)  # straight fingers, 0.16 m reach
    # several frames so the adaptive reach estimate converges
    empty = HandAngles.zeros()
    q20 = None
    for _ in range(5):
        q20 = ret.retarget(empty, keypoints3d=kp)
    # straight hand -> open robot hand: no large flexions
    assert np.abs(q20[[2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]).max() < 15.0
    # adaptive scale puts the targets at the robot's open-hand reach
    assert abs(q20[6]) < 5.0  # index PIP nearly straight
