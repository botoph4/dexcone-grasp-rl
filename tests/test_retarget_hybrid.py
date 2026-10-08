"""Hybrid retargeting: joint-mapping anchor + pinch refinement."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.angles import HandAngles  # noqa: E402
from p24grasp.teleop.retarget import (  # noqa: E402
    HybridRetargeter,
    P24_LIMITS,
    scaling_q16,
)


def _retargeter() -> HybridRetargeter:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    return HybridRetargeter(
        max_nfev=400,
        lateral_calibration_path=tmp / "lateral.json",
        hand_calibration_path=tmp / "hand_calibration.json",
    )


def _angles(flexion: float = 30.0, abduction: float = 0.0) -> HandAngles:
    return HandAngles(flexion=np.full((5, 3), flexion),
                      abduction=np.full(5, abduction),
                      flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                      presence=1.0)


def _straight_keypoints(reach: float = 0.16) -> np.ndarray:
    """Realistic open right hand facing the camera (fingers +z, thumb -x)."""
    kp = np.zeros((21, 3))
    kp[:, 2] = 0.30
    kp[0] = (0.0, 0.0, 0.30)
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
    return kp


def test_missing_keypoints_holds_the_last_command():
    ret = _retargeter()
    angles = _angles(flexion=60.0)
    q_tracked = ret.retarget(angles, keypoints3d=_straight_keypoints())
    # hand leaves the view: the command must HOLD, not switch to the joint
    # mapping (the optimizer's solution and q0 differ -- the thumb has no
    # prior -- so the old fallback jumped the angles on every hand removal)
    q_lost = ret.retarget(angles, keypoints3d=np.full((21, 3), np.nan))
    np.testing.assert_allclose(q_lost, q_tracked, atol=1e-12)


def test_output_respects_limits_and_mimic():
    ret = _retargeter()
    angles = _angles(flexion=150.0, abduction=60.0)
    q20 = ret.retarget(angles, keypoints3d=_straight_keypoints())
    assert np.all(q20 >= P24_LIMITS[:, 0] - 1e-9)
    assert np.all(q20 <= P24_LIMITS[:, 1] + 1e-9)
    for offset in (4, 8, 12, 16):
        np.testing.assert_allclose(q20[offset + 3], q20[offset + 2])  # DIP == PIP


def test_straight_hand_maps_to_open_hand():
    ret = _retargeter()
    angles = _angles(flexion=0.0)
    for _ in range(5):  # adaptive reach converges
        q20 = ret.retarget(angles, keypoints3d=_straight_keypoints())
    assert np.abs(q20[[2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]).max() < 15.0


def test_solution_stays_near_joint_mapping():
    ret = _retargeter()
    angles = _angles(flexion=60.0)
    q0 = scaling_q16(ret.hand, angles)
    # targets consistent with the joint-mapping pose: the anchor and the
    # tips agree, so the refinement stays put
    targets = ret._tips(q0)  # pylint: disable=protected-access
    q20 = ret.solve(targets, q0)
    q_sol = ret._q20_to_active(q20)  # pylint: disable=protected-access
    assert np.abs(q_sol - q0).max() < np.radians(10.0)


def test_pinch_gap_tracks_reachable_targets():
    ret = _retargeter()
    # a reachable robot pinch: thumb + index flexed, others open
    q_pinch = np.zeros(ret.hand.n_active)
    index_s, _ = ret.hand.chain_slices["index"]
    q_pinch[index_s + 0] = np.radians(70.0)  # index MCP
    q_pinch[index_s + 2] = np.radians(70.0)  # index PIP
    thumb_s, _ = ret.hand.chain_slices["thumb"]
    q_pinch[thumb_s + 0] = np.radians(40.0)  # thumb CMC
    q_pinch[thumb_s + 3] = np.radians(70.0)  # thumb MP
    targets = ret._tips(q_pinch)  # pylint: disable=protected-access
    pinch_ref = np.linalg.norm(targets[4] - targets[0])  # thumb vs index
    q20 = ret.solve(targets, q_pinch)
    q_sol = ret._q20_to_active(q20)  # pylint: disable=protected-access
    tips = ret._tips(np.asarray(q_sol))  # pylint: disable=protected-access
    pinch_sol = np.linalg.norm(tips[4] - tips[0])
    assert abs(pinch_sol - pinch_ref) < 8e-3  # pinch preserved within ~8 mm
    assert np.abs(q_sol - q_pinch).max() < np.radians(10.0)  # anchored


def test_open_hand_lateral_does_not_collapse_middle_fingers():
    # regression: a flat open hand must not push the middle three fingers
    # into each other (wrong per-user lateral calibration did exactly that)
    ret = _retargeter()
    kp = np.zeros((21, 3))
    kp[:, 2] = 0.4
    kp[0] = (-0.02, 0.0, 0.4)
    for base, y in ((5, 0.02), (9, -0.02), (13, -0.06), (17, -0.10)):
        kp[base] = (0.06, y, 0.4)
        kp[base + 1] = (0.10, y, 0.4)
        kp[base + 2] = (0.14, y, 0.4)
        kp[base + 3] = (0.17, y, 0.4)
    kp[1] = (0.03, 0.05, 0.4)
    kp[2] = (0.02, 0.09, 0.4)
    kp[3] = (0.02, 0.13, 0.4)
    kp[4] = (0.02, 0.16, 0.4)
    angles = _angles(flexion=0.0)
    for _ in range(80):  # let the auto-calibration collect its frames
        q20 = ret.retarget(angles, keypoints3d=kp)
    # the four lateral joints (URDF 5, 9, 13, 17) stay near neutral
    np.testing.assert_array_less(np.abs(q20[[5, 9, 13, 17]]), 10.0)


def test_warm_start_recovers_after_pose_change():
    # a stale warm start from a flexed pose must not trap the next solve
    ret = _retargeter()
    q_a = np.zeros(ret.hand.n_active)
    index_s, _ = ret.hand.chain_slices["index"]
    q_a[index_s + 2] = np.radians(70.0)  # index PIP flexed
    targets_a = ret._tips(q_a)  # pylint: disable=protected-access
    q20 = ret.solve(targets_a, q_a)
    assert ret._q20_to_active(q20)[index_s + 2] > np.radians(50.0)  # pylint: disable=protected-access
    # now the hand opens completely: targets are the open-hand tips
    q_open = np.zeros(ret.hand.n_active)
    targets_open = ret._tips(q_open)  # pylint: disable=protected-access
    q20 = ret.solve(targets_open, q_open)
    q_sol = ret._q20_to_active(q20)  # pylint: disable=protected-access
    tips_sol = ret._tips(np.asarray(q_sol))  # pylint: disable=protected-access
    np.testing.assert_allclose(tips_sol, targets_open, atol=5e-3)


def test_lateral_disabled_holds_j2_at_neutral():
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    ret = HybridRetargeter(max_nfev=400,
                           lateral_calibration_path=tmp / "lateral.json",
                           hand_calibration_path=tmp / "hand.json",
                           lateral_enabled=False)
    for _ in range(3):
        q20 = ret.retarget(_angles(flexion=40.0),
                           keypoints3d=_straight_keypoints())
    np.testing.assert_allclose(q20[[5, 9, 13, 17]], 0.0, atol=1e-9)


def test_absence_invalidates_stale_target_holds():
    ret = _retargeter()
    kp = _straight_keypoints()
    kp[4] = kp[5] + (0.005, 0.0, 0.0)  # thumb tip covers the index MCP
    ret.retarget(_angles(flexion=40.0), keypoints3d=kp)
    assert ret._last_targets is not None  # pylint: disable=protected-access
    assert ret._last_pip_targets is not None  # pylint: disable=protected-access
    # hand leaves the view: the hold memory must be dropped, or the next
    # detection would keep pulling the index toward the stale pre-absence
    # point and twist the mapping
    ret.retarget(_angles(flexion=40.0), keypoints3d=np.full((21, 3), np.nan))
    assert ret._last_targets is None  # pylint: disable=protected-access
    assert ret._last_pip_targets is None  # pylint: disable=protected-access


def test_palm_frame_snaps_on_large_reorientation():
    # re-entering rotated ~90 deg must snap the palm frame instead of
    # blending it over ~10 frames (the transitional frame twists the
    # mapped pose)
    ret = _retargeter()
    ret.retarget(_angles(flexion=0.0), keypoints3d=_straight_keypoints())
    old = ret._palm_frame.copy()  # pylint: disable=protected-access
    kp = _straight_keypoints()
    a = np.radians(90.0)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0],
                    [np.sin(a), np.cos(a), 0.0],
                    [0.0, 0.0, 1.0]])
    kp = (rot @ (kp - kp[0]).T).T + kp[0]
    ret.retarget(_angles(flexion=0.0), keypoints3d=kp)
    new = ret._palm_frame  # pylint: disable=protected-access
    angle = np.degrees(np.arccos(np.clip(
        (np.trace(new @ old.T) - 1.0) / 2.0, -1.0, 1.0)))
    assert abs(angle - 90.0) < 5.0


def test_scale_update_rejects_glitched_reach_spikes():
    # a wrist depth-lifted onto the background inflates every tip vector
    # uniformly; absorbing the spike into the monotonic reach max would
    # clip the scale to its lower bound -- a persistent fist
    ret = _retargeter()
    ret._human_reach = 0.16  # pylint: disable=protected-access
    rel = np.full((5, 3), 0.16 / np.sqrt(3.0))
    scale = ret._update_scale(rel)  # pylint: disable=protected-access
    assert scale > 1.0
    glitched = rel * 2.5
    ret._update_scale(glitched)  # pylint: disable=protected-access
    assert ret._human_reach <= 0.16 * 1.0001  # pylint: disable=protected-access
    # the mapping is still the open-hand scale, not the fist-clipped 0.8
    assert ret._update_scale(rel) > 1.0  # pylint: disable=protected-access


def test_glitched_wrist_depth_holds_targets():
    ret = _retargeter()
    kp = _straight_keypoints()
    assert ret.targets_from_keypoints(kp) is not None
    bad = kp.copy()
    bad[0, 2] += 0.35  # wrist depth sampled the background behind the hand
    assert ret.targets_from_keypoints(bad) is None
    assert ret.pip_targets_from_keypoints(bad) is None
