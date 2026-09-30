"""Joint-angle extraction from synthetic 3D keypoints."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.angles import angles_from_keypoints, dip_coupling_fixup  # noqa: E402
from p24grasp.teleop.detector import HandDetection  # noqa: E402


def _straight_hand() -> np.ndarray:
    """Open hand in the x-y plane (z = camera depth), fingers along +x.

    MediaPipe ids: 0=wrist; 1-4 thumb CMC/MP/IP/TIP; 5-8 index; 9-12 middle;
    13-16 ring; 17-20 little.  The wrist is collinear with the index ray so a
    straight index reads exactly 0 flexion at all three joints.
    """
    kp = np.zeros((21, 3))
    kp[:, 2] = 0.3
    kp[0] = (0.01, 0.01, 0.3)  # wrist, on the index ray
    for base, y in ((5, 0.01), (9, -0.02), (13, -0.05), (17, -0.08)):
        kp[base] = (0.05, y, 0.3)
        kp[base + 1] = (0.09, y, 0.3)
        kp[base + 2] = (0.13, y, 0.3)
        kp[base + 3] = (0.16, y, 0.3)
    # thumb: CMC near wrist, ray pointing along -y (abducted)
    kp[1] = (0.03, 0.03, 0.3)
    kp[2] = (0.02, 0.07, 0.3)
    kp[3] = (0.02, 0.11, 0.3)
    kp[4] = (0.02, 0.14, 0.3)
    return kp


def _det(kp: np.ndarray) -> HandDetection:
    return HandDetection(keypoints3d=kp, visibility=np.ones(21), presence=1.0)


def test_straight_finger_gives_zero_flexion():
    angles = angles_from_keypoints(_det(_straight_hand()))
    np.testing.assert_allclose(angles.flexion[1], [0, 0, 0], atol=0.5)


def test_pip_bend_gives_90_degrees():
    kp = _straight_hand()
    # bend index PIP (7): distal part points down (-y), right-angle bend
    kp[7] = kp[6] + (0.0, 0.04, 0.0)
    kp[8] = kp[6] + (0.0, 0.07, 0.0)
    angles = angles_from_keypoints(_det(kp))
    np.testing.assert_allclose(angles.flexion[1, 1], 90.0, atol=0.5)
    # proximal flexion unaffected
    np.testing.assert_allclose(angles.flexion[1, 0], 0.0, atol=0.5)


def test_middle_finger_abduction_is_zero_by_definition():
    angles = angles_from_keypoints(_det(_straight_hand()))
    assert angles.abduction[2] == 0.0


def test_abduction_sign_positive_away_from_middle():
    kp = _straight_hand()
    # index finger spreads in +y (away from the middle finger at -y)
    kp[6] = kp[5] + (0.04, 0.06, 0.0)
    kp[7] = kp[5] + (0.08, 0.10, 0.0)
    kp[8] = kp[5] + (0.11, 0.13, 0.0)
    angles = angles_from_keypoints(_det(kp))
    assert angles.abduction[1] > 0.0


def test_missing_keypoints_give_nan_and_zero_vis():
    kp = _straight_hand()
    kp[6] = np.nan  # index PIP hidden: poisons its MCP and PIP flexion
    angles = angles_from_keypoints(_det(kp))
    assert np.isnan(angles.flexion[1, 1])
    assert np.isnan(angles.flexion[1, 0])


def test_dip_coupling_fixup_replaces_hidden_dip():
    angles = angles_from_keypoints(_det(_straight_hand()))
    angles.flexion[1, 1] = 60.0  # PIP
    angles.flexion[1, 2] = np.nan  # DIP hidden
    angles.flexion_vis[1, 2] = 0.0
    out = dip_coupling_fixup(angles)
    np.testing.assert_allclose(out.flexion[1, 2], 0.7 * 60.0, atol=0.5)


def test_dip_coupling_fixup_leaves_visible_dip():
    angles = angles_from_keypoints(_det(_straight_hand()))
    angles.flexion[1, 1] = 60.0
    angles.flexion[1, 2] = 20.0
    angles.flexion_vis[1, 2] = 1.0
    out = dip_coupling_fixup(angles)
    np.testing.assert_allclose(out.flexion[1, 2], 20.0, atol=0.5)


def test_thumb_flexion_detected():
    kp = _straight_hand()
    # curl thumb distal part back along +x: flexion at MP (joint 2)
    kp[3] = kp[2] + (0.05, 0.0, 0.0)
    kp[4] = kp[2] + (0.09, 0.0, 0.0)
    angles = angles_from_keypoints(_det(kp))
    assert angles.flexion[0, 1] > 60.0


def test_visibility_is_min_of_contributing_keypoints():
    kp = _straight_hand()
    det = _det(kp)
    det.visibility[7] = 0.2  # index DIP barely visible
    angles = angles_from_keypoints(det)
    assert angles.flexion_vis[1, 2] == 0.2  # DIP + PIP use kp7
    assert angles.flexion_vis[1, 1] == 0.2
    assert angles.flexion_vis[1, 0] == 1.0  # MCP uses wrist/MCP/PIP only


def test_thumb_over_index_damps_index_visibility():
    kp = _straight_hand()
    kp[4] = kp[5] + (0.01, 0.0, 0.0)  # thumb tip right on the index MCP
    angles = angles_from_keypoints(_det(kp))
    assert angles.flexion_vis[1, 0] <= 0.3
    assert angles.flexion_vis[1, 2] <= 0.3


def test_thumb_away_leaves_visibility_intact():
    angles = angles_from_keypoints(_det(_straight_hand()))
    assert angles.flexion_vis[1, 0] == 1.0
