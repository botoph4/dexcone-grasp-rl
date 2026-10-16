"""dex-retargeting adapter: construction + synthetic retarget (skips without
the package, e.g. the minimal test venv)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("dex_retargeting")
pytest.importorskip("pinocchio")

from p24grasp.teleop.retarget import DexRetargetingAdapter, P24_LIMITS  # noqa: E402


def _straight_keypoints() -> np.ndarray:
    """Realistic open right hand facing the camera (fingers +z, thumb -x);
    same geometry as the hybrid tests' fixture (wrist->tip 0.16 m)."""
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


def test_adapter_constructs_and_retargets():
    adapter = DexRetargetingAdapter()
    q20 = adapter.retarget(None, keypoints3d=_straight_keypoints())
    assert q20.shape == (20,)
    assert np.all(np.isfinite(q20))
    assert np.all(q20 >= P24_LIMITS[:, 0] - 1e-6)
    assert np.all(q20 <= P24_LIMITS[:, 1] + 1e-6)
    # straight human hand: the four fingers stay near the open pose (the
    # robot thumb's rest pose has an out-of-plane mounting the flat
    # fixture cannot express, so the thumb is allowed to move)
    assert np.abs(q20[[4, 6, 8, 10, 12, 14, 16, 18]]).max() < 20.0


def test_flexed_hand_drives_the_fingers():
    # regression: the adapter used to pass the RAW (21, 3) keypoints where
    # the optimizer expects the 7 extracted human vectors -- every solve
    # died in a tensor-size mismatch, the package swallowed the RuntimeError
    # and returned the zero pose, and the straight-hand-only test could not
    # tell the difference (zeros are also a valid open hand).  A curled
    # hand must produce real flexion.
    adapter = DexRetargetingAdapter()
    kp = _straight_keypoints()
    # curl the four fingers: the tips pull back toward the palm (short
    # wrist-relative vectors along the fingers)
    for tip, base in ((8, 5), (12, 9), (16, 13), (20, 17)):
        kp[tip] = kp[base] + (0.0, 0.0, 0.04)
        kp[tip - 1] = kp[base] + (0.0, 0.0, 0.05)
    q20 = adapter.retarget(None, keypoints3d=kp)
    assert np.all(np.isfinite(q20))
    # the index and middle fingers (also weighted by the two pinch
    # vectors) must flex substantially -- the old broken path returned
    # zeros here
    assert q20[[4, 6, 8, 10]].min() > 15.0


def test_adapter_holds_on_missing_keypoints():
    adapter = DexRetargetingAdapter()
    out = adapter.retarget(None, keypoints3d=np.full((21, 3), np.nan))
    np.testing.assert_allclose(out, np.zeros(20))
