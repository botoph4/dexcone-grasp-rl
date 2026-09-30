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
    kp = np.full((21, 3), np.nan)
    kp[0] = (0.0, 0.0, 0.30)
    for tip in (4, 8, 12, 16, 20):
        kp[tip] = kp[0] + (0.0, 0.0, 0.16)
    return kp


def test_adapter_constructs_and_retargets():
    adapter = DexRetargetingAdapter()
    kp = _straight_keypoints()
    q20 = adapter.retarget(None, keypoints3d=kp)
    assert q20.shape == (20,)
    assert np.all(np.isfinite(q20))
    assert np.all(q20 >= P24_LIMITS[:, 0] - 1e-6)
    assert np.all(q20 <= P24_LIMITS[:, 1] + 1e-6)
    # straight human hand: the vector optimizer should stay near the open pose
    assert np.abs(q20).max() < 40.0


def test_adapter_holds_on_missing_keypoints():
    adapter = DexRetargetingAdapter()
    out = adapter.retarget(None, keypoints3d=np.full((21, 3), np.nan))
    np.testing.assert_allclose(out, np.zeros(20))
