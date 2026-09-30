"""Guided gesture calibration: two-point lateral fit, flexion gains, persistence."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.calibration import (  # noqa: E402
    compute_calibration,
    load_hand_calibration,
    save_hand_calibration,
)
from p24grasp.teleop.lateral import (  # noqa: E402
    DEFAULT_LIMITS_DEG,
    LateralEstimator,
    wrap_angle,
)


def test_two_point_lateral_maps_endpoints_to_limits():
    # together at raw -0.35 rad, open spread at +0.15 rad (some finger)
    together = np.array([-0.35, -0.30, 0.20, 0.40])
    open_raw = np.array([0.15, 0.20, 0.70, 0.85])
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=open_raw,
        together_lateral=together,
        fist_flexion=np.full((5, 3), 90.0),
        reach_m=0.16,
    ).lateral
    limits = np.deg2rad(np.asarray(DEFAULT_LIMITS_DEG, dtype=np.float64))
    lo, hi = limits[:, 0], limits[:, 1]
    for finger in range(4):
        mapped_lo = cal.gains[finger] * wrap_angle(
            np.array([together[finger]]) - np.array([cal.offsets_rad[finger]]))[0]
        mapped_hi = cal.gains[finger] * wrap_angle(
            np.array([open_raw[finger]]) - np.array([cal.offsets_rad[finger]]))[0]
        np.testing.assert_allclose(mapped_lo, lo[finger], atol=1e-6)
        np.testing.assert_allclose(mapped_hi, hi[finger], atol=1e-6)


def test_flexion_gains_map_measured_range_to_robot_rom():
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=np.zeros(4),
        together_lateral=np.zeros(4),
        fist_flexion=np.array([[50.0, 70.0, 60.0],   # thumb
                               [60.0, 80.0, 60.0],
                               [60.0, 80.0, 60.0],
                               [60.0, 80.0, 60.0],
                               [60.0, 80.0, 60.0]]),
        reach_m=0.16,
    )
    # MCP: measured 60 deg -> robot 80 deg
    np.testing.assert_allclose(cal.flexion_gains[1, 0], 80.0 / 60.0, atol=1e-6)
    # thumb j1: 50 -> 67
    np.testing.assert_allclose(cal.flexion_gains[0, 0], 67.0 / 50.0, atol=1e-6)
    # distal: total 140 measured -> 0.5*(g*140 - 10.03) = 80 -> g = 1.2145
    expected = (2 * 80.0 + 10.0337) / 140.0
    np.testing.assert_allclose(cal.distal_total_gain, expected, atol=1e-6)


def test_calibration_roundtrip(tmp_path):
    path = tmp_path / "hand_calibration.json"
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=np.ones(4) * 0.3,
        together_lateral=np.zeros(4),
        fist_flexion=np.full((5, 3), 80.0),
        reach_m=0.17,
    )
    save_hand_calibration(cal, path)
    loaded = load_hand_calibration(path)
    assert loaded is not None
    np.testing.assert_allclose(loaded.lateral.offsets_deg, cal.lateral.offsets_deg)
    np.testing.assert_allclose(loaded.flexion_gains, cal.flexion_gains)
    assert loaded.reach_m == 0.17


def test_missing_calibration_loads_none(tmp_path):
    assert load_hand_calibration(tmp_path / "nope.json") is None


def test_estimator_uses_calibrated_lateral():
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=np.array([0.2, 0.2, 0.2, 0.2]),
        together_lateral=np.array([-0.2, -0.2, -0.2, -0.2]),
        fist_flexion=np.full((5, 3), 90.0),
        reach_m=0.16,
    )
    estimator = LateralEstimator(
        calibration=cal.lateral,
        filter_alpha=1.0,
        calibration_path="/nonexistent/cal.json",
    )
    qpos, _ = estimator.update(np.zeros((21, 3)))
    assert np.all(np.isfinite(qpos))
    limits = np.deg2rad(np.asarray(DEFAULT_LIMITS_DEG, dtype=np.float64))
    assert np.all(qpos >= limits[:, 0] - 1e-9)
    assert np.all(qpos <= limits[:, 1] + 1e-9)
