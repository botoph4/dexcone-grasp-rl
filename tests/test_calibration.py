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
    INWARD_LIMITS_DEG,
    OUTWARD_LIMITS_DEG,
    LateralEstimator,
    wrap_angle,
)


def test_two_point_lateral_maps_endpoints_to_limits():
    # together at raw -0.35 rad, open spread at +0.15 rad (radial fingers);
    # ring/little have the reversed raw direction (together > open)
    together = np.array([-0.35, -0.30, 0.40, 0.80])
    open_raw = np.array([0.15, 0.20, 0.20, 0.40])
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=open_raw,
        together_lateral=together,
        fist_flexion=np.full((5, 3), 90.0),
        reach_m=0.16,
    ).lateral
    inward = np.deg2rad(np.asarray(INWARD_LIMITS_DEG, dtype=np.float64))
    outward = np.deg2rad(np.asarray(OUTWARD_LIMITS_DEG, dtype=np.float64))
    for finger in range(4):
        mapped_together = cal.gains[finger] * wrap_angle(
            np.array([together[finger]]) - np.array([cal.offsets_rad[finger]]))[0]
        mapped_open = cal.gains[finger] * wrap_angle(
            np.array([open_raw[finger]]) - np.array([cal.offsets_rad[finger]]))[0]
        np.testing.assert_allclose(mapped_together, inward[finger], atol=1e-6)
        np.testing.assert_allclose(mapped_open, outward[finger], atol=1e-6)


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


def test_calibrated_lateral_starts_at_the_together_mapping():
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
    # before any confident observation the output is the calibrated neutral
    # (the two-point fit maps the together gesture to the inward limits)
    qpos, _ = estimator.update(np.full((21, 3), np.nan))
    inward = np.deg2rad(np.asarray(INWARD_LIMITS_DEG, dtype=np.float64))
    np.testing.assert_allclose(qpos, inward, atol=1e-9)


def test_ulnar_fingers_reversed_raw_direction():
    # ring/little raw angles are LARGER for together than for open (the
    # derotation sign convention flips on the ulnar side); the fit must
    # still send together -> inward and open -> outward physically
    together = np.array([-0.9, -0.6, 0.5, 0.8])
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=np.array([-0.6, -0.2, 0.3, 0.4]),
        together_lateral=together,
        fist_flexion=np.full((5, 3), 90.0),
        reach_m=0.16,
    ).lateral
    inward = np.deg2rad(np.asarray(INWARD_LIMITS_DEG, dtype=np.float64))
    outward = np.deg2rad(np.asarray(OUTWARD_LIMITS_DEG, dtype=np.float64))
    for finger in range(4):
        mapped_together = cal.gains[finger] * wrap_angle(
            np.array([together[finger]]) - np.array([cal.offsets_rad[finger]]))[0]
        mapped_open = cal.gains[finger] * wrap_angle(
            np.array([-0.6 if finger == 0 else -0.2 if finger == 1 else 0.3
                      if finger == 2 else 0.4]) - np.array([cal.offsets_rad[finger]]))[0]
        np.testing.assert_allclose(mapped_together, inward[finger], atol=1e-6)
        np.testing.assert_allclose(mapped_open, outward[finger], atol=1e-6)


def test_reach_nan_falls_back_to_default():
    cal = compute_calibration(
        open_flexion=np.zeros((5, 3)),
        open_lateral=np.zeros(4),
        together_lateral=np.zeros(4),
        fist_flexion=np.full((5, 3), 90.0),
        reach_m=np.nan,
    )
    assert cal.reach_m == 0.16
    assert np.isfinite(cal.reach_m)
