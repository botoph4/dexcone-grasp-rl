"""Lateral (MCP ab/adduction) estimator: palm normal, calibration, gating."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.lateral import (  # noqa: E402
    DEFAULT_LIMITS_DEG,
    LateralCalibration,
    LateralEstimator,
    calibrate_lateral,
    fit_palm_normal,
)


def _flat_hand() -> np.ndarray:
    """Open hand in the x-y plane (camera z = 0.4), fingers along +x.

    MediaPipe ids as in tests/test_angles.py; finger rays fan slightly in y.
    """
    kp = np.zeros((21, 3))
    kp[:, 2] = 0.4
    kp[0] = (0.01, 0.01, 0.4)
    for base, y in ((5, 0.01), (9, -0.02), (13, -0.05), (17, -0.08)):
        kp[base] = (0.05, y, 0.4)
        kp[base + 1] = (0.09, y, 0.4)
        kp[base + 2] = (0.13, y, 0.4)
        kp[base + 3] = (0.16, y, 0.4)
    return kp


def _zero_calibration() -> LateralCalibration:
    return LateralCalibration((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0),
                              DEFAULT_LIMITS_DEG)


def test_fit_palm_normal_points_toward_camera():
    kp = _flat_hand()
    normal = fit_palm_normal(kp)
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)


def test_flat_hand_lateral_measures_fan_geometry():
    # realistic fan: each proximal phalanx is collinear with ITS metacarpal
    # reference (palm center -> MCP), so the measured lateral ~= 0 while the
    # in-plane confidence stays high
    kp = _flat_hand()
    palm_center = np.mean(kp[list((0, 5, 9, 13, 17))], axis=0)
    for base, pip in ((5, 6), (9, 10), (13, 14), (17, 18)):
        reference = kp[base] - palm_center
        reference /= np.linalg.norm(reference)
        kp[pip] = kp[base] + reference * 0.04
        kp[pip + 1] = kp[base] + reference * 0.08
        kp[pip + 2] = kp[base] + reference * 0.11
    estimator = LateralEstimator(_zero_calibration(), filter_alpha=1.0)
    raw, confidence, _ = estimator.measure(kp)
    assert np.all(confidence > 0.95)
    assert np.all(np.abs(raw) < np.radians(2.0))


def test_estimator_output_respects_limits():
    estimator = LateralEstimator(filter_alpha=1.0)
    for _ in range(5):
        qpos, _ = estimator.update(_flat_hand())
    limits = np.deg2rad(np.asarray(DEFAULT_LIMITS_DEG, dtype=np.float64))
    assert np.all(qpos >= limits[:, 0] - 1e-9)
    assert np.all(qpos <= limits[:, 1] + 1e-9)


def test_fist_freezes_lateral():
    # proximal phalanges bent 90 deg out of the palm plane: zero in-plane
    # fraction -> zero confidence -> the estimator freezes its value
    kp = _flat_hand()
    for base in (5, 9, 13, 17):
        kp[base + 1] = kp[base] + (0.0, 0.0, -0.04)  # PIP straight down (-z)
        kp[base + 2] = kp[base] + (0.0, 0.0, -0.08)
    estimator = LateralEstimator(_zero_calibration(), filter_alpha=1.0)
    estimator.update(_flat_hand())
    before = estimator._value.copy()  # pylint: disable=protected-access
    _, confidence, _ = estimator.measure(kp)
    assert np.all(confidence < 0.1)
    estimator.update(kp)
    after = estimator._value  # pylint: disable=protected-access
    np.testing.assert_allclose(after, before)  # frozen


def test_calibrate_lateral_fits_offsets_and_gains():
    frames = [_flat_hand() for _ in range(10)]
    calibration = calibrate_lateral(frames, target_half_range_deg=12.0)
    assert len(calibration.offsets_deg) == 4
    assert np.all(np.isfinite(calibration.offsets_deg))
    assert np.all(np.asarray(calibration.gains) >= 0.2)
    assert np.all(np.asarray(calibration.gains) <= 1.0)


def test_derotation_exact_under_flexion():
    # a rigid phalanx flexed out of plane keeps the same lateral angle under
    # derotation (orthogonal projection would shrink the in-plane norm)
    normal = np.asarray([0.0, 0.0, 1.0])
    reference = np.asarray([1.0, 0.0, 0.0])
    lateral = np.radians(12.0)
    flexion = np.radians(60.0)
    # phalanx = cos(lateral)*r + sin(lateral)*axis with axis = n x r = +y;
    # flexion rotates the r-component toward the palm (-n), leaving the
    # axis component untouched
    cos_l, sin_l = np.cos(lateral), np.sin(lateral)
    cos_f, sin_f = np.cos(flexion), np.sin(flexion)
    flex_axis = np.asarray([0.0, 1.0, 0.0])
    phalanx = cos_l * (reference * cos_f + np.asarray([0.0, 0.0, -1.0]) * sin_f) \
        + sin_l * flex_axis
    from p24grasp.teleop.lateral import _derotation_angle  # noqa: E402

    measured = _derotation_angle(reference, phalanx, normal)
    np.testing.assert_allclose(measured, lateral, atol=1e-9)


def test_auto_calibration_learns_neutral_spread(tmp_path):
    estimator = LateralEstimator(
        LateralCalibration((-19.0, -13.0, 0.5, 7.3), (1.0, 1.0, 1.0, 0.7),
                           DEFAULT_LIMITS_DEG),
        filter_alpha=1.0, auto_calibrate=True, collect_frames=20,
        calibration_path=tmp_path / "lateral.json")
    kp = _flat_hand()
    palm_center = np.mean(kp[list((0, 5, 9, 13, 17))], axis=0)
    for base, pip in ((5, 6), (9, 10), (13, 14), (17, 18)):
        reference = kp[base] - palm_center
        reference /= np.linalg.norm(reference)
        kp[pip] = kp[base] + reference * 0.04
        kp[pip + 1] = kp[base] + reference * 0.08
        kp[pip + 2] = kp[base] + reference * 0.11
    # collection phase: neutral output even with the wrong prior calibration
    for _ in range(10):
        qpos, diag = estimator.update(kp)
        assert np.all(qpos == 0.0)
        assert diag["lateral_auto_calibrating"]
    # finish collection (needs 20 total; 10 collected above)
    for _ in range(11):
        qpos, diag = estimator.update(kp)
    assert not diag["lateral_auto_calibrating"]
    qpos, _ = estimator.update(kp)
    assert np.all(np.abs(qpos) < np.radians(5.0))


def _tilt_hand(angle_deg: float) -> np.ndarray:
    """Flat hand tilted about the camera x axis (palm no longer faces the
    camera head-on; the raw normal's +z flip would fire)."""
    kp = _flat_hand()
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_a, -sin_a],
                         [0.0, sin_a, cos_a]])
    return (rotation @ kp.T).T


def test_normal_continuity_across_tilt():
    reference = fit_palm_normal(_flat_hand())
    tilted = fit_palm_normal(_tilt_hand(120.0), reference=reference)
    # the tilted raw normal points away from the camera; continuity with the
    # reference keeps the same hemisphere (no sign flip -> no lateral mirror)
    assert np.dot(tilted, reference) > 0.0


def test_lateral_recovers_after_tilt_reentry():
    # the real regression: hand leaves tilted and returns flat; the output
    # must come back to the pre-tilt value (no stuck mirror)
    estimator = LateralEstimator(_zero_calibration(), filter_alpha=1.0)
    for _ in range(3):
        before, _ = estimator.update(_flat_hand())
    for _ in range(10):
        estimator.update(_tilt_hand(120.0))
    for _ in range(5):
        after, _ = estimator.update(_flat_hand())
    np.testing.assert_allclose(after, before, atol=np.radians(3.0))


def test_calibration_persists_across_instances(tmp_path):
    path = tmp_path / "lateral.json"
    first = LateralEstimator(auto_calibrate=True, collect_frames=10,
                             calibration_path=path)
    for _ in range(15):
        first.update(_flat_hand())
    assert path.exists()
    second = LateralEstimator(auto_calibrate=True, collect_frames=10,
                              calibration_path=path)
    qpos, diag = second.update(_flat_hand())
    assert not diag["lateral_auto_calibrating"]  # loaded, collection skipped
    assert np.all(np.isfinite(qpos))
