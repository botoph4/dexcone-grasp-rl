"""ManoHandFitter: FK sanity, pose recovery, jitter smoothing, fallback,
bone-length calibration, reset semantics."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.mano_fit import (  # noqa: E402
    N_PARAMS,
    REST_OFFSETS,
    _FLEX_SLOT,
    ManoHandFitter,
)

FX, FY, CX, CY = 600.0, 600.0, 320.0, 240.0


def _truth_pose() -> np.ndarray:
    """A realistic wrist-mount pose: fingers +z (away from the camera),
    index flexed 45 deg, thumb flexed 30 deg."""
    truth = np.zeros(N_PARAMS)
    truth[0:3] = np.array([0.0, np.radians(-80.0), 0.0])
    truth[3:6] = np.array([0.02, -0.01, 0.30])
    truth[_FLEX_SLOT[5]] = np.radians(45.0)
    truth[_FLEX_SLOT[6]] = np.radians(45.0)
    truth[_FLEX_SLOT[1]] = np.radians(30.0)
    return truth


def _project(fitter: ManoHandFitter, truth: np.ndarray):
    """The synthetic 2D/depth measurements of the truth pose."""
    pos = fitter._fk(truth)  # pylint: disable=protected-access
    z = pos[:, 2]
    u = FX * pos[:, 0] / z + CX
    v = FY * pos[:, 1] / z + CY
    return pos, u, v


def test_open_hand_fk_matches_rest_offsets():
    fitter = ManoHandFitter(calib_frames=0)
    pos = fitter._fk(np.zeros(N_PARAMS))  # pylint: disable=protected-access
    np.testing.assert_allclose(pos[0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(pos[1], REST_OFFSETS[1], atol=1e-12)
    np.testing.assert_allclose(pos[5], REST_OFFSETS[5], atol=1e-12)
    # the middle tip = the full palm->tip chain at the open pose
    expected = REST_OFFSETS[9] + REST_OFFSETS[10] + REST_OFFSETS[11] \
        + REST_OFFSETS[12]
    np.testing.assert_allclose(pos[12], expected, atol=1e-12)


def test_cold_start_recovers_a_synthetic_pose():
    fitter = ManoHandFitter(calib_frames=0)
    truth = _truth_pose()
    pos, u, v = _project(fitter, truth)
    lifted = pos.copy()
    lifted[:, 2] += np.random.default_rng(0).normal(0, 0.003, 21)
    out = fitter.fit(u, v, lifted, fx=FX, fy=FY, cx=CX, cy=CY, ts=0.1)
    assert np.linalg.norm(out - pos, axis=1).mean() < 5e-3
    params = fitter._prev  # pylint: disable=protected-access
    assert abs(params[_FLEX_SLOT[5]] - np.radians(45.0)) < np.radians(5.0)
    assert abs(params[_FLEX_SLOT[1]] - np.radians(30.0)) < np.radians(5.0)


def test_warm_started_fit_smooths_frame_rate_jitter():
    # the raw per-point depth noise moves every keypoint independently
    # each frame; the bone-rigid model cannot follow it, so the fitted
    # trajectory must move far less than the raw lifting
    fitter = ManoHandFitter(calib_frames=0)
    truth = _truth_pose()
    pos, u, v = _project(fitter, truth)
    rng = np.random.default_rng(1)
    raw_moves, fit_moves = [], []
    prev_lifted = prev_fit = None
    for i in range(40):
        lifted = pos.copy()
        lifted[:, 2] += rng.normal(0, 0.003, 21)
        uj = u + rng.normal(0, 1.0, 21)
        vj = v + rng.normal(0, 1.0, 21)
        out = fitter.fit(uj, vj, lifted, fx=FX, fy=FY, cx=CX, cy=CY, ts=0.1 + 0.033 * i)
        if prev_lifted is not None:
            raw_moves.append(np.linalg.norm(
                lifted - prev_lifted, axis=1).mean())
            fit_moves.append(np.linalg.norm(out - prev_fit, axis=1).mean())
        prev_lifted, prev_fit = lifted, out
    raw_mean = float(np.mean(raw_moves))
    fit_mean = float(np.mean(fit_moves))
    assert fit_mean < 0.6 * raw_mean  # the model absorbs most of the noise
    assert fit_mean < 3.5e-3  # (the raw lifting moves ~4.2 mm/frame)


def test_garbage_keypoints_fall_back_to_the_raw_lifting():
    fitter = ManoHandFitter(calib_frames=0)
    lifted = np.zeros((21, 3))
    lifted[:, 2] = 0.3
    out = fitter.fit(np.full(21, 0.0), np.full(21, 0.0), lifted,
                     fx=FX, fy=FY, cx=CX, cy=CY, ts=0.1)
    assert out is lifted  # rejected: the raw keypoints pass through
    assert fitter._prev is None  # pylint: disable=protected-access


def test_bone_lengths_auto_calibrate_to_the_user_hand():
    # a synthetic hand whose bones are 1.2x the population lengths
    scaled = ManoHandFitter(calib_frames=0)
    scaled._offsets = REST_OFFSETS * 1.2  # pylint: disable=protected-access
    truth = _truth_pose()
    pos, _, _ = _project(scaled, truth)
    fitter = ManoHandFitter(calib_frames=30)
    rng = np.random.default_rng(2)
    out = None
    for i in range(30):
        lifted = pos.copy()
        lifted[:, 2] += rng.normal(0, 0.002, 21)
        out = fitter.fit(np.zeros(21), np.zeros(21), lifted,
                         fx=FX, fy=FY, cx=CX, cy=CY, ts=0.1 + 0.033 * i)
    assert out is not None
    reach = np.linalg.norm(fitter._offsets[9] + fitter._offsets[10]  # pylint: disable=protected-access
                           + fitter._offsets[11] + fitter._offsets[12])
    expected = 1.2 * np.linalg.norm(
        REST_OFFSETS[9] + REST_OFFSETS[10] + REST_OFFSETS[11] + REST_OFFSETS[12])
    assert abs(reach - expected) < 0.02 * expected


def test_jacobian_matches_finite_differences():
    # the analytic jacobian replaces scipy's 2-point scheme (27 residual
    # evals per jacobian -- far too slow per frame); pin it against
    # central differences on a random pose
    fitter = ManoHandFitter(calib_frames=0)
    rng = np.random.default_rng(3)
    params = np.zeros(N_PARAMS)
    params[0:3] = rng.normal(0, 0.5, 3)
    params[3:6] = np.array([0.02, -0.01, 0.30])
    params[6:] = rng.normal(0, 0.4, 20)
    u = rng.uniform(100, 500, 21)
    v = rng.uniform(50, 400, 21)
    zref = np.full(21, 0.3)
    kwargs = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "dt": 0.033}
    jac = fitter._jacobian(params, u, v, zref, **kwargs)  # pylint: disable=protected-access
    step = 1e-7
    fd = np.zeros_like(jac)
    for k in range(N_PARAMS):
        pp = params.copy()
        pp[k] += step
        pm = params.copy()
        pm[k] -= step
        fd[:, k] = (fitter._residuals(pp, u, v, zref, **kwargs)  # pylint: disable=protected-access
                    - fitter._residuals(pm, u, v, zref, **kwargs)) / (2 * step)  # pylint: disable=protected-access
    assert np.abs(jac - fd).max() < 1e-6


def test_reset_pose_keeps_the_calibrated_lengths():
    fitter = ManoHandFitter(calib_frames=0)
    truth = _truth_pose()
    pos, u, v = _project(fitter, truth)
    fitter.fit(u, v, pos, fx=FX, fy=FY, cx=CX, cy=CY, ts=0.1)
    offsets = fitter._offsets.copy()  # pylint: disable=protected-access
    assert fitter._prev is not None  # pylint: disable=protected-access
    fitter.reset_pose()
    assert fitter._prev is None  # pylint: disable=protected-access
    np.testing.assert_array_equal(fitter._offsets, offsets)  # pylint: disable=protected-access
