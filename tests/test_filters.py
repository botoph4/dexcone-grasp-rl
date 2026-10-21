"""Filter behavior: smoothing, holding, extrapolation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.filters import EMA, KalmanCV, OneEuroFilter  # noqa: E402


def test_one_euro_converges_to_step():
    filt = OneEuroFilter(f_cmin=3.0, beta=0.05)
    out = [filt(np.array([0.0]), 1 / 60)[0] for _ in range(120)]
    # constant input after warmup: filter should converge to it
    out = [filt(np.array([30.0]), 1 / 60)[0] for _ in range(120)]
    assert abs(out[-1] - 30.0) < 0.5


def test_one_euro_holds_nan():
    filt = OneEuroFilter(f_cmin=3.0, beta=0.05)
    filt(np.array([10.0, 20.0]), 1 / 60)
    out = filt(np.array([np.nan, 25.0]), 1 / 60)
    assert out[0] == 10.0  # NaN DOF holds
    assert out[1] != 25.0  # valid DOF is smoothed, not snapped


def test_one_euro_delay_small_at_speed():
    filt = OneEuroFilter(f_cmin=3.0, beta=0.5)
    # ramp 0 -> 100 deg over 0.5 s: lag should be far below 100 ms of motion
    n = 30
    out = [filt(np.array([float(i) / n * 100.0]), 1 / 60)[0] for i in range(n)]
    lag_deg = 100.0 - out[-1]  # at the ramp end, output trails the input
    assert lag_deg < 10.0


def test_ema_holds_on_invalid():
    ema = EMA(alpha=0.7)
    ema(np.array([5.0, 5.0]))
    out = ema(np.array([40.0, np.nan]), valid=np.array([True, False]))
    assert out[1] == 5.0  # invalid DOF holds exactly
    assert 5.0 < out[0] < 40.0  # valid DOF moves


def test_kalman_predict_extrapolates_velocity():
    kf = KalmanCV(dim=1, r_var=2.25)
    dt = 1 / 60
    kf.update(np.array([10.0]), dt)
    kf.update(np.array([11.0]), dt)
    kf.update(np.array([12.0]), dt)  # steady 60 deg/s
    pred = kf.predict(dt)
    assert 12.5 < pred[0] < 13.5  # ~13.0


def test_kalman_gates_outliers():
    kf = KalmanCV(dim=1, r_var=2.25)
    dt = 1 / 60
    for _ in range(30):
        kf.update(np.array([10.0]), dt)
    out = kf.update(np.array([80.0]), dt)  # 70 deg outlier in one frame
    assert abs(out[0] - 10.0) < 5.0
