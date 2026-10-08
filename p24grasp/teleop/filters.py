"""Temporal filters for joint-angle streams (all vectorized per-DOF).

One Euro Filter: tracking-state default smoother (Casiez et al., CHI 2012).
  Use f_cmin=3-4 Hz for finger angles, NOT the 1 Hz defaults -- measured
  hand-tracking studies show 1 Hz eats ~half the motion path length.
EMA: degraded-state fallback; skipping an update for invalid samples makes
  the output hold its last valid value (implicit occlusion hold).
KalmanCV: constant-velocity Kalman (per-DOF, scalar measurement).  Provides
  predict-only extrapolation during occlusion holds and innovation gating
  against outlier measurements.
"""
from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """Speed-adaptive low-pass (Casiez's One Euro); operates per-DOF columns.

    Args:
        f_cmin: minimum cutoff frequency (Hz); 3-4 Hz for finger angles,
            NOT the 1 Hz defaults -- measured hand-tracking studies show
            1 Hz eats ~half the motion path length.
        beta: speed coefficient: cutoff grows with the measured derivative.
        d_cutoff: cutoff (Hz) of the derivative low-pass.
    """

    def __init__(self, f_cmin: float = 3.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.f_cmin = f_cmin
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._prev_x = None
        self._prev_dx = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Filter one sample per DOF column.

        Args:
            x: measurement array of shape (dim,) or (..., dim); the last
                axis holds the per-DOF values.
            dt: seconds since the previous call.

        Returns:
            Smoothed array of the same shape; NaN samples hold the
            previous output per-DOF (the filter never outputs NaN).
        """
        x = np.asarray(x, dtype=np.float64)
        if self._prev_x is None or dt <= 0:
            self._prev_x = x.copy()
            self._prev_dx = np.zeros_like(x)
            return x.copy()
        # NaN samples: hold previous output per-DOF (filter never outputs NaN).
        out = self._prev_x.copy()
        valid = np.isfinite(x)
        if valid.any():
            dx = (x - self._prev_x) / dt
            a_d = self._alpha(self.d_cutoff, dt)
            self._prev_dx = np.where(
                valid, a_d * dx + (1.0 - a_d) * self._prev_dx, self._prev_dx)
            cutoff = self.f_cmin + self.beta * np.abs(self._prev_dx)
            a = self._alpha(cutoff, dt)
            out = np.where(valid, a * x + (1.0 - a) * self._prev_x, out)
        self._prev_x = out
        return out

    def reset(self) -> None:
        """Drop history.  NOTE: do not reset on tracking recovery -- that is
        the #1 cause of post-occlusion jumps; let the filter re-converge."""
        self._prev_x = None
        self._prev_dx = None

    def seed(self, x: np.ndarray) -> None:
        """Re-anchor the filter history at ``x`` (occlusion recovery re-entry).

        Args:
            x: array with the same shape as the filtered samples.
        """
        self._prev_x = np.asarray(x, dtype=np.float64).copy()
        self._prev_dx = np.zeros_like(self._prev_x)


class EMA:
    """Exponential moving average with per-DOF invalid-sample hold.

    Args:
        alpha: update weight per call; can be overridden per-sample via
            the ``alpha`` argument of :meth:`__call__`.
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._prev = None

    def __call__(self, x: np.ndarray, valid: np.ndarray | None = None,
                 alpha: np.ndarray | None = None) -> np.ndarray:
        """Filter one sample; invalid DOFs hold their previous output.

        Args:
            x: measurement array (per-DOF along the last axis).
            valid: optional boolean mask; combined with finiteness of x
                (default: finite x is valid).
            alpha: optional per-DOF update weight (default: ``self.alpha``).

        Returns:
            Smoothed array of the same shape; skipping an update for
            invalid samples makes the output hold its last valid value
            (implicit occlusion hold).
        """
        x = np.asarray(x, dtype=np.float64)
        if valid is None:
            valid = np.isfinite(x)
        else:
            valid = np.asarray(valid, dtype=bool) & np.isfinite(x)
        a = np.broadcast_to(alpha if alpha is not None else self.alpha, x.shape)
        if self._prev is None:
            out = np.where(valid, x, np.nan)
            self._prev = out.copy()
            return out
        out = np.where(valid, a * x + (1.0 - a) * self._prev, self._prev)
        self._prev = out
        return out

    def reset(self) -> None:
        """Drop history; the next sample becomes the new anchor."""
        self._prev = None

    def seed(self, x: np.ndarray) -> None:
        """Re-anchor the history at ``x``.

        Args:
            x: array with the same shape as the filtered samples.
        """
        self._prev = np.asarray(x, dtype=np.float64).copy()


class KalmanCV:
    """Scalar constant-velocity Kalman filter, vectorized over DOFs.

    state: (2, dim) = [position; velocity].  ``update`` runs the full
    predict/measure cycle with innovation gating; ``predict`` extrapolates
    without a measurement (occlusion hold).  Measurement model is the
    identity per DOF, so the vectorized update is exact (no batch matrix
    solve involved).
    """

    def __init__(self, dim: int, r_var: float = 2.25,  # (1.5 deg)^2
                 sigma_accel: float = 60.0):  # deg/s^2, ~max accel / 3
        """Per-DOF constant-velocity Kalman filter (vectorized over DOFs).

        Args:
            dim: number of DOFs (independent scalar filters).
            r_var: measurement noise variance per DOF (deg^2).
            sigma_accel: process-noise acceleration (deg/s^2): how far the
                predict-only extrapolation may drift during an occlusion
                hold.
        """
        self.dim = dim
        self.r_var = r_var
        self.sigma_accel = sigma_accel
        self._x = np.zeros((2, dim))
        self._p = np.full((2, 2, dim), 1e2 * np.eye(2)[..., None])
        self._inited = np.zeros(dim, dtype=bool)  # >= 1 measurement seen
        self._vel_inited = np.zeros(dim, dtype=bool)  # >= 2 measurements seen
        self._last_z = np.full(dim, np.nan)

    def _q(self, dt: float) -> np.ndarray:
        s2 = self.sigma_accel ** 2
        return np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * s2

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        return np.array([[1.0, dt], [0.0, 1.0]])

    def predict(self, dt: float) -> np.ndarray:
        """Predict-only step (used during occlusion holds); returns positions.

        Args:
            dt: seconds since the previous update/predict.

        Returns:
            (dim,) predicted positions; covariance grows with dt so the
            uncertainty reflects the extrapolation time.
        """
        f = self._transition(dt)
        self._x = f @ self._x
        q = self._q(dt)
        # P <- F P F' + Q, per-DOF (2, 2, dim) stack
        self._p = np.einsum("ij,jld,kl->ikd", f, self._p, f) + q[..., None]
        return self._x[0].copy()

    def update(self, z: np.ndarray, dt: float, gate: float = 3.0) -> np.ndarray:
        """Measure z (dim,); first sighting initializes position, the second
        initializes velocity from the finite difference (waiting for Q to
        "grow" a velocity takes hundreds of frames); gated outliers are
        skipped (predict-only) instead of corrupting the state.

        Args:
            z: (dim,) measurement; NaN DOFs are skipped (predict-only).
            dt: seconds since the previous update/predict.
            gate: innovation gating threshold in standard deviations
                (default 3.0).

        Returns:
            (dim,) filtered positions after the update.
        """
        z = np.asarray(z, dtype=np.float64)
        valid_z = np.isfinite(z)
        first = valid_z & ~self._inited
        # Must be computed BEFORE _inited |= first: the second measurement
        # (not the first) is the one that yields a finite difference.
        second = valid_z & self._inited & ~self._vel_inited
        if second.any():
            self._x[1, second] = (z[second] - self._last_z[second]) / dt
            self._vel_inited |= second
        seen = first | second
        self._last_z[seen] = z[seen]
        self._x[0, first] = z[first]
        self._inited |= first

        self.predict(dt)
        s = self._p[0, 0] + self.r_var  # (dim,): H P H' + R with H = [1, 0]
        innovation = z - self._x[0]
        accept = valid_z & ~first & (np.abs(innovation) <= gate * np.sqrt(s))
        if accept.any():
            gain = self._p[:, 0, :] / s  # (2, dim): P H' / S
            corr = gain * innovation[None, :] * accept[None, :]
            self._x += corr
            # Joseph-free covariance update: P -= K (H P), H P_j = P_j[0, :]
            self._p -= gain[:, None, :] * self._p[0:1, :, :] * accept[None, None, :]
        return self._x[0].copy()

    def reset(self) -> None:
        """Drop all state (positions, covariance, init flags); the next
        update restarts from the first-sighting initialization."""
        self._x = np.zeros((2, self.dim))
        self._p = np.full((2, 2, self.dim), 1e2 * np.eye(2)[..., None])
        self._inited = np.zeros(self.dim, dtype=bool)
        self._vel_inited = np.zeros(self.dim, dtype=bool)
        self._last_z = np.full(self.dim, np.nan)
