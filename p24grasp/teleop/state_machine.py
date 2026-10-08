"""Occlusion state machine (docs/HAND_TELEOP_RESEARCH.md section 5.2).

States:
    TRACKING  normal: One Euro filter + ROM clamp + slew limit; KF keeps absorbing
    DEGRADED  hand presence in [t_lo, t_hi) or partially visible joints:
              visibility-weighted EMA + DIP~=0.7*PIP kinematic fixup (soft hold
              per hidden DOF)
    HOLD      presence < t_lo for N_enter frames: CV-Kalman predict-only
              extrapolation (hold_fast s), then FREEZE at the last pose --
              relaxing toward the rest pose made every hand removal look like
              the mapping restarted on re-entry, so the absence is held
              instead; after hold_max s -> LOST (status downgrade only)
    LOST      keeps holding the frozen pose; needs N_exit consecutive good
              frames to recover
    RECOVERY  (transient, first TRACKING frame after HOLD/LOST) slew-limited
              catch-up or exponential blend -- filters are NOT reset (resets
              cause the post-occlusion jump this machine exists to prevent)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from p24grasp.teleop.angles import (
    HandAngles,
    clamp_rom,
    dip_coupling_fixup,
)
from p24grasp.teleop.filters import EMA, KalmanCV, OneEuroFilter

N_ANGLES = 5 * 3 + 5  # 15 flexion + 5 abduction


class State(Enum):
    TRACKING = "tracking"
    DEGRADED = "degraded"
    HOLD = "hold"
    LOST = "lost"


@dataclass
class OcclusionParams:
    """Starting values per the research doc; tune on real data."""

    t_hi: float = 0.7  # presence hysteresis: enter/exit HOLD asymmetric
    t_lo: float = 0.5
    n_enter: int = 3  # consecutive hard-bad frames to enter HOLD
    n_exit: int = 5  # consecutive good frames to leave HOLD/LOST
    hold_fast: float = 0.5  # s of KF extrapolation
    hold_max: float = 3.0  # s after which HOLD is reported as LOST
    slew_max: float = 300.0  # deg/s recovery catch-up limit
    blend_tau: float = 0.1  # s recovery exponential blend
    min_vis: float = 0.5  # per-DOF visibility threshold
    min_dof_frac: float = 0.6  # fraction of DOFs that must be visible to count
    # a frame as "fully tracked" (fingertips near the frame edge are often
    # hidden; the per-DOF hold handles them, full visibility must not be
    # required or the machine would never leave HOLD/LOST)
    ema_alpha: float = 0.7
    one_euro_f_cmin: float = 3.0
    one_euro_beta: float = 0.05
    kf_r_var: float = 2.25
    kf_sigma_accel: float = 60.0
    dip_coupling_k: float = 0.7


@dataclass
class FilteredAngles:
    """Smoothed human angles + machine state (inputs to the retargeter)."""

    angles: HandAngles
    state: State
    tracking_ok: bool  # False only in LOST
    dof_frac: float  # fraction of DOFs considered visible this frame


class OcclusionStateMachine:
    """Consumes raw per-frame :class:`HandAngles`, emits smooth held-safe ones."""

    def __init__(self, params: OcclusionParams | None = None):
        self.params = params or OcclusionParams()
        self.state = State.TRACKING
        self._good = 0
        self._bad = 0
        self._t_hold = 0.0
        self._prev = HandAngles.zeros()
        self._one_euro = OneEuroFilter(self.params.one_euro_f_cmin,
                                       self.params.one_euro_beta)
        self._ema = EMA(self.params.ema_alpha)
        self._ema.seed(np.zeros(N_ANGLES))  # start from the neutral command
        self._kf = KalmanCV(dim=N_ANGLES, r_var=self.params.kf_r_var,
                            sigma_accel=self.params.kf_sigma_accel)

    @staticmethod
    def _flatten(angles: HandAngles) -> np.ndarray:
        return np.concatenate([angles.flexion.ravel(), angles.abduction])

    @staticmethod
    def _unflatten(vec: np.ndarray, angles: HandAngles) -> HandAngles:
        return HandAngles(flexion=vec[:15].reshape(5, 3).copy(),
                          abduction=vec[15:20].copy(),
                          flexion_vis=angles.flexion_vis.copy(),
                          abduction_vis=angles.abduction_vis.copy(),
                          presence=angles.presence)

    def _dof_valid(self, angles: HandAngles) -> np.ndarray:
        """(20,) boolean: is this DOF observed well enough to trust this frame?

        Requires MediaPipe landmark visibility AND a finite raw angle (depth
        lifting may fail under the keypoint even when MediaPipe is confident).
        """
        vis = np.concatenate([angles.flexion_vis.ravel(), angles.abduction_vis])
        raw = np.concatenate([angles.flexion.ravel(), angles.abduction])
        return (vis >= self.params.min_vis) & np.isfinite(raw)

    def _slew_limit(self, prev: HandAngles, target: HandAngles, dt: float) -> HandAngles:
        max_step = self.params.slew_max * dt
        flat = np.clip(self._flatten(target) - self._flatten(prev),
                       -max_step, max_step) + self._flatten(prev)
        return self._unflatten(flat, target)

    def update(self, angles: HandAngles, dt: float) -> FilteredAngles:
        p = self.params
        angles = clamp_rom(dip_coupling_fixup(angles, p.dip_coupling_k))
        dof_frac = self._dof_valid(angles).mean()
        conf_ok = angles.presence >= p.t_hi and dof_frac >= p.min_dof_frac
        hard_bad = angles.presence < p.t_lo

        # Hysteresis counters: only consecutive good/hard-bad frames count.
        if conf_ok:
            self._good += 1
            self._bad = 0
        elif hard_bad:
            self._bad += 1
            self._good = 0
        else:  # degraded zone: neither full nor lost
            self._good = 0

        transitioned = False
        if self.state in (State.TRACKING, State.DEGRADED):
            if self._bad >= p.n_enter:
                self.state = State.HOLD
                self._t_hold = 0.0
            else:
                self.state = State.TRACKING if conf_ok else State.DEGRADED
        else:  # HOLD / LOST
            if self._good >= p.n_exit:
                self.state = State.TRACKING
                self._t_hold = 0.0
                transitioned = True

        if transitioned:
            out = self._recover(angles, dt)
        else:
            out = self._step(angles, dt)
        self._prev = out
        return FilteredAngles(angles=out, state=self.state,
                              tracking_ok=self.state != State.LOST,
                              dof_frac=dof_frac)

    def _step(self, angles: HandAngles, dt: float) -> HandAngles:
        p = self.params
        flat = self._flatten(angles)
        valid = self._dof_valid(angles) & np.isfinite(flat)
        if self.state == State.TRACKING:
            self._kf.update(flat, dt)  # keep absorbing normal measurements
            out = self._unflatten(self._one_euro(flat, dt), angles)
            # Keep the EMA warm at the current command so a transition into
            # DEGRADED holds from here, not from a stale seed.
            self._ema.seed(self._flatten(out))
            return self._slew_limit(self._prev, out, dt)
        if self.state == State.DEGRADED:
            # Visibility-weighted EMA: hidden DOFs drift toward holding.
            # Below t_lo the detection is untrustworthy wholesale: hold all.
            weights = np.concatenate([angles.flexion_vis.ravel(),
                                      angles.abduction_vis])
            presence_weight = 1.0 if angles.presence >= p.t_lo else 0.0
            alpha = np.where(valid, presence_weight * p.ema_alpha * np.clip(weights, 0.0, 1.0),
                             0.0)
            out = self._unflatten(self._ema(flat, valid=valid, alpha=alpha), angles)
            return self._slew_limit(self._prev, out, dt)
        if self.state == State.HOLD:
            self._t_hold += dt
            if self._t_hold <= p.hold_fast:
                out = self._unflatten(self._kf.predict(dt), angles)
            else:
                # freeze at the last commanded pose: a hand leaving the view
                # must not reset the mapping (the old relaxation toward the
                # rest pose made every re-entry look like a fresh start);
                # _recover() re-syncs smoothly when the hand comes back
                if self._t_hold > p.hold_max:
                    self.state = State.LOST
                out = self._unflatten(self._flatten(self._prev), angles)
            return out
        # LOST: keep holding the frozen pose (never command the rest pose --
        # re-opening the hand on every absence is exactly the "mapping
        # restarts" behaviour); recovery is handled by the counters in
        # update().
        return self._unflatten(self._flatten(self._prev), angles)

    def _recover(self, angles: HandAngles, dt: float) -> HandAngles:
        """First TRACKING frame after HOLD/LOST: smooth re-entry, no filter reset."""
        p = self.params
        flat = self._flatten(angles)
        target = self._unflatten(self._one_euro(flat, dt), angles)
        delta = np.abs(self._flatten(target) - self._flatten(self._prev))
        if np.nanmax(delta, initial=0.0) > 30.0:
            # Gesture changed while occluded: fast but rate-limited catch-up.
            out = self._slew_limit(self._prev, target, dt)
        else:
            blend = 1.0 - np.exp(-dt / p.blend_tau)
            out = self._unflatten(self._flatten(self._prev) +
                                  blend * (self._flatten(target) - self._flatten(self._prev)),
                                  angles)
        # Re-seed the smoothers at the blended output so the next frames
        # continue from here instead of from pre-occlusion stale state.
        self._one_euro.seed(self._flatten(out))
        self._ema.seed(self._flatten(out))
        return self._slew_limit(self._prev, out, dt)
