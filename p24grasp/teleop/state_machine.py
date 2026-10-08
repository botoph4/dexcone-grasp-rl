"""Occlusion state machine (docs/HAND_TELEOP_RESEARCH.md section 5.2).

States:
    TRACKING  normal: One Euro filter + ROM clamp + slew limit
    DEGRADED  hand presence in [t_lo, t_hi) or partially visible joints:
              visibility-weighted EMA + DIP~=0.7*PIP kinematic fixup (soft hold
              per hidden DOF)
    HOLD      presence < t_lo for N_enter frames: the hand left the view --
              the command returns to the REST pose (open palm, rate-limited)
              and the filtering history is reset (see :meth:`reset`), so the
              next detection re-runs the first-startup logic; after hold_max
              s -> LOST (status downgrade only)
    LOST      keeps commanding the rest pose; needs N_exit consecutive
              presence-good frames to recover (a re-entering pinch hides
              ~half the DOFs, so the exit must NOT require full DOF
              visibility)
    RECOVERY  (transient, first TRACKING frame after HOLD/LOST) slew-limited
              catch-up or exponential blend from the rest pose -- with the
              filters reset at HOLD entry this reads exactly like a fresh
              startup
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from p24grasp.teleop.angles import (
    HandAngles,
    clamp_rom,
    dip_coupling_fixup,
)
from p24grasp.teleop.filters import EMA, OneEuroFilter

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
    n_exit: int = 5  # consecutive presence-good frames to leave HOLD/LOST
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
    dip_coupling_k: float = 0.7
    rest_pose: HandAngles = field(default_factory=HandAngles.zeros)


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

    def reset(self) -> None:
        """Reset the filtering history as if the pipeline just started.

        Called when the hand leaves the view: the command then returns to
        the rest pose (open palm) and the next detection re-runs the
        first-startup logic.  The current state, hysteresis counters, and
        the previous output are kept, so the return to the rest pose
        stays rate-limited instead of snapping.
        """
        self._one_euro.reset()
        self._ema.seed(np.zeros(N_ANGLES))

    @staticmethod
    def _flatten(angles: HandAngles) -> np.ndarray:
        """Pack :class:`HandAngles` into a (20,) vector [15 flexion, 5 abduction].

        Args:
            angles: the :class:`HandAngles` to pack.

        Returns:
            (20,) float array.
        """
        return np.concatenate([angles.flexion.ravel(), angles.abduction])

    @staticmethod
    def _unflatten(vec: np.ndarray, angles: HandAngles) -> HandAngles:
        """Unpack a (20,) vector back into :class:`HandAngles`.

        Args:
            vec: (20,) array [15 flexion, 5 abduction].
            angles: source of the visibility/presence fields (the angle
                values come from ``vec``).

        Returns:
            :class:`HandAngles` with ``vec``'s values and ``angles``'
            visibility/presence.
        """
        return HandAngles(flexion=vec[:15].reshape(5, 3).copy(),
                          abduction=vec[15:20].copy(),
                          flexion_vis=angles.flexion_vis.copy(),
                          abduction_vis=angles.abduction_vis.copy(),
                          presence=angles.presence)

    def _dof_valid(self, angles: HandAngles) -> np.ndarray:
        """(20,) boolean: is this DOF observed well enough to trust this frame?

        Requires MediaPipe landmark visibility AND a finite raw angle (depth
        lifting may fail under the keypoint even when MediaPipe is confident).

        Args:
            angles: raw (clamped, DIP-fixed) :class:`HandAngles` of this
                frame.

        Returns:
            (20,) boolean mask, one entry per DOF.
        """
        vis = np.concatenate([angles.flexion_vis.ravel(), angles.abduction_vis])
        raw = np.concatenate([angles.flexion.ravel(), angles.abduction])
        return (vis >= self.params.min_vis) & np.isfinite(raw)

    def _slew_limit(self, prev: HandAngles, target: HandAngles, dt: float) -> HandAngles:
        """Limit the per-frame command change to ``slew_max`` deg/s.

        Args:
            prev: the previous output :class:`HandAngles`.
            target: the desired :class:`HandAngles` this frame.
            dt: seconds since the previous frame.

        Returns:
            ``target`` moved toward at most ``slew_max * dt`` degrees per
            DOF (a rate limit, not a filter -- steady commands pass
            through unchanged).
        """
        max_step = self.params.slew_max * dt
        flat = np.clip(self._flatten(target) - self._flatten(prev),
                       -max_step, max_step) + self._flatten(prev)
        return self._unflatten(flat, target)

    def update(self, angles: HandAngles, dt: float) -> FilteredAngles:
        """Consume one raw-angle frame; run the state transitions and the
        per-state filtering, and emit the smooth held-safe output.

        Args:
            angles: raw :class:`HandAngles` of this frame (may contain
                NaN).
            dt: seconds since the previous frame.

        Returns:
            :class:`FilteredAngles` with the smoothed angles, the current
            :class:`State`, ``tracking_ok`` (False only in LOST), and the
            visible-DOF fraction of this frame.
        """
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
        elif self.state in (State.HOLD, State.LOST) and angles.presence >= p.t_hi:
            # Exit-lenient: a re-entering pinch naturally hides ~half the
            # DOFs (the thumb covers the index/middle bases and the
            # visibility rule damps them), so the full dof_frac condition
            # can never fire and the machine would stay stuck in HOLD
            # forever.  Presence-only frames count toward the exit; the
            # per-DOF holds cover the hidden joints.
            self._good += 1
            self._bad = 0
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
        """Per-state filtering for non-transition frames (see the module
        docstring for the per-state behaviour).

        Args:
            angles: raw (clamped, DIP-fixed) :class:`HandAngles`.
            dt: seconds since the previous frame.

        Returns:
            The smoothed output :class:`HandAngles` for this frame.
        """
        p = self.params
        flat = self._flatten(angles)
        valid = self._dof_valid(angles) & np.isfinite(flat)
        if self.state == State.TRACKING:
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
            if self._t_hold > p.hold_max:
                self.state = State.LOST
            # the hand left the view: return to the open palm, rate-limited
            # (no snap); the pipeline resets the filtering/mapping state at
            # HOLD entry so the next detection re-runs the first-startup
            # logic
            return self._slew_limit(self._prev, p.rest_pose, dt)
        # LOST: keep commanding the rest pose; recovery is handled by the
        # counters in update().
        return self._slew_limit(self._prev, p.rest_pose, dt)

    def _recover(self, angles: HandAngles, dt: float) -> HandAngles:
        """First TRACKING frame after HOLD/LOST: smooth re-entry, no filter reset.

        Args:
            angles: raw (clamped, DIP-fixed) :class:`HandAngles` of the
                re-detected hand.
            dt: seconds since the previous frame.

        Returns:
            The blended/slew-limited output; the smoothers are re-seeded
            at it so the next frames continue from here instead of from
            pre-occlusion stale state.
        """
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
