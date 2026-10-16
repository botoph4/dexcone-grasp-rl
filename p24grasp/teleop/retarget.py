"""Human angles -> P24 joint commands.

P24 structure (from p24grasp/assets/p24_hand_right.urdf): 5 fingers x 4
revolute joints (DH).  IMPORTANT: the four fingers' DIP (joint 4) carries a
``<mimic joint=..._dh_joint_3 multiplier=1>`` -- the hardware couples DIP 1:1
to PIP, so the hand has 16 independent DOFs, not 20.  The MCP joints are
independent.  ``couple_dip_pip=True`` (faithful to the URDF) slaves robot
DIP to robot PIP; the MuJoCo build in ``build/hand.xml`` does NOT reproduce
the mimic, so the simulation is free.  Verify against the real hardware
before relying on the coupling.

Retargeting tracks:
  * HybridRetargeter (default): joint-angle mapping as the base posture +
    fingertip/pinch refinement -- the joint mapping anchors the
    underdetermined DOFs (the thumb has 4 joints for 1 fingertip), the
    optimization keeps the pinch geometry.
  * FingertipRetargeter: pure DexPilot-style fingertip-position
    optimization (prone to odd thumb postures; kept for comparison).
  * GeoRtRetargeter (p24grasp.teleop.geort): MLP trained in simulation,
    1 kHz-class inference.
  * DirectAngleScaling: stage-0 placeholder, joint mapping only.
The dex-retargeting (Pinocchio) adapter lives in the stub below; the
HybridRetargeter implements the same AnyTeleop math natively on the repo's
own kinematics (zero extra dependencies).
"""
from __future__ import annotations

# pylint: disable=too-many-lines
# the retargeting backends share one scaffolding class; splitting the file
# would scatter the solver invariants across modules

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.optimize import least_squares

from p24grasp.teleop.angles import HandAngles, thumb_over_finger
from p24grasp.teleop.lateral import (
    LATERAL_FINGER_NAMES,
    LateralEstimator,
    fit_palm_frame,
)

# Joint order == URDF order.
P24_JOINT_NAMES = tuple(
    f"{finger}_dh_joint_{i}" for finger in ("thumb", "index", "middle", "ring", "little")
    for i in range(1, 5)
)
# (lower, upper) degrees, from the URDF <limit> tags.
P24_LIMITS = np.array([
    [0.0, 67.0], [-20.0, 80.0], [-20.0, 20.0], [0.0, 80.0],       # thumb
    [0.0, 80.0], [-15.0, 15.0], [0.0, 80.0], [0.0, 80.0],         # index
    [0.0, 80.0], [-15.0, 15.0], [0.0, 80.0], [0.0, 80.0],         # middle
    [0.0, 80.0], [-15.0, 15.0], [0.0, 80.0], [0.0, 80.0],         # ring
    [0.0, 80.0], [-15.0, 15.0], [0.0, 80.0], [0.0, 80.0],         # little
])
# Human ROM -> robot ROM scale, per flexion column [MCP, PIP, DIP] (fingers) and
# [CMC, MP, IP] (thumb), using HUMAN_FLEXION_ROM from angles.py.
FINGER_FLEX_SCALE = np.array([80.0 / 90.0, 80.0 / 110.0, 80.0 / 80.0])
THUMB_FLEX_SCALE = np.array([67.0 / 60.0, 80.0 / 80.0, 80.0 / 80.0])
ABDUCTION_SCALE = 15.0 / 30.0  # robot +/-15 deg vs human +/-30 deg

# Active-order chain names == HandModel.chains order (alphabetical tip names).
CHAIN_NAMES = ("index", "little", "middle", "ring", "thumb")
# URDF row offset of each finger chain.
URDF_OFFSETS = {"index": 4, "little": 16, "middle": 8, "ring": 12}


class Retargeter(Protocol):
    """Maps filtered human angles to a 20-joint P24 command (deg, URDF order).

    ``keypoints3d`` is the lifted 21-keypoint cloud in the camera frame
    (may be all-NaN); angle-only retargeters ignore it.
    """

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        ...


def scaling_q16(hand, angles: HandAngles,
                calibration: "HandCalibration | None" = None,
                thumb_gain: float = 1.5) -> np.ndarray:
    """DirectAngleScaling output as the 16 active DOFs (rad, active order).

    Args:
        hand: the :class:`HandModel` (chain slices/offsets).
        angles: filtered human :class:`HandAngles`.
        calibration: optional per-user :class:`HandCalibration` (flexion
            offsets/gains and the coupled distal gain).
        thumb_gain: extra amplification of the thumb's flexion mapping
            (the thumb's motion read as too small at 1:1).

    Returns:
        (16,) active-DOF joint angles in radians, in ACTIVE order
        (index, little, middle, ring, thumb).
    """
    if calibration is not None:
        scaling = DirectAngleScaling(
            flexion_offsets_deg=calibration.flexion_offsets_deg,
            flexion_gains=calibration.flexion_gains,
            distal_total_gain=calibration.distal_total_gain,
            thumb_gain=thumb_gain,
        )
    else:
        scaling = DirectAngleScaling(thumb_gain=thumb_gain)
    q20 = np.radians(scaling.retarget(angles))
    out = np.zeros(hand.n_active)
    thumb_s, thumb_e = hand.chain_slices["thumb"]
    out[thumb_s:thumb_e] = q20[0:4]
    for name in ("index", "little", "middle", "ring"):
        offset = URDF_OFFSETS[name]
        start, end = hand.chain_slices[name]
        out[start:end] = q20[[offset, offset + 1, offset + 2]]
    return out


_WRIST_SANITY_REPORTED = False


def _wrist_depth_sane(keypoints3d: np.ndarray, max_distance_m: float) -> bool:
    """Whether the wrist keypoint agrees with the palm keypoints.

    During hand (re-)entry the wrist often sits at the frame edge,
    where its depth window samples the background: the wrist point is
    then lifted far behind the palm, and EVERY wrist-relative tip
    vector is inflated uniformly -- the median cannot reject it, the
    monotonic reach max absorbs it, and the reach scale clips to its
    lower bound, curling the robot into a fist for minutes.  Frames
    with a glitched wrist must hold instead of updating targets/scale.

    Args:
        keypoints3d: (21, 3) camera-frame keypoints.
        max_distance_m: the orientation-independent wrist-to-palm bound.

    Returns:
        True when the wrist lies within ``max_distance_m`` of the median
        MCP position.
    """
    kp = np.asarray(keypoints3d, dtype=np.float64)
    wrist = kp[0]
    mcp = kp[[1, 5, 9, 13, 17]]
    mcp = mcp[np.isfinite(mcp).all(axis=1)]
    if not np.isfinite(wrist).all() or mcp.shape[0] < 3:
        return False
    distance = float(np.linalg.norm(wrist - np.median(mcp, axis=0)))
    global _WRIST_SANITY_REPORTED  # pylint: disable=global-statement
    if distance >= max_distance_m and not _WRIST_SANITY_REPORTED:
        _WRIST_SANITY_REPORTED = True
        print(f"[retarget] 腕部合理性拒绝: 腕-掌距 {distance * 1000:.0f} mm "
              f"> {max_distance_m * 1000:.0f} mm,指尖目标持续保持"
              f"(Leap 路径常见:腕点来自手臂骨,比腕装 RGB-D 相机远;"
              f"可用 wrist_sanity_m=None 关闭)", flush=True)
    return distance < max_distance_m


class _TipSpaceRetargeter:
    """Shared scaffolding: HandModel, chain order, joint bounds, adaptive
    reach scale, and the URDF<->active DOF conversions.

    The adaptive scale is essential: the human hand (wrist->tip ~0.16 m) is
    smaller than the P24 fingertip reach (~0.22 m).  A fixed scale=1 leaves
    the targets short of the robot tips, so the optimizer curls the fingers
    to reach them (straight human -> bent robot).  The adaptive scale tracks
    the human's maximum observed reach and maps it onto the robot reach:
    straight fingers -> open hand, by construction.
    """

    TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
    WRIST_ID = 0
    # URDF joint order per finger: (MCP, abduction, PIP, DIP); the URDF
    # <mimic> couples DIP 1:1 to PIP (joint_4 mimics joint_3), so each finger
    # chain contributes the active (MCP, abd, PIP) = URDF indices 0, 1, 2.
    FINGER_ACTIVE_TO_URDF = (0, 1, 2)

    def __init__(self, scale: float | None = None, max_nfev: int = 200,
                 wrist_sanity_m: float | None = 0.10):
        """Shared solver scaffolding.

        Args:
            scale: fixed human->robot fingertip reach scale; None enables
                the adaptive reach estimate (recommended -- a fixed 1.0
                leaves the targets short of the robot tips, so straight
                human fingers curl the robot hand).
            max_nfev: scipy ``least_squares`` evaluation budget per solve.
            wrist_sanity_m: maximum wrist-to-palm distance for trusting
                the wrist keypoint (None disables the check -- the leap
                wrist comes from the arm bone and can be farther than
                the wrist-mounted RGB-D camera's, with no depth glitches
                to guard against).
        """
        self.wrist_sanity_m = wrist_sanity_m
        from p24grasp.model.urdf import HandModel  # noqa: E402
        from p24grasp.paths import urdf_path  # noqa: E402

        self.scale = scale
        self.max_nfev = max_nfev
        self.hand = HandModel(urdf_path())
        self._chain_names = CHAIN_NAMES
        self._q_prev = np.zeros(self.hand.n_active)
        self._q20_prev = np.zeros(20)  # last command in URDF order, degrees
        self._lb = np.zeros(self.hand.n_active)
        self._ub = np.zeros(self.hand.n_active)
        for name in self._chain_names:
            chain = self._chain(name)
            start, _ = self.hand.chain_slices[name]
            for offset, joint in enumerate(chain.active_joint_names):
                data = self.hand.joints[joint]
                self._lb[start + offset] = data.lower
                self._ub[start + offset] = data.upper
        # The human thumb never hyperextends backward, and the solver
        # otherwise exploits the MP's [-20, 80] range: measured -19 deg
        # constantly on the leap path, in a basin WORSE than MP=0 on
        # every residual.  Clamp the thumb MP's lower bound to 0.
        thumb_start, _ = self.hand.chain_slices["thumb"]
        self._lb[thumb_start + 1] = 0.0
        # robot fingertip reach at the open pose
        open_tip = self.hand.tip_point(self._chain("index"), np.zeros(3))
        self._robot_reach = float(np.linalg.norm(open_tip))
        # adaptive reach estimate: starts conservative, tracks the running max
        # of the observed human wrist->tip distances (slow decay so a hand
        # that never fully extends still converges).
        self._human_reach = 0.12 if scale is None else self._robot_reach / scale
        # remembered for reset(): a hand removal returns the mapping to the
        # first-startup state, including the reach estimate
        self._human_reach_init = float(self._human_reach)
        self._palm_frame: np.ndarray | None = None  # smoothed camera->palm rotation
        # per-finger target memory: the index holds its last targets while
        # the thumb covers it (the keypoints then belong to the thumb)
        self._last_targets: np.ndarray | None = None
        self._last_pip_targets: np.ndarray | None = None

    def _chain(self, name):
        """Look up a finger chain of the hand model by name.

        Args:
            name: chain name (index/little/middle/ring/thumb).

        Returns:
            The :class:`Chain` object.

        Raises:
            KeyError: unknown chain name.
        """
        for chain in self.hand.chains:
            if chain.name == name:
                return chain
        raise KeyError(name)

    def targets_from_keypoints(self, keypoints3d: np.ndarray) -> np.ndarray | None:
        """(5, 3) robot-palm-frame fingertip targets, or None when the needed
        keypoints lack depth.  Updates the adaptive reach/scale estimate.

        Args:
            keypoints3d: (21, 3) camera-frame keypoints.

        Returns:
            (5, 3) fingertip targets in the robot palm frame (chain order),
            or None when the wrist, any fingertip, or a palm-frame input
            (middle MCP / thumb CMC) is non-finite -- a missing frame input
            would fit a garbage palm frame, so the caller holds instead.
            While the thumb tip covers the index MCP, the index target
            holds its last value (the keypoints then belong to the thumb).
        """
        # frame inputs are as critical as the tips: without them the palm
        # frame fit is garbage and the targets would twist the robot hand
        needed = [self.WRIST_ID, 9, 1,
                  *[self.TIP_IDS[name] for name in self._chain_names]]
        if keypoints3d is None or not np.isfinite(keypoints3d[needed]).all():
            # a stale hold must not cross an absence: drop both target
            # memories so the next detection starts with fresh targets
            self._last_targets = None
            self._last_pip_targets = None
            return None
        if not self._wrist_sane(keypoints3d):
            self._last_targets = None
            self._last_pip_targets = None
            return None
        frame = self._palm_frame_from_keypoints(keypoints3d)
        wrist = keypoints3d[self.WRIST_ID]
        rel = np.stack([keypoints3d[self.TIP_IDS[name]] - wrist
                        for name in self._chain_names])
        scale = self._update_scale(rel)
        targets = scale * (frame @ rel.T).T + self.mount_t
        targets = targets @ self._TARGET_ROTATION.T
        if thumb_over_finger(keypoints3d, 5) and self._last_targets is not None:
            index_slot = self._chain_names.index("index")
            targets[index_slot] = self._last_targets[index_slot]
        self._last_targets = targets
        return targets

    def pip_targets_from_keypoints(self, keypoints3d: np.ndarray) -> np.ndarray | None:
        """(5, 3) robot-palm-frame PIP-joint targets (Vector10's proximal half).

        Args:
            keypoints3d: (21, 3) camera-frame keypoints.

        Returns:
            (5, 3) PIP-joint targets in the robot palm frame (chain
            order), or None when the wrist, any PIP keypoint, or a
            palm-frame input (middle MCP / thumb CMC) is non-finite; the
            index PIP holds its last value while the thumb covers it.
        """
        pip_ids = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "little": 18}
        needed = [self.WRIST_ID, 9, 1, *[pip_ids[name] for name in self._chain_names]]
        if keypoints3d is None or not np.isfinite(keypoints3d[needed]).all():
            # drop the memory across an absence (see targets_from_keypoints)
            self._last_targets = None
            self._last_pip_targets = None
            return None
        if not self._wrist_sane(keypoints3d):
            self._last_targets = None
            self._last_pip_targets = None
            return None
        frame = self._palm_frame_from_keypoints(keypoints3d)
        wrist = keypoints3d[self.WRIST_ID]
        rel = np.stack([keypoints3d[pip_ids[name]] - wrist
                        for name in self._chain_names])
        scale = self._update_scale(
            np.stack([keypoints3d[self.TIP_IDS[name]] - wrist
                      for name in self._chain_names]))
        pip_targets = scale * (frame @ rel.T).T + self.mount_t
        pip_targets = pip_targets @ self._TARGET_ROTATION.T
        if thumb_over_finger(keypoints3d, 5) and self._last_pip_targets is not None:
            index_slot = self._chain_names.index("index")
            pip_targets[index_slot] = self._last_pip_targets[index_slot]
        self._last_pip_targets = pip_targets
        return pip_targets

    def _update_scale(self, rel: np.ndarray) -> float:
        """Track the adaptive reach scale from wrist-relative tip vectors.

        Args:
            rel: (5, 3) wrist->fingertip vectors of this frame (chain
                order).

        Returns:
            The robot/human reach ratio, clipped to [0.8, 1.8]; with a
            fixed ``self.scale`` the estimate is not updated and the
            fixed value is returned.
        """
        if self.scale is None:
            observed = float(np.median(np.linalg.norm(rel, axis=1)))
            # Spike rejection: one glitched frame (e.g. wrist depth-lifted
            # onto the background) inflates every tip vector uniformly and
            # the median cannot reject it.  The running max NEVER decays:
            # a decay (x0.999/frame) shrank the estimate from the
            # calibrated 168 mm to 127 mm during a ~7 s closed-hand hold,
            # inflating the scale to 1.76 and pushing the thumb target
            # beyond the robot's reach (MP hyperextension).  Only
            # plausible reaches within the human band and within a 1.6x
            # extension step of the current estimate are accepted.
            if 0.06 <= observed <= 0.30 and observed <= self._human_reach * 1.6:
                self._human_reach = max(self._human_reach, observed)
            return float(np.clip(self._robot_reach / self._human_reach, 0.8, 1.8))
        return self.scale

    def _palm_frame_from_keypoints(self, keypoints3d: np.ndarray) -> np.ndarray:
        """Camera->palm rotation estimated from the keypoints themselves.

        The wrist-mounted camera has an unknown roll angle, so a fixed mount
        matrix maps spread into flexion when the camera rolls.  Instead, the
        palm frame is rebuilt every frame from the two most stable rays:
        x = in-plane wrist->middle-MCP (finger direction), y = in-plane
        wrist->thumb-MCP orthogonalized (thumb side).  The estimate is
        one-pole smoothed and re-orthonormalized (polar decomposition).

        Args:
            keypoints3d: (21, 3) camera-frame keypoints.

        Returns:
            (3, 3) smoothed camera->palm rotation (rows = palm axes);
            keeps the previous frame on degenerate input, identity before
            the first valid frame.
        """
        frame = fit_palm_frame(keypoints3d)
        if not np.isfinite(frame).all():
            # degenerate input (missing keypoints): keep the previous frame
            if self._palm_frame is None:
                return np.eye(3)
            return self._palm_frame
        if self._palm_frame is None:
            self._palm_frame = frame
        else:
            # a large orientation change (hand re-entered rotated) snaps
            # instead of blending: the 0.8/0.2 mix would compute targets in
            # a transitional frame for ~10 frames and twist the robot hand
            rel_angle = float(np.degrees(np.arccos(np.clip(
                (np.trace(frame @ self._palm_frame.T) - 1.0) / 2.0, -1.0, 1.0))))
            if rel_angle > 45.0:
                self._palm_frame = frame
            else:
                self._palm_frame = 0.8 * self._palm_frame + 0.2 * frame
                # re-orthonormalize via polar decomposition
                u, _, vt = np.linalg.svd(self._palm_frame)
                self._palm_frame = u @ vt
                if np.linalg.det(self._palm_frame) < 0:
                    self._palm_frame[-1] *= -1
        return self._palm_frame

    mount_t = np.zeros(3)

    # The robot's fingers extend along +z in its FK (the open index tip
    # sits at (0, 49, 217) mm), while the palm-frame transform maps the
    # human's finger direction (frame x) onto +x.  Without this rotation
    # every target lands in the z=0 plane, 90 degrees off the robot's
    # finger direction, and the solver saturates the MCP joints against
    # their limits trying to reach sideways.  The rotation maps x -> +z
    # and keeps the fan direction (y) as y: (x, y, z) -> (-z, y, x).
    _TARGET_ROTATION = np.array([[0.0, 0.0, -1.0],
                                 [0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0]])

    @staticmethod
    def _wrist_depth_sane(keypoints3d: np.ndarray) -> bool:
        return _wrist_depth_sane(keypoints3d, 0.10)

    def _wrist_sane(self, keypoints3d: np.ndarray) -> bool:
        """The per-instance wrist check (see ``wrist_sanity_m``)."""
        if self.wrist_sanity_m is None:
            return True
        return _wrist_depth_sane(keypoints3d, self.wrist_sanity_m)

    def _tips(self, q16: np.ndarray) -> np.ndarray:
        """(5, 3) robot fingertip positions at q16 (chain order).

        Args:
            q16: (16,) active-DOF angles (rad).

        Returns:
            (5, 3) forward-kinematics fingertip positions in the robot
            base frame, rows in chain order (index..thumb).
        """
        tips = []
        for name in self._chain_names:
            start, end = self.hand.chain_slices[name]
            tips.append(self.hand.tip_point(self._chain(name), q16[start:end]))
        return np.stack(tips)

    def _to_urdf_degrees(self, q16: np.ndarray) -> np.ndarray:
        """Convert 16 active DOFs (rad) -> 20-joint URDF order (deg).

        Args:
            q16: (16,) active-DOF angles (rad).

        Returns:
            (20,) URDF-order joint angles in degrees; the four DIP joints
            are slaved to their PIP (the URDF mimic coupling).
        """
        q20 = np.zeros(20)
        thumb = self.hand.chain_slices["thumb"]
        q20[0:4] = q16[thumb[0]:thumb[1]]  # thumb joints are all active
        for name, offset in zip(("index", "middle", "ring", "little"),
                                (4, 8, 12, 16)):
            start, end = self.hand.chain_slices[name]
            q_chain = q16[start:end]  # (MCP, abd, PIP)
            for k, urdf_idx in enumerate(self.FINGER_ACTIVE_TO_URDF):
                q20[offset + urdf_idx] = q_chain[k]
            q20[offset + 3] = q20[offset + 2]  # URDF mimic: DIP = PIP
        return np.degrees(q20)

    def _active_from_urdf(self) -> np.ndarray:
        """Indices of the 16 active DOFs within the 20-joint URDF vector,
        in ACTIVE order: index, little, middle, ring, thumb (the thumb chain
        is LAST in active order but FIRST in URDF order).

        Returns:
            (16,) integer index array into a 20-joint URDF vector.
        """
        indices = []
        # URDF finger rows in ACTIVE (chain) order: index, little,
        # middle, ring -- (MCP, abd, PIP, DIP=mimic) per finger
        for offset in (4, 16, 8, 12):
            indices.extend([offset + 0, offset + 1, offset + 2])
        indices.extend([0, 1, 2, 3])  # thumb URDF row, last in active order
        return np.array(indices)

    def _q20_to_active(self, q20_deg: np.ndarray) -> np.ndarray:
        """Convert a 20-joint URDF command (deg) to 16 active DOFs (rad).

        Args:
            q20_deg: (20,) URDF-order joint angles in degrees.

        Returns:
            (16,) active-DOF angles in radians (ACTIVE order).
        """
        return np.radians(np.asarray(q20_deg))[self._active_from_urdf()]

    def reset(self) -> None:
        """Reset the mapping state for a fresh start.

        The pipeline calls this when the hand leaves the view: the robot
        returns to the open palm and the next detection re-runs the
        first-startup logic.  Clears the solver warm start, the palm
        frame, the adaptive reach (back to its startup value), and the
        occlusion target holds.
        """
        self._q_prev = np.zeros(self.hand.n_active)
        self._q20_prev = np.zeros(20)
        self._palm_frame = None
        self._last_targets = None
        self._last_pip_targets = None
        self._human_reach = self._human_reach_init


class FingertipRetargeter(_TipSpaceRetargeter):
    """DexPilot-style fingertip-position retargeting on the P24 kinematics.

    min_q  sum_i w_i ||t_i - tip_i(q)||^2 + smooth * ||q - q_prev||^2
    s.t.   URDF joint limits, warm-started from the previous solution.

    Thumb and index residuals carry ``pinch_weight`` so the thumb-index tip
    distance survives measurement noise.  NOTE: the pure fingertip objective
    underdetermines the thumb (4 joints, 1 point), which can drift into odd
    postures -- prefer HybridRetargeter in production.
    """

    def __init__(self, scale: float | None = None, pinch_weight: float = 2.0,
                 smooth_weight: float = 1e-3, max_nfev: int = 200):
        super().__init__(scale=scale, max_nfev=max_nfev)
        self.pinch_weight = pinch_weight
        self.smooth_weight = smooth_weight
        self._weights = np.array(
            [pinch_weight, 1.0, 1.0, 1.0, pinch_weight])  # index..thumb chain order

    def _residuals(self, q16: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Least-squares residual vector for the pure fingertip objective.

        Args:
            q16: (16,) candidate active-DOF angles (rad).
            targets: (5, 3) fingertip targets (chain order).

        Returns:
            Stacked residuals: pinch-weighted tip errors (15,) + a small
            smoothness term pulling toward the previous solution (16,).
        """
        residuals = [self._weights[:, None] * (targets - self._tips(q16))]
        residuals.append(np.sqrt(self.smooth_weight) * (q16 - self._q_prev))
        return np.concatenate([r.ravel() for r in residuals])

    def solve(self, targets: np.ndarray) -> np.ndarray:
        """Solve for the 16 active DOFs (rad); returns the 20-joint command
        in URDF order, degrees, with the mimic coupling applied.

        Args:
            targets: (5, 3) fingertip targets (chain order).

        Returns:
            (20,) URDF-order joint command in degrees, warm-started from
            the previous solution and clamped to the URDF limits.
        """
        result = least_squares(
            self._residuals, self._q_prev, args=(targets,),
            bounds=(self._lb, self._ub),
            jac="2-point", max_nfev=self.max_nfev)
        self._q_prev = np.clip(result.x, self._lb, self._ub)
        self._q20_prev = self._to_urdf_degrees(self._q_prev)
        return self._q20_prev.copy()

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        """Per-frame retarget; holds the previous command when the required
        keypoints are missing (occlusion / depth loss).

        Args:
            angles: filtered human :class:`HandAngles` (unused by this
                pure fingertip backend).
            keypoints3d: (21, 3) camera-frame keypoints; None/NaN yields a
                hold.

        Returns:
            (20,) URDF-order joint command in degrees.
        """
        targets = self.targets_from_keypoints(keypoints3d)
        if targets is None:
            return self._q20_prev.copy()
        return self.solve(targets)


class HybridRetargeter(_TipSpaceRetargeter):
    """Joint-angle mapping base + fingertip/pinch refinement (default).

    Why: pure fingertip-position retargeting underdetermines the thumb
    (4 joints, 1 point) and lets non-pinch fingers drift into odd postures.
    The hybrid anchors the solution at the joint-angle mapping ``q0`` (the
    human MCP/PIP/DIP and thumb angles, scaled to the P24 ranges -- natural
    finger posture by construction) and only corrects what the tips demand:

        min_q  w_tip * sum_i v_i ||t_i - tip_i(q)||
             + w_pinch * |d_ti(q) - d_ti_human|
             + w_reg * ||q - q0||
        s.t.   URDF joint limits, warm-started from the previous frame.

    The objective alone has no self-collision notion: the URDF has no
    inter-finger contact geometry, so a human squeeze maps straight
    into interpenetrating phalanges.  ``solve()`` therefore runs a
    post-solve separation pass (``_separate_fingers``) that opens the
    involved fingers' lateral DOFs until every adjacent pair clears
    ``collision_threshold_m`` (disabled by ``collision_avoidance=False``).

    When the keypoints lack depth (occlusion / hand out of view), the
    retargeter HOLDs the last optimized command -- falling back to the pure
    joint mapping would jump, because the optimizer's solution differs from
    q0 (the thumb carries no prior).
    """

    def __init__(self, scale: float | None = None, tip_weight: float = 0.5,
                 pinch_weight: float = 2.0, reg_weight: float = 10.0, *,
                 pip_weight: float = 0.5,
                 max_nfev: int = 200,
                 prior_weights: dict | None = None,
                 lateral_calibration_path: str | Path | None = None,
                 hand_calibration_path: str | Path | None = None,
                 lateral_enabled: bool = True,
                 thumb_gain: float = 1.5,
                 wrist_sanity_m: float | None = 0.10,
                 collision_avoidance: bool = True,
                 collision_threshold_m: float = 0.022):
        # max_nfev=200 covers hard cold starts; with the cold start at q0
        # (see solve()) typical solves converge in a single iteration.
        # NOTE: do NOT skip solves on "unchanged" targets -- a slowly
        # moving finger shifts its target by <1 mm per frame, and any
        # skip threshold freezes slow motions (a stuck-looking finger).
        # prior_weights: per-joint-class regularization strength (reference
        # workspace configs): {"mcp_flexion": 0.01, "lateral": 0.05,
        # "distal": 0.01}; thumb priors are zero (the optimizer owns the
        # thumb).  Lateral carries the strongest prior because an
        # unconstrained tip objective most often misuses joint_2.
        # reg_weight scales the whole prior block: the reference ratios are
        # preserved, and the global scale compensates for our tip-only
        # objective (their Vector10 also constrains the proximal links).
        super().__init__(scale=scale, max_nfev=max_nfev,
                         wrist_sanity_m=wrist_sanity_m)
        self.tip_weight = tip_weight
        self.pinch_weight = pinch_weight
        self.reg_weight = reg_weight
        self.pip_weight = pip_weight
        self.lateral_enabled = lateral_enabled
        self.thumb_gain = thumb_gain
        self.collision_avoidance = collision_avoidance
        self.collision_threshold_m = collision_threshold_m
        self._lateral_j2 = np.array([5, 9, 13, 17])  # URDF joint_2 slots
        self._weights = np.array(
            [2.0, 1.0, 1.0, 1.0, 2.0])  # index..thumb chain order (pinch side)
        # the thumb's PIP constraint CONFLICTS with its tip target (the
        # robot thumb cannot satisfy both, and the solver hyperextends
        # the MP to its lower bound chasing the PIP) -- zero it; the
        # four fingers keep their proximal constraints
        self._pip_weights = np.array([2.0, 1.0, 1.0, 1.0, 0.0])
        prior_config = {
            "mcp_flexion": 0.01, "lateral": 0.05, "distal": 0.01,
            "thumb_flexion": 0.05, "thumb_abduction": 0.05,
            **(prior_weights or {})}
        self._prior_weights = self._make_prior_weights(prior_config)
        # Guided-gesture calibration takes precedence when present;
        # otherwise the startup auto-calibration learns the user's neutral
        # lateral spread from ~1 s of open-hand frames (and stays neutral
        # until then -- wrong offsets collapse the middle fingers together).
        from p24grasp.teleop.calibration import (  # noqa: E402
            load_hand_calibration,
        )

        self._hand_calibration = load_hand_calibration(
            hand_calibration_path if hand_calibration_path is not None
            else None)
        if self._hand_calibration is not None:
            self._lateral = LateralEstimator(
                calibration=self._hand_calibration.lateral,
                calibration_path=lateral_calibration_path)
            if self.scale is None:
                # seed the adaptive reach with the calibrated open-hand size
                if np.isfinite(self._hand_calibration.reach_m):
                    self._human_reach = self._hand_calibration.reach_m
                else:
                    print("[retarget] 警告: 标定文件中的伸展半径非有限值,"
                          "使用默认值;建议重新运行 --calibrate", flush=True)
            print(f"[retarget] loaded hand calibration "
                  f"(lateral offsets "
                  f"{np.round(self._hand_calibration.lateral.offsets_deg, 1).tolist()}, "
                  f"reach {self._hand_calibration.reach_m * 1000:.0f} mm, "
                  f"thumb gain x{self.thumb_gain})",
                  flush=True)
        else:
            self._lateral = LateralEstimator(
                auto_calibrate=True, calibration_path=lateral_calibration_path)
        # a reset() returns the reach estimate to this startup value (the
        # calibrated open-hand reach when the calibration was loaded)
        self._human_reach_init = float(self._human_reach)

    def reset(self) -> None:
        """Reset the mapping state for a fresh start (see the base class);
        also restarts the lateral estimator back to its startup value."""
        super().reset()
        self._lateral.reset()

    def _make_prior_weights(self, config: dict) -> np.ndarray:
        """Per-DOF prior weights over the 16 active DOFs (rad scale).

        Thumb DOFs get zero prior (the optimizer owns the thumb); the
        four-finger classes use the reference-workspace strengths.

        Args:
            config: dict with keys "mcp_flexion"/"lateral"/"distal"/
                "thumb_flexion" (per-class prior strengths).

        Returns:
            (16,) prior-weight array in ACTIVE order.
        """
        weights = np.zeros(self.hand.n_active)
        for name in self._chain_names:
            chain = self._chain(name)
            start, _ = self.hand.chain_slices[name]
            for offset, joint in enumerate(chain.active_joint_names):
                if joint.startswith("thumb_"):
                    # thumb flexion DOFs get a small prior: the tip+PIP
                    # targets underdetermine the 4-DOF chain and the
                    # unconstrained solver drives the MP hyperextended
                    # (measured -20 deg on the leap path); the
                    # opposition DOF (joint_3) gets the same small
                    # prior toward the measured thumb abduction, with
                    # the pinch free to refine it
                    if joint.endswith("_joint_3"):
                        weights[start + offset] = config["thumb_abduction"]
                    else:
                        weights[start + offset] = config["thumb_flexion"]
                elif joint.endswith("_joint_1"):
                    weights[start + offset] = config["mcp_flexion"]
                elif joint.endswith("_joint_2"):
                    weights[start + offset] = config["lateral"]
                elif joint.endswith("_joint_3"):
                    weights[start + offset] = config["distal"]
        return weights

    # adjacent four-finger pairs whose phalanges must not interpenetrate:
    # the URDF has no inter-finger contact geometry, so nothing else
    # stops the mapping from driving them through each other (the tips
    # converge when the human squeezes or grips something small).  The
    # thumb is excluded -- the pinch residual intentionally brings it
    # to the index tip.
    FINGER_NEIGHBORS = (("index", "middle"), ("middle", "ring"),
                        ("ring", "little"))

    @staticmethod
    def _checkpoint_links(name: str) -> tuple[str, str, str]:
        """PIP/DIP/tip link names of a four-finger chain."""
        return (f"{name}_ip_link", f"{name}_dip_link", f"{name}_tip_link")

    def _finger_checkpoints(self, q16: np.ndarray) -> np.ndarray:
        """(4, 5, 3) PIP/DIP/tip + segment-midpoint positions of the four
        fingers at q16.

        The five checkpoints per finger (PIP, PIP-DIP midpoint, DIP,
        DIP-tip midpoint, tip) sample the phalanx paths; the pairwise
        distances between adjacent fingers' checkpoints feed the
        self-collision penalty (see ``_residuals``).  Joints alone are
        not enough: two phalanges can scissor-cross with every joint
        pair still far apart (measured 1.5 mm segment distance with
        >= 27 mm joint distances).

        Args:
            q16: (16,) active-DOF angles (rad).

        Returns:
            (4, 5, 3) positions in the robot base frame, first axis in
            chain order (index, little, middle, ring), second axis
            PIP/mid/DIP/mid/tip.
        """
        rows = []
        for name in self._chain_names:
            if name == "thumb":
                continue
            start, end = self.hand.chain_slices[name]
            transforms = self.hand.fk_chain(self._chain(name), q16[start:end])
            pip_dip_tip = np.stack([transforms[link][:3, 3]
                                    for link in self._checkpoint_links(name)])
            rows.append(np.stack([
                pip_dip_tip[0],
                (pip_dip_tip[0] + pip_dip_tip[1]) / 2.0,
                pip_dip_tip[1],
                (pip_dip_tip[1] + pip_dip_tip[2]) / 2.0,
                pip_dip_tip[2],
            ]))
        return np.stack(rows)

    def _pair_distances(self, q16: np.ndarray) -> np.ndarray:
        """(3,) minimum checkpoint distance per adjacent finger pair.

        Args:
            q16: (16,) active-DOF angles (rad).

        Returns:
            Minimum over the 25 checkpoint pairs of each neighbor pair,
            in ``FINGER_NEIGHBORS`` order, meters.
        """
        checkpoints = self._finger_checkpoints(q16)
        minima = []
        for a_name, b_name in self.FINGER_NEIGHBORS:
            a = checkpoints[self._chain_names.index(a_name)]
            b = checkpoints[self._chain_names.index(b_name)]
            minima.append(float(np.linalg.norm(
                a[:, None, :] - b[None, :, :], axis=-1).min()))
        return np.array(minima)

    def _separate_fingers(self, q16: np.ndarray) -> np.ndarray:
        """Push apart adjacent fingers whose phalanges interpenetrate.

        The hybrid objective reproduces the human gesture as-is: when
        the human squeezes adjacent fingertips together, the mapped
        robot fingers clip through each other (the robot phalanges are
        wider than the human's and the URDF has no inter-finger
        collision bodies to stop them).  This post-solve pass walks
        the involved fingers' lateral (joint_2) DOFs outward -- the
        fan-opening direction -- with a small coordinate-descent line
        search until every adjacent pair clears
        ``collision_threshold_m``, or no lateral move improves the
        worst pair.

        Lateral-only keeps the pass cheap (a few FK evaluations, only
        while a violation exists) and leaves the tuned objective
        untouched: flexion still follows the human, only the fan
        angle widens where the phalanges would overlap.

        Args:
            q16: (16,) solved active-DOF angles (rad), within bounds.

        Returns:
            (16,) adjusted angles (rad, within bounds); unchanged when
            no pair violates the threshold or the search cannot improve
            the worst pair.
        """
        lateral_steps = np.radians(np.array([0.0, 4.0, 8.0, 12.0, 16.0,
                                             20.0, 24.0]))
        q = np.asarray(q16, dtype=np.float64).copy()
        for _ in range(3):  # at most three sweep rounds per pair
            distances = self._pair_distances(q)
            worst = int(np.argmin(distances))
            if distances[worst] >= self.collision_threshold_m:
                break
            a_name, b_name = self.FINGER_NEIGHBORS[worst]
            a_idx = self._chain_names.index(a_name)
            b_idx = self._chain_names.index(b_name)
            a_slot = self.hand.chain_slices[a_name][0] + 1  # joint_2 slot
            b_slot = self.hand.chain_slices[b_name][0] + 1
            # the fan-opening direction of each lateral: the sign of
            # joint_2 that shifts the finger's checkpoints away from
            # the neighbor's centroid
            checkpoints = self._finger_checkpoints(q)
            a_center = checkpoints[a_idx].mean(axis=0)
            b_center = checkpoints[b_idx].mean(axis=0)
            away_a = a_center - b_center
            away_b = b_center - a_center

            def _lateral_sign(slot: int, idx: int, center: np.ndarray,
                              away: np.ndarray) -> float:
                probe = q.copy()
                probe[slot] = min(probe[slot] + 1e-3, self._ub[slot])
                shifted = self._finger_checkpoints(probe)[idx].mean(axis=0)
                return 1.0 if float((shifted - center) @ away) > 0.0 else -1.0

            sign_a = _lateral_sign(a_slot, a_idx, a_center, away_a)
            sign_b = _lateral_sign(b_slot, b_idx, b_center, away_b)
            # coordinate descent: sweep finger A's lateral outward, then
            # finger B's, keeping the better of the two; accept a step
            # only when it improves the WORST pair distance (it cannot
            # be allowed to hurt any other pair either -- the global
            # min is what the check below re-evaluates)
            improved = False
            for slot, sign in ((a_slot, sign_a), (b_slot, sign_b)):
                best_q = None
                best_d = distances.min()
                for step in lateral_steps:
                    trial = q.copy()
                    trial[slot] += sign * step
                    trial[slot] = np.clip(trial[slot], self._lb[slot],
                                          self._ub[slot])
                    if trial[slot] == q[slot]:
                        continue
                    d_trial = self._pair_distances(trial).min()
                    if d_trial > best_d:
                        best_d = d_trial
                        best_q = trial
                if best_q is not None:
                    q = best_q
                    distances = self._pair_distances(q)
                    improved = True
                    if distances.min() >= self.collision_threshold_m:
                        return q
            if not improved:
                break
        return q

    def _residuals(self, q16: np.ndarray, targets: np.ndarray,
                   q0: np.ndarray, pinch_human: float,
                   pip_targets: np.ndarray | None) -> np.ndarray:
        """Least-squares residual vector for the hybrid objective
        (tips + optional PIPs + pinch distance + joint-mapping prior).

        Args:
            q16: (16,) candidate active-DOF angles (rad).
            targets: (5, 3) fingertip targets (chain order).
            q0: (16,) joint-mapping prior (active order, rad).
            pinch_human: human thumb-index tip distance (m).
            pip_targets: optional (5, 3) PIP targets (chain order).

        Returns:
            Stacked residuals: pinch-weighted tip errors (15,) + optional
            PIP errors (15,) + scalar pinch-gap error (1,) + per-class
            regularization toward q0 (16,).
        """
        tips = self._tips(q16)
        residuals = [self.tip_weight * self._weights[:, None] * (targets - tips)]
        checkpoints = self._finger_checkpoints(q16)
        if pip_targets is not None:
            # Vector10's proximal half: constrain the middle joints so the
            # fingertips alone cannot be satisfied by odd postures.  The
            # four-finger PIP positions come from the checkpoints (no
            # second FK pass); the thumb row carries weight zero.
            pip_positions = np.zeros((5, 3))
            pip_positions[:4] = checkpoints[:, 0]
            residuals.append(
                self.pip_weight * self._pip_weights[:, None]
                * (pip_targets - pip_positions))
        index_tip = tips[self._chain_names.index("index")]
        thumb_tip = tips[self._chain_names.index("thumb")]
        pinch_robot = float(np.linalg.norm(thumb_tip - index_tip))
        residuals.append(np.array([self.pinch_weight * (pinch_robot - pinch_human)]))
        # per-joint-class prior toward the analytical mapping (thumb: none);
        # reg_weight scales the whole prior block (default 1.0)
        residuals.append(
            self.reg_weight * self._prior_weights[:, None] * (q16 - q0))
        return np.concatenate([r.ravel() for r in residuals])

    def solve(self, targets: np.ndarray, q0: np.ndarray,
              pip_targets: np.ndarray | None = None) -> np.ndarray:
        """Refine the joint-mapping prior q0 against the targets; returns the
        20-joint command in URDF order, degrees.

        Args:
            targets: (5, 3) fingertip targets (chain order).
            q0: (16,) joint-mapping prior (active order, rad; clipped to
                the URDF bounds exactly).
            pip_targets: optional (5, 3) PIP targets (chain order).

        Returns:
            (20,) URDF-order joint command in degrees; warm-started from
            whichever of the previous solution and q0 has the lower
            residual (method="dogbox" -- trf stalls when the warm start
            sits exactly on a joint bound).
        """
        # the prior must respect the bounds exactly: the URDF limits are
        # truncated decimals and radians(80 deg) exceeds them by ~1e-10
        q0 = np.clip(np.asarray(q0, dtype=np.float64), self._lb, self._ub)

        pinch_human = float(np.linalg.norm(
            targets[self._chain_names.index("thumb")] -
            targets[self._chain_names.index("index")]))
        # Warm-start sanity: after an occlusion the stale previous solution
        # can be far from the new targets; start from whichever of the stale
        # solution and the analytical prior has the lower residual.  (A
        # short-circuit for the post-reset zero vector changed the re-entry
        # solution for some gestures -- the comparison must always run.)
        x0 = self._q_prev.copy()
        cost_stale = float(np.sum(
            self._residuals(x0, targets, q0, pinch_human, pip_targets) ** 2))
        cost_prior = float(np.sum(
            self._residuals(q0, targets, q0, pinch_human, pip_targets) ** 2))
        if cost_prior < cost_stale:
            x0 = q0.copy()
        # debug snapshot for the live-path diagnostics
        self.last_debug = {
            "x0": x0.copy(), "q0": q0.copy(),
            "targets": np.asarray(targets).copy(),
            "cost_stale": cost_stale, "cost_prior": cost_prior,
        }
        # method="dogbox": trf stalls when the warm start sits exactly on a
        # joint bound (the zero pose is the lower bound for 12 of 16 DOFs).
        result = least_squares(
            self._residuals, x0,
            args=(targets, q0, pinch_human, pip_targets),
            bounds=(self._lb, self._ub), method="dogbox",
            jac="2-point", max_nfev=self.max_nfev)
        self._q_prev = np.clip(result.x, self._lb, self._ub)
        # the objective has no self-collision notion -- the URDF has no
        # inter-finger contact bodies -- so a human squeeze maps straight
        # into interpenetrating phalanges; separate them afterwards
        if self.collision_avoidance:
            separated = self._separate_fingers(self._q_prev)
            self.last_debug["collision_min_mm"] = (
                float(self._pair_distances(separated).min()) * 1000.0)
            self._q_prev = separated
        self._q20_prev = self._to_urdf_degrees(self._q_prev)
        return self._q20_prev.copy()

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        """Per-frame hybrid retarget: joint-mapping prior + tip/pinch
        refinement; holds the last command when the keypoints are missing.

        Args:
            angles: filtered human :class:`HandAngles` (builds the joint
                mapping prior q0; the calibrated lateral estimator writes
                the four joint_2 values into q0, or neutral zeros with
                ``lateral_enabled=False``).
            keypoints3d: (21, 3) camera-frame keypoints; all-NaN yields a
                hold of the last optimized command.

        Returns:
            (20,) URDF-order joint command in degrees.
        """
        q0 = scaling_q16(self.hand, angles, self._hand_calibration,
                         thumb_gain=self.thumb_gain)
        if self.lateral_enabled:
            # Shared calibrated lateral: always write the estimator's robot
            # joint_2 values into the prior (it holds its calibrated initial
            # value without observations), so the no-hand fallback also starts
            # at the user's neutral spread instead of zero.
            lateral_qpos, _ = self._lateral.update(keypoints3d)
            for name in ("index", "little", "middle", "ring"):
                start, _ = self.hand.chain_slices[name]
                q0[start + 1] = lateral_qpos[LATERAL_FINGER_NAMES.index(name)]
        else:
            # lateral disabled: hold the four joint_2 DOFs at neutral
            for name in ("index", "little", "middle", "ring"):
                start, _ = self.hand.chain_slices[name]
                q0[start + 1] = 0.0
        targets = self.targets_from_keypoints(keypoints3d)
        if targets is None:
            # keypoints lost (hand out of view / depth lifting failed): HOLD
            # the last optimized command.  Do NOT fall back to the joint
            # mapping q0 -- the optimizer's solution and q0 differ (the
            # thumb has no prior, so its opposition/CMC are tip-driven), and
            # switching between the two is exactly the angle jump seen when
            # the hand leaves the view.  solve() warm-starts from this held
            # pose, so re-entry continues smoothly.
            return self._q20_prev.copy()
        pip_targets = self.pip_targets_from_keypoints(keypoints3d)
        q20 = self.solve(targets, q0, pip_targets)
        if not self.lateral_enabled:
            q20[self._lateral_j2] = 0.0
            self._q20_prev = q20.copy()
        return q20


@dataclass
class DirectAngleScaling:
    """Joint-angle mapping baseline (BEHAVIOR-style, ported from the
    agile-hand-retarget workspace).

    Four fingers: MCP -> j1, lateral -> j2.  The P2.4 PIP/DIP pair is a 1:1
    mimic, so the human PIP+DIP total distal bend is split half/half across
    j3/j4 after subtracting the robot-side zero-geometry offset (the P2.4
    zero pose is not collinear: FK gives ~10.03 deg of neutral bend, so a
    straight human finger would otherwise read as a bent robot finger).
    Thumb: the three unsigned bends map to j1/j2/j4; j3 (CMC abduction) is
    neutral in this baseline -- the hybrid optimizer refines j1/j3 from the
    fingertip targets.
    """

    couple_dip_pip: bool = True
    distal_zero_deg: float = 10.0337
    flexion_offsets_deg: np.ndarray | None = None  # (5, 3) per-user zero baseline
    flexion_gains: np.ndarray | None = None  # (5, 3) per-user ROM gains
    distal_total_gain: float | None = None  # per-user coupled distal gain
    thumb_gain: float = 1.5  # extra amplification of the thumb flexion
    _limits = P24_LIMITS

    def _calibrated(self, flexion: np.ndarray) -> np.ndarray:
        """Apply the per-user calibration when provided (guided gestures),
        falling back to the anatomical population averages.

        Args:
            flexion: (5, 3) human flexion angles (deg).

        Returns:
            (5, 3) calibrated flexion = (flexion - offsets) * gains, ready
            for the URDF limit clamp.
        """
        offsets = np.zeros((5, 3)) if self.flexion_offsets_deg is None \
            else np.asarray(self.flexion_offsets_deg)
        if self.flexion_gains is None:
            gains = np.vstack([THUMB_FLEX_SCALE,
                               np.tile(FINGER_FLEX_SCALE, (4, 1))])
        else:
            gains = np.asarray(self.flexion_gains).copy()
        # extra thumb amplification (its motion read as too small at 1:1)
        gains[0] *= self.thumb_gain
        return (flexion - offsets) * gains

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        """Pure joint-angle mapping (no optimization): human angles ->
        20-joint URDF command (deg).

        Args:
            angles: filtered human :class:`HandAngles` (NaN read as 0).
            keypoints3d: ignored (kept for the :class:`Retargeter`
                protocol).

        Returns:
            (20,) URDF-order joint command in degrees, clamped to the
            URDF limits.
        """
        raw = np.nan_to_num(angles.flexion, nan=0.0)
        flexion = self._calibrated(raw)
        abduction = np.nan_to_num(angles.abduction, nan=0.0)
        q = np.zeros(20)
        # Thumb: three unsigned bends -> j1, j2, j4; the base abduction
        # (opposition/spread, j3) follows the measured thumb abduction --
        # the leap path subtracts its neutral resting offset upstream, so
        # only the spread/tuck motion around neutral drives the joint.
        # The measured positive = spreading AWAY from the palm, while the
        # robot's +j3 adducts TOWARD it (FK-measured): negate.
        q[0] = flexion[0, 0]
        q[1] = flexion[0, 1]
        q[2] = -abduction[0] * ABDUCTION_SCALE
        q[3] = flexion[0, 2]
        # Fingers: rows 1-4, cols [MCP, PIP, DIP].
        for row, finger_offset in enumerate(range(4, 20, 4)):
            q[finger_offset] = flexion[1 + row, 0]
            q[finger_offset + 1] = abduction[1 + row] * ABDUCTION_SCALE
            if self.couple_dip_pip:
                # split the compensated total distal bend across the 1:1
                # mimic pair; the coupled branch consumes the RAW total
                # (per-DOF gains are irrelevant here -- the distal total
                # gain from the guided calibration covers it)
                total = raw[1 + row, 1] + raw[1 + row, 2]
                gain = self.distal_total_gain if self.distal_total_gain is not None \
                    else 1.0
                coupled = 0.5 * max(0.0, gain * total - self.distal_zero_deg)
                q[finger_offset + 2] = coupled
                q[finger_offset + 3] = coupled
            else:
                q[finger_offset + 2] = flexion[1 + row, 1]
                q[finger_offset + 3] = flexion[1 + row, 2]
        return self.clamp(q)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp a 20-joint command to the URDF limits (deg).

        Args:
            q: (20,) joint angles (deg).

        Returns:
            (20,) clipped command.
        """
        return np.clip(q, self._limits[:, 0], self._limits[:, 1])


class DexRetargetingAdapter:
    """dex-retargeting VectorOptimizer (AnyTeleop math) on the P24 URDF.

    Vectors: 5x (palm -> fingertip) + 2x (thumb tip -> index/middle tip) for
    the pinch semantics, same convention as the official teleop configs.
    The URDF mimic (DIP = PIP) is handled by the package's
    MimicJointKinematicAdaptor automatically.  Requires pinocchio +
    dex-retargeting (installed in the base env); on machines without them,
    construction raises -- use HybridRetargeter there (it implements the
    same optimization natively on the repo kinematics).
    """

    def __init__(self, scaling: float = 1.4, low_pass_alpha: float = 0.5):
        from dex_retargeting.retargeting_config import (  # noqa: E402
            RetargetingConfig,
        )
        from p24grasp.paths import urdf_path  # noqa: E402

        config = RetargetingConfig.from_dict({
            "type": "vector",
            "urdf_path": str(urdf_path()),
            # exclude the four mimicked DIP joints (package requirement)
            "target_joint_names": [name for name in P24_JOINT_NAMES
                                   if not name.endswith("joint_4")],
            "target_origin_link_names": ["palm"] * 5 + ["thumb_tip_link"] * 2,
            "target_task_link_names": [
                "index_tip_link", "middle_tip_link", "ring_tip_link",
                "little_tip_link", "thumb_tip_link",
                "index_tip_link", "middle_tip_link",
            ],
            "target_link_human_indices": [
                [0, 0, 0, 0, 0, 4, 4],
                [8, 12, 16, 20, 4, 8, 12],
            ],
            "scaling_factor": scaling,
            "low_pass_alpha": low_pass_alpha,
        })
        self._seq = config.build()
        # dof_joint_names excludes zero-DOF dummy joints (e.g. 'universe')
        joint_names = list(self._seq.optimizer.robot.dof_joint_names)
        self._order = np.array([joint_names.index(name) for name in P24_JOINT_NAMES])
        # the 7 human vector pairs: (origin, task) keypoint indices,
        # 5x (wrist -> fingertip) + 2x (thumb tip -> index/middle tip),
        # matching target_origin/task_link_names above
        self._origin_ids = np.array([0, 0, 0, 0, 0, 4, 4])
        self._task_ids = np.array([8, 12, 16, 20, 4, 8, 12])
        self._q20_prev = np.zeros(20)
        self._tip_ids = np.array([4, 8, 12, 16, 20])

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        """One dex-retargeting optimization step: human tip vectors ->
        20-joint URDF command (deg).

        The optimizer consumes the 7 human vectors (wrist -> fingertips,
        thumb tip -> index/middle tips), NOT the raw keypoints -- it
        subtracts them from the robot's FK vectors directly.  The
        vectors are expressed in the robot base frame via the same
        per-frame palm fit + target rotation as the hybrid retargeter
        (fingers +z); the config's scaling_factor (1.4) covers the
        hand-size difference.

        Args:
            angles: filtered human :class:`HandAngles` (unused; the
                dex-retargeting optimizer works on tip vectors only).
            keypoints3d: (21, 3) camera-frame keypoints; all-NaN yields a
                hold of the previous command.

        Returns:
            (20,) URDF-order joint command in degrees (mimic applied by
            the package's MimicJointKinematicAdaptor).
        """
        # frame inputs (wrist / middle MCP / thumb CMC) are as critical
        # as the tips: without them the palm-frame fit is garbage and
        # the vectors would twist the robot hand
        needed = [0, 9, 1, *self._tip_ids]
        if keypoints3d is None or not np.isfinite(keypoints3d[needed]).all():
            return self._q20_prev.copy()
        frame = fit_palm_frame(keypoints3d)  # camera -> palm rotation
        wrist = keypoints3d[0]
        palm = (frame @ (keypoints3d - wrist).T).T  # fingers +x, thumb +y
        human_vecs = palm[self._task_ids] - palm[self._origin_ids]
        # palm frame (fingers +x) -> robot base frame (fingers +z):
        # the same rotation the hybrid retargeter applies to its targets
        human_vecs = human_vecs @ _TipSpaceRetargeter._TARGET_ROTATION.T  # pylint: disable=protected-access
        fixed = np.zeros(len(self._seq.optimizer.idx_pin2fixed))
        qpos = self._seq.retarget(human_vecs, fixed_qpos=fixed)  # (dof,) rad
        # the mimic adaptor can push a joint a hair past its URDF bound
        q20 = np.clip(np.degrees(qpos[self._order]),
                      P24_LIMITS[:, 0], P24_LIMITS[:, 1])
        self._q20_prev = q20
        return q20


class GeoRtAdapter:
    """GeoRT neural retargeter: see p24grasp.teleop.geort.GeoRtRetargeter."""

    def __init__(self, model_path: str | None = None):  # pylint: disable=unused-argument
        raise NotImplementedError(
            "use p24grasp.teleop.geort.GeoRtRetargeter (trained in simulation)"
        )
