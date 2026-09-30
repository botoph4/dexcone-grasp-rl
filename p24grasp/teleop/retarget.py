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

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from p24grasp.teleop.angles import HandAngles
from p24grasp.teleop.lateral import (
    LATERAL_FINGER_NAMES,
    LateralEstimator,
    fit_palm_normal,
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
                 calibration: "HandCalibration | None" = None) -> np.ndarray:
    """DirectAngleScaling output as the 16 active DOFs (rad, active order)."""
    if calibration is not None:
        scaling = DirectAngleScaling(
            flexion_offsets_deg=calibration.flexion_offsets_deg,
            flexion_gains=calibration.flexion_gains,
            distal_total_gain=calibration.distal_total_gain,
        )
    else:
        scaling = DirectAngleScaling()
    q20 = np.radians(scaling.retarget(angles))
    out = np.zeros(hand.n_active)
    thumb_s, thumb_e = hand.chain_slices["thumb"]
    out[thumb_s:thumb_e] = q20[0:4]
    for name in ("index", "little", "middle", "ring"):
        offset = URDF_OFFSETS[name]
        start, end = hand.chain_slices[name]
        out[start:end] = q20[[offset, offset + 1, offset + 2]]
    return out


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

    def __init__(self, scale: float | None = None, max_nfev: int = 200):
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
        # robot fingertip reach at the open pose
        open_tip = self.hand.tip_point(self._chain("index"), np.zeros(3))
        self._robot_reach = float(np.linalg.norm(open_tip))
        # adaptive reach estimate: starts conservative, tracks the running max
        # of the observed human wrist->tip distances (slow decay so a hand
        # that never fully extends still converges).
        self._human_reach = 0.12 if scale is None else self._robot_reach / scale
        self._palm_frame: np.ndarray | None = None  # smoothed camera->palm rotation

    def _chain(self, name):
        for chain in self.hand.chains:
            if chain.name == name:
                return chain
        raise KeyError(name)

    def targets_from_keypoints(self, keypoints3d: np.ndarray) -> np.ndarray | None:
        """(5, 3) robot-palm-frame fingertip targets, or None when the needed
        keypoints lack depth.  Updates the adaptive reach/scale estimate."""
        needed = [self.WRIST_ID, *[self.TIP_IDS[name] for name in self._chain_names]]
        if keypoints3d is None or not np.isfinite(keypoints3d[needed]).all():
            return None
        frame = self._palm_frame_from_keypoints(keypoints3d)
        wrist = keypoints3d[self.WRIST_ID]
        rel = np.stack([keypoints3d[self.TIP_IDS[name]] - wrist
                        for name in self._chain_names])
        scale = self._update_scale(rel)
        targets = scale * (frame @ rel.T).T + self.mount_t
        return targets

    def pip_targets_from_keypoints(self, keypoints3d: np.ndarray) -> np.ndarray | None:
        """(5, 3) robot-palm-frame PIP-joint targets (Vector10's proximal half)."""
        pip_ids = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "little": 18}
        needed = [self.WRIST_ID, *[pip_ids[name] for name in self._chain_names]]
        if keypoints3d is None or not np.isfinite(keypoints3d[needed]).all():
            return None
        frame = self._palm_frame_from_keypoints(keypoints3d)
        wrist = keypoints3d[self.WRIST_ID]
        rel = np.stack([keypoints3d[pip_ids[name]] - wrist
                        for name in self._chain_names])
        scale = self._update_scale(
            np.stack([keypoints3d[self.TIP_IDS[name]] - wrist
                      for name in self._chain_names]))
        return scale * (frame @ rel.T).T + self.mount_t

    def _update_scale(self, rel: np.ndarray) -> float:
        if self.scale is None:
            observed = float(np.median(np.linalg.norm(rel, axis=1)))
            # track the running max (with slow decay) as the full-extension reach
            self._human_reach = max(self._human_reach * 0.999, observed)
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
        """
        wrist = keypoints3d[self.WRIST_ID]
        normal_reference = None if self._palm_frame is None else self._palm_frame[2]
        normal = fit_palm_normal(keypoints3d, reference=normal_reference)
        x_dir = keypoints3d[9] - wrist  # middle MCP
        x_dir -= np.dot(x_dir, normal) * normal
        x_norm = np.linalg.norm(x_dir)
        if not np.isfinite(x_dir).all() or x_norm < 1e-9:
            # degenerate input (missing keypoints): keep the previous frame
            if self._palm_frame is None:
                return np.eye(3)
            return self._palm_frame
        x_dir /= x_norm
        y_dir = keypoints3d[1] - wrist  # thumb CMC
        y_dir -= np.dot(y_dir, normal) * normal
        y_dir -= np.dot(y_dir, x_dir) * x_dir  # orthogonalize against x
        y_norm = np.linalg.norm(y_dir)
        if not np.isfinite(y_dir).all() or y_norm < 1e-9:
            if self._palm_frame is None:
                return np.eye(3)
            return self._palm_frame
        y_dir /= y_norm
        z_dir = np.cross(x_dir, y_dir)
        frame = np.stack([x_dir, y_dir, z_dir])
        if self._palm_frame is None:
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

    def _tips(self, q16: np.ndarray) -> np.ndarray:
        """(5, 3) robot fingertip positions at q16 (chain order)."""
        tips = []
        for name in self._chain_names:
            start, end = self.hand.chain_slices[name]
            tips.append(self.hand.tip_point(self._chain(name), q16[start:end]))
        return np.stack(tips)

    def _to_urdf_degrees(self, q16: np.ndarray) -> np.ndarray:
        """16 active DOFs -> 20-joint URDF order (deg), DIP slaved to PIP."""
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
        is LAST in active order but FIRST in URDF order)."""
        indices = []
        for offset in (4, 8, 12, 16):  # fingers: (MCP, abd, PIP, DIP=mimic)
            indices.extend([offset + 0, offset + 1, offset + 2])
        indices.extend([0, 1, 2, 3])  # thumb URDF row, last in active order
        return np.array(indices)

    def _q20_to_active(self, q20_deg: np.ndarray) -> np.ndarray:
        """20-joint URDF command (deg) -> 16 active DOFs (rad)."""
        return np.radians(np.asarray(q20_deg))[self._active_from_urdf()]


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
        residuals = [self._weights[:, None] * (targets - self._tips(q16))]
        residuals.append(np.sqrt(self.smooth_weight) * (q16 - self._q_prev))
        return np.concatenate([r.ravel() for r in residuals])

    def solve(self, targets: np.ndarray) -> np.ndarray:
        """Solve for the 16 active DOFs (rad); returns the 20-joint command
        in URDF order, degrees, with the mimic coupling applied."""
        from scipy.optimize import least_squares  # noqa: E402

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
        keypoints are missing (occlusion / depth loss)."""
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

    When the keypoints lack depth, the retarget degrades gracefully to the
    pure joint mapping (``retarget`` returns the scaling command).
    """

    def __init__(self, scale: float | None = None, tip_weight: float = 0.5,
                 pinch_weight: float = 2.0, reg_weight: float = 10.0, *,
                 pip_weight: float = 0.5,
                 max_nfev: int = 200,
                 prior_weights: dict | None = None,
                 lateral_calibration_path: str | Path | None = None,
                 hand_calibration_path: str | Path | None = None):
        # max_nfev=200 covers the cold start (first frame from the zero
        # pose); warm-started frames converge in a single iteration.
        # prior_weights: per-joint-class regularization strength (reference
        # workspace configs): {"mcp_flexion": 0.01, "lateral": 0.05,
        # "distal": 0.01}; thumb priors are zero (the optimizer owns the
        # thumb).  Lateral carries the strongest prior because an
        # unconstrained tip objective most often misuses joint_2.
        # reg_weight scales the whole prior block: the reference ratios are
        # preserved, and the global scale compensates for our tip-only
        # objective (their Vector10 also constrains the proximal links).
        super().__init__(scale=scale, max_nfev=max_nfev)
        self.tip_weight = tip_weight
        self.pinch_weight = pinch_weight
        self.reg_weight = reg_weight
        self.pip_weight = pip_weight
        self._weights = np.array(
            [2.0, 1.0, 1.0, 1.0, 2.0])  # index..thumb chain order (pinch side)
        prior_config = {
            "mcp_flexion": 0.01, "lateral": 0.05, "distal": 0.01,
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
                self._human_reach = self._hand_calibration.reach_m
            print(f"[retarget] loaded hand calibration "
                  f"(lateral offsets "
                  f"{np.round(self._hand_calibration.lateral.offsets_deg, 1).tolist()}, "
                  f"reach {self._hand_calibration.reach_m * 1000:.0f} mm)",
                  flush=True)
        else:
            self._lateral = LateralEstimator(
                auto_calibrate=True, calibration_path=lateral_calibration_path)

    def _make_prior_weights(self, config: dict) -> np.ndarray:
        """Per-DOF prior weights over the 16 active DOFs (rad scale).

        Thumb DOFs get zero prior (the optimizer owns the thumb); the
        four-finger classes use the reference-workspace strengths.
        """
        weights = np.zeros(self.hand.n_active)
        for name in self._chain_names:
            chain = self._chain(name)
            start, _ = self.hand.chain_slices[name]
            for offset, joint in enumerate(chain.active_joint_names):
                if joint.startswith("thumb_"):
                    weights[start + offset] = 0.0
                elif joint.endswith("_joint_1"):
                    weights[start + offset] = config["mcp_flexion"]
                elif joint.endswith("_joint_2"):
                    weights[start + offset] = config["lateral"]
                elif joint.endswith("_joint_3"):
                    weights[start + offset] = config["distal"]
        return weights

    PIP_LINK_NAMES = {
        "index": "index_ip_link", "little": "little_ip_link",
        "middle": "middle_ip_link", "ring": "ring_ip_link",
        "thumb": "thumb_ip_link",
    }

    def _pip_positions(self, q16: np.ndarray) -> np.ndarray:
        """(5, 3) robot PIP-joint positions at q16 (chain order)."""
        positions = []
        for name in self._chain_names:
            start, end = self.hand.chain_slices[name]
            transforms = self.hand.fk_chain(self._chain(name), q16[start:end])
            positions.append(transforms[self.PIP_LINK_NAMES[name]][:3, 3])
        return np.stack(positions)

    def _residuals(self, q16: np.ndarray, targets: np.ndarray,
                   q0: np.ndarray, pinch_human: float,
                   pip_targets: np.ndarray | None) -> np.ndarray:
        tips = self._tips(q16)
        residuals = [self.tip_weight * self._weights[:, None] * (targets - tips)]
        if pip_targets is not None:
            # Vector10's proximal half: constrain the middle joints so the
            # fingertips alone cannot be satisfied by odd postures
            residuals.append(
                self.pip_weight * self._weights[:, None]
                * (pip_targets - self._pip_positions(q16)))
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
        20-joint command in URDF order, degrees."""
        from scipy.optimize import least_squares  # noqa: E402

        # the prior must respect the bounds exactly: the URDF limits are
        # truncated decimals and radians(80 deg) exceeds them by ~1e-10
        q0 = np.clip(np.asarray(q0, dtype=np.float64), self._lb, self._ub)

        pinch_human = float(np.linalg.norm(
            targets[self._chain_names.index("thumb")] -
            targets[self._chain_names.index("index")]))
        # Warm-start sanity: after an occlusion the stale previous solution
        # can be far from the new targets; start from whichever of the stale
        # solution and the analytical prior has the lower residual.
        x0 = self._q_prev.copy()
        cost_stale = float(np.sum(
            self._residuals(x0, targets, q0, pinch_human, pip_targets) ** 2))
        cost_prior = float(np.sum(
            self._residuals(q0, targets, q0, pinch_human, pip_targets) ** 2))
        if cost_prior < cost_stale:
            x0 = q0.copy()
        # method="dogbox": trf stalls when the warm start sits exactly on a
        # joint bound (the zero pose is the lower bound for 12 of 16 DOFs).
        result = least_squares(
            self._residuals, x0,
            args=(targets, q0, pinch_human, pip_targets),
            bounds=(self._lb, self._ub), method="dogbox",
            jac="2-point", max_nfev=self.max_nfev)
        self._q_prev = np.clip(result.x, self._lb, self._ub)
        self._q20_prev = self._to_urdf_degrees(self._q_prev)
        return self._q20_prev.copy()

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        q0 = scaling_q16(self.hand, angles, self._hand_calibration)
        # Shared calibrated lateral: always write the estimator's robot
        # joint_2 values into the prior (it holds its calibrated initial
        # value without observations), so the no-hand fallback also starts
        # at the user's neutral spread instead of zero.
        lateral_qpos, _ = self._lateral.update(keypoints3d)
        for name in ("index", "little", "middle", "ring"):
            start, _ = self.hand.chain_slices[name]
            q0[start + 1] = lateral_qpos[LATERAL_FINGER_NAMES.index(name)]
        targets = self.targets_from_keypoints(keypoints3d)
        if targets is None:
            # depth loss: fall back to the joint mapping (graceful)
            self._q20_prev = self._to_urdf_degrees(q0)
            return self._q20_prev.copy()
        pip_targets = self.pip_targets_from_keypoints(keypoints3d)
        return self.solve(targets, q0, pip_targets)


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
    _limits = P24_LIMITS

    def _calibrated(self, flexion: np.ndarray) -> np.ndarray:
        """Apply the per-user calibration when provided (guided gestures),
        falling back to the anatomical population averages."""
        offsets = np.zeros((5, 3)) if self.flexion_offsets_deg is None \
            else np.asarray(self.flexion_offsets_deg)
        if self.flexion_gains is None:
            gains = np.vstack([THUMB_FLEX_SCALE,
                               np.tile(FINGER_FLEX_SCALE, (4, 1))])
        else:
            gains = np.asarray(self.flexion_gains)
        return (flexion - offsets) * gains

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        raw = np.nan_to_num(angles.flexion, nan=0.0)
        flexion = self._calibrated(raw)
        abduction = np.nan_to_num(angles.abduction, nan=0.0)
        q = np.zeros(20)
        # Thumb: three unsigned bends -> j1, j2, j4; j3 neutral (the hybrid
        # optimizer covers thumb abduction/opposition from the tip targets).
        q[0] = flexion[0, 0]
        q[1] = flexion[0, 1]
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
        """Clamp to URDF limits (deg)."""
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
        # camera frame (x right, y down, z forward) -> palm frame
        # (x along fingers, y across the palm, z up)
        self._mount_rot = np.array([[0.0, 0.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, -1.0, 0.0]])
        self._q20_prev = np.zeros(20)
        self._tip_ids = np.array([4, 8, 12, 16, 20])

    def retarget(self, angles: HandAngles,
                 keypoints3d: np.ndarray | None = None) -> np.ndarray:
        needed = np.concatenate([[0], self._tip_ids])
        if keypoints3d is None or not np.isfinite(keypoints3d[needed]).all():
            return self._q20_prev.copy()
        wrist = keypoints3d[0]
        ref = keypoints3d - wrist  # wrist frame, camera orientation
        ref = (self._mount_rot @ ref.T).T  # palm-frame orientation
        fixed = np.zeros(len(self._seq.optimizer.idx_pin2fixed))
        qpos = self._seq.retarget(ref, fixed_qpos=fixed)  # (dof,) rad, mimic applied
        q20 = np.degrees(qpos[self._order])
        self._q20_prev = q20
        return q20


class GeoRtAdapter:
    """GeoRT neural retargeter: see p24grasp.teleop.geort.GeoRtRetargeter."""

    def __init__(self, model_path: str | None = None):  # pylint: disable=unused-argument
        raise NotImplementedError(
            "use p24grasp.teleop.geort.GeoRtRetargeter (trained in simulation)"
        )
