"""MANO-topology kinematic hand fitting (CPU, no external weights).

The leap path's quality edge over the RGB-D path comes from model-based
tracking: the Gemini service fits a fixed-bone-length hand model to its
stereo stream, while the MediaPipe path lifts each of the 21 keypoints
independently -- the per-point depth noise flows straight into the
targets and the mapped fingers jitter.  This fitter replicates the
model-based idea on CPU:

- a 21-joint hand skeleton in the MediaPipe/MANO topology, with rest
  offsets in the palm frame (population bone directions; the per-user
  bone LENGTHS are auto-calibrated from ~1 s of lifted keypoints);
- parameterized per frame by the camera->palm rotation (axis-angle),
  the wrist translation, and scalar flexion/abduction angles per joint
  -- the anatomically plausible subspace of MANO's 45 axis-angle
  parameters (flexion around the palm-frame thumb-side axis, abduction
  around the palm normal);
- fitted with scipy ``least_squares`` to the 2D reprojection (dominant:
  the RGB keypoints are reliable) + a soft depth agreement + a temporal
  smoothness prior + the DIP ~ 0.7*PIP coupling hint.

The full MANO blend-shape model is NOT used: its weights are gated and
the shape blend adds nothing here (the hand shape is fixed per user --
the calibrated bone lengths cover it).

The fit is warm-started from the previous frame; a failed fit (mean 2D
reprojection error above ``fallback_px``) returns the raw lifted
keypoints and keeps the warm start, so the pipeline degrades gracefully
to the pre-fitting behavior.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from p24grasp.teleop.lateral import fit_palm_frame

N_JOINTS = 21

# parent joint of each MediaPipe keypoint (the MANO tree); -1 = root
PARENTS = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11,
                    0, 13, 14, 15, 0, 17, 18, 19])

# palm-frame rest offsets (parent -> child, meters), average adult right
# hand: x along the fingers, y toward the thumb, z out of the palm.
# Only the DIRECTIONS matter for the zero pose; the lengths are
# auto-calibrated per user (the population lengths seed the fallback).
REST_OFFSETS = np.array([
    [0.000, 0.000, 0.000],  # 0 wrist
    [0.005, 0.042, 0.000],  # 1 thumb CMC
    [0.008, 0.033, -0.002],  # 2 thumb MCP
    [0.002, 0.030, 0.000],  # 3 thumb IP
    [0.000, 0.024, 0.000],  # 4 thumb tip
    [0.055, 0.021, 0.000],  # 5 index MCP
    [0.040, 0.000, 0.000],  # 6 index PIP
    [0.024, 0.000, 0.000],  # 7 index DIP
    [0.018, 0.000, 0.000],  # 8 index tip
    [0.062, 0.002, 0.000],  # 9 middle MCP
    [0.045, 0.000, 0.000],  # 10 middle PIP
    [0.028, 0.000, 0.000],  # 11 middle DIP
    [0.020, 0.000, 0.000],  # 12 middle tip
    [0.057, -0.018, 0.000],  # 13 ring MCP
    [0.041, 0.000, 0.000],  # 14 ring PIP
    [0.026, 0.000, 0.000],  # 15 ring DIP
    [0.019, 0.000, 0.000],  # 16 ring tip
    [0.048, -0.036, 0.000],  # 17 little MCP
    [0.033, 0.000, 0.000],  # 18 little PIP
    [0.019, 0.000, 0.000],  # 19 little DIP
    [0.016, 0.000, 0.000],  # 20 little tip
])

# joints with a flexion scalar (all but the tips) and with an abduction
# scalar (the thumb CMC + the four finger MCPs), in parameter order
FLEX_JOINTS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
ABD_JOINTS = (1, 5, 9, 13, 17)
N_PARAMS = 6 + len(FLEX_JOINTS) + len(ABD_JOINTS)  # rot + trans + angles
_FLEX_SLOT = {joint: 6 + i for i, joint in enumerate(FLEX_JOINTS)}
_ABD_SLOT = {joint: 6 + len(FLEX_JOINTS) + i for i, joint in enumerate(ABD_JOINTS)}
# the joint each angle parameter rotates (flexion params first, then
# abduction), matching the parameter layout
_ANGLE_PARAM_JOINTS = FLEX_JOINTS + ABD_JOINTS
N_ANGLES = len(_ANGLE_PARAM_JOINTS)

def _ancestors(joint: int) -> set[int]:
    """The joint and all its ancestors (for the subtree construction)."""
    chain = {joint}
    parent = PARENTS[joint]
    while parent >= 0:
        chain.add(int(parent))
        parent = PARENTS[int(parent)]
    return chain


# tree-order descendant lists per joint (for the angle-derivative chains);
# every parent index is below its children, so range order is tree order
_SUBTREES: tuple[tuple[int, ...], ...] = tuple(
    tuple(j for j in range(1, N_JOINTS) if k in _ancestors(j))
    for k in range(N_JOINTS))

# DIP ~ 0.7 * PIP coupling hint (per-finger pairs, weak)
_COUPLE_PAIRS = ((7, 6), (11, 10), (15, 14), (19, 18))

_DEPTH_WEIGHT = 50.0  # per meter: 1 mm of depth error costs 0.05 residual
_SMOOTH_WEIGHT = 1.0  # s/rad: ~halves the frame-rate jitter (measured
# fit/raw movement ratio 0.47 with 1 px / 3 mm noise) while a 90 deg/s
# flexion lags only ~1.4 mm; the 2D data still dominates real motion
_COUPLE_WEIGHT = 0.3  # per rad of DIP deviation from 0.7 * PIP
_DEPTH_REF_ALPHA = 0.4  # EMA rate of the depth reference (the scale anchor)


def _rot_y(theta: float) -> np.ndarray:
    """3x3 rotation around the palm-frame y axis (the flexion axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(theta: float) -> np.ndarray:
    """3x3 rotation around the palm-frame z axis (the abduction axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _d_rot_y(theta: float) -> np.ndarray:
    """d/dtheta of :func:`_rot_y`."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]])


def _d_rot_z(theta: float) -> np.ndarray:
    """d/dtheta of :func:`_rot_z`."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 cross-product matrix of ``v``."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _so3_left_jac(rotvec: np.ndarray) -> np.ndarray:
    """Left jacobian of SO(3) at ``rotvec`` (the Exp(.) derivative).

    J = I + (1-cos th)/th^2 [w]_x + (th - sin th)/th^3 [w]_x^2, with
    th = ||w|| and the small-angle limit I.
    """
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-9:
        return np.eye(3)
    a = _skew(rotvec) / theta
    return (np.eye(3) + (1.0 - np.cos(theta)) / theta * a
            + (theta - np.sin(theta)) / theta * (a @ a))


class ManoHandFitter:
    """Model-based 3D hand fitter: MediaPipe 2D + depth -> smooth,
    bone-rigid 21 keypoints (see the module docstring).

    Args:
        calib_frames: lifted-keypoint frames collected to calibrate the
            per-user bone lengths (0 = skip, use the population lengths).
        fallback_px: mean 2D reprojection error (px) above which the fit
            is rejected and the raw lifted keypoints are returned.
        smooth_weight / depth_weight: temporal-prior and depth-agreement
            strengths (see the module constants for the units).
    """

    def __init__(self, calib_frames: int = 30, fallback_px: float = 12.0,
                 smooth_weight: float = _SMOOTH_WEIGHT,
                 depth_weight: float = _DEPTH_WEIGHT):
        self.calib_frames = calib_frames
        self.fallback_px = fallback_px
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self._offsets = REST_OFFSETS.copy()
        self._calibrated = calib_frames <= 0
        self._calib_count = 0
        self._segments: list[list[float]] = [[] for _ in range(N_JOINTS)]
        self._prev: np.ndarray | None = None
        self._last_ts: float | None = None
        self._depth_ref: np.ndarray | None = None  # EMA'd depth reference
        self._lb, self._ub = self._param_bounds()

    # ------------------------------------------------------------------ #
    # parameterization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _param_bounds() -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bounds for the 26-parameter vector."""
        lb = np.empty(N_PARAMS)
        ub = np.empty(N_PARAMS)
        lb[0:3], ub[0:3] = -np.pi / 2.0, np.pi / 2.0  # palm rotation
        lb[3:6], ub[3:6] = (-0.4, -0.4, 0.02), (0.4, 0.4, 1.0)  # wrist
        for slot in _FLEX_SLOT.values():
            lb[slot], ub[slot] = np.radians(-10.0), np.radians(100.0)
        for slot in _ABD_SLOT.values():
            lb[slot], ub[slot] = np.radians(-20.0), np.radians(20.0)
        return lb, ub

    @staticmethod
    def _initial_params(lifted: np.ndarray) -> np.ndarray:
        """Cold-start parameters: the fitted palm frame, the lifted wrist
        position (or a front-of-camera default), and the open hand."""
        params = np.zeros(N_PARAMS)
        params[3:6] = np.array([0.0, 0.0, 0.3])
        finite = lifted[np.isfinite(lifted).all(axis=1)]
        if finite.size:
            # fit_palm_frame returns camera->palm (rows = palm axes in
            # camera coordinates); the FK needs the inverse, palm->camera
            frame = fit_palm_frame(lifted).T
            params[0:3] = Rotation.from_matrix(frame).as_rotvec()
        if np.isfinite(lifted[0]).all():
            params[3:6] = lifted[0]
        return params

    # ------------------------------------------------------------------ #
    # forward kinematics
    # ------------------------------------------------------------------ #
    def _fk(self, params: np.ndarray) -> np.ndarray:
        """(21, 3) camera-frame joint positions for ``params``.

        The skeleton lives in the palm frame (x fingers, y thumb side,
        z palm normal); each joint's flexion/abduction rotates ITS OWN
        frame (and therefore every bone below it), never the bone
        above -- the joint position is the pivot.  Then the palm pose
        maps everything into the camera.
        """
        rot = Rotation.from_rotvec(params[0:3]).as_matrix()
        trans = params[3:6]
        world = np.empty((N_JOINTS, 3, 3))
        pos = np.empty((N_JOINTS, 3))
        world[0] = rot
        pos[0] = trans
        for joint in range(1, N_JOINTS):
            parent = PARENTS[joint]
            local = np.eye(3)
            if joint in _ABD_SLOT:
                local = local @ _rot_z(params[_ABD_SLOT[joint]])
            if joint in _FLEX_SLOT:
                local = local @ _rot_y(params[_FLEX_SLOT[joint]])
            world[joint] = world[parent] @ local
            pos[joint] = pos[parent] + world[parent] @ self._offsets[joint]
        return pos

    def _fk_with_derivs(self, params: np.ndarray):
        """Forward pass plus the position jacobians for the residual.

        Returns:
            (pos, dpos_drot, dpos_dangle): the (21, 3) camera-frame
            positions, their (21, 3, 3) jacobian wrt the palm rotation
            (axis-angle columns), and their (21, 3, N_ANGLES) jacobian
            wrt the joint angles.  The translation jacobian is the
            identity for every joint (assembled by the caller).
        """
        rot = Rotation.from_rotvec(params[0:3]).as_matrix()
        trans = params[3:6]
        world = np.empty((N_JOINTS, 3, 3))
        pos = np.empty((N_JOINTS, 3))
        local = np.empty((N_JOINTS, 3, 3))
        world[0] = rot
        pos[0] = trans
        local[0] = np.eye(3)
        for joint in range(1, N_JOINTS):
            parent = PARENTS[joint]
            rl = np.eye(3)
            if joint in _ABD_SLOT:
                rl = rl @ _rot_z(params[_ABD_SLOT[joint]])
            if joint in _FLEX_SLOT:
                rl = rl @ _rot_y(params[_FLEX_SLOT[joint]])
            local[joint] = rl
            world[joint] = world[parent] @ rl
            pos[joint] = pos[parent] + world[parent] @ self._offsets[joint]
        # rotation columns: d(R0)/d rot_k = R0 [J_r e_k]_x (Rodrigues;
        # J_r is the right jacobian = the transpose of J_l)
        j_l = _so3_left_jac(params[0:3])
        d_world_rot = np.zeros((N_JOINTS, 3, 3, 3))
        d_pos_rot = np.zeros((N_JOINTS, 3, 3))
        for k in range(3):
            d_world_rot[0, :, :, k] = rot @ _skew(j_l[k, :])
        for joint in range(1, N_JOINTS):
            parent = PARENTS[joint]
            d_world_rot[joint] = np.einsum("abk,bc->ack",
                                           d_world_rot[parent], local[joint])
            d_pos_rot[joint] = d_pos_rot[parent] + np.einsum(
                "abk,b->ak", d_world_rot[parent], self._offsets[joint])
        # angle columns: a joint angle moves its own frame (and every
        # frame below it), never the joint position itself -- the pivot
        d_world_ang = np.zeros((N_JOINTS, 3, 3, N_ANGLES))
        d_pos_ang = np.zeros((N_JOINTS, 3, N_ANGLES))
        for k, joint in enumerate(_ANGLE_PARAM_JOINTS):
            parent = PARENTS[joint]
            if k < len(FLEX_JOINTS):
                d_local = _d_rot_y(params[6 + k])
                if joint in _ABD_SLOT:
                    d_local = _rot_z(params[_ABD_SLOT[joint]]) @ d_local
            else:
                d_local = _d_rot_z(params[6 + k])
                d_local = d_local @ _rot_y(params[_FLEX_SLOT[joint]])
            d_world_ang[joint, :, :, k] = world[parent] @ d_local
            for desc in _SUBTREES[joint]:
                if desc == joint:
                    continue  # the base case above, not the recursion
                p = PARENTS[desc]
                d_world_ang[desc, :, :, k] = (d_world_ang[p, :, :, k]
                                              @ local[desc])
                d_pos_ang[desc, :, k] = d_pos_ang[p, :, k] + \
                    d_world_ang[p, :, :, k] @ self._offsets[desc]
        return pos, d_pos_rot, d_pos_ang

    def _jacobian(self, params: np.ndarray, u_px: np.ndarray,
                  v_px: np.ndarray, z_ref: np.ndarray, *,
                  fx: float, fy: float, cx: float, cy: float,
                  dt: float) -> np.ndarray:
        """(93, 26) analytic jacobian of :meth:`_residuals` (the residual
        order matches exactly; the scipy default finite differences cost
        27 residual evaluations per jacobian -- far too slow per frame).
        """
        pos, d_pos_rot, d_pos_ang = self._fk_with_derivs(params)
        jac = np.zeros((42 + N_JOINTS + N_PARAMS + len(_COUPLE_PAIRS),
                        N_PARAMS))
        for joint in range(N_JOINTS):
            x, y, z = pos[joint]
            du = np.array([fx / z, 0.0, -fx * x / (z * z)])
            dv = np.array([0.0, fy / z, -fy * y / (z * z)])
            row_u = 2 * joint
            row_v = 2 * joint + 1
            jac[row_u, 0:3] = du @ d_pos_rot[joint]
            jac[row_u, 3:6] = du
            jac[row_u, 6:26] = du @ d_pos_ang[joint]
            jac[row_v, 0:3] = dv @ d_pos_rot[joint]
            jac[row_v, 3:6] = dv
            jac[row_v, 6:26] = dv @ d_pos_ang[joint]
            row_z = 42 + joint
            jac[row_z, 0:3] = self.depth_weight * d_pos_rot[joint, 2, :]
            jac[row_z, 3:6] = np.array([0.0, 0.0, self.depth_weight])
            jac[row_z, 6:26] = self.depth_weight * d_pos_ang[joint, 2, :]
        if self._prev is not None and dt > 0:
            jac[42 + N_JOINTS:42 + N_JOINTS + N_PARAMS,
                0:N_PARAMS] = np.eye(N_PARAMS) * (self.smooth_weight / dt)
        for m, (dip, pip) in enumerate(_COUPLE_PAIRS):
            row = 42 + N_JOINTS + N_PARAMS + m
            jac[row, _FLEX_SLOT[dip]] = _COUPLE_WEIGHT
            jac[row, _FLEX_SLOT[pip]] = -0.7 * _COUPLE_WEIGHT
        return jac

    # ------------------------------------------------------------------ #
    # bone-length calibration
    # ------------------------------------------------------------------ #
    def _calibrate_lengths(self, lifted: np.ndarray) -> bool:
        """Collect one frame of segment lengths; finish after
        ``calib_frames`` frames and return True (the median per segment
        replaces the population lengths, clipped to sane bounds)."""
        for joint in range(1, N_JOINTS):
            parent = PARENTS[joint]
            if np.isfinite(lifted[joint]).all() and np.isfinite(lifted[parent]).all():
                length = float(np.linalg.norm(lifted[joint] - lifted[parent]))
                if 0.005 <= length <= 0.1:  # 5..100 mm sanity window
                    self._segments[joint].append(length)
        self._calib_count += 1
        if self._calib_count < self.calib_frames:
            return False
        for joint in range(1, N_JOINTS):
            if not self._segments[joint]:
                continue
            length = float(np.median(self._segments[joint]))
            norm = np.linalg.norm(self._offsets[joint])
            if norm > 1e-9:
                self._offsets[joint] = self._offsets[joint] / norm * length
        self._calibrated = True
        reach = float(np.linalg.norm(
            self._offsets[9] + self._offsets[10] + self._offsets[11]
            + self._offsets[12]))
        print(f"[mano-fit] bone lengths calibrated over "
              f"{self._calib_count} frames (palm->tip {reach * 1000:.0f} mm)",
              flush=True)
        return True

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def _residuals(self, params: np.ndarray, u_px: np.ndarray,
                   v_px: np.ndarray, z_ref: np.ndarray, *,
                   fx: float, fy: float, cx: float, cy: float,
                   dt: float) -> np.ndarray:
        """Stacked residuals: 2D reprojection (42) + soft depth (21) +
        temporal smoothness (26) + DIP-PIP coupling (4)."""
        pos = self._fk(params)
        z = pos[:, 2]
        u_proj = fx * pos[:, 0] / z + cx
        v_proj = fy * pos[:, 1] / z + cy
        res2d = np.stack([u_proj - u_px, v_proj - v_px], axis=1).ravel()
        # soft depth agreement against the EMA'd reference, only where a
        # reference exists (the smoothed scale anchor -- the 2D keeps the
        # pose tracking instant, the depth is too noisy to track raw)
        res_depth = self.depth_weight * (
            z - np.where(np.isfinite(z_ref), z_ref, z))
        if self._prev is None or dt <= 0:
            res_smooth = np.zeros(N_PARAMS)
        else:
            res_smooth = self.smooth_weight * (params - self._prev) / dt
        res_couple = [
            _COUPLE_WEIGHT
            * (params[_FLEX_SLOT[dip]] - 0.7 * params[_FLEX_SLOT[pip]])
            for dip, pip in _COUPLE_PAIRS]
        return np.concatenate(
            [res2d, res_depth, res_smooth, np.asarray(res_couple)])

    def fit(self, u_px: np.ndarray, v_px: np.ndarray, lifted: np.ndarray, *,
            fx: float, fy: float, cx: float, cy: float,
            ts: float) -> np.ndarray:
        """Fit the model to one frame; returns the smooth (21, 3)
        camera-frame keypoints, or the raw ``lifted`` keypoints while
        the bone lengths calibrate or when the fit is rejected.

        Args:
            u_px/v_px: (21,) MediaPipe pixel keypoints.
            lifted: (21, 3) raw lifted keypoints (meters, camera frame;
                NaN rows where the depth window was empty) -- the depth
                reference and the fallback output.
            fx/fy/cx/cy: camera intrinsics (pixels).
            ts: frame timestamp (seconds) for the temporal prior.

        Returns:
            (21, 3) camera-frame keypoints (meters).
        """
        dt = ts - self._last_ts if self._last_ts is not None else 0.0
        self._last_ts = ts
        if not self._calibrated:
            self._calibrate_lengths(lifted)
            return lifted
        # EMA the measured depths: the depth residual is the SCALE anchor
        # (the 2D drives the pose instantaneously); the raw per-point
        # depth jitter must not pull the wrist frame-to-frame
        if self._depth_ref is None:
            self._depth_ref = lifted[:, 2].copy()
        else:
            finite = np.isfinite(lifted[:, 2])
            self._depth_ref = np.where(
                finite,
                _DEPTH_REF_ALPHA * lifted[:, 2]
                + (1.0 - _DEPTH_REF_ALPHA) * self._depth_ref,
                self._depth_ref)
        x0 = self._prev.copy() if self._prev is not None \
            else self._initial_params(lifted)
        # a degenerate cold start (e.g. a zero-depth wrist sample) can
        # sit outside the bounds -- clip it in rather than letting
        # least_squares raise
        x0 = np.clip(x0, self._lb, self._ub)
        result = least_squares(
            self._residuals, x0,
            args=(u_px, v_px, self._depth_ref),
            kwargs={"fx": fx, "fy": fy, "cx": cx, "cy": cy, "dt": dt},
            jac=self._jacobian,
            bounds=(self._lb, self._ub), method="dogbox",
            max_nfev=60)
        pos = self._fk(result.x)
        err = float(np.mean(np.hypot(
            fx * pos[:, 0] / pos[:, 2] + cx - u_px,
            fy * pos[:, 1] / pos[:, 2] + cy - v_px)))
        if err > self.fallback_px:
            # the fit diverged (garbage keypoints / fast motion): keep the
            # last good warm start and degrade to the raw lifting
            return lifted
        self._prev = result.x
        return pos

    def reset_pose(self) -> None:
        """Drop the pose warm start (the pipeline calls this when the
        hand leaves the view -- the next entry cold-starts, the same
        first-startup logic as the retargeters).  The calibrated bone
        lengths survive: the hand has not changed."""
        self._prev = None
        self._last_ts = None
        self._depth_ref = None
