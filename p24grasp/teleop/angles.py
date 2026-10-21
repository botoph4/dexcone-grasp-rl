"""Human hand joint angles from 3D keypoints (MediaPipe 21-joint layout).

Angles are scale/translation invariant, which is what makes wrist-mounted
angle driving robust: only the relative directions of adjacent bones matter.
Layout:

    flexion    (5, 3) degrees: row 0 = thumb [CMC, MP, IP],
                               rows 1-4 = index..little [MCP, PIP, DIP]
    abduction  (5,)   degrees: thumb CMC abduction, then finger MCP abductions
                               (signed; positive = away from the middle finger)

Flexion: all joints (thumb CMC/MP/IP, finger MCP/PIP/DIP) are inter-segment
angles (0 = straight, 90 = right-angle bend) -- rotation-invariant and
sign-correct for any hand orientation.  The palm-plane MCP variant
(``signed_flexion_from_palm_plane``, kept for reference) flips sign when
the hand turns (the fitted normal follows the rotation), which broke the
fist measurement of the wrist-camera setup; its fan-geometry bias is
removed per user by the guided calibration's open-hand baseline instead.

Abduction is the signed angle in the palm plane between the finger ray and
the wrist->middle-MCP ray (positive = away from the middle finger).  It
carries a per-user offset (relaxed fingers show +15-35 deg, not 0); for the
robot joint-2 values prefer the calibrated LateralEstimator
(p24grasp.teleop.lateral), which subtracts neutral offsets, applies gains,
and freezes when the in-plane observation collapses during a fist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p24grasp.teleop.detector import HandDetection

FINGER_IDS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16),
              (17, 18, 19, 20))  # (MCP_or_CMC, PIP_or_MP, DIP_or_IP, TIP) per finger
WRIST = 0

# Anatomical flexion ranges (deg) used for clamping and retarget scaling.
HUMAN_FLEXION_ROM = np.array([(60.0, 80.0, 80.0),   # thumb CMC/MP/IP
                              (90.0, 110.0, 80.0),  # index MCP/PIP/DIP
                              (90.0, 110.0, 80.0),
                              (90.0, 110.0, 80.0),
                              (90.0, 110.0, 80.0)])
HUMAN_ABDUCTION_ROM = 30.0  # deg, thumb CMC / finger MCP side spread


@dataclass
class HandAngles:
    """Human joint angles + per-DOF visibility (min of contributing keypoints)."""

    flexion: np.ndarray  # (5, 3) deg
    abduction: np.ndarray  # (5,) deg
    flexion_vis: np.ndarray  # (5, 3)
    abduction_vis: np.ndarray  # (5,)
    presence: float

    @classmethod
    def zeros(cls, presence: float = 0.0) -> "HandAngles":
        return cls(flexion=np.zeros((5, 3)), abduction=np.zeros(5),
                   flexion_vis=np.zeros((5, 3)), abduction_vis=np.zeros(5),
                   presence=presence)


def _bone_angle_deg(p: np.ndarray, center: np.ndarray, q: np.ndarray) -> float:
    """Inter-segment flexion angle (deg) at ``center``, NaN-safe.

    Flexion = 180 - angle(center->p, center->q): 0 deg straight, 90 deg
    right-angle bend.  The sign-free inter-segment form is rotation
    invariant, so it reads the same for any hand orientation (unlike
    palm-plane projections, which flip when the hand turns).

    Args:
        p: proximal keypoint (3,) in the camera frame.
        center: the joint's keypoint (3,).
        q: distal keypoint (3,).

    Returns:
        Flexion angle in degrees, or NaN when either bone is degenerate
        (zero length).
    """
    a = p - center  # from the joint back toward the proximal keypoint
    b = q - center  # from the joint toward the distal keypoint
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return 180.0 - np.degrees(np.arccos(cos))


def _signed_angle_deg(dir_vec: np.ndarray, ref_vec: np.ndarray,
                      plane_normal: np.ndarray) -> float:
    """Signed angle (deg) from ``ref_vec`` to ``dir_vec`` around ``plane_normal``.

    Positive = counter-clockwise seen from the normal's tip (right-hand
    rule); used for the palm-plane abduction measurement.

    Args:
        dir_vec: measured direction (3,).
        ref_vec: reference direction (3,).
        plane_normal: rotation axis (3,), typically the palm normal.

    Returns:
        Signed angle in degrees in (-180, 180], NaN when either vector is
        degenerate.
    """
    nd, nr = np.linalg.norm(dir_vec), np.linalg.norm(ref_vec)
    if nd < 1e-9 or nr < 1e-9:
        return np.nan
    cos = float(np.clip(np.dot(dir_vec, ref_vec) / (nd * nr), -1.0, 1.0))
    sign = float(np.sign(np.dot(plane_normal, np.cross(ref_vec, dir_vec))))
    return sign * np.degrees(np.arccos(cos))


def _palm_plane(kp: np.ndarray) -> np.ndarray:
    """Unit palm-plane normal fitted from the wrist + 5 MCP keypoints.

    Least-squares plane via SVD over the valid points; the normal is
    oriented toward the camera (+z) so signed angles around it have a
    stable sign convention.  Keypoint rows containing NaN are skipped.

    Args:
        kp: (21, 3) camera-frame keypoints.

    Returns:
        (3,) unit normal; zeros when fewer than 3 valid points remain.
    """
    pts = kp[np.array([WRIST, 1, 5, 9, 13, 17])]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return np.zeros(3)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    normal = vt[2]
    if normal[2] < 0:  # palm faces the wrist-mounted camera: +z forward
        normal = -normal
    return normal


def signed_flexion_from_palm_plane(
    finger_vector: np.ndarray,
    palm_normal: np.ndarray,
    epsilon: float = 1e-9,
) -> float:
    """MCP flexion of a proximal phalanx relative to the palm plane (deg).

    Ported from the agile-hand-retarget workspace (BEHAVIOR-style mapping):
    measuring the proximal phalanx against the palm plane removes the
    fan-shaped metacarpal geometry inside the palm, which the wrist->MCP
    reference mixes into the flexion.  The palm normal points toward the
    camera (out of the palm), so flexion (toward the palm) is positive.
    NOTE: rejected for this wrist-camera setup -- when the hand turns, the
    fitted normal follows the rotation and the measured flexion flips sign.

    Args:
        finger_vector: proximal phalanx direction (3,) in the camera frame.
        palm_normal: unit palm normal (3,) pointing out of the palm.
        epsilon: degenerate-norm guard (m).

    Returns:
        Flexion in degrees (positive toward the palm), NaN on degenerate
        input.
    """
    vector = np.asarray(finger_vector, dtype=np.float64)
    normal = np.asarray(palm_normal, dtype=np.float64)
    vector_norm = float(np.linalg.norm(vector))
    normal_norm = float(np.linalg.norm(normal))
    if vector_norm < epsilon or normal_norm < epsilon:
        return np.nan
    unit_normal = normal / normal_norm
    normal_component = float(np.dot(vector, unit_normal))
    in_plane_component = vector - normal_component * unit_normal
    in_plane_norm = float(np.linalg.norm(in_plane_component))
    if in_plane_norm < epsilon:
        return np.nan
    return float(np.degrees(np.arctan2(-normal_component, in_plane_norm)))


def thumb_over_finger(keypoints3d: np.ndarray, base_index: int,
                       distance_m: float = 0.025) -> bool:
    """Whether the thumb tip covers a finger base's neighbourhood.

    Used for thumb-over-index/middle occlusion: when the thumb tip covers
    a finger's proximal joints, MediaPipe's keypoints (and the depth under
    them) belong to the thumb, so an unbent finger would measure spurious
    flexion.  The affected finger's visibility is then damped below the
    state machine's threshold so those DOFs are held instead of followed.

    Args:
        keypoints3d: (21, 3) camera-frame keypoints.
        base_index: MCP keypoint index of the finger to check (e.g. 5).
        distance_m: cover radius (m); default 0.025.

    Returns:
        True when the thumb tip lies within ``distance_m`` of the base.
    """
    kp = np.asarray(keypoints3d, dtype=np.float64)
    thumb_tip = kp[4]
    if not np.isfinite(thumb_tip).all() or not np.isfinite(kp[base_index]).all():
        return False
    return bool(np.linalg.norm(thumb_tip - kp[base_index]) < distance_m)


def angles_from_keypoints(det: HandDetection) -> HandAngles:
    """Flexion + abduction + per-DOF visibility from a 21-keypoint detection.

    Flexions (thumb CMC/MP/IP, finger MCP/PIP/DIP) are inter-segment
    angles -- rotation-invariant and sign-correct for any hand orientation
    (the palm-plane MCP formulation was rejected for this wrist-camera
    setup: when the hand turns, the fitted normal follows the rotation and
    the measured flexion flips sign).  Abduction is the signed in-palm-
    plane angle between each finger ray and the wrist->middle-MCP
    reference (positive = away from the middle finger).  The thumb-over-
    finger occlusion rule damps index/middle visibility before the per-DOF
    visibility minima are taken.

    Args:
        det: a :class:`HandDetection` with keypoints3d (21, 3), visibility
            (21,), and presence.

    Returns:
        :class:`HandAngles` with flexion (5, 3) deg, abduction (5,) deg,
        per-DOF visibility (min of the contributing keypoints), and
        presence; joints whose keypoints are missing yield NaN angles with
        0 visibility (the occlusion state machine treats them as
        unreliable and holds/estimates).
    """
    kp = det.keypoints3d
    flexion = np.full((5, 3), np.nan)
    for row, (base, mid, dip, tip) in enumerate(FINGER_IDS):
        flexion[row, 0] = _bone_angle_deg(kp[WRIST], kp[base], kp[mid])
        flexion[row, 1] = _bone_angle_deg(kp[base], kp[mid], kp[dip])
        flexion[row, 2] = _bone_angle_deg(kp[mid], kp[dip], kp[tip])

    abduction = np.full(5, np.nan)
    normal = _palm_plane(kp)
    if np.linalg.norm(normal) > 1e-9:
        ref = kp[9] - kp[WRIST]  # toward middle-finger MCP, in palm plane
        for row, (base, mid, _, _) in enumerate(FINGER_IDS):
            dir_vec = kp[mid] - kp[base]
            abduction[row] = _signed_angle_deg(dir_vec, ref, normal)
        # middle finger is its own reference: report 0 by definition
        abduction[2] = 0.0

    # Per-DOF visibility = min visibility of the keypoints that built the angle.
    vis = det.visibility
    # Thumb-over-finger occlusion: when the thumb tip covers a finger's
    # proximal joints, MediaPipe's keypoints (and the depth under them)
    # belong to the thumb, so an unbent finger measures spurious flexion.
    # Lower the affected finger's visibility below the state machine's
    # threshold so those DOFs are held instead of followed.
    for occluded_base in (5, 9):  # index, middle MCP
        if thumb_over_finger(kp, occluded_base):
            vis[occluded_base:occluded_base + 4] = np.minimum(
                vis[occluded_base:occluded_base + 4], 0.3)
    flexion_vis = np.zeros((5, 3))
    abduction_vis = np.zeros(5)
    for row, (base, mid, dip, tip) in enumerate(FINGER_IDS):
        flexion_vis[row, 0] = min(vis[WRIST], vis[base], vis[mid])
        flexion_vis[row, 1] = min(vis[base], vis[mid], vis[dip])
        flexion_vis[row, 2] = min(vis[mid], vis[dip], vis[tip])
        abduction_vis[row] = min(vis[base], vis[mid], vis[9])

    return HandAngles(flexion=flexion, abduction=abduction,
                      flexion_vis=flexion_vis, abduction_vis=abduction_vis,
                      presence=det.presence)


def dip_coupling_fixup(angles: HandAngles, k: float = 0.7) -> HandAngles:
    """Blend each DIP toward the anatomical coupling DIP ~= k*PIP.

    Blends the measured DIP toward k*PIP weighted by (1 - vis): a hidden
    DIP inherits the coupled estimate, a fully visible DIP is left
    untouched.  A kinematic fallback applied before the occlusion state
    machine (the temporal filters hold hidden DOFs afterwards).

    Args:
        angles: measured :class:`HandAngles` (may contain NaN); not
            modified.
        k: coupling constant (default 0.7).

    Returns:
        Copy with ``flexion[1:, 2]`` fixed up; NaN DIPs become k*PIP,
        clamped to [0, 90] deg.
    """
    out = HandAngles(flexion=angles.flexion.copy(), abduction=angles.abduction.copy(),
                     flexion_vis=angles.flexion_vis.copy(),
                     abduction_vis=angles.abduction_vis.copy(),
                     presence=angles.presence)
    pip = angles.flexion[1:, 1]
    dip = angles.flexion[1:, 2]
    vis = angles.flexion_vis[1:, 2]
    coupled = np.clip(k * np.where(np.isnan(pip), 0.0, pip), 0.0, 90.0)
    both = np.isfinite(dip) & np.isfinite(pip)
    fixed = np.where(both & (vis < 1.0),
                     vis * dip + (1.0 - vis) * coupled,
                     np.where(np.isnan(dip), coupled, dip))
    out.flexion[1:, 2] = fixed
    return out


def clamp_rom(angles: HandAngles) -> HandAngles:
    """Clamp all angles to anatomical ranges (NaN-safe: NaN stays NaN).

    Args:
        angles: :class:`HandAngles` to clamp (not modified).

    Returns:
        Copy with flexion clipped to [0, HUMAN_FLEXION_ROM] and abduction
        to +/-HUMAN_ABDUCTION_ROM per DOF.
    """
    out = HandAngles(flexion=angles.flexion.copy(), abduction=angles.abduction.copy(),
                     flexion_vis=angles.flexion_vis.copy(),
                     abduction_vis=angles.abduction_vis.copy(),
                     presence=angles.presence)
    out.flexion = np.clip(out.flexion, 0.0, HUMAN_FLEXION_ROM)
    out.abduction = np.clip(out.abduction, -HUMAN_ABDUCTION_ROM, HUMAN_ABDUCTION_ROM)
    return out
