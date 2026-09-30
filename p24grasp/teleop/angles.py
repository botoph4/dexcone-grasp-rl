"""Human hand joint angles from 3D keypoints (MediaPipe 21-joint layout).

Angles are scale/translation invariant, which is what makes wrist-mounted
angle driving robust: only the relative directions of adjacent bones matter.
Layout:

    flexion    (5, 3) degrees: row 0 = thumb [CMC, MP, IP],
                               rows 1-4 = index..little [MCP, PIP, DIP]
    abduction  (5,)   degrees: thumb CMC abduction, then finger MCP abductions
                               (signed; positive = away from the middle finger)

Flexion: four-finger MCP is measured against the fitted palm plane
(BEHAVIOR-style, removes the fan-geometry bias); PIP/DIP and the thumb
CMC/MP/IP are inter-segment angles (0 = straight, 90 = right-angle bend).

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
    """Flexion at ``center`` = angle between proximal and distal bone
    directions (0 = straight, 90 = right-angle bend); NaN-safe."""
    a = p - center  # from the joint back toward the proximal keypoint
    b = q - center  # from the joint toward the distal keypoint
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return 180.0 - np.degrees(np.arccos(cos))


def _signed_angle_deg(dir_vec: np.ndarray, ref_vec: np.ndarray,
                      plane_normal: np.ndarray) -> float:
    """Signed angle from ``ref_vec`` to ``dir_vec`` around ``plane_normal`` (deg)."""
    nd, nr = np.linalg.norm(dir_vec), np.linalg.norm(ref_vec)
    if nd < 1e-9 or nr < 1e-9:
        return np.nan
    cos = float(np.clip(np.dot(dir_vec, ref_vec) / (nd * nr), -1.0, 1.0))
    sign = float(np.sign(np.dot(plane_normal, np.cross(ref_vec, dir_vec))))
    return sign * np.degrees(np.arccos(cos))


def _palm_plane(kp: np.ndarray) -> np.ndarray:
    """Unit normal of the best-fit palm plane (wrist + 5 MCPs).

    Oriented toward the camera (+z), so signed abduction around it has a
    stable sign convention.  Keypoint rows containing NaN are skipped.
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


def angles_from_keypoints(det: HandDetection) -> HandAngles:
    """Compute flexion + abduction angles from a 21-keypoint detection.

    Four-finger MCP flexion is measured relative to the fitted palm plane
    (BEHAVIOR-style); PIP/DIP and the thumb remain inter-segment angles.
    Joints whose keypoints are missing yield NaN angles and 0 visibility;
    the occlusion state machine treats them as unreliable and holds/estimates.
    """
    kp = det.keypoints3d
    flexion = np.full((5, 3), np.nan)
    normal = _palm_plane(kp)
    for row, (base, mid, dip, tip) in enumerate(FINGER_IDS):
        if row == 0:  # thumb: three unsigned inter-segment bends
            flexion[row, 0] = _bone_angle_deg(kp[WRIST], kp[base], kp[mid])
        else:  # four fingers: palm-plane MCP flexion
            flexion[row, 0] = signed_flexion_from_palm_plane(kp[mid] - kp[base], normal)
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
    """Apply the anatomical DIP ~= k*PIP coupling where DIP is unreliable.

    Blends the measured DIP toward k*PIP weighted by (1 - vis): a hidden DIP
    inherits the coupled estimate, a fully visible DIP is left untouched.
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
    """Clamp all angles to anatomical ranges (NaN-safe: NaN stays NaN)."""
    out = HandAngles(flexion=angles.flexion.copy(), abduction=angles.abduction.copy(),
                     flexion_vis=angles.flexion_vis.copy(),
                     abduction_vis=angles.abduction_vis.copy(),
                     presence=angles.presence)
    out.flexion = np.clip(out.flexion, 0.0, HUMAN_FLEXION_ROM)
    out.abduction = np.clip(out.abduction, -HUMAN_ABDUCTION_ROM, HUMAN_ABDUCTION_ROM)
    return out
