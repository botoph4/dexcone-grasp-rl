"""Shared four-finger lateral (MCP ab/adduction) estimator for the P24 hand.

Ported from the agile-hand-retarget workspace and adapted to MediaPipe-21
keypoints in the camera frame (the OpenXR-26 metacarpal landmarks do not
exist in MediaPipe, so the reference mode is ``palm_center``).

The estimator turns one 21-keypoint frame into four robot ``*_dh_joint_2``
angles (radians) with:

- a robust palm normal fitted from the wrist and the four MCP landmarks;
- a signed lateral angle between each finger's palm-center reference and its
  proximal phalanx, measured by de-rotating the phalanx back into the palm
  plane about its flexion axis (``method="derotation"`` -- exact under
  flexion, unlike orthogonal projection which shrinks and destabilizes near
  a 90-degree fist); the confidence is the in-plane fraction of the
  proximal phalanx;
- a per-finger linear calibration ``joint_2 = gain * (angle - offset)``
  (see :func:`calibrate_lateral`);
- confidence-gated one-pole smoothing (freeze when unobservable) and
  per-finger joint limits (asymmetric for index/little: inward adduction
  collides with the neighbouring finger).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LATERAL_FINGER_NAMES = ("index", "middle", "ring", "little")

# MediaPipe-21 indices: MCP bases of the four non-thumb fingers (the
# metacarpal landmarks do not exist; the palm center substitutes).
MCP_INDICES = (5, 9, 13, 17)
PIP_INDICES = (6, 10, 14, 18)
DIP_INDICES = (7, 11, 15, 19)
PALM_FIT_INDICES = (0, 5, 9, 13, 17)  # wrist + four MCPs

# Physical P2.4 joint_2 limits in degrees (index..little).  Index/little are
# asymmetric: positive joint_2 rotates inward and collides with the
# neighbour, so inward adduction is clamped while outward abduction keeps
# the full range.
DEFAULT_LIMITS_DEG = ((-15.0, 5.0), (-15.0, 15.0), (-15.0, 15.0), (-15.0, 0.0))

# Defaults from the reference workspace configs (lateral_common.yml).
DEFAULT_CALIBRATION_PATH = Path.home() / ".cache" / "p24grasp" / "lateral_calibration.json"
DEFAULT_FILTER_ALPHA = 0.15
DEFAULT_CONFIDENCE_LOW = 0.3
DEFAULT_CONFIDENCE_HIGH = 0.6
DEFAULT_OFFSETS_DEG = (-19.0553, -12.9812, 0.4986, 7.2642)
DEFAULT_GAINS = (1.0, 1.0, 1.0, 0.7282)


@dataclass(frozen=True)
class LateralCalibration:
    """Per-finger linear lateral mapping and output limits."""

    offsets_deg: tuple[float, float, float, float]
    gains: tuple[float, float, float, float]
    limits_deg: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.offsets_deg) != 4 or len(self.gains) != 4 or len(self.limits_deg) != 4:
            raise ValueError("Lateral calibration requires four values per field.")

    @classmethod
    def from_config(cls, config: dict) -> "LateralCalibration":
        offsets = _as_tuple4(config["offsets_deg"], "offsets_deg")
        gains = _as_tuple4(config.get("gains", [1.0] * 4), "gains")
        limits = config.get("limits_deg", DEFAULT_LIMITS_DEG)
        if len(limits) != 4:
            raise ValueError("limits_deg must contain four [min, max] pairs.")
        return cls(
            offsets_deg=tuple(float(value) for value in offsets),
            gains=tuple(float(value) for value in gains),
            limits_deg=tuple((float(lo), float(hi)) for lo, hi in limits),
        )

    @property
    def offsets_rad(self) -> np.ndarray:
        return np.deg2rad(self.offsets_deg)

    @property
    def limits_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.limits_deg, dtype=np.float64))


def _as_tuple4(values, name: str):
    values = list(values)
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values.")
    return values


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-pi, pi]`` elementwise."""
    angle = np.asarray(angle, dtype=np.float64)
    return np.arctan2(np.sin(angle), np.cos(angle))


def fit_palm_normal(positions: np.ndarray, epsilon: float = 1e-9,
                    reference: np.ndarray | None = None) -> np.ndarray:
    """Fit the palm plane normal from the wrist and MCP landmarks.

    Orientation: matches ``reference`` when given (continuity across frames
    -- a hand re-entering the view at a tilted angle must NOT flip the
    normal, or every lateral angle changes sign and the fingers mirror);
    the very first frame falls back to the camera +z convention.
    """
    points = np.asarray(positions, dtype=np.float64)[list(PALM_FIT_INDICES)]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        if reference is not None:
            return np.asarray(reference, dtype=np.float64).copy()
        return np.asarray([0.0, 0.0, 1.0])
    centered = points - points.mean(axis=0)
    if np.linalg.norm(centered) < epsilon:
        if reference is not None:
            return np.asarray(reference, dtype=np.float64).copy()
        return np.asarray([0.0, 0.0, 1.0])
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if reference is not None and np.linalg.norm(reference) > 1e-9 \
            and np.dot(normal, reference) < 0.0:
        normal = -normal
    elif reference is None and normal[2] < 0.0:
        normal = -normal
    return normal


def _signed_angle(first: np.ndarray, second: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from ``first`` to ``second`` about ``axis``; 0 when degenerate."""
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    axis_norm = float(np.linalg.norm(axis))
    if min(first_norm, second_norm, axis_norm) < 1e-9:
        return 0.0
    sine = float(np.dot(axis / axis_norm, np.cross(first, second)) / (first_norm * second_norm))
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.arctan2(sine, cosine))


def _derotation_angle(
    reference_in_plane: np.ndarray, current_vector: np.ndarray, normal: np.ndarray
) -> float:
    """Lateral angle after de-rotating the phalanx back into the palm plane.

    Decompose ``current_vector`` in the orthonormal frame ``{r, axis, n}``
    where ``r`` is the in-plane reference direction, ``n`` the palm normal,
    and ``axis = n x r`` the flexion axis.  Flexion rotates the phalanx in
    the ``r-n`` plane and leaves the ``axis`` component untouched, so the
    true lateral is ``atan2(dot(cur, axis), hypot(dot(cur, r), dot(cur, n)))``
    -- exact for a rigid phalanx under flexion.
    """
    ref_norm = float(np.linalg.norm(reference_in_plane))
    if ref_norm < 1e-9:
        return 0.0
    r = reference_in_plane / ref_norm
    axis = np.cross(normal, r)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return 0.0
    axis = axis / axis_norm
    lateral = float(np.dot(current_vector, axis))
    forward = float(np.dot(current_vector, r))
    out_of_plane = float(np.dot(current_vector, normal))
    return float(np.arctan2(lateral, np.hypot(forward, out_of_plane)))


class LateralEstimator:
    """Estimate four robot lateral joint angles from a 21-keypoint frame."""

    def __init__(
        self,
        calibration: LateralCalibration | None = None,
        *,
        filter_alpha: float = DEFAULT_FILTER_ALPHA,
        confidence_low: float = DEFAULT_CONFIDENCE_LOW,
        confidence_high: float = DEFAULT_CONFIDENCE_HIGH,
        method: str = "derotation",
        auto_calibrate: bool = False,
        collect_frames: int = 60,
        calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    ) -> None:
        if method not in ("projection", "derotation"):
            raise ValueError("method must be 'projection' or 'derotation'.")
        if not 0.0 <= filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in [0, 1].")
        if not 0.0 <= confidence_low < confidence_high <= 1.0:
            raise ValueError("Expected 0 <= confidence_low < confidence_high <= 1.")
        if collect_frames < 1:
            raise ValueError("collect_frames must be positive.")
        self.calibration = calibration or LateralCalibration(
            DEFAULT_OFFSETS_DEG, DEFAULT_GAINS, DEFAULT_LIMITS_DEG)
        self.method = method
        self.filter_alpha = float(filter_alpha)
        self.confidence_low = float(confidence_low)
        self.confidence_high = float(confidence_high)
        self._value: np.ndarray | None = None
        # continuity reference for the palm normal (see fit_palm_normal):
        # prevents sign flips when the hand re-enters the view tilted
        self._last_normal: np.ndarray | None = None
        # Auto-calibration: lateral offsets are per-user.  While enabled the
        # estimator collects high-confidence frames (an open hand has maximal
        # in-plane confidence; a fist collapses it) and takes the median raw
        # angle as this user's neutral spread.  During collection the output
        # stays neutral instead of applying someone else's calibration --
        # wrong offsets collapse the middle fingers into each other.
        self._auto_calibrate = auto_calibrate
        if not self._auto_calibrate:
            # With an explicit (loaded) calibration the lateral joints start
            # at the user's natural neutral spread -- the together-gesture
            # mapping (the two-point fit sends the together pose to the
            # lower limits) -- instead of zero.
            self._value = self.calibration.limits_rad[:, 0].copy()
        self._collect_frames = collect_frames
        self._calibration_path = (
            Path(calibration_path) if calibration_path is not None
            else DEFAULT_CALIBRATION_PATH)
        self._raw_buffer: list[np.ndarray] = []
        if auto_calibrate:
            loaded = self._load_calibration()
            if loaded is not None:
                # persist across sessions: skip the startup collection
                self.calibration = loaded
                self._auto_calibrate = False
                print(f"[lateral] loaded saved calibration from "
                      f"{self._calibration_path}", flush=True)

    @classmethod
    def from_config(cls, config: dict) -> "LateralEstimator":
        return cls(
            LateralCalibration.from_config(config),
            filter_alpha=float(config.get("filter_alpha", DEFAULT_FILTER_ALPHA)),
            confidence_low=float(config.get("confidence_low", DEFAULT_CONFIDENCE_LOW)),
            confidence_high=float(config.get("confidence_high", DEFAULT_CONFIDENCE_HIGH)),
            method=config.get("method", "derotation"),
        )

    def reset(self) -> None:
        self._value = None
        self._last_normal = None
        self._raw_buffer.clear()

    def measure(self, keypoints3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw lateral angles (rad), confidence, and the palm normal."""
        positions = np.asarray(keypoints3d, dtype=np.float64)
        if positions.shape != (21, 3):
            raise ValueError(f"Expected keypoints shape (21, 3), got {positions.shape}.")
        normal = fit_palm_normal(positions, reference=self._last_normal)
        self._last_normal = normal.copy()
        palm_points = positions[list(PALM_FIT_INDICES)]
        palm_points = palm_points[np.isfinite(palm_points).all(axis=1)]
        palm_center = (palm_points.mean(axis=0) if len(palm_points)
                       else np.zeros(3))

        angles = np.zeros(4, dtype=np.float64)
        confidence = np.zeros(4, dtype=np.float64)
        for index, (mcp, pip, _) in enumerate(zip(MCP_INDICES, PIP_INDICES, DIP_INDICES)):
            reference_vector = positions[mcp] - palm_center
            current_vector = positions[pip] - positions[mcp]  # proximal phalanx
            reference_in_plane = reference_vector - np.dot(reference_vector, normal) * normal
            current_in_plane = current_vector - np.dot(current_vector, normal) * normal
            current_norm = float(np.linalg.norm(current_vector))
            if current_norm < 1e-9 or not np.isfinite(current_vector).all():
                confidence[index] = 0.0
                continue
            confidence[index] = float(
                np.clip(np.linalg.norm(current_in_plane) / current_norm, 0.0, 1.0)
            )
            if self.method == "derotation":
                angles[index] = _derotation_angle(reference_in_plane, current_vector, normal)
            else:
                angles[index] = _signed_angle(reference_in_plane, current_in_plane, normal)
        return angles, confidence, normal

    def update(self, keypoints3d: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return four calibrated, smoothed, clamped robot joint_2 values (rad)."""
        raw, confidence, normal = self.measure(keypoints3d)

        if self._auto_calibrate:
            if np.all(confidence > self.confidence_high):
                self._raw_buffer.append(raw.copy())
            if len(self._raw_buffer) < self._collect_frames:
                diagnostics = {
                    "raw_lateral_rad": raw.tolist(),
                    "lateral_confidence": confidence.tolist(),
                    "lateral_target_rad": np.zeros(4).tolist(),
                    "lateral_smoothed_rad": np.zeros(4).tolist(),
                    "palm_normal": normal.tolist(),
                    "lateral_auto_calibrating": True,
                    "lateral_collected_frames": len(self._raw_buffer),
                }
                return np.zeros(4), diagnostics
            offsets = np.median(np.asarray(self._raw_buffer), axis=0)
            self.calibration = LateralCalibration(
                offsets_deg=tuple(float(value) for value in np.rad2deg(offsets)),
                gains=(1.0, 1.0, 1.0, 1.0),
                limits_deg=self.calibration.limits_deg,
            )
            self._auto_calibrate = False
            self._save_calibration()
            print(f"[lateral] auto-calibrated neutral offsets (deg): "
                  f"{np.round(np.rad2deg(offsets), 1).tolist()}", flush=True)

        target = self.calibration.gains * wrap_angle(
            raw - self.calibration.offsets_rad)

        gate = np.clip(
            (confidence - self.confidence_low) / (self.confidence_high - self.confidence_low),
            0.0, 1.0)
        effective_alpha = self.filter_alpha * gate

        if self._value is None and not np.any(gate > 0):
            # nothing observable (degenerate palm fit / missing keypoints):
            # stay neutral instead of feeding calibration offsets into a
            # prior that would drag the optimizer sideways
            diagnostics = {
                "raw_lateral_rad": raw.tolist(),
                "lateral_confidence": confidence.tolist(),
                "lateral_target_rad": target.tolist(),
                "lateral_smoothed_rad": np.zeros(4).tolist(),
                "palm_normal": normal.tolist(),
            }
            return np.zeros(4), diagnostics

        if self._value is None:
            self._value = target.copy()
        else:
            self._value += effective_alpha * (target - self._value)

        qpos = np.clip(
            self._value, self.calibration.limits_rad[:, 0], self.calibration.limits_rad[:, 1])
        diagnostics = {
            "raw_lateral_rad": raw.tolist(),
            "lateral_confidence": confidence.tolist(),
            "lateral_target_rad": target.tolist(),
            "lateral_smoothed_rad": qpos.tolist(),
            "palm_normal": normal.tolist(),
            "lateral_auto_calibrating": False,
        }
        return qpos, diagnostics


    def _load_calibration(self) -> LateralCalibration | None:
        try:
            data = json.loads(self._calibration_path.read_text(encoding="utf-8"))
            return LateralCalibration.from_config(data)
        except (OSError, ValueError, KeyError):
            return None

    def _save_calibration(self) -> None:
        try:
            self._calibration_path.parent.mkdir(parents=True, exist_ok=True)
            self._calibration_path.write_text(
                json.dumps({
                    "offsets_deg": list(self.calibration.offsets_deg),
                    "gains": list(self.calibration.gains),
                    "limits_deg": [list(lim) for lim in self.calibration.limits_deg],
                }, indent=2),
                encoding="utf-8",
            )
            print(f"[lateral] calibration saved to {self._calibration_path}",
                  flush=True)
        except OSError as exc:
            print(f"[lateral] could not save calibration: {exc}", flush=True)


def calibrate_lateral(
    keypoints_list,
    *,
    method: str = "derotation",
    open_quantile: float = 0.2,
    target_half_range_deg: float = 12.0,
    min_gain: float = 0.2,
    max_gain: float = 1.0,
) -> LateralCalibration:
    """Fit per-finger offsets and gains from a sequence of 21-keypoint frames.

    The neutral offset is the median raw angle over the most-open frames
    (lowest mean four-finger distal bend).  The gain maps the observed spread
    of ``angle - offset`` to ``target_half_range_deg`` so natural abduction
    stays inside the robot joint_2 range.  High-confidence frames only:
    during a deep fist the in-plane projection is tiny and unstable.
    """
    from p24grasp.teleop.angles import angles_from_keypoints  # noqa: E402
    from p24grasp.teleop.detector import HandDetection  # noqa: E402

    probe = LateralEstimator(
        LateralCalibration((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0),
                           DEFAULT_LIMITS_DEG),
        filter_alpha=1.0, method=method,
    )
    raw_all, confidence_all, openness_all = [], [], []
    for keypoints in keypoints_list:
        keypoints = np.asarray(keypoints, dtype=np.float64)
        raw, confidence, _ = probe.measure(keypoints)
        raw_all.append(raw)
        confidence_all.append(confidence)
        flexion = angles_from_keypoints(HandDetection(
            keypoints3d=keypoints, visibility=np.ones(21), presence=1.0)).flexion
        openness_all.append(float(np.nanmean(flexion[1:, 1] + flexion[1:, 2])))

    raw_all = np.asarray(raw_all, dtype=np.float64)
    confidence_all = np.asarray(confidence_all, dtype=np.float64)
    openness_all = np.asarray(openness_all, dtype=np.float64)
    if raw_all.shape[0] == 0:
        raise ValueError("calibrate_lateral requires at least one frame.")

    threshold = float(np.quantile(openness_all, open_quantile))
    open_mask = openness_all <= threshold
    offsets = np.median(raw_all[open_mask], axis=0)

    confident = confidence_all >= 0.5
    residual = wrap_angle(raw_all - offsets)
    residual[~confident] = np.nan
    spread = np.nanpercentile(np.abs(residual), 95, axis=0)
    spread = np.nan_to_num(spread, nan=np.deg2rad(target_half_range_deg))
    gains = np.clip(
        np.deg2rad(target_half_range_deg) / np.maximum(spread, 1e-6), min_gain, max_gain)

    return LateralCalibration(
        offsets_deg=tuple(float(value) for value in np.rad2deg(offsets)),
        gains=tuple(float(value) for value in gains),
        limits_deg=DEFAULT_LIMITS_DEG,
    )


__all__ = [
    "DEFAULT_GAINS",
    "DEFAULT_LIMITS_DEG",
    "DEFAULT_OFFSETS_DEG",
    "LATERAL_FINGER_NAMES",
    "LateralCalibration",
    "LateralEstimator",
    "calibrate_lateral",
    "fit_palm_normal",
    "wrap_angle",
]
