"""Guided per-user hand calibration (five-finger gestures).

Three gestures give everything the mapping needs:

  open     fingers extended and spread  -> max abduction (lateral upper
            bound), zero-flexion baseline, hand reach
  together fingers adducted together   -> min abduction (lateral lower
            bound = the natural neutral spread)
  fist     full flexion                -> max flexion per joint class

The lateral estimator is fitted two-point: the measured [together, open]
angles map exactly onto the robot joint_2 limits, so the user's own spread
range is used end to end.  Flexion gains map the measured [open, fist]
ranges onto the robot ROM (MCP 80 deg, coupled PIP=DIP 80 deg each, thumb
67/80/80 deg), replacing the anatomical population averages.

Results persist to ~/.cache/p24grasp/hand_calibration.json and are loaded
automatically by the retargeting backends (auto-calibration remains the
fallback when the file is absent).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from p24grasp.teleop.lateral import (
    DEFAULT_LIMITS_DEG,
    LateralCalibration,
    LateralEstimator,
    wrap_angle,
)

DEFAULT_PATH = Path.home() / ".cache" / "p24grasp" / "hand_calibration.json"

# Robot-side flexion ROM (deg) per joint class, from the P24 URDF.
ROBOT_MCP_ROM = 80.0
ROBOT_DISTAL_ROM = 80.0  # each of the coupled PIP/DIP pair
ROBOT_THUMB_ROM = (67.0, 80.0, 80.0)  # j1, j2, j4
DISTAL_ZERO_DEG = 10.0337  # robot zero-pose neutral distal bend

GESTURES = (
    ("open", "五指伸直并张开(侧摆最大),掌心朝向相机"),
    ("together", "五指并拢(侧摆最小)"),
    ("fist", "握拳(屈曲最大)"),
)


@dataclass
class HandCalibration:
    """Per-user calibration derived from the guided gestures."""

    lateral: LateralCalibration
    flexion_offsets_deg: np.ndarray  # (5, 3) zero-flexion baseline per DOF
    flexion_gains: np.ndarray  # (5, 3) robot ROM / measured range per DOF
    distal_total_gain: float  # for the coupled PIP=DIP split formula
    reach_m: float  # wrist->tip distance at the open gesture
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "lateral": {
                "offsets_deg": list(self.lateral.offsets_deg),
                "gains": list(self.lateral.gains),
                "limits_deg": [list(lim) for lim in self.lateral.limits_deg],
            },
            "flexion_offsets_deg": self.flexion_offsets_deg.tolist(),
            "flexion_gains": self.flexion_gains.tolist(),
            "distal_total_gain": self.distal_total_gain,
            "reach_m": self.reach_m,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HandCalibration":
        return cls(
            lateral=LateralCalibration.from_config(data["lateral"]),
            flexion_offsets_deg=np.asarray(data["flexion_offsets_deg"], dtype=np.float64),
            flexion_gains=np.asarray(data["flexion_gains"], dtype=np.float64),
            distal_total_gain=float(data["distal_total_gain"]),
            reach_m=float(data["reach_m"]),
            created_at=data.get("created_at", ""),
        )


def save_hand_calibration(calibration: HandCalibration,
                          path: str | Path = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    print(f"[calibrate] hand calibration saved to {path}")


def load_hand_calibration(path: str | Path = DEFAULT_PATH) -> HandCalibration | None:
    try:
        return HandCalibration.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return None


def compute_calibration(
    open_flexion: np.ndarray,  # (5, 3) median flexion at the open gesture
    open_lateral: np.ndarray,  # (4,) median raw lateral at the open gesture
    together_lateral: np.ndarray,  # (4,) median raw lateral at the together gesture
    fist_flexion: np.ndarray,  # (5, 3) median flexion at the fist gesture
    reach_m: float,
) -> HandCalibration:
    """Fit the lateral two-point mapping and the per-joint flexion gains."""
    # Lateral: [together, open] raw angles -> robot [lo, hi] limits exactly.
    limits = np.deg2rad(np.asarray(DEFAULT_LIMITS_DEG, dtype=np.float64))
    lo, hi = limits[:, 0], limits[:, 1]
    span_raw = wrap_angle(open_lateral - together_lateral)
    span_raw = np.where(np.abs(span_raw) < 1e-3, 1e-3, span_raw)
    gains = (hi - lo) / span_raw
    offsets = together_lateral - lo / gains
    lateral = LateralCalibration(
        offsets_deg=tuple(float(value) for value in np.rad2deg(offsets)),
        gains=tuple(float(value) for value in np.clip(gains, 0.1, 3.0)),
        limits_deg=DEFAULT_LIMITS_DEG,
    )

    # Flexion: measured [open, fist] range -> robot ROM per DOF.
    offsets_flex = np.nan_to_num(open_flexion, nan=0.0)
    measured_range = np.maximum(np.nan_to_num(fist_flexion, nan=0.0) - offsets_flex, 1.0)
    flexion_gains = np.ones((5, 3), dtype=np.float64)
    for row in range(1, 5):  # fingers: MCP 80 deg
        flexion_gains[row, 0] = ROBOT_MCP_ROM / measured_range[row, 0]
        flexion_gains[row, 1] = 1.0  # PIP/DIP handled by the distal total gain
        flexion_gains[row, 2] = 1.0
    thumb_rom = np.asarray(ROBOT_THUMB_ROM, dtype=np.float64)
    flexion_gains[0] = np.clip(thumb_rom / measured_range[0], 0.1, 3.0)
    # coupled distal: 0.5 * (g * total - 10.03) = 80 at the fist
    total_measured = np.maximum(
        np.median((fist_flexion[1:, 1] + fist_flexion[1:, 2])
                  - (offsets_flex[1:, 1] + offsets_flex[1:, 2])),
        1.0,
    )
    distal_total_gain = (2.0 * ROBOT_DISTAL_ROM + DISTAL_ZERO_DEG) / total_measured

    return HandCalibration(
        lateral=lateral,
        flexion_offsets_deg=offsets_flex,
        flexion_gains=flexion_gains,
        distal_total_gain=float(np.clip(distal_total_gain, 0.1, 3.0)),
        reach_m=float(reach_m),
    )


class GuidedCalibration:
    """Collect gesture statistics from the live camera and fit the
    calibration; prints prompts and a countdown between gestures."""

    def __init__(self, source, detector, seconds_per_gesture: float = 3.0,
                 path: str | Path = DEFAULT_PATH):
        self.source = source
        self.detector = detector
        self.seconds_per_gesture = seconds_per_gesture
        self.path = Path(path)
        self._lateral_probe = LateralEstimator(
            LateralCalibration((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0),
                               DEFAULT_LIMITS_DEG),
            filter_alpha=1.0,
        )

    def run(self) -> HandCalibration:
        from p24grasp.teleop.angles import angles_from_keypoints  # noqa: E402
        from p24grasp.teleop.pipeline import TeleopPipeline  # noqa: E402

        pipeline = TeleopPipeline(self.source, self.detector)
        stats: dict[str, dict] = {}
        try:
            for name, prompt in GESTURES:
                print(f"\n[calibrate] 手势「{name}」:{prompt}", flush=True)
                for count in range(3, 0, -1):
                    print(f"[calibrate]   {count}...", flush=True)
                    time.sleep(1.0)
                flexion_list, lateral_list, reach_list = [], [], []
                end = time.monotonic() + self.seconds_per_gesture
                while time.monotonic() < end:
                    out = pipeline.step()
                    if out.frame is None:
                        continue
                    if out.detection.presence <= 0:
                        continue
                    angles = angles_from_keypoints(out.detection)
                    flexion_list.append(angles.flexion)
                    raw_lat, confidence, _ = self._lateral_probe.measure(
                        out.detection.keypoints3d)
                    if np.all(confidence > 0.6):
                        lateral_list.append(raw_lat)
                    tips = out.detection.keypoints3d[[4, 8, 12, 16, 20]]
                    wrist = out.detection.keypoints3d[0]
                    reach_list.append(
                        np.median(np.linalg.norm(tips - wrist, axis=1)))
                if not flexion_list:
                    raise RuntimeError(
                        f"gesture '{name}' collected no frames -- keep the hand "
                        "in the camera view")
                stats[name] = {
                    "flexion": np.nanmedian(np.asarray(flexion_list), axis=0),
                    "lateral": np.median(np.asarray(lateral_list), axis=0)
                    if lateral_list else np.zeros(4),
                    "reach": float(np.median(np.asarray(reach_list))),
                }
                print(f"[calibrate]   「{name}」采集完成 "
                      f"({len(flexion_list)} 帧)", flush=True)
        finally:
            pipeline.close()

        calibration = compute_calibration(
            open_flexion=stats["open"]["flexion"],
            open_lateral=stats["open"]["lateral"],
            together_lateral=stats["together"]["lateral"],
            fist_flexion=stats["fist"]["flexion"],
            reach_m=stats["open"]["reach"],
        )
        save_hand_calibration(calibration, self.path)
        print("[calibrate] 标定摘要:")
        print(f"  侧摆偏移(deg): {np.round(calibration.lateral.offsets_deg, 1)}")
        print(f"  侧摆增益:      {np.round(calibration.lateral.gains, 2)}")
        print(f"  MCP 屈曲增益:  {np.round(calibration.flexion_gains[1:, 0], 2)}")
        print(f"  远端总增益:    {calibration.distal_total_gain:.2f}")
        print(f"  手部伸展半径:  {calibration.reach_m * 1000:.0f} mm")
        return calibration
