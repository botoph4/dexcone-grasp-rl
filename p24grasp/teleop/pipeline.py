"""End-to-end teleoperation pipeline: source -> detect -> angles -> machine -> retarget."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p24grasp.teleop.angles import HandAngles, angles_from_keypoints
from p24grasp.teleop.camera import Frame, FrameSource
from p24grasp.teleop.detector import HandDetection, HandDetector
from p24grasp.teleop.retarget import Retargeter
from p24grasp.teleop.state_machine import (
    FilteredAngles,
    OcclusionParams,
    OcclusionStateMachine,
    State,
)


@dataclass
class TeleopFrame:
    """Everything one pipeline step produces (for logging, sim, and debugging)."""

    frame: Frame | None  # None when the source is exhausted
    detection: HandDetection
    raw_angles: HandAngles
    filtered: FilteredAngles
    command: np.ndarray  # 20-joint P24 command (deg, URDF order)
    dt: float


class TeleopPipeline:
    """Frame-by-frame wearable hand teleoperation pipeline.

    Call ``step()`` repeatedly; ``step()`` returns ``TeleopFrame`` with
    ``frame=None`` when the source ends.  Threads dt from frame timestamps
    (frame-rate independent, per the research doc).
    """

    def __init__(self, source: FrameSource, detector: HandDetector,
                 state_machine: OcclusionStateMachine | None = None,
                 retargeter: Retargeter | None = None):
        """Wire the perception->mapping chain.

        Args:
            source: any :class:`FrameSource` (camera or replay).
            detector: a :class:`HandDetector` over the same frames.
            state_machine: optional occlusion state machine (default:
                one with default :class:`OcclusionParams`).
            retargeter: optional backend (default:
                :class:`HybridRetargeter`).
        """
        self.source = source
        self.detector = detector
        self.state_machine = state_machine or OcclusionStateMachine(OcclusionParams())
        if retargeter is None:
            from p24grasp.teleop.retarget import HybridRetargeter  # noqa: E402

            retargeter = HybridRetargeter()
        self.retargeter = retargeter
        self._last_ts: float | None = None
        self.last_output: TeleopFrame | None = None

    def step(self) -> TeleopFrame:
        """Read one frame and run detect -> angles -> state machine ->
        retarget.

        Returns:
            :class:`TeleopFrame` with the frame (None = source exhausted),
            detection, raw/filtered angles, the 20-joint command, and dt.
        """
        frame = self.source.read()
        dt = self._dt(frame)
        detection = self.detector.detect(frame) if frame is not None else \
            HandDetection(keypoints3d=np.full((21, 3), np.nan),
                          visibility=np.zeros(21), presence=0.0)
        raw = angles_from_keypoints(detection)
        filtered = self.state_machine.update(raw, dt)
        command = self.retargeter.retarget(filtered.angles, detection.keypoints3d)
        out = TeleopFrame(frame=frame, detection=detection, raw_angles=raw,
                          filtered=filtered, command=command, dt=dt)
        self.last_output = out
        return out

    def _dt(self, frame: Frame | None) -> float:
        """Seconds since the previous frame (timestamp-based, frame-rate
        independent).

        Args:
            frame: the current frame, or None at end of stream.

        Returns:
            Positive dt; falls back to 1/60 s on the first frame, on
            timestamp regressions, and at end of stream.
        """
        if frame is None:
            return 1.0 / 60.0
        if self._last_ts is None:
            self._last_ts = frame.ts
            return 1.0 / 60.0
        dt = frame.ts - self._last_ts
        self._last_ts = frame.ts
        return dt if dt > 0 else 1.0 / 60.0

    def close(self) -> None:
        """Release the frame source and the detector."""
        self.source.close()
        self.detector.close()


def format_angles(angles: HandAngles, state: State, command: np.ndarray) -> str:
    """Compact human-readable per-step summary (thumb/index/middle only).

    ``--`` marks joints whose keypoints never got valid depth (unmeasured);
    the retargeter works on the raw keypoints, so ``--`` does not mean the
    hand stopped tracking.

    Args:
        angles: the filtered :class:`HandAngles` to summarize.
        state: the current state-machine state.
        command: the (20,) joint command (deg).

    Returns:
        Multi-line status string.
    """
    def row(finger: int, names: tuple[str, str, str]) -> str:
        flex = angles.flexion[finger]
        return " ".join(
            f"{n}={v:5.1f}" if np.isfinite(v) else f"{n}=  --"
            for n, v in zip(names, flex))
    lines = [
        f"state={state.value:<8} thumb  " + row(0, ("CMC", "MP", "IP")),
        "              index  " + row(1, ("MCP", "PIP", "DIP")),
        "              middle " + row(2, ("MCP", "PIP", "DIP")),
        f"q_p24(thumb/index) = {np.round(command[:8], 1)}",
    ]
    return "\n".join(lines)
