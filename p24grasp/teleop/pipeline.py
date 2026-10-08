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

# Max commanded joint speed (deg/s) applied to the retargeter's output.
# The state machine already rate-limits the filtered human angles, but the
# retargeter solves against live fingertip targets, so its output can jump
# when the hand (re-)enters the view even while the angles are held --
# without this bound the robot hand snaps to the new pose like a cold
# start instead of transitioning from the held pose.  Matches the state
# machine's slew budget (docs: 假肢安全上限).
COMMAND_SLEW_DEG_S = 300.0


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
                 retargeter: Retargeter | None = None,
                 command_slew_deg_s: float = COMMAND_SLEW_DEG_S):
        """Wire the perception->mapping chain.

        Args:
            source: any :class:`FrameSource` (camera or replay).
            detector: a :class:`HandDetector` over the same frames.
            state_machine: optional occlusion state machine (default:
                one with default :class:`OcclusionParams`).
            retargeter: optional backend (default:
                :class:`HybridRetargeter`).
            command_slew_deg_s: max per-joint command speed (deg/s); the
                command starts from the neutral pose and is rate-limited
                every frame, so a re-entering hand transitions smoothly
                from the held pose instead of snapping like a cold start.
        """
        self.source = source
        self.detector = detector
        self.state_machine = state_machine or OcclusionStateMachine(OcclusionParams())
        if retargeter is None:
            from p24grasp.teleop.retarget import HybridRetargeter  # noqa: E402

            retargeter = HybridRetargeter()
        self.retargeter = retargeter
        self.command_slew_deg_s = float(command_slew_deg_s)
        self._last_ts: float | None = None
        # command history for the rate limit: starts at the neutral pose
        self._last_command = np.zeros(20)
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
        command = self._slew_limit_command(command, dt)
        self._last_command = command.copy()
        out = TeleopFrame(frame=frame, detection=detection, raw_angles=raw,
                          filtered=filtered, command=command, dt=dt)
        self.last_output = out
        return out

    def _slew_limit_command(self, command: np.ndarray, dt: float) -> np.ndarray:
        """Rate-limit the 20-joint command toward the retargeter's output.

        The retargeter solves against live fingertip targets, so its output
        can change instantly when the hand (re-)enters the view even though
        the filtered angles are held -- without this bound the robot hand
        snaps to the new pose as if the hand had just entered for the first
        time, instead of transitioning smoothly from the held pose.

        Args:
            command: the retargeter's (20,) output in degrees.
            dt: seconds since the previous frame.

        Returns:
            ``command`` moved toward the previous command by at most
            ``command_slew_deg_s * dt`` degrees per joint.
        """
        max_step = self.command_slew_deg_s * dt
        return np.clip(command - self._last_command, -max_step, max_step) \
            + self._last_command

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
