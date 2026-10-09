"""End-to-end teleoperation pipeline: source -> detect -> angles -> machine -> retarget."""
from __future__ import annotations

import threading
import time
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
        self._in_absence = False
        self.last_output: TeleopFrame | None = None
        # per-stage durations (ms, EMA) for the frequency diagnostics:
        # read (camera wait included), detect, angles+state, retarget
        self._timing_ms = {"read": 0.0, "detect": 0.0,
                           "angles": 0.0, "retarget": 0.0}

    @property
    def timing_ms(self) -> dict:
        """Per-stage durations in ms (exponential moving averages), used by
        the viewers' frequency statistics: read / detect / angles+state /
        retarget."""
        return dict(self._timing_ms)

    def _update_timing(self, read_s: float, detect_s: float,
                       angles_s: float, retarget_s: float) -> None:
        for key, value in (("read", read_s), ("detect", detect_s),
                           ("angles", angles_s), ("retarget", retarget_s)):
            ms = value * 1000.0
            self._timing_ms[key] = ms if not self._timing_ms[key] else \
                0.9 * self._timing_ms[key] + 0.1 * ms

    def step(self) -> TeleopFrame:
        """Read one frame and run detect -> angles -> state machine ->
        retarget.

        Returns:
            :class:`TeleopFrame` with the frame (None = source exhausted),
            detection, raw/filtered angles, the 20-joint command, and dt.
        """
        t0 = time.perf_counter()
        frame = self.source.read()
        dt = self._dt(frame)
        t1 = time.perf_counter()
        detection = self.detector.detect(frame) if frame is not None else \
            HandDetection(keypoints3d=np.full((21, 3), np.nan),
                          visibility=np.zeros(21), presence=0.0)
        t2 = time.perf_counter()
        raw = angles_from_keypoints(detection)
        filtered = self.state_machine.update(raw, dt)
        t3 = time.perf_counter()
        if filtered.state in (State.HOLD, State.LOST):
            if not self._in_absence:
                # the hand left the view: reset to the open palm (the state
                # machine slews the angles to the rest pose) and re-run the
                # first-startup logic when it comes back (fresh filters and
                # mapping state)
                self._in_absence = True
                self.state_machine.reset()
                reset = getattr(self.retargeter, "reset", None)
                if reset is not None:
                    reset()
        else:
            self._in_absence = False
        command = self.retargeter.retarget(filtered.angles, detection.keypoints3d)
        command = self._slew_limit_command(command, dt)
        self._last_command = command.copy()
        self._update_timing(t1 - t0, t2 - t1, t3 - t2, time.perf_counter() - t3)
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


class FixedRateControl:
    """Fixed-rate control loop over the teleop pipeline (producer/consumer).

    A producer thread runs the pipeline at the camera's own frame rate and
    publishes each output; ``run()`` consumes the latest output at a fixed
    ``control_hz`` (60 Hz default), re-issuing the last command on ticks
    without a newer frame (a hold), and measures both rates for the
    on-screen statistics.

    Threads: the producer owns the pipeline (detector / state machine /
    retargeter); the consumer -- the caller of ``run`` -- owns the display,
    so MuJoCo/GL contexts stay on their creation thread.
    """

    def __init__(self, pipeline: TeleopPipeline, control_hz: float = 60.0):
        """Wrap ``pipeline`` in a fixed-rate control loop.

        Args:
            pipeline: the :class:`TeleopPipeline` to run at camera rate.
            control_hz: the fixed control (tick) frequency.
        """
        self._pipeline = pipeline
        self.control_hz = float(control_hz)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: TeleopFrame | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        # consumer-side statistics
        self._last_issued_seq = -1
        self._command = np.zeros(20)
        self._control_rate = 0.0
        self._map_rate = 0.0
        self._stats = {"control_hz": self.control_hz, "control_rate": 0.0,
                       "map_rate": 0.0, "fresh": False}

    def start(self) -> None:
        """Start the producer thread (the pipeline runs at camera rate)."""
        self._thread = threading.Thread(target=self._produce, daemon=True,
                                        name="teleop-producer")
        self._thread.start()

    def _produce(self) -> None:
        """Producer: step the pipeline as fast as frames arrive."""
        try:
            while not self._stop.is_set():
                out = self._pipeline.step()
                if out.frame is None:
                    break
                with self._lock:
                    self._latest = out
                    self._seq += 1
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # close() unblocking a blocked camera read

    def run(self, apply, stop: threading.Event | None = None) -> None:
        """Fixed-rate consumer loop; blocks until ``apply`` returns False,
        ``stop`` is set, or the producer ends (source exhausted).

        Args:
            apply: callback ``apply(command, out, stats) -> bool | None``
                called once per tick with the 20-joint command (degrees),
                the latest pipeline output (repeated between new frames),
                and the statistics dict (keys: ``control_hz``,
                ``control_rate``, ``map_rate``, ``fresh``).  Return False
                to stop the loop.
            stop: optional external stop event.
        """
        self.start()
        period = 1.0 / self.control_hz
        next_tick = time.perf_counter()
        tick_prev = time.perf_counter()
        new_prev: float | None = None  # wall clock of the last NEW frame
        while not (stop is not None and stop.is_set()):
            with self._lock:
                out, seq = self._latest, self._seq
            if out is None:
                # camera not delivering yet: wait for the first frame
                next_tick = time.perf_counter() + period
                time.sleep(period)
                continue
            fresh = seq != self._last_issued_seq
            if fresh:
                self._last_issued_seq = seq
                self._command = out.command
                now = time.perf_counter()
                if new_prev is not None and now > new_prev:
                    instant = 1.0 / (now - new_prev)
                    self._map_rate = (instant if not self._map_rate
                                      else 0.9 * self._map_rate + 0.1 * instant)
                new_prev = now
            self._stats["map_rate"] = self._map_rate
            self._stats["control_rate"] = self._control_rate
            self._stats["fresh"] = fresh
            if apply(self._command, out, self._stats) is False:
                break
            # fixed-rate scheduling: sleep the remainder of the tick
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:  # the tick overran: resync the schedule
                next_tick = time.perf_counter()
            now = time.perf_counter()
            if now > tick_prev:
                instant = 1.0 / (now - tick_prev)
                self._control_rate = (instant if not self._control_rate
                                      else 0.9 * self._control_rate + 0.1 * instant)
            tick_prev = now
            if not fresh and self._thread is not None \
                    and not self._thread.is_alive():
                break  # source exhausted and nothing new will ever arrive
        self.close()

    def close(self) -> None:
        """Stop the producer and release the pipeline (idempotent)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._pipeline.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
