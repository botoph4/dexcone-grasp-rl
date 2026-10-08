"""TeleopPipeline: detect -> angles -> state machine -> retarget -> command slew.

The pipeline's command rate limit (300 deg/s, matching the state machine's
slew) is what makes a re-entering hand transition smoothly from the held
pose instead of snapping like a cold start -- these tests pin that down
with fake sources/detectors/retargeters.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.camera import Frame  # noqa: E402
from p24grasp.teleop.detector import HandDetection  # noqa: E402
from p24grasp.teleop.pipeline import TeleopPipeline  # noqa: E402


class _FakeSource:
    """A fixed-rate stream of ``n`` frames at 60 fps."""

    def __init__(self, n: int):
        self._n = n
        self._ts = 0.0

    def read(self):
        if self._n <= 0:
            return None
        self._n -= 1
        self._ts += 1.0 / 60.0
        return Frame(color=np.zeros((4, 4, 3), np.uint8),
                     depth=np.full((4, 4), 0.3, np.float32),
                     ts=self._ts, fx=1.0, fy=1.0, cx=2.0, cy=2.0)

    def close(self):
        pass


class _FakeDetector:
    """A detector whose hand presence the test toggles."""

    def __init__(self):
        self.hand = True

    def detect(self, frame):
        if self.hand:
            return HandDetection(keypoints3d=np.zeros((21, 3)),
                                 visibility=np.ones(21), presence=1.0)
        return HandDetection(keypoints3d=np.full((21, 3), np.nan),
                             visibility=np.zeros(21), presence=0.0)

    def close(self):
        pass


class _JumpRetargeter:
    """Outputs 80 deg on the thumb CMC whenever keypoints exist."""

    def retarget(self, angles, keypoints3d=None):
        q = np.zeros(20)
        q[0] = 80.0
        return q


class _SwitchRetargeter:
    """Outputs ``target`` deg while the hand is present; holds otherwise."""

    def __init__(self):
        self.target = 80.0
        self._prev = np.zeros(20)

    def retarget(self, angles, keypoints3d=None):
        if keypoints3d is None or not np.isfinite(keypoints3d).any():
            return self._prev.copy()
        q = np.zeros(20)
        q[0] = self.target
        self._prev = q
        return q


def test_command_change_is_rate_limited():
    pipeline = TeleopPipeline(_FakeSource(40), _FakeDetector(),
                              retargeter=_JumpRetargeter())
    prev = np.zeros(20)
    steps = []
    for _ in range(40):
        out = pipeline.step()
        steps.append(float(np.abs(out.command - prev).max()))
        prev = out.command
    pipeline.close()
    # 300 deg/s at 60 fps: at most 5 deg per joint per frame
    assert max(steps) <= 5.0 + 1e-6
    # and the command eventually reaches the retargeter's output
    np.testing.assert_allclose(out.command[0], 80.0, atol=5.0)


def test_absence_holds_the_command():
    det = _FakeDetector()
    ret = _SwitchRetargeter()
    pipeline = TeleopPipeline(_FakeSource(90), det, retargeter=ret)
    for _ in range(30):  # hand present: ramp to 80
        out = pipeline.step()
    assert out.command[0] > 75.0
    det.hand = False
    for _ in range(30):  # absence: the command must hold, not decay
        out = pipeline.step()
    assert out.command[0] > 75.0
    det.hand = True
    for _ in range(30):  # same gesture returns: no movement at all
        out = pipeline.step()
    assert out.command[0] > 75.0
    pipeline.close()


def test_reentry_transitions_smoothly_from_the_held_pose():
    det = _FakeDetector()
    ret = _SwitchRetargeter()
    pipeline = TeleopPipeline(_FakeSource(120), det, retargeter=ret)
    for _ in range(30):  # hand present: ramp to 80
        out = pipeline.step()
    assert out.command[0] > 75.0
    det.hand = False
    for _ in range(30):  # absence: hold at 80
        out = pipeline.step()
    assert out.command[0] > 75.0
    ret.target = 0.0  # the hand returns open: the target jumps to 0
    det.hand = True
    prev = out.command[0]
    steps = []
    for _ in range(30):
        out = pipeline.step()
        steps.append(abs(out.command[0] - prev))
        prev = out.command[0]
    # the transition from the held pose is rate-limited, not a cold-start
    # snap to the new pose
    assert max(steps) <= 5.0 + 1e-6
    assert out.command[0] < 5.0  # converged to the new (open) target
    pipeline.close()


class _ResettableRetargeter(_SwitchRetargeter):
    """Counts resets; after one it holds zeros (the open palm) when absent."""

    def __init__(self):
        super().__init__()
        self.resets = 0

    def reset(self):
        self.resets += 1
        self._prev = np.zeros(20)


def test_hand_removal_resets_to_the_open_palm():
    det = _FakeDetector()
    ret = _ResettableRetargeter()
    pipeline = TeleopPipeline(_FakeSource(70), det, retargeter=ret)
    for _ in range(30):  # hand present: ramp to 80
        out = pipeline.step()
    assert out.command[0] > 75.0
    det.hand = False
    for _ in range(3):  # n_enter hard-bad frames -> HOLD entry -> reset
        out = pipeline.step()
    assert out.filtered.state.value == "hold"
    assert ret.resets == 1
    # the command returns to the open palm, rate-limited (no snap)
    prev = out.command[0]
    steps = []
    for _ in range(30):
        out = pipeline.step()
        steps.append(abs(out.command[0] - prev))
        prev = out.command[0]
    assert max(steps) <= 5.0 + 1e-6
    np.testing.assert_allclose(out.command[0], 0.0, atol=1e-9)
    pipeline.close()
