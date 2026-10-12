"""Leap Motion (Gemini) backend: bone->keypoint conversion + driver glue.

The `leap` module is injected as a fake so the tests run without the
Ultraleap tracking service.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.leap import (  # noqa: E402
    LeapDetector,
    LeapSource,
    keypoints_from_hand,
)


class _V:
    """A vector-like with .x/.y/.z."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Bone:
    def __init__(self, prev, nxt):
        self._prev, self._next = prev, nxt

    def prev_joint(self):
        return self._prev

    def next_joint(self):
        return self._next


class _Digit:
    def __init__(self, bones):
        self._bones = bones

    def bones(self):
        return self._bones


def _digit(points):
    """points: 5 joint positions -> 4 bones."""
    return _Digit([_Bone(points[i], points[i + 1]) for i in range(4)])


class _Hand:
    """A synthetic hand floating above the Leap (y up), fingers along +y."""

    def __init__(self):
        self.type = "HandType.Right"
        self.arm = _Bone(_V(0.0, 0.16, 0.0), _V(0.0, 0.20, 0.0))  # -> wrist
        thumb = [_V(0.03, 0.20, 0.02), _V(0.03, 0.23, 0.0), _V(0.04, 0.26, 0.0),
                 _V(0.05, 0.28, 0.0), _V(0.05, 0.30, 0.0)]
        index = [_V(0.01, 0.22, 0.0), _V(0.0, 0.24, 0.0), _V(0.0, 0.28, 0.0),
                 _V(0.0, 0.31, 0.0), _V(0.0, 0.34, 0.0)]
        middle = [_V(-0.01, 0.22, 0.0), _V(-0.01, 0.24, 0.0), _V(-0.01, 0.28, 0.0),
                  _V(-0.01, 0.31, 0.0), _V(-0.01, 0.33, 0.0)]
        ring = [_V(-0.02, 0.21, 0.0), _V(-0.02, 0.23, 0.0), _V(-0.02, 0.26, 0.0),
                _V(-0.02, 0.29, 0.0), _V(-0.02, 0.31, 0.0)]
        little = [_V(-0.03, 0.20, 0.0), _V(-0.03, 0.22, 0.0), _V(-0.03, 0.24, 0.0),
                  _V(-0.03, 0.26, 0.0), _V(-0.03, 0.28, 0.0)]
        self.digits = [_digit(thumb), _digit(index), _digit(middle),
                       _digit(ring), _digit(little)]


def test_keypoints_from_hand_maps_bones_to_21_keypoints():
    keypoints = keypoints_from_hand(_Hand())
    assert keypoints.shape == (21, 3)
    # wrist from the arm bone, converted (x, -y, -z)
    np.testing.assert_allclose(keypoints[0], (0.0, -0.20, 0.0))
    # thumb: CMC/MP/IP/TIP
    np.testing.assert_allclose(keypoints[1], (0.03, -0.20, -0.02))
    np.testing.assert_allclose(keypoints[2], (0.03, -0.23, 0.0))
    np.testing.assert_allclose(keypoints[3], (0.04, -0.26, 0.0))
    np.testing.assert_allclose(keypoints[4], (0.05, -0.30, 0.0))
    # index: MCP/PIP/DIP/TIP (metacarpal next -> proximal next -> ...)
    np.testing.assert_allclose(keypoints[5], (0.0, -0.24, 0.0))
    np.testing.assert_allclose(keypoints[6], (0.0, -0.28, 0.0))
    np.testing.assert_allclose(keypoints[7], (0.0, -0.31, 0.0))
    np.testing.assert_allclose(keypoints[8], (0.0, -0.34, 0.0))
    # the little finger's MCP
    np.testing.assert_allclose(keypoints[17], (-0.03, -0.22, 0.0))


class _FakeConnection:
    def __init__(self):
        self.listeners = []
        self.mode = None

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        pass

    def open(self):
        return _FakeOpen()

    def set_tracking_mode(self, mode):
        self.mode = mode


class _FakeOpen:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeListenerBase:
    """Minimal stand-in for leap.Listener (the connection dispatches via
    on_event; the detector subclass overrides on_tracking_event)."""

    def on_event(self, event):
        pass


def _install_fake_leap():
    fake = types.ModuleType("leap")
    fake.Connection = _FakeConnection
    fake.TrackingMode = type("TrackingMode", (), {"Desktop": "desktop"})
    fake.Listener = _FakeListenerBase
    sys.modules["leap"] = fake
    return fake


def _event(hands):
    return types.SimpleNamespace(hands=hands)


def test_detector_converts_tracking_events():
    _install_fake_leap()
    detector = LeapDetector()
    connection = detector._connection  # pylint: disable=protected-access
    listener = connection.listeners[0]
    # no hand yet
    empty = detector.detect(None)
    assert empty.presence == 0.0
    # a hand arrives
    listener.on_tracking_event(_event([_Hand()]))
    detection = detector.detect(None)
    assert detection.presence == 1.0
    assert detection.handedness == "Right"
    np.testing.assert_allclose(detection.keypoints3d[8], (0.0, -0.34, 0.0))
    # the tracking mode was set to desktop
    assert connection.mode == "desktop"
    detector.close()


def test_source_returns_stub_frames_per_event():
    _install_fake_leap()
    detector = LeapDetector()
    source = LeapSource(detector, width=320, height=240)
    listener = detector._connection.listeners[0]  # pylint: disable=protected-access
    listener.on_tracking_event(_event([_Hand()]))
    frame = source.read()
    assert frame is not None
    assert frame.color.shape == (240, 320, 3)
    assert np.isnan(frame.depth).all()
    assert frame.ts > 0.0
    source.close()
