"""Camera sources: construction and device-less failure behavior."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.camera import (  # noqa: E402
    Frame,
    OrbbecSource,
    RealsenseSource,
    ReplaySource,
    record_frames,
)
from p24grasp.teleop.run import make_camera_source  # noqa: E402


class _StubSource:
    def __init__(self, n=3):
        self.i, self.n = 0, n

    def read(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return Frame(color=np.zeros((4, 4, 3), np.uint8), depth=np.full((4, 4), 0.3),
                     ts=self.i / 60.0, fx=1.0, fy=1.0, cx=2.0, cy=2.0)

    def close(self):
        pass


def test_camera_factory():
    rs = make_camera_source("realsense", 640, 480, 60)
    ob = make_camera_source("orbbec", 848, 480, 60)
    assert isinstance(rs, RealsenseSource)
    assert isinstance(ob, OrbbecSource)
    with pytest.raises(ValueError):
        make_camera_source("nope", 640, 480, 60)


def test_read_before_start_raises():
    for source in (RealsenseSource(), OrbbecSource()):
        with pytest.raises(RuntimeError):
            source.read()


def test_record_and_replay_roundtrip(tmp_path):
    saved = record_frames(_StubSource(), tmp_path, 3)
    assert saved == 3
    replay = ReplaySource(tmp_path)
    frames = [replay.read() for _ in range(4)]
    assert frames[0].ts == 1 / 60
    assert frames[2].depth.shape == (4, 4)
    assert frames[3] is None  # source exhausted


def test_replay_loop_restarts(tmp_path):
    record_frames(_StubSource(n=1), tmp_path, 1)
    replay = ReplaySource(tmp_path, loop=True)
    for _ in range(5):
        assert replay.read() is not None  # 1-frame loop never exhausts
