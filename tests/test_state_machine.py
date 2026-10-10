"""Occlusion state machine: hold without jumps, smooth recovery."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.angles import HandAngles  # noqa: E402
from p24grasp.teleop.state_machine import (  # noqa: E402
    OcclusionParams,
    OcclusionStateMachine,
    State,
)

DT = 1 / 60


def _angles(presence: float = 1.0, flexion: float = 20.0) -> HandAngles:
    return HandAngles(flexion=np.full((5, 3), flexion),
                      abduction=np.zeros(5),
                      flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                      presence=presence)


def _warm(sm: OcclusionStateMachine, flexion: float = 60.0, n: int = 90) -> None:
    """Drive TRACKING for n frames so the slew ramp + smoothers converge."""
    for _ in range(n):
        sm.update(_angles(flexion=flexion), DT)


def test_tracking_with_full_confidence():
    sm = OcclusionStateMachine()
    out = sm.update(_angles(), DT)
    assert out.state == State.TRACKING
    assert out.tracking_ok


def test_enters_hold_after_n_bad_frames():
    sm = OcclusionStateMachine()
    _warm(sm)
    for _ in range(OcclusionParams().n_enter - 1):
        out = sm.update(_angles(presence=0.1), DT)
        assert out.state != State.HOLD
    out = sm.update(_angles(presence=0.1), DT)
    assert out.state == State.HOLD


def test_hold_returns_to_the_rest_pose():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=60.0)
    for _ in range(OcclusionParams().n_enter):
        out = sm.update(_angles(presence=0.0, flexion=60.0), DT)
    steps = []
    prev = out.angles.flexion[0, 0]  # the HOLD-entry frame already stepped
    for _ in range(30):  # 0.5 s in HOLD
        out = sm.update(_angles(presence=0.0, flexion=60.0), DT)
        steps.append(abs(out.angles.flexion[0, 0] - prev))
        prev = out.angles.flexion[0, 0]
    assert out.state == State.HOLD
    assert max(steps) <= 5.0 + 1e-6  # rate-limited, no snap
    np.testing.assert_allclose(out.angles.flexion, 0.0)  # open palm


def test_reset_clears_filter_history():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=60.0)
    sm.reset()
    assert sm._one_euro._prev_x is None  # pylint: disable=protected-access
    np.testing.assert_allclose(sm._ema._prev, 0.0)  # pylint: disable=protected-access


def test_recovery_has_no_jump():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=60.0)
    for _ in range(OcclusionParams().n_enter + 30):  # 0.5 s hold
        sm.update(_angles(presence=0.0, flexion=60.0), DT)
    before = sm.update(_angles(presence=0.0, flexion=60.0), DT)
    after = None
    for _ in range(OcclusionParams().n_exit):
        after = sm.update(_angles(presence=1.0, flexion=60.0), DT)
    assert after.state == State.TRACKING
    step = np.max(np.abs(after.angles.flexion - before.angles.flexion))
    # one frame at 60 fps may move at most slew_max * dt = 5 deg
    assert step <= 5.0 + 1e-6


def test_lost_commands_the_rest_pose():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=60.0)
    for _ in range(OcclusionParams().n_enter):
        sm.update(_angles(presence=0.0, flexion=60.0), DT)
    out = None
    for _ in range(200):  # past hold_max (3 s)
        out = sm.update(_angles(presence=0.0, flexion=60.0), DT)
    assert out.state == State.LOST
    assert not out.tracking_ok
    # the hand left the view: the command is the open palm (rest pose)
    np.testing.assert_allclose(out.angles.flexion, 0.0)


def test_recovery_does_not_require_full_dof_visibility():
    # a re-entering pinch hides ~half the DOFs (the thumb covers the
    # index/middle bases), so the exit must count presence-only frames --
    # the full-visibility rule would leave the machine stuck in HOLD forever
    sm = OcclusionStateMachine()
    _warm(sm, flexion=60.0)
    for _ in range(OcclusionParams().n_enter + 200):  # into LOST
        sm.update(_angles(presence=0.0, flexion=60.0), DT)
    assert sm.state == State.LOST
    after = None
    for _ in range(OcclusionParams().n_exit):
        angles = _angles(presence=1.0, flexion=60.0)
        angles.flexion_vis[:] = 0.3  # pinch-like: every DOF below min_vis
        angles.abduction_vis[:] = 0.3
        after = sm.update(angles, DT)
    assert after.state != State.LOST


def test_degraded_partial_visibility_holds_hidden_dof():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=30.0)
    angles = _angles(flexion=80.0)
    angles.flexion_vis[:, 1:3] = 0.0  # all PIP+DIP hidden: 10/20 DOFs
    angles.abduction_vis[:] = 0.0  # 15/20 DOFs hidden -> degraded
    out = sm.update(angles, DT)
    assert out.state == State.DEGRADED
    # hidden DIP moves much less than the visible MCP
    dip_move = abs(out.angles.flexion[1, 2] - 30.0)
    mcp_move = abs(out.angles.flexion[1, 0] - 30.0)
    assert dip_move < mcp_move


def test_tracking_tolerates_a_few_hidden_dofs():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=30.0)
    angles = _angles(flexion=80.0)
    angles.flexion_vis[:, 2] = 0.0  # only DIPs hidden: 15/20 DOFs visible
    out = sm.update(angles, DT)
    assert out.state == State.TRACKING


def test_recovery_allows_partial_visibility():
    sm = OcclusionStateMachine()
    _warm(sm, flexion=60.0)
    for _ in range(OcclusionParams().n_enter + 200):  # into LOST
        sm.update(_angles(presence=0.0, flexion=60.0), DT)
    assert sm.state == State.LOST
    after = None
    for _ in range(OcclusionParams().n_exit):
        angles = _angles(presence=1.0, flexion=60.0)
        angles.flexion_vis[:, 2] = 0.0  # fingertips hidden: still recoverable
        after = sm.update(angles, DT)
    assert after.state == State.TRACKING


def test_tracking_holds_hidden_dofs():
    # regression: TRACKING used to filter every DOF regardless of the
    # per-DOF visibility, so an index covered by the thumb still followed
    # the contaminated raw angles and coupled the robot index to the thumb
    sm = OcclusionStateMachine()
    _warm(sm, flexion=30.0)
    angles = _angles(flexion=80.0)
    angles.flexion_vis[:, 2] = 0.0  # only the DIPs hidden: still TRACKING
    out = sm.update(angles, DT)
    assert out.state == State.TRACKING
    dip = out.angles.flexion[1, 2]
    assert abs(dip - 30.0) < abs(dip - 80.0)  # the hidden DIP holds
    assert out.angles.flexion[1, 0] > 30.0  # the visible MCP still tracks
