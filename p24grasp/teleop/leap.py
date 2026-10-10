"""Leap Motion (Ultraleap Gemini) hand tracking source + detector.

The Gemini tracking service delivers a 26-bone hand model: each of the
five digits contributes four bones (metacarpal, proximal, intermediate,
distal), plus the palm and the arm bone.  Bone ENDPOINTS are the joint
positions (``Bone.prev_joint`` / ``Bone.next_joint``), so the 27-joint
skeleton maps directly onto the pipeline's MediaPipe-style 21-keypoint
layout:

    wrist(0)      = arm bone next joint (the wrist)
    thumb(1-4)    = CMC, MP, IP, TIP          (digit 0)
    index(5-8)    = MCP, PIP, DIP, TIP        (digit 1)
    middle(9-12)  = digit 2, ring(13-16) = digit 3, little(17-20) = digit 4
      fingers: MCP = metacarpal.next_joint (== proximal.prev_joint)
               PIP = proximal.next_joint, DIP = intermediate.next_joint
               TIP = distal.next_joint
      thumb:   CMC = metacarpal.prev_joint, MP = metacarpal.next_joint
               IP = proximal.next_joint, TIP = distal.next_joint

Coordinates: the Leap frame is x right, y up, z toward the user; the
pipeline's camera frame is x right, y down, z forward, so keypoints are
converted with (x, -y, -z).  Everything downstream is source-agnostic:
the joint angles are inter-segment (rotation-invariant) and the palm
frame is refit from the keypoints, so a desk-mounted Leap needs no
mount calibration.

Driver requirements (official bindings):
    pip install leapc-python-bindings   (https://github.com/ultraleap/
    leapc-python-bindings) + the Gemini Ultraleap Hand Tracking service
    (https://developer.leapmotion.com/tracking-software-download) with a
    Leap Motion Controller 2.  Platforms: Windows, Linux x64, Linux ARM
    (precompiled modules for py3.8-3.11), Darwin (LeapSDK inside the
    "Ultraleap Hand Tracking" app; older precompiled modules -- build
    leapc_cffi yourself for newer Python versions).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from p24grasp.teleop.camera import Frame, FrameSource
from p24grasp.teleop.detector import HandDetection

# MediaPipe-style keypoint row per digit: (first, last) exclusive.
# digits[0]=thumb .. digits[4]=little; keypoint ids: thumb 1-4,
# index 5-8, middle 9-12, ring 13-16, little 17-20.
_DIGIT_KEYPOINT_RANGES = ((1, 5), (5, 9), (9, 13), (13, 17), (17, 21))


def _vector3(value) -> Optional[np.ndarray]:
    """Vector-like (x, y, z) -> camera-frame (x, -y, -z); None when the
    value is unavailable.  Accepts objects exposing .x/.y/.z."""
    if value is None:
        return None
    try:
        return np.array([value.x, -value.y, -value.z], dtype=np.float64)
    except AttributeError:
        return None


def _call_or_attr(obj, name):
    """The bindings expose accessors as methods; accept properties too."""
    value = getattr(obj, name, None)
    if value is None:
        return None
    return value() if callable(value) else value


def keypoints_from_hand(hand) -> np.ndarray:
    """Convert one Gemini hand to the (21, 3) keypoints in the pipeline's
    camera frame (meters); NaN rows for unavailable bones.

    Args:
        hand: a leap hand object (``hand.digits``, ``hand.arm``,
            ``hand.palm``; each digit has ``bones()`` and each bone
            ``prev_joint()`` / ``next_joint()``).

    Returns:
        (21, 3) keypoints: wrist, thumb CMC/MP/IP/TIP, index..little
        MCP/PIP/DIP/TIP.
    """
    keypoints = np.full((21, 3), np.nan)
    # wrist: the arm bone's distal endpoint; fall back to the palm center
    arm = _call_or_attr(hand, "arm")
    wrist = _vector3(_call_or_attr(arm, "next_joint")) if arm is not None else None
    if wrist is None:
        palm = _call_or_attr(hand, "palm")
        wrist = _vector3(_call_or_attr(palm, "position")) if palm is not None else None
    if wrist is not None:
        keypoints[0] = wrist

    digits = _call_or_attr(hand, "digits")
    if digits is None:
        return keypoints
    for digit_index, (first, _last) in enumerate(_DIGIT_KEYPOINT_RANGES):
        try:
            digit = digits[digit_index]
        except (IndexError, TypeError):
            continue
        bones = _call_or_attr(digit, "bones")
        if bones is None or len(bones) < 4:
            continue
        metacarpal, proximal, intermediate, distal = bones[:4]
        joints = {
            "meta_prev": _vector3(_call_or_attr(metacarpal, "prev_joint")),
            "meta_next": _vector3(_call_or_attr(metacarpal, "next_joint")),
            "prox_next": _vector3(_call_or_attr(proximal, "next_joint")),
            "int_next": _vector3(_call_or_attr(intermediate, "next_joint")),
            "dist_next": _vector3(_call_or_attr(distal, "next_joint")),
        }
        if digit_index == 0:  # thumb: CMC, MP, IP, TIP
            keypoints[first] = joints["meta_prev"]
            keypoints[first + 1] = joints["meta_next"]
            keypoints[first + 2] = joints["prox_next"]
            keypoints[first + 3] = joints["dist_next"]
        else:  # fingers: MCP, PIP, DIP, TIP
            keypoints[first] = joints["meta_next"]
            keypoints[first + 1] = joints["prox_next"]
            keypoints[first + 2] = joints["int_next"]
            keypoints[first + 3] = joints["dist_next"]
    return keypoints


class _LeapState:
    """Shared between the source (waits for events) and the detector
    (converts the latest hand); the Gemini listener runs on the
    tracking service's thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.latest = None  # the latest tracking event
        self.frame_id = 0
        self.last_ts = time.perf_counter()


class LeapDetector:
    """HandDetector-compatible wrapper over the Gemini tracking events.

    Opens the Leap connection and registers a listener; ``detect()``
    converts the latest hand into the pipeline's 21-keypoint
    :class:`HandDetection` (presence = 1.0 when a hand is tracked; the
    per-joint visibility is not exposed by the binding, so all joints
    report 1.0 and the occlusion machine relies on the finite-angle
    checks).
    """

    def __init__(self, tracking_mode: str = "Desktop"):
        try:
            import leap  # noqa: E402
        except ImportError as exc:
            raise RuntimeError(
                "Leap Motion 驱动不可用: 需要安装官方 Gemini 绑定 "
                "(pip install leapc-python-bindings) 并运行 Ultraleap "
                "Hand Tracking 服务;详见 p24grasp/teleop/leap.py 模块说明"
            ) from exc
        self._leap = leap
        self._state = _LeapState()
        self._connection = leap.Connection()
        self._connection.add_listener(self._Listener(self._state))
        # keep the connection open for the process lifetime (the binding
        # returns a context manager; enter it explicitly)
        self._open = self._connection.open()
        if hasattr(self._open, "__enter__"):
            self._open.__enter__()
        mode = getattr(leap.TrackingMode, tracking_mode, None)
        if mode is not None:
            self._connection.set_tracking_mode(mode)

    class _Listener:
        """Feeds the shared state from the service thread."""

        def __init__(self, state: "_LeapState") -> None:
            self._state = state

        def on_tracking_event(self, event) -> None:
            with self._state._lock:
                self._state.latest = event
                self._state.frame_id += 1
                self._state.last_ts = time.perf_counter()
            self._state._event.set()

    def detect(self, frame: Frame) -> HandDetection:
        """Return the latest hand as a 21-keypoint detection.

        Args:
            frame: ignored (the Leap delivers 3D directly; the frame is
                only the pipeline's clock).

        Returns:
            :class:`HandDetection` with the converted keypoints;
            all-NaN keypoints + presence 0.0 when no hand is tracked.
        """
        with self._state._lock:
            event = self._state.latest
        hands = getattr(event, "hands", None) if event is not None else None
        if not hands:
            return HandDetection(keypoints3d=np.full((21, 3), np.nan),
                                 visibility=np.zeros(21), presence=0.0)
        hand = hands[0]
        keypoints = keypoints_from_hand(hand)
        handedness = None
        hand_type = getattr(hand, "type", None)
        if hand_type is not None:
            handedness = str(hand_type).rsplit(".", 1)[-1]
        return HandDetection(keypoints3d=keypoints,
                             visibility=np.ones(21), presence=1.0,
                             handedness=handedness)

    def close(self) -> None:
        """Close the Leap connection (safe to call repeatedly)."""
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        try:
            if hasattr(self, "_open") and self._open is not None \
                    and hasattr(self._open, "__exit__"):
                self._open.__exit__(None, None, None)
            connection.remove_listener(self._Listener(self._state))
        except Exception:  # pylint: disable=broad-exception-caught
            pass


class LeapSource(FrameSource):
    """FrameSource over the Leap: one stub frame per tracking event.

    The pipeline needs a :class:`Frame` clock; the Leap delivers the 3D
    skeleton directly, so the frame is a black canvas with NaN depth
    (the camera panel renders black with the skeleton overlay).  The
    frame rate is the Leap's own tracking rate (~90-120 Hz).
    """

    def __init__(self, detector: LeapDetector, width: int = 640,
                 height: int = 480):
        self._detector = detector
        self.width = width
        self.height = height
        self._last_frame_id = 0
        self._closed = False

    def read(self) -> Optional[Frame]:
        """Block for the next tracking event and return its stub frame.

        Returns:
            :class:`Frame` with a black color canvas, NaN depth, and the
            event's monotonic timestamp; None only after :meth:`close`.
        """
        while not self._closed:
            with self._detector._state._lock:
                frame_id = self._detector._state.frame_id
                latest = self._detector._state.latest
                last_ts = self._detector._state.last_ts
            if latest is not None and frame_id != self._last_frame_id:
                self._last_frame_id = frame_id
                height, width = self.height, self.width
                return Frame(
                    color=np.zeros((height, width, 3), np.uint8),
                    depth=np.full((height, width), np.nan, np.float32),
                    ts=last_ts, fx=float(width), fy=float(width),
                    cx=width / 2.0, cy=height / 2.0)
            self._detector._state._event.wait(timeout=1.0)
            self._detector._state._event.clear()
        return None

    def close(self) -> None:
        self._closed = True
        self._detector._state._event.set()
