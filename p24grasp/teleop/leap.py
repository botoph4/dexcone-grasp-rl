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

Driver requirements (official bindings, NOT on PyPI -- install from
source, https://github.com/ultraleap/leapc-python-bindings):

    git clone https://github.com/ultraleap/leapc-python-bindings
    cd leapc-python-bindings
    pip install -r requirements.txt
    pip install -e leapc-python-api
    pip install build cffi && python -m build leapc-cffi
    pip install leapc-cffi/dist/leapc_cffi-*.tar.gz

plus the Gemini Ultraleap Hand Tracking service
(https://developer.leapmotion.com/tracking-software-download) with a
Leap Motion Controller 2.  Platforms: Windows, Linux x64, Linux ARM
(precompiled modules for py3.8-3.11), Darwin (LeapSDK inside the
"Ultraleap Hand Tracking" app; build leapc_cffi yourself for newer
Python versions).

Operational note: KEEP the "Ultraleap Hand Tracking" app RUNNING (it can
be minimized) while teleoperating.  On macOS the app hosts the tracking
service: while it runs, the detector opens the device and subscribes to
its events (a second client opening the device is allowed), and hand
data flows at the camera's frame rate; once the app quits, the service
stops streaming entirely and no client receives frames.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from p24grasp.teleop.angles import HandAngles
from p24grasp.teleop.camera import Frame, FrameSource
from p24grasp.teleop.detector import HandDetection
from p24grasp.teleop.lateral import fit_palm_normal

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


def _unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None


def _angle_deg(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    """Angle between two unit-ish directions, degrees; 0 on degeneracy."""
    if first is None or second is None:
        return 0.0
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def direct_angles_from_hand(hand, palm_guard: bool = False) -> HandAngles:
    """Direct joint angles from the Gemini bones, ported from the
    leap_motion_sync_with_ffh scheme (lpV2SyncFfh::calPubFingerAngle).

    Every joint is the angle between two ADJACENT bone directions (the
    bone endpoints are the joints), using the finger's REAL metacarpal
    (palm -> MCP), not a wrist-ray proxy:

      fingers:  MCP = angle(metacarpal, proximal)       [handBone vs prox]
                PIP = angle(proximal, middle)           [prox vs middle]
                DIP = angle(middle, distal)             [middle vs distal]
      thumb:    CMC = the metacarpal's elevation above the palm plane
                MP  = angle(metacarpal, proximal)
                IP  = angle(middle, distal)             [their FbDir]
      abduction: the in-palm-plane signed angle between the metacarpal
                and the proximal phalanx (calFingerAdduction: both
                directions are projected onto the palm plane first).

    Args:
        hand: a leap hand object (digits with bones(), prev_joint/
            next_joint); the palm normal is fitted from the converted
            keypoints (the Python binding does not expose palm.normal).
        palm_guard: the reference's palm-away guard (bones with a
            negative dot product against the palm normal read as
            straight).  It is mounting-dependent: it was written for a
            head-on camera where the fingers point toward the sensor;
            with the DESK-mounted leap (hand above the camera) every
            bone points away from the palm and the guard zeroes ALL
            flexion.  Off by default.

    Returns:
        :class:`HandAngles` with the directly measured flexion/abduction
        (visibility all 1.0; presence 1.0).
    """
    flexion = np.zeros((5, 3))
    abduction = np.zeros(5)
    keypoints = keypoints_from_hand(hand)
    normal = fit_palm_normal(keypoints)
    if np.linalg.norm(normal) < 1e-9:
        normal = np.array([0.0, 0.0, 1.0])

    def direction(bone) -> Optional[np.ndarray]:
        nxt = _vector3(_call_or_attr(bone, "next_joint"))
        prev = _vector3(_call_or_attr(bone, "prev_joint"))
        if nxt is None or prev is None:
            return None
        return _unit(nxt - prev)

    digits = _call_or_attr(hand, "digits")
    if digits is None:
        return HandAngles(flexion=flexion, abduction=abduction,
                          flexion_vis=np.ones((5, 3)),
                          abduction_vis=np.ones(5), presence=1.0)
    for digit_index in range(min(5, len(digits))):
        bones = _call_or_attr(digits[digit_index], "bones")
        if bones is None or len(bones) < 4:
            continue
        metacarpal, proximal, middle, distal = bones[:4]
        meta_dir = direction(metacarpal)
        prox_dir = direction(proximal)
        middle_dir = direction(middle)
        distal_dir = direction(distal)
        if digit_index == 0:  # thumb
            if meta_dir is not None:
                flexion[0, 0] = max(0.0, 90.0 - _angle_deg(meta_dir, normal))
            flexion[0, 1] = _angle_deg(meta_dir, prox_dir)
            flexion[0, 2] = _angle_deg(middle_dir, distal_dir)
            if palm_guard:
                if prox_dir is not None and np.dot(prox_dir, normal) < 0:
                    flexion[0, 1] = 0.0
                if distal_dir is not None and np.dot(distal_dir, normal) < 0:
                    flexion[0, 2] = 0.0
            # thumb abduction: the metacarpal's in-plane angle vs the
            # palm direction (wrist -> middle MCP)
            palm_dir = _unit(keypoints[9] - keypoints[0]) \
                if np.isfinite(keypoints[9]).all() else None
            if meta_dir is not None and palm_dir is not None:
                meta_in = _unit(meta_dir - np.dot(meta_dir, normal) * normal)
                palm_in = _unit(palm_dir - np.dot(palm_dir, normal) * normal)
                if meta_in is not None and palm_in is not None:
                    sign = np.sign(np.dot(normal, np.cross(palm_in, meta_in)))
                    abduction[0] = sign * _angle_deg(palm_in, meta_in)
        else:  # fingers
            row = digit_index
            flexion[row, 0] = _angle_deg(meta_dir, prox_dir)
            flexion[row, 1] = _angle_deg(prox_dir, middle_dir)
            flexion[row, 2] = _angle_deg(middle_dir, distal_dir)
            if palm_guard:
                if prox_dir is not None and np.dot(prox_dir, normal) < 0:
                    flexion[row, 0] = 0.0
                if middle_dir is not None and np.dot(middle_dir, normal) < 0:
                    flexion[row, 1] = 0.0
                if distal_dir is not None and np.dot(distal_dir, normal) < 0:
                    flexion[row, 2] = 0.0
            if meta_dir is not None and prox_dir is not None:
                meta_in = _unit(meta_dir - np.dot(meta_dir, normal) * normal)
                prox_in = _unit(prox_dir - np.dot(prox_dir, normal) * normal)
                if meta_in is not None and prox_in is not None:
                    sign = np.sign(np.dot(normal, np.cross(meta_in, prox_in)))
                    abduction[row] = sign * _angle_deg(meta_in, prox_in)
    return HandAngles(flexion=flexion, abduction=abduction,
                      flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                      presence=1.0)


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
        self.hands_seen = False


class LeapDetector:
    """HandDetector-compatible wrapper over the Gemini tracking events.

    Opens the Leap connection and registers a listener; ``detect()``
    converts the latest hand into the pipeline's 21-keypoint
    :class:`HandDetection` (presence = 1.0 when a hand is tracked; the
    per-joint visibility is not exposed by the binding, so all joints
    report 1.0 and the occlusion machine relies on the finite-angle
    checks).
    """

    def __init__(self, tracking_mode: str = "Desktop",
                 direct_angles: bool = True):
        """Open the Leap connection and register the listener.

        Args:
            tracking_mode: Gemini tracking mode ("Desktop" etc.).
            direct_angles: compute the joint angles directly from the
                bone directions (the leap_motion_sync_with_ffh scheme)
                and attach them to each detection; when False, the
                pipeline falls back to the keypoint-derived angles.
        """
        self._direct_angles = direct_angles
        try:
            import leap  # noqa: E402
        except ImportError as exc:
            raise RuntimeError(
                "Leap Motion 驱动不可用: 绑定不在 PyPI 上,请从源码安装:\n"
                "  git clone https://github.com/ultraleap/leapc-python-bindings\n"
                "  cd leapc-python-bindings && pip install -r requirements.txt\n"
                "  pip install -e leapc-python-api\n"
                "  pip install build cffi && python -m build leapc-cffi\n"
                "  pip install leapc-cffi/dist/leapc_cffi-*.tar.gz\n"
                "并安装 Gemini 追踪服务(Ultraleap Hand Tracking);详见 "
                "p24grasp/teleop/leap.py 模块说明"
            ) from exc
        self._leap = leap
        self._state = _LeapState()
        self._connection = leap.Connection()
        # the listener must subclass leap.Listener: the connection's
        # dispatch calls on_event(), which the base class routes to the
        # per-event handlers (a plain class raises AttributeError on
        # every event)
        state = self._state
        connection = self._connection

        class _LeapListener(leap.Listener):
            """Feeds the shared state from the service thread.

            Opens the device and SUBSCRIBES to its events: without the
            subscription the service delivers tracking frames with
            nHands = 0 (the hand payloads go to subscribed clients only
            -- measured: 2223 frames / 2093 with hands after
            subscribing).  The app must be running (it hosts the
            service on macOS); a second client opening the device is
            allowed while it runs.
            """

            def __init__(self) -> None:
                self._device_open = None

            def on_device_event(self, event) -> None:
                try:
                    self._device_open = event.device.open()
                    if hasattr(self._device_open, "__enter__"):
                        self._device_open.__enter__()
                    connection.subscribe_events(event.device)
                    print("[leap] device opened + events subscribed", flush=True)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"[leap] device open/subscribe failed: {exc}", flush=True)

            def on_tracking_event(self, event) -> None:
                with state._lock:
                    had_hands = state.hands_seen
                    hands = getattr(event, "hands", None)
                    state.hands_seen = bool(hands)
                    state.latest = event
                    state.frame_id += 1
                    state.last_ts = time.perf_counter()
                    if state.hands_seen and not had_hands:
                        hand = hands[0]
                        print(f"[leap] hand detected: id={hand.id} "
                              f"type={hand.type}", flush=True)
                    elif not state.hands_seen and had_hands:
                        print("[leap] hand lost", flush=True)
                state._event.set()

        self._listener = _LeapListener()
        self._connection.add_listener(self._listener)
        self._diag_calls = 0
        self._diag_done = False
        self._diag_enabled = True
        # keep the connection open for the process lifetime (the binding
        # returns a context manager; enter it explicitly)
        self._open = self._connection.open()
        if hasattr(self._open, "__enter__"):
            self._open.__enter__()
        mode = getattr(leap.TrackingMode, tracking_mode, None)
        if mode is not None:
            self._connection.set_tracking_mode(mode)

    def _diagnose(self, hands) -> None:
        """One-shot diagnostics: reconcile detect() with the listener."""
        if self._diag_done or not self._diag_enabled:
            return
        self._diag_calls += 1
        if self._diag_calls < 10 or not hands:
            self._diag_done = self._diag_calls >= 10 and not hands
            if not hands or self._diag_calls == 10:
                with self._state._lock:
                    frame_id = self._state.frame_id
                    seen = self._state.hands_seen
                print(f"[leap] detect() hands={len(hands) if hands is not None else 'None'} "
                      f"frame_id={frame_id} listener_hands_seen={seen} "
                      f"diag_calls={self._diag_calls}", flush=True)
        else:
            self._diag_done = True

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
        # diagnostic: is the shared event the same one the listener saw?
        self._diagnose(hands)
        if not hands:
            return HandDetection(keypoints3d=np.full((21, 3), np.nan),
                                 visibility=np.zeros(21), presence=0.0)
        hand = hands[0]
        keypoints = keypoints_from_hand(hand)
        detection = HandDetection(keypoints3d=keypoints,
                                  visibility=np.ones(21), presence=1.0)
        hand_type = getattr(hand, "type", None)
        if hand_type is not None:
            detection.handedness = str(hand_type).rsplit(".", 1)[-1]
        if self._direct_angles:
            # the leap's own joint angles (adjacent-bone directions,
            # FFH-style) override the keypoint-derived ones downstream
            detection.hand_angles = direct_angles_from_hand(hand)
        return detection

    def close(self) -> None:
        """Close the Leap connection (safe to call repeatedly)."""
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        try:
            listener = getattr(self, "_listener", None)
            device_open = getattr(listener, "_device_open", None)
            if device_open is not None and hasattr(device_open, "__exit__"):
                device_open.__exit__(None, None, None)
            if hasattr(self, "_open") and self._open is not None \
                    and hasattr(self._open, "__exit__"):
                self._open.__exit__(None, None, None)
            connection.remove_listener(listener)
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
