"""Hand keypoint detection: MediaPipe HandLandmarker (Tasks API) + depth lifting.

MediaPipe 1.0.x ships PyPI wheels for linux-aarch64 and macOS-arm64, so the
same code runs CPU-only on the deployment board and on the dev machine.
The model file (``hand_landmarker.task``) is not bundled; :func:`ensure_model`
downloads it on first use (override with ``model_path`` for offline boards).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from p24grasp.teleop.camera import Frame

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "p24grasp" / "hand_landmarker.task"

N_JOINTS = 21


def ensure_model(model_path: str | Path | None = None) -> Path:
    """Return the hand_landmarker.task path, downloading it if missing.

    Args:
        model_path: explicit model file; None = the shared cache path
            (~/.cache/p24grasp/hand_landmarker.task).

    Returns:
        The existing (or freshly downloaded) model file path.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading hand_landmarker.task to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


@dataclass
class HandDetection:
    """21 hand keypoints lifted to camera-frame 3D (meters) + per-point visibility.

    ``keypoints3d[i]`` = (x, y, z) right/down/forward in the camera frame.
    ``visibility[i]`` in [0, 1].  ``presence`` is binary: 1.0 when a hand was
    detected this frame, 0.0 otherwise (the Tasks API exposes no continuous
    hand-presence score; handedness confidence is a left/right classifier,
    NOT a presence measure, and must not be used for the occlusion logic).
    All-NaN keypoints mean no hand was detected (or no valid depth under it).
    """

    keypoints3d: np.ndarray  # (21, 3)
    visibility: np.ndarray  # (21,)
    presence: float
    handedness: Optional[str] = None


def sanitize_visibility(values: np.ndarray) -> np.ndarray:
    """NaN visibility -> 1.0 (visible).

    The hand_landmarker.task model variant does not emit per-landmark
    visibility: the attribute exists but is NaN for every landmark, which
    would zero out every DOF's confidence in the state machine.  Treat
    missing data as "visible"; DOF validity then falls back to whether the
    angle itself is finite (depth lifting succeeded).

    Args:
        values: raw per-landmark visibility array from MediaPipe.

    Returns:
        Same-shape array with non-finite entries replaced by 1.0.
    """
    out = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(out), out, 1.0)


class HandDetector:
    """MediaPipe HandLandmarker wrapper: pixel keypoints + depth lifting."""

    def __init__(self, model_path: str | Path | None = None, *,
                 num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 depth_median_window: int = 3):
        """Create the detector (the MediaPipe landmarker initializes lazily
        on the first :meth:`detect` call).

        Args:
            model_path: hand_landmarker.task path (auto-downloaded when
                missing).
            num_hands: maximum tracked hands (the wrist-mounted setup
                takes the closest/first).
            min_detection_confidence / min_tracking_confidence /
                min_presence_confidence: MediaPipe landmarker thresholds.
            depth_median_window: median-filter window size (px) used to
                sample depth under each keypoint.
        """
        self.model_path = ensure_model(model_path)
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_presence_confidence = min_presence_confidence
        self.depth_median_window = depth_median_window
        self._landmarker = None
        self._last_ts_ms = -1

    def _init(self):
        """Create the MediaPipe HandLandmarker (VIDEO mode, CPU delegate).

        Returns:
            The mediapipe module (also stored on ``self._landmarker``).
        """
        import mediapipe as mp  # noqa: E402
        from mediapipe.tasks import python as mp_python  # noqa: E402
        from mediapipe.tasks.python import vision  # noqa: E402

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path),
                # CPU-only: the ARM deployment target has no usable GPU
                # delegate, and Metal initialization fails in some macOS
                # contexts (headless / new OS).  XNNPACK CPU is fast enough
                # for 21-keypoint tracking (tens of fps on a modern core).
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        return mp

    def detect(self, frame: Frame) -> HandDetection:
        """Detect the hand in ``frame``; returns an empty detection when absent.

        Args:
            frame: one aligned RGB-D :class:`Frame`.

        Returns:
            :class:`HandDetection` with (21, 3) lifted keypoints (meters,
            camera frame), per-landmark visibility, presence 1.0/0.0, and
            the handedness label; all-NaN keypoints when no hand (or no
            valid depth) is present.
        """
        empty = HandDetection(keypoints3d=np.full((N_JOINTS, 3), np.nan),
                              visibility=np.zeros(N_JOINTS), presence=0.0)
        if frame.color.size == 0:
            return empty
        if self._landmarker is None:
            mp = self._init()
        else:
            import mediapipe as mp  # noqa: E402
        ts_ms = int(frame.ts * 1000.0)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1  # VIDEO mode requires strictly increasing ts
        self._last_ts_ms = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.color)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.hand_landmarks:
            return empty

        landmarks = result.hand_landmarks[0]  # wrist-mounted: take the closest/first hand
        u = np.array([lm.x for lm in landmarks]) * frame.color.shape[1]
        v = np.array([lm.y for lm in landmarks]) * frame.color.shape[0]
        vis = sanitize_visibility(
            np.array([getattr(lm, "visibility", 1.0) for lm in landmarks]))
        handedness = result.handedness[0][0].category_name if result.handedness else None

        keypoints3d = self._lift(frame, u, v)
        return HandDetection(keypoints3d=keypoints3d, visibility=vis,
                             presence=1.0, handedness=handedness)

    def _lift(self, frame: Frame, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Lift pixel keypoints to camera-frame 3D via median depth sampling.

        Args:
            frame: the aligned RGB-D frame (depth meters, intrinsics).
            u: (21,) keypoint x pixels.
            v: (21,) keypoint y pixels.

        Returns:
            (21, 3) camera-frame keypoints (x right, y down, z forward in
            meters); NaN rows where the sampled depth window has no valid
            pixels.
        """
        pts = np.full((N_JOINTS, 3), np.nan)
        h, w = frame.depth.shape
        half = self.depth_median_window // 2
        ui = np.clip(np.rint(u).astype(int), 0, w - 1)
        vi = np.clip(np.rint(v).astype(int), 0, h - 1)
        for i in range(N_JOINTS):
            r0, r1 = max(0, vi[i] - half), min(h, vi[i] + half + 1)
            c0, c1 = max(0, ui[i] - half), min(w, ui[i] + half + 1)
            window = frame.depth[r0:r1, c0:c1]
            valid = window[np.isfinite(window)]
            if valid.size == 0:
                continue
            z = float(np.median(valid))
            pts[i, 0] = (ui[i] - frame.cx) * z / frame.fx
            pts[i, 1] = (vi[i] - frame.cy) * z / frame.fy
            pts[i, 2] = z
        return pts

    def close(self) -> None:
        """Release the MediaPipe landmarker (safe to call repeatedly)."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
