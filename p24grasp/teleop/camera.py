"""Frame sources: live RealSense D405 / Orbbec Gemini 305, recorded replay, recorder.

``pyrealsense2`` wheel availability (checked 2026-09-28): Linux aarch64
wheels exist since 2.58.4 (manylinux2014, cp39/cp310/cp312), but there are
NO macOS wheels -- on this dev machine the bindings are built from source
(librealsense + ``-DBUILD_PYTHON_BINDINGS=ON``).

``pyorbbecsdk2`` 2.1.2 ships macOS-arm64 and linux-aarch64 wheels (module
name ``pyorbbecsdk``); the Gemini 305 (4 cm min depth, 68 g) is purpose-built
for wrist mounting.

All camera-dependent code lives behind :class:`FrameSource` so the rest of
the pipeline is platform-independent: develop on macOS against recorded
frames, deploy on ARM with :class:`RealsenseSource`/:class:`OrbbecSource`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np


@dataclass
class Frame:
    """One aligned RGB-D frame, in meters/seconds/pixel coordinates.

    ``depth`` is meters, NaN where invalid.  ``fx/fy/cx/cy`` are the color
    (aligned) intrinsics used to lift pixels to 3D.
    """

    color: np.ndarray  # (H, W, 3) uint8 RGB
    depth: np.ndarray  # (H, W) float32 meters
    ts: float  # seconds, monotonic
    fx: float
    fy: float
    cx: float
    cy: float


class FrameSource(Protocol):
    """Anything that yields aligned RGB-D frames; ``None`` signals end of stream."""

    def read(self) -> Optional[Frame]:
        ...

    def close(self) -> None:
        ...


class RealsenseSource:
    """Live RealSense D405: 640x480@60 RGB8 + Z16 depth, aligned to color.

    D405 (global shutter, ~7 cm min depth) is required; D435-class cameras do
    not work wrist-mounted (min depth ~28 cm, rolling shutter).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 60):
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0

    def start(self) -> None:
        import pyrealsense2 as rs  # noqa: E402

        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        device = profile.get_device()
        device.first_depth_sensor().set_option(rs.option.depth_units, 0.0001)
        self._depth_scale = device.first_depth_sensor().get_depth_scale()
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

    def _intrinsics(self, frame):
        import pyrealsense2 as rs  # noqa: E402

        profile = frame.profile.as_video_stream_profile()
        if profile.stream_type() != rs.stream.color:
            raise RuntimeError("aligned frames expected; color intrinsics unavailable")
        intr = profile.get_intrinsics()
        return intr.fx, intr.fy, intr.ppx, intr.ppy

    def read(self) -> Optional[Frame]:
        if self._pipeline is None:
            raise RuntimeError("start() must be called before read()")
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())
        depth = depth_u16.astype(np.float32) * self._depth_scale
        depth[depth_u16 == 0] = np.nan
        fx, fy, cx, cy = self._intrinsics(color_frame)
        ts = aligned.get_timestamp() / 1000.0
        return Frame(color=color, depth=depth, ts=ts, fx=fx, fy=fy, cx=cx, cy=cy)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class OrbbecSource:
    """Live Orbbec Gemini 305: hardware depth-to-color aligned RGB-D.

    Gemini 305: 4 cm min working distance, 68 g, 42x42 mm, built for
    wrist-mounted robotics.  Uses ``pyorbbecsdk2`` (pip-installable on
    macOS-arm64 and linux-aarch64; the import module is ``pyorbbecsdk``).
    Depth stream is HW-aligned to color, so both arrive at the color
    resolution with the color intrinsics.
    """

    def __init__(self, width: int = 848, height: int = 480, fps: int = 60,
                 frame_sync: bool = False, align_mode: str = "hw"):
        # frame_sync defaults OFF: on this camera/SDK/macOS combination the
        # sync filter intermittently never assembles a complete set (the
        # color/depth timestamp domains drift), which stalls the pipeline.
        # align_mode: "hw" (device-side D2C) or "sw" (host-side D2C); switch
        # to "sw" if the hardware align delivers depth-only framesets.
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_sync = frame_sync
        self.align_mode = align_mode
        self._pipeline = None
        self._fx = self._fy = self._cx = self._cy = np.nan
        self._timeouts = 0

    def _build_config(self, pipeline):
        """Enable color+depth streams and D2C alignment; raises with a readable
        message when the requested mode is unsupported."""
        import pyorbbecsdk as ob  # noqa: E402

        config = ob.Config()
        for sensor, fmt in ((ob.OBSensorType.COLOR_SENSOR, ob.OBFormat.RGB),
                            (ob.OBSensorType.DEPTH_SENSOR, ob.OBFormat.Y16)):
            profiles = pipeline.get_stream_profile_list(sensor)
            if profiles is None:
                raise RuntimeError(f"no {sensor} stream profiles (device connected?)")
            profile = profiles.get_video_stream_profile(self.width, self.height, fmt, self.fps)
            if profile is None:  # requested mode unsupported: fall back to default
                profile = profiles.get_default_video_stream_profile()
                if profile is None:
                    raise RuntimeError(f"no default {sensor} profile")
            config.enable_stream(profile)
        mode = ob.OBAlignMode.HW_MODE if self.align_mode == "hw" \
            else ob.OBAlignMode.SW_MODE
        config.set_align_mode(mode)
        return config

    def start(self) -> None:
        import pyorbbecsdk as ob  # noqa: E402

        try:
            pipeline = ob.Pipeline()
        except RuntimeError as exc:
            if "uvc_open failed" in str(exc):
                # Since macOS 12, UVC cameras opened through libuvc-style SDKs
                # (bypassing Apple's camera-permission prompt) require root:
                #   sudo <python> scripts/teleop_camera.py --camera orbbec
                # See OrbbecSDK issue #9 and libuvc issue #194.
                raise RuntimeError(
                    f"{exc}\n"
                    "macOS requires root to open the UVC camera with the Orbbec "
                    "SDK.  Run with sudo, e.g.:\n"
                    "  sudo /path/to/python scripts/teleop_camera.py --camera orbbec"
                ) from exc
            raise
        config = self._build_config(pipeline)
        if self.frame_sync:
            pipeline.enable_frame_sync()
        pipeline.start(config)
        intrinsics = pipeline.get_camera_param().rgb_intrinsic
        self._fx, self._fy = intrinsics.fx, intrinsics.fy
        self._cx, self._cy = intrinsics.cx, intrinsics.cy
        self._pipeline = pipeline
        self._warmup()

    def _warmup(self, budget_s: float = 15.0) -> None:
        """Block until the first complete RGB-D frameset arrives.

        The color stream lags depth by up to ~5 s at startup (measured: the
        probe sees depth-only framesets first, color joins later).  Absorbing
        that lag here keeps every consumer free of the warmup race.
        """
        print("[orbbec] waiting for the color stream to warm up...", flush=True)
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if frames.get_color_frame() is not None \
                    and frames.get_depth_frame() is not None:
                print("[orbbec] warmup complete", flush=True)
                return
        raise RuntimeError(
            "camera started but never delivered a complete RGB-D frameset "
            f"within {budget_s:.0f} s")

    def probe(self) -> None:
        """Step-by-step bring-up diagnostics (run without a detector).

        Prints one line per stage so a stalled camera can be located:
        device open -> profile negotiation -> stream start -> first frameset.
        """
        import pyorbbecsdk as ob  # noqa: E402

        try:
            pipeline = ob.Pipeline()
        except RuntimeError as exc:
            print(f"[1/5] FAILED at pipeline creation: {exc}", flush=True)
            return
        print("[1/5] pipeline created", flush=True)
        try:
            config = self._build_config(pipeline)
        except RuntimeError as exc:
            print(f"[2/5] FAILED at stream config: {exc}", flush=True)
            return
        print("[2/5] stream config ok", flush=True)
        for sensor, label in ((ob.OBSensorType.COLOR_SENSOR, "color"),
                              (ob.OBSensorType.DEPTH_SENSOR, "depth")):
            profiles = pipeline.get_stream_profile_list(sensor)
            print(f"     {label} profiles ({profiles.get_count()}):", flush=True)
            for i in range(profiles.get_count()):
                video = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                print(f"       {video.get_width()}x{video.get_height()}"
                      f"@{video.get_fps()} fmt={video.get_format()}", flush=True)
        pipeline.start(config)
        print("[3/5] stream started; waiting up to 5 s for the first frameset...",
              flush=True)
        frames = pipeline.wait_for_frames(5000)
        print("[4/5] first frameset:", "arrived" if frames is not None
              else "TIMEOUT (no frames in 5 s)", flush=True)
        if frames is not None:
            got_color = False
            for attempt in range(30):
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if color is not None:
                    got_color = True
                    print(f"[5/5] color frame arrived on attempt {attempt}: "
                          f"{color.get_width()}x{color.get_height()} "
                          f"fmt={color.get_format()}", flush=True)
                    print(f"      depth: {depth.get_width()}x{depth.get_height()} "
                          f"scale={depth.get_depth_scale()}", flush=True)
                    break
                if attempt == 0:
                    depth_desc = (f"depth {depth.get_width()}x{depth.get_height()}"
                                  f" scale={depth.get_depth_scale()}"
                                  if depth is not None else "no depth frame")
                    print(f"[5/5] first frameset has NO color frame | {depth_desc}",
                          flush=True)
                    print("      polling up to 3 s for the color stream...", flush=True)
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    break
            if not got_color:
                print("[5/5] color stream never produced a frame", flush=True)
        print("probe done, stopping pipeline...", flush=True)
        pipeline.stop()
        print("pipeline stopped cleanly", flush=True)

    def read(self) -> Optional[Frame]:
        if self._pipeline is None:
            raise RuntimeError("start() must be called before read()")
        # wait_for_frames has a 1 s timeout, and early framesets may carry
        # depth before the color stream warms up.  Keep waiting with a
        # warning instead of surfacing None (which the pipeline reads as
        # end-of-stream and would silently terminate the run).
        patterns = []
        for _ in range(5):
            frames = self._pipeline.wait_for_frames(1000)
            color_frame = frames.get_color_frame() if frames is not None else None
            depth_frame = frames.get_depth_frame() if frames is not None else None
            if color_frame is not None and depth_frame is not None:
                patterns.append("complete")
                self._timeouts = 0
                break
            patterns.append("none" if frames is None else
                            "depth_only" if color_frame is None else "color_only")
            self._timeouts += 1
            if self._timeouts == 1:
                if frames is None:
                    detail = "no frameset at all (stream not delivering?)"
                else:
                    missing = "color" if color_frame is None else "depth"
                    detail = f"frameset without {missing} (warmup lag?)"
                print(f"[orbbec] waiting: {detail}; align={self.align_mode} "
                      f"sync={'on' if self.frame_sync else 'off'}", flush=True)
            elif self._timeouts % 5 == 1:
                print(f"[orbbec] still waiting after {self._timeouts}s "
                      f"(align={self.align_mode}, "
                      f"sync={'on' if self.frame_sync else 'off'})...", flush=True)
        else:
            # Batch failed: report the delivery pattern of the last 5 s so the
            # failure mode (no frames vs depth-only vs intermittent) is visible.
            print(f"[orbbec] last 5s delivery: {patterns.count('complete')} complete / "
                  f"{patterns.count('depth_only')} depth-only / "
                  f"{patterns.count('color_only')} color-only / "
                  f"{patterns.count('none')} no-frameset "
                  f"(align={self.align_mode})", flush=True)
            return None
        width, height = color_frame.get_width(), color_frame.get_height()
        color = np.ascontiguousarray(np.asanyarray(color_frame.get_data()),
                                     dtype=np.uint8).reshape(height, width, 3)
        depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        # HW-aligned: depth shares the color resolution.
        depth_u16 = depth_u16.reshape(height, width)
        depth_mm = depth_u16.astype(np.float32) * depth_frame.get_depth_scale()
        depth = depth_mm / 1000.0  # meters
        depth[depth_u16 == 0] = np.nan
        ts = frames.get_system_timestamp_us() / 1e6
        return Frame(color=color, depth=depth, ts=ts,
                     fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class ReplaySource:
    """Replay frames recorded by :func:`record_frames` (sorted *.npz)."""

    def __init__(self, directory: str | Path, loop: bool = False):
        self._paths = sorted(Path(directory).glob("*.npz"))
        if not self._paths:
            raise FileNotFoundError(f"no .npz frames in {directory}")
        self._loop = loop
        self._i = 0

    def read(self) -> Optional[Frame]:
        if self._i >= len(self._paths):
            return None
        data = np.load(self._paths[self._i])
        self._i += 1
        if self._loop and self._i >= len(self._paths):
            self._i = 0
        return Frame(color=data["color"], depth=data["depth"], ts=float(data["ts"]),
                     fx=float(data["fx"]), fy=float(data["fy"]),
                     cx=float(data["cx"]), cy=float(data["cy"]))

    def close(self) -> None:
        pass


def record_frames(source: FrameSource, out_dir: str | Path, n_frames: int,
                  print_every: int = 100) -> int:
    """Record ``n_frames`` from ``source`` as *.npz files (for replay later)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in range(n_frames):
        frame = source.read()
        if frame is None:
            break
        np.savez_compressed(
            out / f"{i:06d}.npz",
            color=frame.color, depth=frame.depth, ts=np.float64(frame.ts),
            fx=np.float64(frame.fx), fy=np.float64(frame.fy),
            cx=np.float64(frame.cx), cy=np.float64(frame.cy),
        )
        saved += 1
        if (saved % print_every) == 0:
            print(f"recorded {saved}/{n_frames}")
    return saved


def iter_frames(source: FrameSource, wall_clock: bool = False) -> Iterator[Frame]:
    """Yield frames; optionally throttle to the recorded frame timestamps."""
    prev_ts = None
    start = time.perf_counter()
    while True:
        frame = source.read()
        if frame is None:
            return
        if wall_clock:
            if prev_ts is not None:
                delay = frame.ts - prev_ts
                elapsed = time.perf_counter() - start
                if delay > elapsed:
                    time.sleep(delay - elapsed)
            prev_ts = frame.ts
        yield frame
