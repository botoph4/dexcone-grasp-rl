"""viser-based debug page for the teleop pipeline.

Shows the live color frame with the 21 hand keypoints overlaid, a depth
colormap, and a text panel with the detection confidence, state-machine
state, joint angles and the P24 command.  Run with
``p24-teleop-camera --camera orbbec --view`` and open the printed URL.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from p24grasp.teleop.pipeline import FixedRateControl, TeleopPipeline, format_angles
from p24grasp.teleop.state_machine import State

# MediaPipe 21-keypoint bone pairs (wrist=0, thumb=1-4, index=5-8, ...).
BONES = ((0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16),
         (13, 17), (17, 18), (18, 19), (19, 20),
         (0, 17))


def overlay_keypoints(color: np.ndarray, uv: np.ndarray, visibility: np.ndarray,
                      presence: float, state: State) -> np.ndarray:
    """RGB frame with keypoints/bones drawn; visibility -> color."""
    image = Image.fromarray(color)
    draw = ImageDraw.Draw(image)
    _, w = color.shape[:2]
    for a, b in BONES:
        if np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all():
            draw.line([tuple(uv[a]), tuple(uv[b])], fill=(0, 180, 255), width=2)
    for i in range(21):
        if not np.isfinite(uv[i]).all():
            continue
        vis = visibility[i]
        fill = (0, 220, 80) if vis >= 0.5 else (255, 170, 0) if vis > 0 else (255, 60, 60)
        r = 3 if vis >= 0.5 else 2
        x, y = uv[i]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    draw.rectangle([0, 0, w, 34], fill=(0, 0, 0))
    draw.text((6, 8), f"presence={presence:.2f}  state={state.value}", fill=(255, 255, 255))
    return np.asarray(image)


_JET_LUT: np.ndarray | None = None


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Jet colormap of valid depth, meters; invalid pixels near-black.

    A precomputed 256-entry LUT replaces matplotlib's per-frame colormap
    call (~30 ms on 848x480 -- far too slow for a 60 Hz display loop);
    the LUT is built once from matplotlib on first use, the percentile
    scaling subsamples every 4th valid pixel (full sorts dominate the
    cost), and the mapping indexes the LUT over the whole array (fancy
    assignment through the validity mask costs ~8 ms extra).
    """
    global _JET_LUT
    valid = np.isfinite(depth_m)
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), np.uint8)
    if _JET_LUT is None:
        import matplotlib  # one-time LUT build

        rgba = matplotlib.colormaps["jet"](np.linspace(0.0, 1.0, 256))
        _JET_LUT = (rgba[..., :3] * 255).astype(np.uint8)
    values = depth_m[valid]
    sampled = values[::4]
    lo, hi = np.percentile(sampled, 1), np.percentile(sampled, 99)
    span = max(hi - lo, 1e-6)
    scaled = np.clip((depth_m - lo) * (1.0 / span), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)
    img = _JET_LUT[(scaled * 255.0).astype(np.uint8)]
    img[~valid] = 15
    return img


def _cjk_font(size: int = 16):
    """A TTF font with CJK coverage (PIL's default bitmap font renders no
    Chinese); falls back to the default font when none is installed."""
    from functools import lru_cache

    from PIL import ImageFont  # noqa: E402

    @lru_cache(maxsize=None)
    def resolve(font_size: int):
        candidates = (
            "/System/Library/Fonts/PingFang.ttc",          # macOS
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # macOS
            "/System/Library/Fonts/STHeiti Light.ttc",     # macOS
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",          # Linux
            "C:/Windows/Fonts/msyh.ttc",                   # Windows
            "C:/Windows/Fonts/simhei.ttf",                 # Windows
        )
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, font_size)
                except OSError:
                    continue
        return ImageFont.load_default()

    return resolve(size)


def render_calibration_frame(color: np.ndarray, uv: np.ndarray,
                             visibility: np.ndarray, presence: float,
                             title: str, *, lines: list[str]) -> np.ndarray:
    """Camera view + keypoints + a calibration info bar (used by the guided
    gesture calibration window)."""
    image = overlay_keypoints(color, uv, visibility, presence, State.TRACKING)
    pil = Image.fromarray(image)
    bar_height = 26 + 22 * len(lines)
    info_bar = Image.new("RGB", (pil.width, bar_height), (20, 20, 20))
    draw_bar = ImageDraw.Draw(info_bar)
    draw_bar.text((10, 8), title, fill=(255, 220, 80), font=_cjk_font(18))
    for i, line in enumerate(lines):
        draw_bar.text((10, 30 + 20 * i), line, fill=(255, 255, 255),
                      font=_cjk_font(16))
    return np.asarray(np.vstack([pil, info_bar]))


def _uv_from_keypoints(keypoints3d: np.ndarray, frame) -> np.ndarray:
    """Project lifted camera-frame keypoints back to pixel coordinates."""
    pts = np.full((21, 2), np.nan)
    valid = np.isfinite(keypoints3d).all(axis=1)
    x, y, z = keypoints3d[valid].T
    pts[valid, 0] = frame.fx * x / np.maximum(z, 1e-6) + frame.cx
    pts[valid, 1] = frame.fy * y / np.maximum(z, 1e-6) + frame.cy
    return pts


def _placeholder(text: str = "waiting for camera frames...") -> np.ndarray:
    """Visible placeholder so a frame-less viewer page is not blank."""
    img = np.full((240, 360, 3), 30, np.uint8)
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((24, 110), text, fill=(255, 255, 255))
    return np.asarray(pil)


def _with_lights(xml_path) -> str:
    """Inject directional lights into the hand MJCF (the bare hand.xml has
    only the headlight; extra lights make the mesh readable)."""
    lights = (
        '<light directional="true" diffuse="0.9 0.9 0.9" specular="0.15 0.15 0.15"'
        ' pos="0.1 0.05 0.35" dir="0 0 -1"/>\n'
        '<light directional="true" diffuse="0.45 0.45 0.45"'
        ' pos="-0.15 0.05 0.2" dir="1 0 -0.4"/>\n'
        '<light directional="true" diffuse="0.35 0.35 0.35"'
        ' pos="0.35 0.0 0.2" dir="-1 0 -0.4"/>\n'
    )
    text = Path(xml_path).read_text(encoding="utf-8")
    if "<light" in text:
        return text  # already has custom lights
    # <light> is a worldbody child in the MJCF schema, not a root element.
    return text.replace("</worldbody>", lights + "</worldbody>")


def _brighten_background(image: np.ndarray, threshold: int = 25) -> np.ndarray:
    """Standard-viewer look: light vertical gradient background + brighter mesh.

    The offscreen render has no skybox, so the background comes out black.
    Replace it with a MuJoCo-viewer-style light gray gradient and boost the
    mesh brightness (the headlight-only hand reads very dark).
    """
    out = image.copy().astype(np.float32)
    background = out.max(axis=2) < threshold
    # MuJoCo viewer's standard light blue-gray sky gradient.
    rows = np.linspace(1.0, 0.0, out.shape[0])[:, None, None]
    top = np.array([219.0, 227.0, 240.0])
    bottom = np.array([176.0, 186.0, 209.0])
    gradient = np.empty(out.shape, np.float32)
    gradient[:] = top * rows + bottom * (1.0 - rows)
    out[background] = gradient[background]
    out[~background] = np.clip(out[~background] * 1.7, 0, 255)  # brighten mesh
    return out.astype(np.uint8)


class P24MeshRenderer:
    """Offscreen MuJoCo render of the P24 hand at a mapped joint command.

    Joint order is resolved by name against the generated ``build/hand.xml``
    (the MJCF build drops the URDF ``mimic`` coupling, so the 20 joints are
    independent -- the command already encodes the coupling when enabled).
    Renders return None (panel skipped) when mujoco or the GL context is
    unavailable, e.g. on a headless board.
    """

    def __init__(self, height: int = 220):
        self._model = None
        self._error = None
        self._height = height
        try:
            import mujoco  # noqa: E402
            from p24grasp.paths import ensure_hand_xml  # noqa: E402
            from p24grasp.teleop.retarget import P24_JOINT_NAMES  # noqa: E402

            model = mujoco.MjModel.from_xml_string(_with_lights(ensure_hand_xml()))
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=height)
            self._qpos_ids = np.array(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in P24_JOINT_NAMES], dtype=int)
            if (self._qpos_ids < 0).any():
                raise RuntimeError("some P24 joints are missing from the MJCF build")
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self._camera)
            # Same viewing convention as the repo's own render_scene (demo.py):
            # default free camera + azimuth 180 / elevation -20, looking at
            # the hand visual centroid (fingers dominate the mesh); distance
            # 0.30 fills most of the panel with the hand.
            self._camera.lookat[:] = (0.10, 0.03, 0.135)
            self._camera.distance = 0.30
            self._camera.azimuth = 180.0
            self._camera.elevation = -20.0
            self._model, self._data, self._renderer = model, data, renderer
        except Exception as exc:  # headless / GL-less / broken build: skip panel
            self._error = exc

    def render(self, command_deg: np.ndarray) -> np.ndarray | None:
        if self._model is None:
            return None
        try:
            import mujoco  # noqa: E402

            self._data.qpos[self._qpos_ids] = np.radians(command_deg)
            mujoco.mj_forward(self._model, self._data)
            self._renderer.update_scene(self._data, camera=self._camera)
            return _brighten_background(self._renderer.render())
        except Exception as exc:  # once GL dies, never recover: disable panel
            self._model = None
            self._error = exc
            return None


def skeleton_arrays(keypoints3d: np.ndarray, visibility: np.ndarray):
    """3D hand skeleton for the viser scene + per-point colors.

    Returns (points, point_colors, segments, segment_colors) in viser
    coordinates (y up): the camera frame (x right, y down, z forward) maps
    to (x, z, -y).  Invalid keypoints are dropped; bones that need a dropped
    joint are skipped.
    """
    valid = np.isfinite(keypoints3d).all(axis=1)
    points = keypoints3d[valid]
    points_viser = np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)
    vis = np.asarray(visibility, dtype=np.float64)
    colors = np.where(vis[valid, None] >= 0.5, (0, 220, 80), (255, 170, 0)).astype(np.uint8)
    index_of = np.full(21, -1, dtype=int)
    index_of[valid] = np.arange(int(valid.sum()))
    segments, seg_colors = [], []
    for a, b in BONES:
        if index_of[a] >= 0 and index_of[b] >= 0:
            segments.append([points_viser[index_of[a]], points_viser[index_of[b]]])
            seg_colors.append((0, 180, 255))
    segments = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)
    seg_colors = np.asarray(seg_colors, dtype=np.uint8).reshape(-1, 1, 3)
    seg_colors = np.repeat(seg_colors, 2, axis=1)  # viser wants (N, 2, 3)
    return (points_viser.astype(np.float32), colors,
            segments, seg_colors, bool(valid.any()))


def compose_display(rgb_overlay: np.ndarray, depth_rgb: np.ndarray,
                    p24_rgb: np.ndarray | None,
                    text_lines: list[str]) -> np.ndarray:
    """One display frame as a 2x2 grid:
    color+keypoints | depth; P24 hand | status text."""
    height, width = rgb_overlay.shape[:2]
    depth = np.asarray(Image.fromarray(depth_rgb).resize((width, height)))
    if p24_rgb is not None:
        p24 = np.asarray(Image.fromarray(p24_rgb).resize((width, height)))
    else:
        p24 = np.full((height, width, 3), 30, np.uint8)
    status = np.full((height, width, 3), 20, np.uint8)
    pil = Image.fromarray(status)
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(text_lines):
        draw.text((12, 20 + 26 * i), line, fill=(255, 255, 255))
    return np.vstack([np.hstack([rgb_overlay, depth]),
                      np.hstack([p24, np.asarray(pil)])])


def _status_lines(out, stats) -> list[str]:
    filtered = out.filtered
    return [
        f"map={stats.get('map_rate', 0):.0f}Hz  "
        f"ctrl={stats.get('control_rate', 0):.0f}/"
        f"{stats.get('control_hz', 60):.0f}Hz  "
        f"presence={out.detection.presence:.2f}  "
        f"state={filtered.state.value}  ok={filtered.tracking_ok}  "
        f"dof_frac={filtered.dof_frac:.2f}",
        *format_angles(filtered.angles, filtered.state, out.command).splitlines(),
    ]


def compose_camera_display(rgb_overlay: np.ndarray, depth_rgb: np.ndarray,
                           text_lines: list[str]) -> np.ndarray:
    """Camera-window frame: color+keypoints | depth on top, a status strip
    below (the sim hand lives in the separate standard MuJoCo popup)."""
    height, width = rgb_overlay.shape[:2]
    depth = np.asarray(Image.fromarray(depth_rgb).resize((width, height)))
    status = np.full((70, 2 * width, 3), 20, np.uint8)
    pil = Image.fromarray(status)
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(text_lines):
        draw.text((12, 8 + 22 * i), line, fill=(255, 255, 255))
    return np.vstack([np.hstack([rgb_overlay, depth]), np.asarray(pil)])


def run_viewers(source, detector, retargeter=None, *,
                mujoco_popup: bool = False, cv2_window: bool = False,
                control_hz: float = 60.0) -> None:
    """The sim popups over the fixed-rate control loop.

    The pipeline runs on the control loop's producer thread at the
    camera's frame rate; the consumer ticks at the fixed ``control_hz``
    and only does light work: applying the latest command to the
    standard MuJoCo viewer popup (``launch_passive`` -- no custom mesh
    rendering) and, throttled to ~15 Hz, the cv2 camera window
    (color+keypoints | depth | status -- no sim panel).  Keeping the
    heavy 2x2 grid render out of the consumer is what lets the pipeline
    reach its full frame rate.

    Args:
        source: the open :class:`FrameSource`.
        detector: the :class:`HandDetector`.
        retargeter: the mapping backend.
        mujoco_popup: open the interactive standard MuJoCo viewer popup.
        cv2_window: open the cv2 camera window.
        control_hz: fixed control tick frequency.
    """
    import mujoco  # noqa: E402
    import mujoco.viewer  # noqa: E402,F401  # submodule is not auto-imported
    from p24grasp.paths import ensure_hand_xml  # noqa: E402
    from p24grasp.teleop.retarget import P24_JOINT_NAMES  # noqa: E402

    pipeline = TeleopPipeline(source, detector, retargeter=retargeter)
    model = data = viewer = None
    qpos_ids = None
    if mujoco_popup:
        model = mujoco.MjModel.from_xml_string(_with_lights(ensure_hand_xml()))
        data = mujoco.MjData(model)
        qpos_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
             for name in P24_JOINT_NAMES], dtype=int)
        if (qpos_ids < 0).any():
            raise RuntimeError("some P24 joints are missing from the MJCF build")
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except RuntimeError as exc:
            if "mjpython" not in str(exc):
                raise
            # macOS: the interactive popup needs the Cocoa main thread, so
            # it only runs under the mjpython trampoline -- degrade to the
            # camera window and tell the user how to get the popup
            print("[viewer] macOS 上交互式 MuJoCo 窗口需要 mjpython;"
                  "本次仅显示相机窗口。打开仿真弹窗请用:\n"
                  f"[viewer]   sudo {Path(sys.executable).parent / 'mjpython'} "
                  "scripts/teleop_camera.py --local --mujoco-view ...",
                  flush=True)
            mujoco_popup = False
            viewer = None
    cv2 = None
    window_name = "p24 camera"
    if cv2_window:
        import cv2  # noqa: E402

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    control = FixedRateControl(pipeline, control_hz=control_hz)
    last_render = 0.0
    last_print = 0.0

    def apply(command: np.ndarray, out, stats: dict) -> bool:
        nonlocal last_render, last_print
        if mujoco_popup:
            data.qpos[qpos_ids] = np.radians(command)
            mujoco.mj_forward(model, data)
            viewer.sync()
            if not viewer.is_running():
                return False
        if cv2_window:
            now = time.perf_counter()
            if now - last_render >= 1.0 / 15.0:  # ~15 Hz camera window
                last_render = now
                uv = _uv_from_keypoints(out.detection.keypoints3d, out.frame)
                display = compose_camera_display(
                    overlay_keypoints(out.frame.color, uv,
                                      out.detection.visibility,
                                      out.detection.presence, out.filtered.state),
                    colorize_depth(out.frame.depth),
                    _status_lines(out, stats))
                cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # Esc or q
                    return False
        now = time.perf_counter()
        if now - last_print >= 2.0:
            last_print = now
            timing = pipeline.timing_ms
            print(f"[teleop] map={stats['map_rate']:.0f}Hz "
                  f"ctrl={stats['control_rate']:.0f}Hz "
                  f"read={timing['read']:.0f}ms det={timing['detect']:.0f}ms "
                  f"solve={timing['retarget']:.0f}ms "
                  f"presence={out.detection.presence:.2f} "
                  f"state={out.filtered.state.value}", flush=True)
        return True

    try:
        control.run(apply)
    except KeyboardInterrupt:
        print("\nviewer stopped")
    finally:
        control.close()
        if viewer is not None:
            viewer.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


def run_http_viewer(source, detector, retargeter=None, port: int = 8080) -> None:
    """Minimal MJPEG page: one <img> streaming the 2x2 teleop grid.

    Replaces the viser-based page (its GUI panels repeatedly failed to
    render on some macOS/browser combinations).  MJPEG over plain HTTP
    works in every browser with zero client-side complexity.
    """
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pipeline = TeleopPipeline(source, detector, retargeter=retargeter)

    state = {"jpeg": None}
    condition = threading.Condition()
    stop = threading.Event()

    def producer() -> None:
        # The MuJoCo renderer owns a GL context that is NOT thread-safe:
        # it must be created and used inside this thread (creating it
        # elsewhere and calling render() from here hangs in CGL).  The
        # pipeline itself runs on the control loop's producer thread.
        p24 = P24MeshRenderer()
        p24_error_panel = None
        if p24._error is not None:  # pylint: disable=protected-access
            p24_error_panel = _placeholder(f"P24 render failed: {p24._error}")
            print(f"[viewer] P24 mesh panel disabled: {p24._error}", flush=True)
        control = FixedRateControl(pipeline, control_hz=60.0)
        last_push = 0.0
        last_print = 0.0

        def apply(command: np.ndarray, out, stats: dict) -> bool:
            nonlocal last_push, last_print
            now = time.perf_counter()
            if now - last_print >= 2.0:
                last_print = now
                timing = pipeline.timing_ms
                print(f"[teleop] map={stats['map_rate']:.0f}Hz "
                      f"ctrl={stats['control_rate']:.0f}Hz "
                      f"read={timing['read']:.0f}ms det={timing['detect']:.0f}ms "
                      f"solve={timing['retarget']:.0f}ms "
                      f"presence={out.detection.presence:.2f} "
                      f"state={out.filtered.state.value}", flush=True)
            if now - last_push >= 1.0 / 15.0:  # ~15 Hz stream
                uv = _uv_from_keypoints(out.detection.keypoints3d, out.frame)
                hand_img = p24.render(command)
                if hand_img is None:
                    hand_img = p24_error_panel
                grid = compose_display(
                    overlay_keypoints(out.frame.color, uv,
                                      out.detection.visibility,
                                      out.detection.presence,
                                      out.filtered.state),
                    colorize_depth(out.frame.depth),
                    hand_img,
                    _status_lines(out, stats))
                if grid.shape[1] > 1280:
                    scale = 1280 / grid.shape[1]
                    grid = np.asarray(Image.fromarray(grid).resize(
                        (1280, int(grid.shape[0] * scale))))
                buffer = io.BytesIO()
                Image.fromarray(grid).save(buffer, format="JPEG", quality=85)
                with condition:
                    state["jpeg"] = buffer.getvalue()
                    last_push = now
                    condition.notify_all()
            return not stop.is_set()

        try:
            control.run(apply, stop=stop)
        except KeyboardInterrupt:
            pass
        finally:
            control.close()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # pylint: disable=invalid-name
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while not stop.is_set():
                    with condition:
                        condition.wait(timeout=0.5)
                        jpeg = state["jpeg"]
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return
            page = (
                "<!DOCTYPE html><html><head><title>p24 teleop</title>"
                "<style>body{margin:0;background:#111;display:flex;"
                "align-items:center;justify-content:center;min-height:100vh}"
                "img{max-width:100vw;max-height:100vh}</style></head>"
                "<body><img src='/stream' alt='teleop grid'></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *args):  # keep the console clean
            del args

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    print(f"camera viewer: http://127.0.0.1:{port} (open in a browser)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviewer stopped")
    finally:
        stop.set()
        server.shutdown()


def run_camera_viewer(source, detector, port: int = 8080, host: str = "0.0.0.0",
                      retargeter=None) -> None:
    """Stream the live pipeline to a browser debug page (blocks until Ctrl-C)."""
    import viser  # lazy: optional dependency (web extra)

    server = viser.ViserServer(host=host, port=port)
    print(f"camera viewer: http://127.0.0.1:{port} (open in a browser)")
    # One page, four equal quadrants: camera+keypoints | depth / P24 | status,
    # composited into a single 2x2 image (same layout as the local window).
    placeholder = _placeholder()
    grid_panel = server.gui.add_image(placeholder, label="teleop (2x2)")
    p24 = P24MeshRenderer()
    p24_error_panel = None
    if p24._error is not None:  # pylint: disable=protected-access
        p24_error_panel = _placeholder(f"P24 render failed: {p24._error}")
        print(f"[viewer] P24 mesh panel disabled: {p24._error}", flush=True)
    empty = np.zeros((0, 3), np.float32)
    skeleton_points = server.scene.add_point_cloud(
        "/hand/keypoints", empty, colors=np.zeros((0, 3), np.uint8),
        point_size=0.004, point_shape="circle")
    skeleton_lines = server.scene.add_line_segments(
        "/hand/bones", np.zeros((0, 2, 3), np.float32),
        colors=np.zeros((0, 2, 3), np.uint8), thickness=0.002)
    # Full-JPEG panel pushes at 100 fps stall the browser page; ~15 Hz is
    # plenty for a debug view.
    update_step = 6
    pipeline = TeleopPipeline(source, detector, retargeter=retargeter)
    fps = 0.0
    start = time.perf_counter()
    n_frames = 0
    prev_presence = -1.0
    try:
        while True:
            t0 = time.perf_counter()
            out = pipeline.step()
            if out.frame is None:
                # camera warmup / transient gap: keep the page informative
                elapsed = time.perf_counter() - start
                grid_panel.value = _placeholder(
                    f"waiting for camera frames... {elapsed:.0f}s elapsed")
                time.sleep(0.05)
                continue
            n_frames += 1
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6)
            if prev_presence <= 0 < out.detection.presence:
                # one-shot dump per detection burst: where do the DOFs fail?
                vis = out.detection.visibility
                lifted = int(np.isfinite(out.detection.keypoints3d).all(axis=1).sum())
                print(f"[detect] hand found: mean_vis={vis.mean():.2f} "
                      f"vis>=0.5: {int((vis >= 0.5).sum())}/21 "
                      f"depth-lifted: {lifted}/21 "
                      f"dof_frac={out.filtered.dof_frac:.2f}", flush=True)
            prev_presence = out.detection.presence
            if n_frames % 60 == 1:  # terminal heartbeat: frames flow without the browser
                print(f"[orbbec] frame {n_frames}: fps={fps:.0f} "
                      f"presence={out.detection.presence:.2f} "
                      f"dof_frac={out.filtered.dof_frac:.2f} "
                      f"state={out.filtered.state.value}", flush=True)
            uv = _uv_from_keypoints(out.detection.keypoints3d, out.frame)
            if n_frames % update_step == 0:
                hand_img = p24.render(out.command)
                if hand_img is None:
                    hand_img = p24_error_panel
                grid = compose_display(
                    overlay_keypoints(out.frame.color, uv, out.detection.visibility,
                                      out.detection.presence, out.filtered.state),
                    colorize_depth(out.frame.depth),
                    hand_img,
                    _status_lines(out, {"map_rate": fps, "control_rate": fps,
                                        "control_hz": fps}))
                # the full grid is 1696x960: downscale before pushing so the
                # browser websocket/render loop keeps up (stalled pages show
                # the stale placeholder)
                if grid.shape[1] > 1280:
                    scale = 1280 / grid.shape[1]
                    grid = np.asarray(Image.fromarray(grid).resize(
                        (1280, int(grid.shape[0] * scale))))
                grid_panel.value = grid
                if n_frames % 300 == 0:
                    print(f"[viewer] web grid updated (frame {n_frames}, "
                          f"{grid.shape[1]}x{grid.shape[0]})", flush=True)
                (points, pcolors, segments, scolor, has_hand) = skeleton_arrays(
                    out.detection.keypoints3d, out.detection.visibility)
                skeleton_points.points = points
                skeleton_points.colors = pcolors
                skeleton_lines.points = segments
                skeleton_lines.colors = scolor
                skeleton_points.visible = has_hand
                skeleton_lines.visible = has_hand
    except KeyboardInterrupt:
        print("\nviewer stopped")
    finally:
        pipeline.close()
        server.stop()
