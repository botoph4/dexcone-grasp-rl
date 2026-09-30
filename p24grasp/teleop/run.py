"""CLI entry points: live camera, recorded replay, and recording."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from p24grasp.teleop.camera import (
    OrbbecSource,
    RealsenseSource,
    ReplaySource,
    record_frames,
)
from p24grasp.teleop.detector import HandDetector
from p24grasp.teleop.pipeline import TeleopPipeline, format_angles


def _auto_detect_source(width: int, height: int, fps: int,
                        frame_sync: bool, align: str):
    """Try the RealSense backend first, then fall back to Orbbec."""
    errors = []
    for kind in ("realsense", "orbbec"):
        source = make_camera_source(kind, width, height, fps,
                                    frame_sync=frame_sync, align=align)
        try:
            source.start()
            print(f"[{kind}] device detected", flush=True)
            return source
        except RuntimeError as exc:
            errors.append(f"{kind}: {exc}")
            try:
                source.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    raise RuntimeError("no RGB-D camera detected: " + " | ".join(errors))


def make_retargeter(kind: str, hand_calibration_path=None, lateral_enabled=True):
    """Factory for the retargeting backends."""
    if kind == "hybrid":
        from p24grasp.teleop.retarget import HybridRetargeter  # noqa: E402

        return HybridRetargeter(hand_calibration_path=hand_calibration_path,
                                lateral_enabled=lateral_enabled)
    if kind == "fingertip":
        from p24grasp.teleop.retarget import FingertipRetargeter  # noqa: E402

        return FingertipRetargeter()
    if kind == "scaling":
        from p24grasp.teleop.retarget import DirectAngleScaling  # noqa: E402

        return DirectAngleScaling()
    if kind == "geort":
        from p24grasp.teleop.geort import GeoRtRetargeter  # noqa: E402

        return GeoRtRetargeter()
    if kind == "dex":
        from p24grasp.teleop.retarget import DexRetargetingAdapter  # noqa: E402

        return DexRetargetingAdapter()
    raise ValueError(f"unknown retargeter: {kind}")


def make_camera_source(kind: str, width: int, height: int, fps: int, *,
                       frame_sync: bool = False, align: str = "hw"):
    """Factory for the two live RGB-D backends."""
    if kind == "realsense":
        return RealsenseSource(width, height, fps)
    if kind == "orbbec":
        return OrbbecSource(width, height, fps, frame_sync=frame_sync,
                            align_mode=align)
    raise ValueError(f"unknown camera kind: {kind} (realsense|orbbec)")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None,
                        help="hand_landmarker.task path (auto-downloaded if missing)")
    parser.add_argument("--print-every", type=int, default=10,
                        help="print one summary every N frames")
    parser.add_argument("--csv", default=None, help="append per-frame angles to CSV")
    parser.add_argument("--retarget",
                        choices=("hybrid", "fingertip", "scaling", "geort", "dex"),
                        default="hybrid",
                        help="mapping: hybrid (joint mapping + pinch refinement, "
                             "default) | fingertip (pure tip optimization) | "
                             "scaling (joint angles only) | geort (neural, "
                             "trained in sim) | dex (dex-retargeting optimizer)")


CSV_HEADER = ["ts", "state", "thumb_cmc", "thumb_mp", "index_mcp", "index_pip",
              "index_dip", "middle_pip", *[f"q{i}" for i in range(20)]]


def _run(source, args: argparse.Namespace, csv_row) -> int:
    detector = HandDetector(model_path=args.model)
    retargeter = make_retargeter(args.retarget, args.calibration,
                                 lateral_enabled=not args.no_lateral)
    pipeline = TeleopPipeline(source, detector, retargeter=retargeter)
    frames = 0
    try:
        while True:
            out = pipeline.step()
            if out.frame is None:
                break
            frames += 1
            if csv_row is not None:
                a = out.filtered.angles
                csv_row((out.frame.ts, out.filtered.state.value,
                         *a.flexion[0, :2], *a.flexion[1, :3], a.flexion[2, 1],
                         *out.command))
            if frames % args.print_every == 1:
                print(f"--- frame {frames} dt={out.dt*1000:.0f}ms "
                      f"presence={out.detection.presence:.2f}")
                print(format_angles(out.filtered.angles, out.filtered.state, out.command))
    except KeyboardInterrupt:
        print(f"\nstopped after {frames} frames")
    finally:
        pipeline.close()
    return frames


def replay_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the teleop pipeline on recorded frames")
    parser.add_argument("--dir", required=True, help="directory of recorded *.npz frames")
    parser.add_argument("--loop", action="store_true")
    _add_common(parser)
    args = parser.parse_args(argv)
    source = ReplaySource(args.dir, loop=args.loop)
    _run_with_csv(source, args)
    return 0


def camera_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live teleop from a RealSense D405 "
                                                 "or Orbbec Gemini 305")
    parser.add_argument("--camera", choices=("auto", "realsense", "orbbec"),
                        default="auto",
                        help="RGB-D backend (default: auto-detect)")
    parser.add_argument("--probe", action="store_true",
                        help="run step-by-step camera bring-up diagnostics and exit "
                             "(orbbec only)")
    parser.add_argument("--view", action="store_true",
                        help="serve the 2x2 teleop grid as an MJPEG page "
                             "(camera+keypoints | depth / P24 | status)")
    parser.add_argument("--local", action="store_true",
                        help="show the live pipeline in a local cv2 window "
                             "(works under sudo; the web page is unreliable "
                             "on some macOS/browser combinations)")
    parser.add_argument("--no-lateral", action="store_true",
                        help="disable the lateral (ab/adduction) mapping: the "
                             "four joint_2 DOFs stay at neutral zero")
    parser.add_argument("--calibration", default=None,
                        help="hand calibration file (default: "
                             "~/.cache/p24grasp/hand_calibration.json); "
                             "written by --calibrate, read by every teleop run")
    parser.add_argument("--calibrate", action="store_true",
                        help="guided gesture calibration: open / together / "
                             "fist poses calibrate the lateral bounds and the "
                             "flexion range, saved for later runs")
    parser.add_argument("--mujoco-view", action="store_true",
                        help="open a separate interactive MuJoCo viewer window "
                             "for the mapped P24 hand (rotate/zoom, standard "
                             "sim_viewer look); combines with --local")
    parser.add_argument("--port", type=int, default=8080,
                        help="viewer port (with --view)")
    parser.add_argument("--frame-sync", action="store_true",
                        help="enable Orbbec frame sync (off by default; the sync "
                             "filter can stall on this camera)")
    parser.add_argument("--align", choices=("hw", "sw"), default="hw",
                        help="Orbbec depth-to-color alignment: hw (device) or "
                             "sw (host); try sw if hw delivers depth-only sets")
    parser.add_argument("--record-dir", default=None, help="also record frames to this dir")
    parser.add_argument("--record-frames", type=int, default=0,
                        help="if set with --record-dir, record N frames then exit")
    parser.add_argument("--width", type=int, default=None,
                        help="color width (default: 640 for realsense, 848 for orbbec)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    _add_common(parser)
    args = parser.parse_args(argv)
    if args.camera == "auto":
        source = _auto_detect_source(args.width or 848, args.height, args.fps,
                                     args.frame_sync, args.align)
        args.camera = "orbbec" if isinstance(source, OrbbecSource) else "realsense"
        width = source.width
    else:
        width = args.width if args.width else (848 if args.camera == "orbbec" else 640)
        source = make_camera_source(args.camera, width, args.height, args.fps,
                                    frame_sync=args.frame_sync, align=args.align)
        source.start()
    if args.calibrate:
        from p24grasp.teleop.calibration import (  # noqa: E402
            DEFAULT_PATH,
            GuidedCalibration,
        )

        detector = HandDetector(model_path=args.model)
        try:
            GuidedCalibration(source, detector,
                              path=args.calibration or DEFAULT_PATH).run()
        finally:
            source.close()
        return 0
    if args.probe:
        if not isinstance(source, OrbbecSource):
            print(f"--probe is only supported for the orbbec backend, not {args.camera}")
            return 2
        source.probe()
        return 0
    print(f"[{args.camera}] camera started ({width}x{args.height}@{args.fps}), "
          "waiting for frames...", flush=True)
    if args.view or args.local:
        from p24grasp.teleop.viewer import (  # noqa: E402
            run_http_viewer,
            run_local_viewer,
            run_mujoco_viewer,
        )

        retargeter = make_retargeter(args.retarget, args.calibration,
                                     lateral_enabled=not args.no_lateral)
        detector = HandDetector(model_path=args.model)
        try:
            if args.mujoco_view and not args.local:
                run_mujoco_viewer(source, detector, retargeter=retargeter)
            elif args.local:
                run_local_viewer(source, detector, retargeter=retargeter)
            else:
                run_http_viewer(source, detector, retargeter=retargeter,
                                port=args.port)
        except KeyboardInterrupt:
            pass  # second Ctrl-C during shutdown: ignore, cleanup continues
        finally:
            try:
                source.close()
            except KeyboardInterrupt:
                pass
        return 0
    try:
        if args.record_dir and args.record_frames:
            record_frames(source, args.record_dir, args.record_frames)
            print(f"recorded {args.record_frames} frames to {args.record_dir}")
            return 0
        _run_with_csv(source, args)
    finally:
        source.close()
    return 0


def _run_with_csv(source, args: argparse.Namespace) -> None:
    """Run the pipeline, appending to CSV when --csv was given."""
    if args.csv:
        with open(args.csv, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)
            _run(source, args, writer.writerow)
    else:
        _run(source, args, None)


def record_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record RGB-D frames for offline replay (dev machines without camera)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--camera", choices=("auto", "realsense", "orbbec"),
                        default="auto",
                        help="RGB-D backend (default: auto-detect)")
    parser.add_argument("--width", type=int, default=None,
                        help="color width (default: 640 for realsense, 848 for orbbec)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frame-sync", action="store_true",
                        help="enable Orbbec frame sync (off by default)")
    args = parser.parse_args(argv)
    if args.camera == "auto":
        source = _auto_detect_source(args.width or 848, args.height, args.fps,
                                     args.frame_sync, "hw")
    else:
        width = args.width if args.width else (848 if args.camera == "orbbec" else 640)
        source = make_camera_source(args.camera, width, args.height, args.fps,
                                    frame_sync=args.frame_sync)
        source.start()
    try:
        saved = record_frames(source, args.out, args.frames)
        print(f"recorded {saved} frames to {Path(args.out).resolve()}")
    finally:
        source.close()
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("replay", "camera", "record"):
        print("usage: python -m p24grasp.teleop.run {replay,camera,record} ...")
        return 2
    mode = sys.argv[1]
    rest = sys.argv[2:]
    if mode == "replay":
        return replay_main(rest)
    if mode == "camera":
        return camera_main(rest)
    return record_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
