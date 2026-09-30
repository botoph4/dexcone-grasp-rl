"""Thin launcher: python scripts/teleop_calibrate.py --camera orbbec"""
from p24grasp.teleop.run import camera_main

if __name__ == "__main__":
    raise SystemExit(camera_main())
