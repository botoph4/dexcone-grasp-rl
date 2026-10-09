"""Standard MuJoCo viewer popup driven by 20-joint commands on stdin.

Each stdin line is 20 space-separated joint angles in degrees (URDF
order).  On macOS the interactive popup needs the Cocoa main thread, so
it runs under ``mjpython`` as a subprocess of the teleop viewer (the
cv2 camera window needs the main thread of the parent interpreter; the
two GUI toolkits cannot share one process):

    mjpython scripts/mujoco_popup.py

The parent sends one line per control tick (60 Hz); the popup applies
the command, steps the model, and syncs the viewer window (the standard
``launch_passive`` look: rotate/zoom with the mouse).  Closing the
window or closing stdin exits.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402,F401  # not auto-imported by `import mujoco`

from p24grasp.paths import ensure_hand_xml  # noqa: E402
from p24grasp.teleop.retarget import P24_JOINT_NAMES  # noqa: E402
from p24grasp.teleop.viewer import _with_lights  # noqa: E402


def main() -> int:
    model = mujoco.MjModel.from_xml_string(_with_lights(ensure_hand_xml()))
    data = mujoco.MjData(model)
    qpos_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
         for name in P24_JOINT_NAMES], dtype=int)
    if (qpos_ids < 0).any():
        raise RuntimeError("some P24 joints are missing from the MJCF build")
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        for line in sys.stdin:
            try:
                command = np.fromstring(line, sep=" ")
            except ValueError:
                continue
            if command.shape != (20,):
                continue
            data.qpos[qpos_ids] = np.radians(command)
            mujoco.mj_forward(model, data)
            viewer.sync()
            if not viewer.is_running():
                break
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
