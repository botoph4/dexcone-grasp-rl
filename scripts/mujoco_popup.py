"""Standard MuJoCo viewer popup: the P24 hand + the detected human skeleton.

Each stdin line carries the 20-joint command (degrees, URDF order)
followed by the 21 camera-frame keypoints (63 x/y/z meters, "nan" when
missing)::

    q0 .. q19 k0x k0y k0z .. k20x k20y k20z

The human skeleton (21 joint spheres + 20 ellipsoid bones) is drawn
beside the robot hand -- the camera frame (x right, y down, z forward)
maps into the scene (x right, y left, z up) with a fixed offset, so one
window shows both hands live.  On macOS the interactive popup needs the
Cocoa main thread, so it runs under ``mjpython`` as a subprocess of the
teleop viewer:

    mjpython scripts/mujoco_popup.py

The parent sends one line per control tick (60 Hz); the popup applies
the command, steps the model, updates the skeleton, and syncs the
viewer window (the standard ``launch_passive`` look: rotate/zoom with
the mouse).  Closing the window or closing stdin exits.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402,F401  # not auto-imported by `import mujoco`

from p24grasp.paths import ensure_hand_xml  # noqa: E402
from p24grasp.teleop.retarget import P24_JOINT_NAMES  # noqa: E402
from p24grasp.teleop.viewer import BONES, _with_lights  # noqa: E402

JOINT_RGBA = "0.2 0.9 0.5 0.9"  # green joint spheres
BONE_RGBA = "0.1 0.8 1.0 0.75"  # light-blue bone ellipsoids

# Camera frame -> scene placement: the skeleton floats up-right of the
# robot hand (x: depth, y: -camera x, z: up = -camera y).
SKELETON_OFFSET = np.array([0.30, 0.0, 0.42])
SKELETON_SCALE = 0.9


def _backdrop_xml() -> str:
    """A closed white room (six planes, normals facing inward): the popup
    background reads white from every camera angle inside the room --
    the GLFW viewer's own clear color is not configurable from Python."""
    planes = (
        ("bg_floor", "0 0 -0.2", "0 0 0"),      # normal +z
        ("bg_ceiling", "0 0 1.2", "180 0 0"),   # normal -z
        ("bg_back", "1.2 0 0.5", "0 -90 0"),    # normal -x
        ("bg_front", "-1.2 0 0.5", "0 90 0"),   # normal +x
        ("bg_right", "0 1.2 0.5", "90 0 0"),    # normal -y
        ("bg_left", "0 -1.2 0.5", "-90 0 0"),   # normal +y
    )
    return "".join(
        f'<geom name="{name}" type="plane" size="3 3 0.01" pos="{pos}" '
        f'euler="{euler}" rgba="1 1 1 1" contype="0" conaffinity="0" '
        f'group="2"/>\n'
        for name, pos, euler in planes)


def _skeleton_geoms_xml() -> str:
    """The 21 joint spheres + 20 bone ellipsoids as worldbody children
    (free geoms: no body, no collisions, visual-only)."""
    joints = "".join(
        f'<geom name="skel_j{i}" type="sphere" size="0.006" '
        f'rgba="{JOINT_RGBA}" contype="0" conaffinity="0" group="2"/>\n'
        for i in range(21))
    bones = "".join(
        f'<geom name="skel_b{i}" type="ellipsoid" size="0.003 0.003 0.012" '
        f'rgba="{BONE_RGBA}" contype="0" conaffinity="0" group="2"/>\n'
        for i in range(len(BONES)))
    return joints + bones


def build_model():
    """The P24 hand model + the skeleton geoms; returns
    (model, data, qpos_ids, joint_ids, bone_ids)."""
    text = _with_lights(ensure_hand_xml())
    text = text.replace("</worldbody>",
                        _backdrop_xml() + _skeleton_geoms_xml() + "</worldbody>")
    model = mujoco.MjModel.from_xml_string(text)
    data = mujoco.MjData(model)
    qpos_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
         for name in P24_JOINT_NAMES], dtype=int)
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"skel_j{i}")
         for i in range(21)], dtype=int)
    bone_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"skel_b{i}")
         for i in range(len(BONES))], dtype=int)
    if (qpos_ids < 0).any() or (joint_ids < 0).any() or (bone_ids < 0).any():
        raise RuntimeError("some P24/skeleton elements are missing from the MJCF")
    return model, data, qpos_ids, joint_ids, bone_ids


def _rotation_z_to(direction: np.ndarray) -> np.ndarray:
    """(3, 3) rotation mapping the z axis onto ``direction`` (unit)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(z, direction))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sine
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def apply_frame(model, data, *, qpos_ids, joint_ids, bone_ids,
                command: np.ndarray, keypoints3d: np.ndarray) -> None:
    """Apply the 20-joint command and update the skeleton geoms.

    Args:
        model/data: the compiled model and its data.
        qpos_ids/joint_ids/bone_ids: geom/joint id lookups.
        command: (20,) joint angles in degrees (URDF order).
        keypoints3d: (21, 3) camera-frame keypoints (meters, NaN = missing).
    """
    data.qpos[qpos_ids] = np.radians(command)
    mujoco.mj_forward(model, data)
    # camera frame -> scene coords (x right, y down, z forward -> z up)
    pts = keypoints3d * SKELETON_SCALE
    scene = np.empty_like(pts)
    scene[:, 0] = SKELETON_OFFSET[0] + pts[:, 2]
    scene[:, 1] = SKELETON_OFFSET[1] - pts[:, 0]
    scene[:, 2] = SKELETON_OFFSET[2] - pts[:, 1]
    for i in range(21):
        if np.isfinite(scene[i]).all():
            data.geom_xpos[joint_ids[i]] = scene[i]
            model.geom_rgba[joint_ids[i], 3] = 0.9
        else:
            model.geom_rgba[joint_ids[i], 3] = 0.0  # hidden
    for bone, (start, end) in enumerate(BONES):
        a, b = scene[start], scene[end]
        if np.isfinite(a).all() and np.isfinite(b).all():
            delta = b - a
            length = float(np.linalg.norm(delta))
            if length > 1e-6:
                data.geom_xpos[bone_ids[bone]] = 0.5 * (a + b)
                data.geom_xmat[bone_ids[bone]] = _rotation_z_to(delta / length).reshape(9)
                model.geom_size[bone_ids[bone]] = [0.003, 0.003, max(0.5 * length, 0.003)]
                model.geom_rgba[bone_ids[bone], 3] = 0.75
                continue
        model.geom_rgba[bone_ids[bone], 3] = 0.0


def main() -> int:
    model, data, qpos_ids, joint_ids, bone_ids = build_model()
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        for line in sys.stdin:
            try:
                values = np.fromstring(line, sep=" ")
            except ValueError:
                continue
            if values.shape[0] < 20:
                continue
            command = values[:20]
            keypoints = np.full((21, 3), np.nan)
            if values.shape[0] >= 20 + 63:
                keypoints = values[20:83].reshape(21, 3)
            apply_frame(model, data, qpos_ids=qpos_ids, joint_ids=joint_ids,
                        bone_ids=bone_ids, command=command, keypoints3d=keypoints)
            viewer.sync()
            if not viewer.is_running():
                break
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
