#!/usr/bin/env python3
"""Browser-based interactive grasp: drag the object, watch the hand hold it.

The MuJoCo physics runs locally (same scene as ``sim_viewer.py``); the browser
only renders the scene through viser and sends back the dragged pose.  This
gives the web equivalent of the desktop viewer's placement mode, plus real GUI
tabs/sliders instead of the keyboard.

Browser controls:
  * drag the gizmo on the cylinder to move it (position pin in ADJUST mode,
    spring pull during the grasp),
  * GUI tab "摆位": sliders for object x/y/z, axis tilt (±360 deg), size,
    and the "释放抓取" button,
  * GUI tab "抓取": live status (drift / contacts / held), "回到摆位" and
    "复位" buttons.

Run:
  python sim_web.py [--shape cylinder|sphere] [--radius 0.032] [--weight 0.85]
  then open the printed URL (default http://localhost:8080) in a browser.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import mujoco  # noqa: E402
import faulthandler  # noqa: E402

faulthandler.enable()
# a long-running viewer segfault is impossible to debug without a stack; dump
# all thread tracebacks every 90 s so the next crash names the culprit
faulthandler.dump_traceback_later(90.0, repeat=True)

from p24grasp.env.obs import actuator_joint_order  # noqa: E402
from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.paths import (  # noqa: E402
    assets_dir,
    build_dir,
    ensure_hand_xml,
    urdf_path,
)
from p24grasp.kinematics.ik import HALF_LENGTH, fist_center, wrap_aligned_center  # noqa: E402
from p24grasp.sim.scene import (  # noqa: E402
    OBJECT_BODY,
    OBJECT_JOINT,
    _body_id,
    _joint_id,
    build_interactive_scene,
    object_inertia,
)

FRAME_DT = 1.0 / 60.0

# spring coupling between the dragged gizmo and the free object
SPRING_K = 300.0     # N/m
SPRING_D = 8.0       # N/(m/s)
SPRING_FMAX = 10.0   # N
SPRING_ENGAGE = 0.008  # m - beyond this gap the gizmo is being dragged

# rotational spring: aligning the object to the gizmo's rotated orientation
ROT_K = 0.05    # N*m/rad
ROT_D = 0.0015  # N*m/(rad/s)
ROT_MAX = 0.03  # N*m
ROT_ALIGN = np.radians(2.0)  # rad - below this the gizmo snaps to the object


def tilts_from_quat(q) -> tuple[float, float]:
    """Extract (tilt_x, tilt_y) from a wxyz quaternion.

    Only the direction of the body's +z axis matters for a cylinder (a
    rotation about that axis is physically invisible), so the two tilts are
    recovered from the axis alone: Rx(tx)Ry(ty) maps +z to
    (sin ty, -sin tx cos ty, cos tx cos ty).
    """
    w, x, y, z = q
    ax = 2 * (x * z + w * y)
    ay = 2 * (y * z - w * x)
    az = 1 - 2 * (x * x + y * y)
    ty = float(np.arcsin(np.clip(ax, -1.0, 1.0)))
    tx = float(np.arctan2(-ay, az))
    return tx, ty


def quat_angle(q1, q2) -> float:
    """Angular distance [rad] between two wxyz quaternions."""
    dot = abs(float(np.clip(np.dot(np.asarray(q1), np.asarray(q2)), -1.0, 1.0)))
    return 2.0 * np.arccos(dot)


def rotation_torque(q_obj, q_gizmo, k=ROT_K, omega=None, d=ROT_D,
                    max_tau=ROT_MAX):
    """World-frame torque that aligns the object quaternion to the gizmo's.

    ``q_obj`` / ``q_gizmo`` are wxyz world orientations; the torque is
    k * (SO(3) log of q_gizmo * q_obj^-1), optionally with angular-velocity
    damping, clamped to ``max_tau``.
    """
    w1, x1, y1, z1 = q_gizmo
    w2, x2, y2, z2 = q_obj
    q_rel = np.array([
        w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
        -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2,
        -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2,
        -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2,
    ])  # q_gizmo * conj(q_obj)
    w, x, y, z = q_rel
    w = float(np.clip(w, -1.0, 1.0))
    ang = 2.0 * np.arccos(abs(w))
    if ang < 1e-4:
        return np.zeros(3)
    axis = np.array([x, y, z]) / np.sin(ang / 2.0)
    if w < 0.0:
        axis = -axis
    tau = axis * (k * ang)
    if omega is not None:
        tau = tau - d * np.asarray(omega)
    return np.clip(tau, -max_tau, max_tau)

# auto-recover: if the object falls this far from the clasp it is re-placed.
# Besides the UX, a free-falling object left running is the state in which a
# rare MuJoCo constraint-solver crash (segfault inside mj_fwdConstraint) was
# observed once; respawning keeps the simulation away from it.
DROP_RESPAWN = 0.12  # m - the cradle extrudes the object in ~10 s; catch it early


class WebGrasp:
    """MuJoCo physics + grasp logic, mirroring sim_viewer's ADJUST/RUN modes."""

    def __init__(self, shape: str, radius: float, weight: float):
        self.shape = shape
        self.radius = radius
        self.mass = weight
        self.hand = HandModel(str(urdf_path()))
        hand_xml = ensure_hand_xml().read_text(encoding="utf-8")
        self.center = fist_center(self.hand)

        scene = build_interactive_scene(
            hand_xml, radius=radius, mass=weight, center=self.center,
            timestep=0.002, shape=shape,
        )
        self._scene_path = build_dir() / "_scene_web.xml"
        self._scene_path.write_text(scene, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self.data = mujoco.MjData(self.model)

        self.joint_order = actuator_joint_order(self.model)
        self.n_joint_act = len(self.joint_order)
        self.q_flex = np.array([self.hand.joints[n].upper for n in self.joint_order])
        self._is_thumb = np.array([n.startswith("thumb") for n in self.joint_order])
        self.obj_body = _body_id(self.model, OBJECT_BODY)
        self.obj_joint = _joint_id(self.model, OBJECT_JOINT)
        self.obj_gid = next(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.obj_body
        )
        self.joint_qadr = np.array([
            self.model.jnt_qposadr[_joint_id(self.model, n)] for n in self.joint_order
        ])
        self.joint_names = self.joint_order
        # ViserUrdf needs every revolute joint in its config, mimics included
        self.vis_joint_names = []
        for chain in self.hand.chains:
            self.vis_joint_names.extend(
                j.name for j in chain.joints if j.type == "revolute"
            )

        self.adjusting = True
        self.obj_target = self._default_target()
        # horizontal initial pose: cylinder axis along +y across the palm
        self.tilt_x = -np.pi / 2.0 if shape == "cylinder" else 0.0
        self.tilt_y = 0.0
        self.elapsed = 0.0
        self.squeeze = 1.0  # multiplier on the base grasp pose (re-squeeze feedback)
        self.reset_placement()

    # ------------------------------------------------------------------ #
    def _default_target(self) -> np.ndarray:
        """Default placement: wrap-aligned for cylinders (see kinematics)."""
        if self.shape == "cylinder":
            return wrap_aligned_center(self.hand, self.radius)
        return np.asarray(self.center, dtype=float).copy()

    def grasp_pose(self) -> np.ndarray:
        # thumb 0.90: at 0.60 the thumb only acted as an end-stop after the
        # object slid 55 mm; at 0.90 it presses the object's top from t=0.3 s
        # and cuts the worst drift from 76 to ~60 mm
        return np.where(self._is_thumb, 0.90 * self.q_flex, 0.90 * self.q_flex)

    def object_quat(self) -> np.ndarray:
        ax, ay = self.tilt_x / 2.0, self.tilt_y / 2.0
        qx = np.array([np.cos(ax), np.sin(ax), 0.0, 0.0])
        qy = np.array([np.cos(ay), 0.0, np.sin(ay), 0.0])
        w1, x1, y1, z1 = qx
        w2, x2, y2, z2 = qy
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def pin_object(self):
        m, d = self.model, self.data
        adr = m.jnt_qposadr[self.obj_joint]
        d.qpos[adr:adr + 3] = self.obj_target
        d.qpos[adr + 3:adr + 7] = self.object_quat()
        d.qvel[m.jnt_dofadr[self.obj_joint]:m.jnt_dofadr[self.obj_joint] + 6] = 0.0

    def set_radius(self, radius: float):
        radius = float(np.clip(radius, 0.010, 0.060))
        if abs(radius - self.radius) < 1e-9:
            return
        self.radius = radius
        m = self.model
        m.geom_size[self.obj_gid, 0] = radius
        if self.shape == "cylinder":
            m.geom_size[self.obj_gid, 1] = HALF_LENGTH
        i_a, i_b, i_c = object_inertia(radius, self.mass, self.shape)
        m.body_inertia[self.obj_body] = (i_a, i_b, i_c)

    def reset_placement(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.obj_target = self._default_target()
        self.tilt_x = -np.pi / 2.0 if self.shape == "cylinder" else 0.0
        self.tilt_y = 0.0
        self.pin_object()
        q = np.clip(self.grasp_pose(),
                    self.model.actuator_ctrlrange[:self.n_joint_act, 0],
                    self.model.actuator_ctrlrange[:self.n_joint_act, 1])
        d.ctrl[:self.n_joint_act] = q
        mujoco.mj_forward(m, d)
        self.elapsed = 0.0
        self.squeeze = 1.0

    def start_run(self):
        """Release the object and start the automatic grasp from a clean state."""
        m, d = self.model, self.data
        want_radius = self.radius
        mujoco.mj_resetData(m, d)
        self.set_radius(want_radius)
        q = np.clip(self.grasp_pose(),
                    self.model.actuator_ctrlrange[:self.n_joint_act, 0],
                    self.model.actuator_ctrlrange[:self.n_joint_act, 1])
        d.qpos[self.joint_qadr] = q
        d.qvel[:] = 0.0
        d.ctrl[:self.n_joint_act] = q
        adr = m.jnt_qposadr[self.obj_joint]
        d.qpos[adr:adr + 3] = self.obj_target
        d.qpos[adr + 3:adr + 7] = self.object_quat()
        mujoco.mj_forward(m, d)
        self.adjusting = False
        self.elapsed = 0.0
        self.squeeze = 1.0

    def back_to_placement(self):
        d = self.data
        adr = self.model.jnt_qposadr[self.obj_joint]
        self.obj_target = d.qpos[adr:adr + 3].copy()
        self.adjusting = True
        self.elapsed = 0.0

    # ------------------------------------------------------------------ #
    def hand_cfg(self) -> dict[str, float]:
        return {
            name: float(self.data.qpos[
                self.model.jnt_qposadr[_joint_id(self.model, name)]
            ])
            for name in self.vis_joint_names
        }

    def contact_count(self) -> int:
        gids = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.obj_gid:
                gids.add(c.geom2)
            elif c.geom2 == self.obj_gid:
                gids.add(c.geom1)
        return len(gids)

    def finger_forces(self) -> dict[str, tuple[float, float, int]]:
        """Per-finger (normal force [N], friction force [N], #contacts).

        Normal force is the component along the contact frame's normal axis,
        friction the magnitude of the two tangential components.
        """
        m, d = self.model, self.data
        out = {ch.name: [0.0, 0.0, 0] for ch in self.hand.chains}
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != self.obj_gid and c.geom2 != self.obj_gid:
                continue
            other = c.geom2 if c.geom1 == self.obj_gid else c.geom1
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            finger = next((ch.name for ch in self.hand.chains
                           if name.startswith(ch.name + "_")), None)
            if finger is None:
                continue
            mujoco.mj_contactForce(m, d, i, buf)
            out[finger][0] += abs(float(buf[0]))
            out[finger][1] += float(np.linalg.norm(buf[1:3]))
            out[finger][2] += 1
        return out

    def step(self, pull: np.ndarray | None = None, torque: np.ndarray | None = None):
        m, d = self.model, self.data
        nsub = max(1, int(round(FRAME_DT / m.opt.timestep)))
        dt = m.opt.timestep
        for _ in range(nsub):
            if self.adjusting:
                d.ctrl[:self.n_joint_act] = np.clip(
                    self.grasp_pose() * self.squeeze,
                    m.actuator_ctrlrange[:self.n_joint_act, 0],
                    m.actuator_ctrlrange[:self.n_joint_act, 1],
                )
                self.pin_object()
            else:
                # Re-squeeze feedback: a fixed-pose cage slowly extrudes the
                # object - contact-solver jitter and gravity creep it out of
                # the wrap, the wrap shrinks, friction capacity drops, and it
                # slips out (measured 13.8/12.1/15.8 s for kp 0.5/2/5, so
                # servo stiffness barely matters).  Closing the loop on drift
                # and contact count re-tightens the grip the moment the object
                # starts to creep.
                drift = float(np.linalg.norm(d.xpos[self.obj_body] - self.obj_target))
                nc = self.contact_count()
                if drift > 0.050 or nc < 4:
                    self.squeeze = min(1.12, self.squeeze + 0.10 * dt)
                elif drift < 0.035 and nc >= 6:
                    self.squeeze = max(1.0, self.squeeze - 0.04 * dt)
                d.ctrl[:self.n_joint_act] = np.clip(
                    self.grasp_pose() * self.squeeze,
                    m.actuator_ctrlrange[:self.n_joint_act, 0],
                    m.actuator_ctrlrange[:self.n_joint_act, 1],
                )
            d.xfrc_applied[:] = 0.0
            if pull is not None:
                d.xfrc_applied[self.obj_body, :3] = pull
            if torque is not None:
                d.xfrc_applied[self.obj_body, 3:6] = torque
            mujoco.mj_step(m, d)
            self.elapsed += m.opt.timestep


def make_hand_urdf() -> Path:
    """URDF with mesh paths fixed for yourdfpy/ViserUrdf loading.

    The URDF's mesh paths are relative to the repo root, and the full-res palm
    (11 MB) is swapped for the decimated one used by MuJoCo.
    """
    urdf = urdf_path().read_text(encoding="utf-8")
    # absolute mesh paths: the temp URDF lives in build/, and yourdfpy resolves
    # relative paths against the CWD, not the URDF location
    mesh_root = assets_dir() / "p24_meshes"
    urdf = urdf.replace('filename="meshes/', f'filename="{mesh_root}/')
    urdf = urdf.replace(f"{mesh_root}/visual/P2.4_Hand_R_Palm.STL",
                        f"{mesh_root}/mujoco/P2.4_Hand_R_Palm_180000.STL")
    out = build_dir() / "_web_hand.urdf"
    out.write_text(urdf, encoding="utf-8")
    return out


TEXTS = {
    "hint": {
        "zh": "拖拽物体上的手柄移动/旋转它；滑块调整姿态与尺寸。",
        "en": "Drag the gizmo on the object to move/rotate it; use the sliders for pose and size.",
    },
    "slider_x": {"zh": "x [m]", "en": "x [m]"},
    "slider_y": {"zh": "y [m]", "en": "y [m]"},
    "slider_z": {"zh": "z [m]", "en": "z [m]"},
    "slider_tx": {"zh": "tilt x [rad]", "en": "tilt x [rad]"},
    "slider_ty": {"zh": "tilt y [rad]", "en": "tilt y [rad]"},
    "slider_size": {"zh": "尺寸 [mm]", "en": "size [mm]"},
    "btn_release": {"zh": "释放抓取", "en": "Release grasp"},
    "btn_back": {"zh": "回到摆位", "en": "Back to placement"},
    "btn_reset": {"zh": "复位", "en": "Reset"},
    "btn_lang": {"zh": "English", "en": "中文"},
    "sec_status": {"zh": "### 抓取状态", "en": "### Grasp status"},
    "sec_forces": {"zh": "### 各指接触力", "en": "### Per-finger forces"},
    "sec_angles": {"zh": "### 关节角度", "en": "### Joint angles"},
    "adjusting": {"zh": "摆位中（物体已钉住）", "en": "Placement (object pinned)"},
    "pos": {"zh": "位置", "en": "position"},
    "radius": {"zh": "半径", "en": "radius"},
    "tilt": {"zh": "倾斜", "en": "tilt"},
    "col_finger": {"zh": "手指", "en": "finger"},
    "col_normal": {"zh": "法向 N", "en": "normal N"},
    "col_friction": {"zh": "摩擦 N", "en": "friction N"},
    "col_contacts": {"zh": "接触", "en": "contacts"},
    "total": {"zh": "合计", "en": "total"},
    "col_t": {"zh": "t", "en": "t"},
    "col_drift": {"zh": "漂移", "en": "drift"},
    "col_state": {"zh": "状态", "en": "state"},
    "held": {"zh": "已握持", "en": "held"},
    "not_held": {"zh": "未握持", "en": "not held"},
    "joints_deg": {"zh": "关节角 (deg)", "en": "joint angles (deg)"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=("cylinder", "sphere"), default="cylinder")
    ap.add_argument("--radius", type=float, default=0.026,
                    help="cylinder/sphere radius [m]; 0.026 is the largest "
                         "cylinder the fingers can wrap (see FINDINGS)")
    ap.add_argument("--weight", type=float, default=0.85)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--auto-run", action="store_true",
                    help="release the grasp immediately on startup (for testing "
                         "and to watch the hand grab without clicking)")
    args = ap.parse_args()

    import trimesh
    import viser
    from viser.extras import ViserUrdf

    g = WebGrasp(args.shape, args.radius, args.weight)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene = server.scene
    scene.set_up_direction("+z")

    # hand (static meshes, per-frame joint updates); the colour override makes
    # the hand black to match the desktop viewer's hand.xml
    urdf_path = make_hand_urdf()
    hand_vis = ViserUrdf(server, urdf_path, root_node_name="/hand",
                         mesh_color_override=(0.12, 0.12, 0.13))

    def object_mesh():
        if args.shape == "cylinder":
            return trimesh.creation.cylinder(radius=g.radius, height=2.0 * HALF_LENGTH,
                                             sections=48)
        return trimesh.creation.icosphere(radius=g.radius, subdivisions=2)

    obj_node = scene.add_mesh_trimesh(
        "/object", object_mesh(), position=g.data.xpos[g.obj_body],
        wxyz=g.data.xquat[g.obj_body],
    )
    gizmo = scene.add_transform_controls("/gizmo", scale=0.14,
                                         position=g.obj_target, wxyz=g.object_quat())

    # ---------------- GUI (single flat panel, language-toggleable) ----------------
    gui = server.gui
    ui: dict[str, object] = {}
    lang = {"current": "zh"}

    def tr(key: str) -> str:
        return TEXTS[key][lang["current"]]

    def sync_sliders_from_state():
        ui["s_x"].value, ui["s_y"].value, ui["s_z"].value = tuple(g.obj_target)
        ui["s_tx"].value, ui["s_ty"].value = g.tilt_x, g.tilt_y
        ui["s_size"].value = g.radius * 1000

    def on_release(_):
        g.start_run()
        sync_sliders_from_state()

    def on_back(_):
        g.back_to_placement()
        sync_sliders_from_state()

    def on_reset(_):
        g.reset_placement()
        sync_sliders_from_state()

    def on_toggle_lang(_):
        lang["current"] = "en" if lang["current"] == "zh" else "zh"
        build_gui()

    def build_gui():
        old_vals = {k: ui[k].value for k in
                    ("s_x", "s_y", "s_z", "s_tx", "s_ty", "s_size") if k in ui}
        for handle in list(ui.values()):
            handle.remove()
        ui.clear()
        gui.add_markdown(tr("hint"))
        ui["s_x"] = gui.add_slider(tr("slider_x"), g.center[0] - 0.08,
                                   g.center[0] + 0.08, 0.0005,
                                   old_vals.get("s_x", g.obj_target[0]))
        ui["s_y"] = gui.add_slider(tr("slider_y"), g.center[1] - 0.08,
                                   g.center[1] + 0.08, 0.0005,
                                   old_vals.get("s_y", g.obj_target[1]))
        ui["s_z"] = gui.add_slider(tr("slider_z"), g.center[2] - 0.08,
                                   g.center[2] + 0.08, 0.0005,
                                   old_vals.get("s_z", g.obj_target[2]))
        ui["s_tx"] = gui.add_slider(tr("slider_tx"), -2 * np.pi, 2 * np.pi, 0.02,
                                    old_vals.get("s_tx", g.tilt_x))
        ui["s_ty"] = gui.add_slider(tr("slider_ty"), -2 * np.pi, 2 * np.pi, 0.02,
                                    old_vals.get("s_ty", g.tilt_y))
        ui["s_size"] = gui.add_slider(tr("slider_size"), 15.0, 50.0, 0.5,
                                      old_vals.get("s_size", g.radius * 1000))
        ui["btn_release"] = gui.add_button(tr("btn_release"))
        ui["btn_release"].on_click(on_release)
        ui["btn_back"] = gui.add_button(tr("btn_back"))
        ui["btn_back"].on_click(on_back)
        ui["btn_reset"] = gui.add_button(tr("btn_reset"))
        ui["btn_reset"].on_click(on_reset)
        ui["btn_lang"] = gui.add_button(tr("btn_lang"))
        ui["btn_lang"].on_click(on_toggle_lang)
        gui.add_markdown(tr("sec_status"))
        ui["status_md"] = gui.add_markdown("—")
        gui.add_markdown(tr("sec_forces"))
        ui["forces_md"] = gui.add_markdown("—")
        gui.add_markdown(tr("sec_angles"))
        ui["angles_md"] = gui.add_markdown("—")

    build_gui()

    if args.auto_run:
        g.start_run()
    print(f"[sim_web] open http://localhost:{args.port} in a browser "
          f"(shape={args.shape}, r={args.radius * 1000:.0f} mm, m={args.weight} kg, "
          f"mode={'RUN' if args.auto_run else 'ADJUST'})")

    # The transform control's pose only changes while the user drags it; viser
    # delivers those changes through on_update, which is authoritative (polling
    # .position/.wxyz misses drags between frames).
    gizmo_update = {"pending": False, "pos": None, "quat": None}

    def on_gizmo_update(_event):
        gizmo_update["pending"] = True
        gizmo_update["pos"] = np.asarray(gizmo.position, dtype=float).copy()
        gizmo_update["quat"] = np.asarray(gizmo.wxyz, dtype=float).copy()

    gizmo.on_update(on_gizmo_update)

    # ---------------- main loop ----------------
    prev_gizmo_pos = np.asarray(gizmo.position, dtype=float)
    prev_gizmo_quat = np.asarray(gizmo.wxyz, dtype=float)
    obj_vel = np.zeros(3)
    prev_obj_pos = g.data.xpos[g.obj_body].copy()
    last_status = 0.0

    def slider_state():
        return (ui["s_x"].value, ui["s_y"].value, ui["s_z"].value,
                ui["s_tx"].value, ui["s_ty"].value, ui["s_size"].value)

    prev_slider = slider_state()

    try:
        while True:
            t0 = time.time()
            gizmo_pos = np.asarray(gizmo.position, dtype=float)
            gizmo_vel = (gizmo_pos - prev_gizmo_pos) / FRAME_DT
            cur_slider = slider_state()
            slider_changed = [abs(c - p) > 1e-9 for c, p in zip(cur_slider, prev_slider)]

            # consume live drag updates from the transform control
            user_drag = gizmo_update["pending"]
            if user_drag:
                gizmo_update["pending"] = False
                gizmo_pos = gizmo_update["pos"]
                gizmo_quat_drag = gizmo_update["quat"]

            if g.adjusting:
                # sliders and gizmo both feed the placement; the last one moved
                # wins.  The gizmo drags position AND orientation.
                target = np.array(cur_slider[:3])
                if user_drag:
                    target = gizmo_pos
                    ui["s_x"].value, ui["s_y"].value, ui["s_z"].value = tuple(target)
                    g.tilt_x, g.tilt_y = tilts_from_quat(gizmo_quat_drag)
                    ui["s_tx"].value, ui["s_ty"].value = g.tilt_x, g.tilt_y
                g.obj_target = target
                g.tilt_x, g.tilt_y = float(ui["s_tx"].value), float(ui["s_ty"].value)
                if abs(ui["s_size"].value / 1000 - g.radius) > 1e-9:
                    g.set_radius(ui["s_size"].value / 1000)
                    obj_node.remove()
                    obj_node = scene.add_mesh_trimesh(
                        "/object", object_mesh(), position=g.obj_target,
                        wxyz=g.object_quat(),
                    )
                g.step()
                # anchor the gizmo to the pinned object
                gizmo.position = g.obj_target
                gizmo.wxyz = g.object_quat()
                obj_node.position = g.obj_target
                obj_node.wxyz = g.object_quat()
                hand_vis.update_cfg(g.hand_cfg())
            else:
                # RUN mode.  Moving a slider teleports the object to the new
                # pose; dragging the gizmo pulls it with a spring; rotating
                # the gizmo's rings rotates the object kinematically so the
                # effect is immediate and visible.
                adr = g.model.jnt_qposadr[g.obj_joint]
                dadr = g.model.jnt_dofadr[g.obj_joint]
                if any(slider_changed):
                    g.tilt_x, g.tilt_y = float(ui["s_tx"].value), float(ui["s_ty"].value)
                    g.data.qpos[adr:adr + 3] = np.asarray(cur_slider[:3])
                    g.data.qpos[adr + 3:adr + 7] = g.object_quat()
                    g.data.qvel[dadr:dadr + 6] = 0.0
                    if abs(ui["s_size"].value / 1000 - g.radius) > 1e-9:
                        g.set_radius(ui["s_size"].value / 1000)
                        obj_node.remove()
                        obj_node = scene.add_mesh_trimesh(
                            "/object", object_mesh(), position=np.asarray(cur_slider[:3]),
                            wxyz=g.object_quat(),
                        )
                    gizmo.position = np.asarray(cur_slider[:3])
                    gizmo.wxyz = g.object_quat()
                    g.step()
                    obj_vel[:] = 0.0
                    prev_obj_pos = g.data.xpos[g.obj_body].copy()
                else:
                    obj_pos = g.data.xpos[g.obj_body]
                    delta = gizmo_pos - obj_pos
                    pull = np.zeros(3)
                    if float(np.linalg.norm(delta)) > SPRING_ENGAGE:
                        pull = np.clip(SPRING_K * delta + SPRING_D * (gizmo_vel - obj_vel),
                                       -SPRING_FMAX, SPRING_FMAX)
                        # keep the position sliders in sync so a later slider
                        # nudge starts from the object's current pose
                        ui["s_x"].value, ui["s_y"].value, ui["s_z"].value = tuple(gizmo_pos)
                    else:
                        gizmo.position = obj_pos
                    if user_drag and quat_angle(g.data.xquat[g.obj_body],
                                                gizmo_quat_drag) > 1e-3:
                        # kinematic rotation: instant and visible
                        g.data.qpos[adr + 3:adr + 7] = gizmo_quat_drag
                        g.data.qvel[dadr + 3:dadr + 6] = 0.0
                        gizmo.wxyz = gizmo_quat_drag
                        g.tilt_x, g.tilt_y = tilts_from_quat(gizmo_quat_drag)
                        ui["s_tx"].value, ui["s_ty"].value = g.tilt_x, g.tilt_y
                    g.step(pull=pull)
                    obj_pos = g.data.xpos[g.obj_body]
                    obj_vel = (obj_pos - prev_obj_pos) / FRAME_DT
                    prev_obj_pos = obj_pos
                obj_node.position = g.data.xpos[g.obj_body]
                obj_node.wxyz = g.data.xquat[g.obj_body]
                hand_vis.update_cfg(g.hand_cfg())

                if (np.linalg.norm(g.data.xpos[g.obj_body] - g.center) > DROP_RESPAWN
                        and g.contact_count() == 0):
                    print("\n[sim_web] object dropped - re-placing it", flush=True)
                    g.reset_placement()
                    sync_sliders_from_state()
                    gizmo.position = g.obj_target
                    gizmo.wxyz = g.object_quat()
                    prev_obj_pos = g.data.xpos[g.obj_body].copy()
                    obj_vel[:] = 0.0

            prev_gizmo_pos = np.asarray(gizmo.position, dtype=float)
            prev_gizmo_quat = np.asarray(gizmo.wxyz, dtype=float)
            prev_slider = cur_slider

            if g.elapsed - last_status > 0.25:
                last_status = g.elapsed
                cfg = g.hand_cfg()
                alines = [f"| {tr('col_finger')} | {tr('joints_deg')} |", "| --- | --- |"]
                for chain in g.hand.chains:
                    names = [j.name for j in chain.joints if j.type == "revolute"]
                    alines.append(
                        "| " + chain.name + " | "
                        + " ".join(f"{np.degrees(cfg[n]):+6.1f}" for n in names) + " |"
                    )
                ui["angles_md"].content = "\n".join(alines)

                forces = g.finger_forces()
                flines = [f"| {tr('col_finger')} | {tr('col_normal')} | "
                          f"{tr('col_friction')} | {tr('col_contacts')} |",
                          "| --- | --- | --- | --- |"]
                total = [0.0, 0.0, 0]
                for ch in g.hand.chains:
                    nrm, frc, ncon = forces.get(ch.name, (0.0, 0.0, 0))
                    flines.append(f"| {ch.name} | {nrm:.2f} | {frc:.2f} | {ncon} |")
                    total[0] += nrm
                    total[1] += frc
                    total[2] += ncon
                flines.append(f"| **{tr('total')}** | **{total[0]:.2f}** | "
                              f"**{total[1]:.2f}** | **{total[2]}** |")
                ui["forces_md"].content = "\n".join(flines)

                if g.adjusting:
                    ui["status_md"].content = (
                        f"**{tr('adjusting')}**\n\n"
                        f"- {tr('pos')} ({g.obj_target[0]:+.3f}, {g.obj_target[1]:+.3f}, "
                        f"{g.obj_target[2]:+.3f})\n"
                        f"- {tr('radius')} {g.radius * 1000:.1f} mm\n"
                        f"- {tr('tilt')} x {np.degrees(g.tilt_x):+.0f}° / "
                        f"y {np.degrees(g.tilt_y):+.0f}°"
                    )
                else:
                    drift = float(np.linalg.norm(
                        g.data.xpos[g.obj_body] - g.obj_target)) * 1000
                    nc = g.contact_count()
                    held = drift < 75 and nc >= 3
                    ui["status_md"].content = (
                        f"| {tr('col_t')} | {tr('col_drift')} | {tr('col_contacts')} | "
                        f"{tr('col_state')} |\n| --- | --- | --- | --- |\n"
                        f"| {g.elapsed:.1f} s | {drift:.1f} mm | {nc} | "
                        f"{tr('held' if held else 'not_held')} |"
                    )
                    print(f"\r[sim_web] t={g.elapsed:4.1f}s drift={drift:5.1f}mm "
                          f"contacts={nc}  ", end="", flush=True)

            dt = FRAME_DT - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    finally:
        print("[sim_web] closing")


if __name__ == "__main__":
    main()
