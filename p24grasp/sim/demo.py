"""Static demo rendering: build a scene, drive the IK pose, render PNGs.

This is the offline/preview path of the original ``sim_grasp.py`` - no
dynamics, just pose + render (matplotlib or MuJoCo offscreen).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import mujoco
from PIL import Image

from p24grasp.model.urdf import HandModel
from p24grasp.kinematics.ik import CENTER, HALF_LENGTH, solve_angles
from p24grasp.sim.scene import object_geom_xml
from p24grasp.paths import assets_dir, build_dir, ensure_hand_xml, outputs_dir

def build_scene(hand_xml: str, radius: float, weight: float,
                shape: str = "cylinder") -> str:
    mass = weight / 9.81
    object_body = f"""
    <body name="object" pos="{CENTER[0]} {CENTER[1]} {CENTER[2]}">
{object_geom_xml(radius, mass, shape)}
    </body>
"""
    scene = hand_xml.replace("</worldbody>", object_body + "  </worldbody>", 1)
    extras = """
  <option gravity="0 0 -9.81" cone="pyramidal" impratio="5"/>
  <visual>
    <global offwidth="960" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
"""
    scene = scene.replace("</mujoco>", extras + "</mujoco>", 1)
    return scene


def set_hand_pose(model: mujoco.MjModel, data: mujoco.MjData, hand_model: HandModel, q_sol: np.ndarray):
    # build joint_name -> value for all revolute joints (including mimics)
    values = {}
    for chain in hand_model.chains:
        a, b = hand_model.chain_slice(chain.name)
        values.update(hand_model.joint_values(chain, q_sol[a:b]))

    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname in values:
            qpos_addr = model.jnt_qposadr[jid]
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[qpos_addr] = values[jname]

    # position actuators track the solved angles
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        jname = aname.removeprefix("act_")
        if jname in values:
            data.ctrl[aid] = values[jname]


def render_scene(scene: str, hand_model: HandModel, q_sol: np.ndarray, out_path: Path):
    tmp = build_dir() / "_scene_demo.xml"
    tmp.write_text(scene, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(tmp))
    data = mujoco.MjData(model)

    set_hand_pose(model, data, hand_model, q_sol)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = CENTER
    cam.distance = 0.42
    cam.azimuth = 180.0
    cam.elevation = -20.0
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    Image.fromarray(rgb).save(out_path)
    renderer.close()
    return out_path




# --------------------------------------------------------------------------- #
# headless matplotlib preview (no GL context required)
# --------------------------------------------------------------------------- #
def read_stl_vertices(path: Path, max_points: int = 6000):
    import struct
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    step = max(1, n // max_points)
    verts = []
    for i in range(0, n, step):
        off = 84 + i * 50
        tri = struct.unpack("<12fH", data[off:off + 50])
        verts.extend(
            [tri[0:3], tri[3:6], tri[6:9]]
        )
    return np.asarray(verts, dtype=float)


def link_visual_files(urdf_path: Path, mesh_root: Path):
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    files = {}
    for link in root.findall("link"):
        name = link.get("name")
        vf = []
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is not None:
                vf.append(mesh.get("filename"))
        files[name] = vf
    return files


def mesh_path_for(visual_file, mesh_root):
    parts = Path(visual_file).parts
    if parts and parts[0] == "meshes":
        parts = parts[1:]
    name = parts[-1]
    if name == "P2.4_Hand_R_Palm.STL":
        parts = ("mujoco", "P2.4_Hand_R_Palm_180000.STL")
    elif name == "P2.4_Hand_L_Palm.STL":
        parts = ("mujoco", "P2.4_Hand_L_Palm_180000.STL")
    return mesh_root.joinpath(*parts)


def preview_scene(hand_model, q_sol, radius, weight, out_path, urdf_path, mesh_root,
                  shape: str = "cylinder", center=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    center = CENTER if center is None else np.asarray(center, dtype=float)
    visuals = link_visual_files(Path(urdf_path), Path(mesh_root))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # palm
    for vf in visuals.get("palm", []):
        path = mesh_path_for(vf, Path(mesh_root))
        if not path.exists():
            continue
        verts = read_stl_vertices(path, 4000)
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.02, c="#b9bdc9", alpha=0.5)

    # fingers with FK transforms
    for chain in hand_model.chains:
        a, b = hand_model.chain_slice(chain.name)
        transforms = hand_model.fk_chain(chain, q_sol[a:b])
        for link in chain.links:
            if link.name not in transforms:
                continue
            T = transforms[link.name]
            for vf in visuals.get(link.name, []):
                path = mesh_path_for(vf, Path(mesh_root))
                if not path.exists():
                    continue
                verts = read_stl_vertices(path, 2000)
                p = np.hstack([verts, np.ones((len(verts), 1))]) @ T.T
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.02, c="#8a919e", alpha=0.5)

    # object surface
    if shape == "sphere":
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 16)
        ax.plot_surface(
            radius * np.outer(np.cos(u), np.sin(v)) + center[0],
            radius * np.outer(np.sin(u), np.sin(v)) + center[1],
            radius * np.outer(np.ones_like(u), np.cos(v)) + center[2],
            color="#3e8bd1", alpha=0.55, linewidth=0,
        )
    else:
        theta = np.linspace(0, 2 * np.pi, 36)
        x = np.linspace(-HALF_LENGTH, HALF_LENGTH, 9)
        X, T = np.meshgrid(x, theta)
        ax.plot_surface(
            X + center[0],
            radius * np.sin(T) + center[1],
            radius * np.cos(T) + center[2],
            color="#3e8bd1", alpha=0.55, linewidth=0,
        )

    rng = np.array([-0.08, 0.16]), np.array([-0.10, 0.16]), np.array([0.05, 0.27])
    ax.set_xlim(*rng[0]); ax.set_ylim(*rng[1]); ax.set_zlim(*rng[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"P24 hand grasp  {shape} radius={radius*1000:.0f} mm  "
                 f"weight={weight*9.81:.2f} N")
    ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect((0.24, 0.26, 0.22))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(assets_dir() / "p24_hand_right.urdf"))
    ap.add_argument("--outdir", default=str(outputs_dir("render")))
    ap.add_argument("--squeeze-gain", type=float, default=0.0008)
    ap.add_argument("--shape", choices=("cylinder", "sphere"), default="cylinder")
    ap.add_argument("--preview", action="store_true", help="use matplotlib headless preview instead of MuJoCo GL renderer")
    args = ap.parse_args()

    hand_xml = ensure_hand_xml().read_text(encoding="utf-8")
    hand_model = HandModel(args.urdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    configs = [
        (0.015, 0.2),
        (0.025, 0.5),
        (0.035, 1.0),
        (0.035, 2.0),
    ]
    for radius, weight in configs:
        q_sol = solve_angles(hand_model, radius, weight, squeeze_gain=args.squeeze_gain,
                             shape=args.shape)
        scene = build_scene(hand_xml, radius, weight, shape=args.shape)
        name = f"r{int(radius*1000):03d}_w{int(weight*1000):03d}.png"
        if args.preview:
            path = preview_scene(hand_model, q_sol, radius, weight, outdir / name, args.urdf,
                                 assets_dir() / "p24_meshes", shape=args.shape)
        else:
            path = render_scene(scene, hand_model, q_sol, outdir / name)
        print(f"[sim_grasp] radius={radius:.3f} weight={weight:.2f} -> {path.name}")

    print("[sim_grasp] done")


if __name__ == "__main__":
    main()
