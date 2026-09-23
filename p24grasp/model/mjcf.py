#!/usr/bin/env python3
"""Convert p24_hand_right.urdf into a MuJoCo MJCF hand model.

The generated ``hand.xml`` keeps the URDF link/joint hierarchy, uses the
original STL meshes as visual/collision geoms, and adds position actuators on
active joints plus equality constraints for the URDF mimic joints.
"""
from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def rpy_to_rot(rpy: Tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    # standard matrix -> [w, x, y, z]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def fmt(vals) -> str:
    return " ".join(f"{float(v):.9g}" for v in vals)


def read_stl_vertices(path: Path) -> np.ndarray:
    """All triangle vertices of a binary or ASCII STL.

    MuJoCo itself loads these files fine, but the hand's finger meshes are
    curved shells whose convex hull is a poor collider (see ``_emit_geoms``),
    so the generator needs the raw geometry to size capsule colliders.
    """
    raw = path.read_bytes()
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + n * 50 == len(raw):
            tri = np.frombuffer(raw[84:], dtype=np.uint8).reshape(n, 50)
            return tri[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(np.float64)
    verts = []
    for line in raw.decode("utf-8", "ignore").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            verts.append([float(x) for x in parts[1:]])
    return np.asarray(verts, dtype=np.float64)


def parse_origin(elem) -> Tuple[np.ndarray, np.ndarray]:
    if elem is None:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    xyz = [float(x) for x in elem.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in elem.get("rpy", "0 0 0").split()]
    return np.asarray(xyz), rot_to_quat(rpy_to_rot(tuple(rpy)))


class URDF2MJCF:
    def __init__(self, urdf: Path, mesh_root: Path):
        self.urdf = urdf
        self.mesh_root = mesh_root
        self.tree = ET.parse(urdf)
        self.root = self.tree.getroot()
        self.links: Dict[str, dict] = {}
        self.joints: Dict[str, dict] = {}
        self.children: Dict[str, List[str]] = {}
        self.mesh_assets: Dict[str, str] = {}
        self.active_joints: List[str] = []
        self.mimic_joints: List[dict] = []
        self._vert_cache: Dict[str, Optional[np.ndarray]] = {}
        self._parse()

    def _parse(self) -> None:
        for link in self.root.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            data = {"name": name, "inertial": None, "visuals": []}
            if inertial is not None:
                pos, quat = parse_origin(inertial.find("origin"))
                mass = float(inertial.find("mass").get("value"))
                it = inertial.find("inertia")
                inertia = (
                    float(it.get("ixx", "0")),
                    float(it.get("iyy", "0")),
                    float(it.get("izz", "0")),
                    float(it.get("ixy", "0")),
                    float(it.get("ixz", "0")),
                    float(it.get("iyz", "0")),
                )
                data["inertial"] = {"pos": pos, "quat": quat, "mass": mass, "inertia": inertia}
            for vis in link.findall("visual"):
                geom = vis.find("geometry")
                mesh = geom.find("mesh") if geom is not None else None
                if mesh is not None:
                    data["visuals"].append(mesh.get("filename"))
            self.links[name] = data
            self.children.setdefault(name, [])

        for joint in self.root.findall("joint"):
            name = joint.get("name")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            jtype = joint.get("type")
            pos, quat = parse_origin(joint.find("origin"))
            axis_el = joint.find("axis")
            axis = [1.0, 0.0, 0.0] if axis_el is None else [float(x) for x in axis_el.get("xyz", "1 0 0").split()]
            limit = joint.find("limit")
            lower = upper = 0.0
            if limit is not None:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
            mimic = joint.find("mimic")
            j = {
                "name": name, "parent": parent, "child": child, "type": jtype,
                "pos": pos, "quat": quat, "axis": axis, "lower": lower, "upper": upper,
                "mimic": None,
            }
            if mimic is not None:
                j["mimic"] = {
                    "joint": mimic.get("joint"),
                    "multiplier": float(mimic.get("multiplier", "1")),
                    "offset": float(mimic.get("offset", "0")),
                }
                self.mimic_joints.append(j)
            else:
                if jtype == "revolute":
                    self.active_joints.append(name)
            self.joints[name] = j
            self.children.setdefault(parent, []).append(child)

    def _mesh_file_for(self, visual_file: str) -> str:
        from p24grasp.paths import assets_dir

        rel = Path(visual_file)
        parts = rel.parts
        if parts and parts[0] == "meshes":
            parts = parts[1:]

        name = parts[-1]
        # MuJoCo limits a mesh to 200k faces; use the decimated palm mesh
        if name == "P2.4_Hand_R_Palm.STL":
            parts = ("mujoco", "P2.4_Hand_R_Palm_180000.STL")
        elif name == "P2.4_Hand_L_Palm.STL":
            parts = ("mujoco", "P2.4_Hand_L_Palm_180000.STL")
        # absolute paths: the generated XML lives in build/, and MuJoCo
        # resolves mesh paths relative to the XML location
        return str((assets_dir() / "p24_meshes").joinpath(*parts))

    def _asset_name(self, link: str, i: int) -> str:
        return f"mesh_{link}_{i}"

    def _mesh_vertices(self, visual_file: str) -> Optional[np.ndarray]:
        if visual_file not in self._vert_cache:
            path = Path(self._mesh_file_for(visual_file))
            try:
                self._vert_cache[visual_file] = read_stl_vertices(path)
            except (OSError, ValueError):
                self._vert_cache[visual_file] = None
        return self._vert_cache[visual_file]

    def _capsule_for(self, link: dict) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Capsule approximating a finger phalanx, along its bone.

        MuJoCo collides a mesh geom using its convex hull.  The finger meshes
        are curved shells with a pad bulge (a phalanx measures ~77 x 24 x 38 mm),
        so their hulls reach well past the bone and swallow any object small
        enough to grasp: measured penetrations of 20-40 mm into a 32 mm-radius
        cylinder, which flings it out of the hand on the first contact solve.
        A capsule along the bone is a far tighter collider.
        """
        child = next((j for j in self.joints.values() if j["parent"] == link["name"]), None)
        if child is None or not link["visuals"]:
            return None
        bone = np.asarray(child["pos"], dtype=float)
        length = float(np.linalg.norm(bone))
        if length < 1e-6:
            return None
        verts = self._mesh_vertices(link["visuals"][0])
        if verts is None or len(verts) == 0:
            return None

        axis = bone / length
        proj = verts @ axis
        perp = np.linalg.norm(verts - np.outer(proj, axis), axis=1)
        # The mesh is offset from the bone toward the pad side, so a capsule
        # centred on the bone leaves the visual pad sticking out past the
        # collision surface and the object sinks into the rendered mesh.  Put
        # the capsule axis where the meat is (the mesh centroid's perpendicular
        # offset) and grow the radius to the 75th percentile so its surface
        # tracks the pad.  p50 + 5 mm caps outliers like the thumb base's
        # webbing, whose full spread would make the capsule absurdly fat.
        centroid = verts.mean(axis=0)
        offset = centroid - (centroid @ axis) * axis
        radius = float(min(np.percentile(perp, 75),
                           np.percentile(perp, 50) + 0.005))
        lo, hi = float(proj.min()), float(proj.max())
        return axis * lo + offset, axis * hi + offset, radius

    def _emit_geoms(self, out: List[str], link: dict, ind: str) -> None:
        if not link["visuals"]:
            # keep invisible/tip links visible as a small contact sphere
            out.append(
                f'{ind}<geom name="{link["name"]}_tip" type="sphere" size="0.008" '
                f'rgba="0.5 0.5 0.55 0.5" density="0" contype="1" conaffinity="1"/>'
            )
            return

        is_palm = link["name"] == "palm"
        for i, vf in enumerate(link["visuals"]):
            asset = self._asset_name(link["name"], i)
            path = self._mesh_file_for(vf)
            self.mesh_assets[asset] = path
            is_wrist = "Wrist" in Path(vf).name
            if is_palm and not is_wrist:
                # the palm is the opposition surface of a power grasp, and its
                # convex hull is a decent stand-in for the real palm
                out.append(
                    f'{ind}<geom name="{link["name"]}_mesh{i}" type="mesh" mesh="{asset}" '
                    f'rgba="0.12 0.12 0.13 1" density="0" contype="1" conaffinity="1" '
                    f'friction="0.7 0.02 0.02"/>'
                )
            else:
                # visual only; collision comes from the capsule below
                out.append(
                    f'{ind}<geom name="{link["name"]}_mesh{i}" type="mesh" mesh="{asset}" '
                    f'rgba="0.12 0.12 0.13 1" density="0" contype="0" conaffinity="0"/>'
                )

        if not is_palm:
            cap = self._capsule_for(link)
            if cap is not None:
                p0, p1, radius = cap
                out.append(
                    f'{ind}<geom name="{link["name"]}_cap" type="capsule" '
                    f'fromto="{fmt(p0)} {fmt(p1)}" size="{radius:.6g}" '
                    f'rgba="0 0 0 0" density="0" '
                    f'contype="1" conaffinity="1" friction="0.7 0.02 0.02"/>'
                )

    def _emit_inertial(self, out: List[str], link: dict, ind: str) -> None:
        if link["inertial"] is None:
            return
        it = link["inertial"]
        pos = it["pos"]
        mass = it["mass"]
        ixx, iyy, izz, ixy, ixz, iyz = it["inertia"]
        out.append(
            f'{ind}<inertial pos="{fmt(pos)}" mass="{mass:.9g}" '
            f'fullinertia="{ixx:.9g} {iyy:.9g} {izz:.9g} {ixy:.9g} {ixz:.9g} {iyz:.9g}"/>'
        )

    def _emit_body(self, out: List[str], link_name: str, ind: str) -> None:
        link = self.links[link_name]
        # find the joint whose child is this link
        joint = next((j for j in self.joints.values() if j["child"] == link_name), None)

        pos = joint["pos"] if joint is not None else np.zeros(3)
        quat = joint["quat"] if joint is not None else np.array([1.0, 0.0, 0.0, 0.0])

        if link_name == "palm":
            out.append(f'<body name="{link_name}" pos="0 0 0" quat="1 0 0 0">')
        else:
            out.append(f'{ind}<body name="{link_name}" pos="{fmt(pos)}" quat="{fmt(quat)}">')

        child_ind = ind + "  "
        if joint is not None and joint["type"] == "revolute":
            axis = joint["axis"]
            out.append(
                f'{child_ind}<joint name="{joint["name"]}" type="hinge" axis="{fmt(axis)}" '
                f'limited="true" range="{joint["lower"]:.9g} {joint["upper"]:.9g}" damping="0.001" armature="0.0001"/>'
            )

        self._emit_inertial(out, link, child_ind)
        self._emit_geoms(out, link, child_ind)

        for child in self.children.get(link_name, []):
            self._emit_body(out, child, child_ind)

        out.append(f"{ind}</body>")

    def build(self) -> str:
        out: List[str] = []
        out.append('<mujoco model="p24_r_hand">')
        out.append('  <compiler angle="radian" autolimits="false"/>')
        out.append("  <asset>")
        # assets are collected while emitting bodies
        out.append("  </asset>")
        out.append("  <worldbody>")
        self._emit_body(out, "palm", "    ")
        out.append("  </worldbody>")

        if self.active_joints:
            out.append("  <actuator>")
            for jname in self.active_joints:
                j = self.joints[jname]
                out.append(
                    f'    <position name="act_{jname}" joint="{jname}" '
                    f'ctrllimited="true" ctrlrange="{j["lower"]:.9g} {j["upper"]:.9g}" '
                    f'forcelimited="true" forcerange="-1 1" kp="0.5" kv="0.05"/>'
                )
            out.append("  </actuator>")

        if self.mimic_joints:
            out.append("  <equality>")
            for j in self.mimic_joints:
                m = j["mimic"]
                driver, mimic = m["joint"], j["name"]
                mult, offset = m["multiplier"], m["offset"]
                # MuJoCo's <joint> equality constrains
                #   joint1 = polycoef[0] + polycoef[1]*joint2 + polycoef[2]*joint2^2 + ...
                # i.e. joint1 as a polynomial *of joint2*, the opposite of the
                # c0 + c1*q1 + c2*q2 reading the name "polycoef" suggests.
                # Writing the intuitive "offset multiplier -1 0 0" therefore does
                # NOT reproduce the URDF mimic: it silently yields the quadratic
                # q_driver - q_mimic + q_mimic^2 = 0, which leaves the mimic joint
                # effectively unconstrained and drifting under gravity.
                if abs(mult) > 1e-12:
                    # URDF mimic  q_mimic = offset + mult*q_driver
                    #   ->  q_driver = (-offset/mult) + (1/mult)*q_mimic
                    coeffs = f"{-offset / mult + 0.0:.9g} {1.0 / mult:.9g} 0 0 0"
                else:
                    # constant mimic  q_mimic = offset  ->  q_mimic = offset + 0*q_driver
                    driver, mimic = mimic, driver
                    coeffs = f"{offset:.9g} 0 0 0 0"
                out.append(
                    f'    <joint joint1="{driver}" joint2="{mimic}" polycoef="{coeffs}"/>'
                )
            out.append("  </equality>")

        out.append("</mujoco>")

        # replace placeholder asset block with collected mesh assets
        asset_block = []
        for asset, path in self.mesh_assets.items():
            # use path relative to output XML location; keep absolute for robustness here
            asset_block.append(f'    <mesh name="{asset}" file="{path}"/>')
        text = "\n".join(out)
        text = text.replace(
            '  <asset>\n  </asset>',
            "  <asset>\n" + "\n".join(asset_block) + "\n  </asset>",
            1,
        )
        return text + "\n"


def build(out_path=None) -> Path:
    """Generate hand.xml into the build directory (or ``out_path``)."""
    from p24grasp.paths import assets_dir, hand_xml_path

    out = Path(out_path) if out_path is not None else hand_xml_path()
    conv = URDF2MJCF(assets_dir() / "p24_hand_right.urdf", assets_dir() / "p24_meshes")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(conv.build(), encoding="utf-8")
    print(f"[build_mjcf] wrote {out}")
    print(f"[build_mjcf] active joints: {len(conv.active_joints)}, "
          f"mimic joints: {len(conv.mimic_joints)}")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="URDF -> MJCF generator")
    ap.add_argument("--out", default=None, help="output path (default: build/hand.xml)")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
