"""Minimal URDF parser + kinematics/dynamics for the P24 right hand.

Only what the grasp controller needs is implemented:
  * serial-chain forward kinematics per finger
  * numeric tip Jacobian (point contact at fingertip)
  * potential energy and gravity torque (either from URDF inertial data or
    from an identified mass/first-moment parameter vector)
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# small rigid-body helpers
# --------------------------------------------------------------------------- #
def _rpy_to_rot(rpy: Tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    k = axis / n
    K = _skew(k)
    c, s = math.cos(angle), math.sin(angle)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _origin_to_transform(pos: np.ndarray, rpy: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rot(rpy)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def _parse_origin(elem: Optional[ET.Element]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if elem is None:
        return np.zeros(3), (0.0, 0.0, 0.0)
    xyz = elem.get("xyz", "0 0 0")
    rpy = elem.get("rpy", "0 0 0")
    pos = np.array([float(x) for x in xyz.split()], dtype=float)
    rpyv = tuple(float(x) for x in rpy.split())
    return pos, rpyv


# --------------------------------------------------------------------------- #
# data structures
# --------------------------------------------------------------------------- #
@dataclass
class LinkData:
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    origin_com: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class JointData:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    velocity: float
    mimic_joint: Optional[str] = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class FingerChain:
    name: str
    joints: List[JointData]
    links: List[LinkData]
    active_joint_names: List[str]
    tip_link: str
    active_indices: List[int]  # index into joints for each active joint
    mimic_index: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
class HandModel:
    """Kinematic/dynamic model of the P24 hand built from its URDF."""

    def __init__(self, urdf_path: str | Path, gravity: np.ndarray | None = None):
        self.urdf_path = Path(urdf_path)
        self.gravity = (
            np.array([0.0, 0.0, -9.81]) if gravity is None else np.asarray(gravity, dtype=float)
        )
        self.links: Dict[str, LinkData] = {}
        self.joints: Dict[str, JointData] = {}
        self.child_to_joint: Dict[str, JointData] = {}
        self.root_link = "palm"
        self.chains: List[FingerChain] = []

        self._parse()

        # global active-joint order: finger chains concatenated
        self.active_joint_names: List[str] = []
        self.chain_slices: Dict[str, Tuple[int, int]] = {}
        for chain in self.chains:
            start = len(self.active_joint_names)
            self.active_joint_names.extend(chain.active_joint_names)
            self.chain_slices[chain.name] = (start, len(self.active_joint_names))
        self.n_active = len(self.active_joint_names)

        # links that can contribute gravity torque (base "palm" excluded)
        self.inertial_links: List[Tuple[str, LinkData]] = []
        for chain in self.chains:
            for link in chain.links:
                if link.mass > 0.0:
                    self.inertial_links.append((chain.name, link))

    # ------------------------------------------------------------------ #
    def _parse(self) -> None:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for link_elem in root.findall("link"):
            name = link_elem.get("name")
            link = LinkData(name=name)
            inertial = link_elem.find("inertial")
            if inertial is not None:
                pos, rpy = _parse_origin(inertial.find("origin"))
                link.origin_com = _origin_to_transform(pos, rpy)
                link.mass = float(inertial.find("mass").get("value"))
                inertia = inertial.find("inertia")
                if inertia is not None:
                    vals = {
                        "ixx": float(inertia.get("ixx", "0")),
                        "ixy": float(inertia.get("ixy", "0")),
                        "ixz": float(inertia.get("ixz", "0")),
                        "iyy": float(inertia.get("iyy", "0")),
                        "iyz": float(inertia.get("iyz", "0")),
                        "izz": float(inertia.get("izz", "0")),
                    }
                    link.inertia = np.array(
                        [
                            [vals["ixx"], vals["ixy"], vals["ixz"]],
                            [vals["ixy"], vals["iyy"], vals["iyz"]],
                            [vals["ixz"], vals["iyz"], vals["izz"]],
                        ]
                    )
                link.com = np.asarray(pos, dtype=float)
            self.links[name] = link

        for joint_elem in root.findall("joint"):
            name = joint_elem.get("name")
            jtype = joint_elem.get("type")
            parent = joint_elem.find("parent").get("link")
            child = joint_elem.find("child").get("link")
            pos, rpy = _parse_origin(joint_elem.find("origin"))
            axis_elem = joint_elem.find("axis")
            axis = (
                np.array([1.0, 0.0, 0.0])
                if axis_elem is None
                else np.array([float(x) for x in axis_elem.get("xyz", "1 0 0").split()])
            )
            limit = joint_elem.find("limit")
            lower = upper = effort = velocity = 0.0
            if limit is not None:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
                effort = float(limit.get("effort", "0"))
                velocity = float(limit.get("velocity", "0"))

            mimic = joint_elem.find("mimic")
            mj = mm = mo = None
            if mimic is not None:
                mj = mimic.get("joint")
                mm = float(mimic.get("multiplier", "1"))
                mo = float(mimic.get("offset", "0"))

            joint = JointData(
                name=name,
                type=jtype,
                parent=parent,
                child=child,
                origin=_origin_to_transform(pos, rpy),
                axis=axis,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
                mimic_joint=mj,
                mimic_multiplier=mm,
                mimic_offset=mo,
            )
            self.joints[name] = joint
            self.child_to_joint[child] = joint

        self._build_chains()

    def _build_chains(self) -> None:
        tip_links = sorted([n for n in self.links if n.endswith("_tip_link")])
        for tip in tip_links:
            chain_name = tip.replace("_tip_link", "")
            joints: List[JointData] = []
            links: List[LinkData] = []
            cur = tip
            while cur != self.root_link:
                joint = self.child_to_joint[cur]
                joints.insert(0, joint)
                links.insert(0, self.links[cur])
                cur = joint.parent

            active_names: List[str] = []
            active_indices: List[int] = []
            for idx, j in enumerate(joints):
                if j.type == "revolute" and j.mimic_joint is None:
                    active_names.append(j.name)
                    active_indices.append(idx)

            self.chains.append(
                FingerChain(
                    name=chain_name,
                    joints=joints,
                    links=links,
                    active_joint_names=active_names,
                    tip_link=tip,
                    active_indices=active_indices,
                )
            )

    # ------------------------------------------------------------------ #
    # forward kinematics
    # ------------------------------------------------------------------ #
    def chain_slice(self, chain_name: str) -> Tuple[int, int]:
        return self.chain_slices[chain_name]

    def joint_values(self, chain: FingerChain, q_chain: np.ndarray) -> Dict[str, float]:
        vals: Dict[str, float] = {}
        active_pos = {name: i for i, name in enumerate(chain.active_joint_names)}
        for j in chain.joints:
            if j.type != "revolute":
                continue
            if j.mimic_joint is None:
                vals[j.name] = float(q_chain[active_pos[j.name]])
        # resolve mimics iteratively (single-level mimic in this URDF)
        for j in chain.joints:
            if j.type == "revolute" and j.mimic_joint is not None:
                vals[j.name] = (
                    j.mimic_multiplier * vals[j.mimic_joint] + j.mimic_offset
                )
        return vals

    def fk_chain(
        self, chain: FingerChain, q_chain: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return 4x4 transforms of each child link, keyed by link name."""
        vals = self.joint_values(chain, q_chain)
        T = np.eye(4)
        out: Dict[str, np.ndarray] = {}
        for j in chain.joints:
            T = T @ j.origin
            if j.type == "revolute":
                R4 = np.eye(4)
                R4[:3, :3] = _rot_about_axis(j.axis, vals[j.name])
                T = T @ R4
            elif j.type != "fixed":
                raise NotImplementedError(f"joint type {j.type} not supported")
            out[j.child] = T.copy()
        return out

    def link_com_world(
        self, chain: FingerChain, link: LinkData, transforms: Dict[str, np.ndarray]
    ) -> np.ndarray:
        T_link = transforms[link.name]
        return (T_link @ link.origin_com)[:3, 3]

    # ------------------------------------------------------------------ #
    # tip kinematics
    # ------------------------------------------------------------------ #
    def tip_pose(self, chain: FingerChain, q_chain: np.ndarray) -> np.ndarray:
        T = self.fk_chain(chain, q_chain)[chain.tip_link]
        return T

    def tip_point(
        self,
        chain: FingerChain,
        q_chain: np.ndarray,
        contact_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        T = self.tip_pose(chain, q_chain)
        p = T[:3, 3].copy()
        if contact_offset is not None:
            p = p + T[:3, :3] @ np.asarray(contact_offset, dtype=float)
        return p

    def tip_jacobian(
        self,
        chain: FingerChain,
        q_chain: np.ndarray,
        contact_offset: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        n = len(chain.active_joint_names)
        J = np.zeros((3, n))
        for i in range(n):
            qp = q_chain.copy()
            qm = q_chain.copy()
            qp[i] += eps
            qm[i] -= eps
            J[:, i] = (
                self.tip_point(chain, qp, contact_offset)
                - self.tip_point(chain, qm, contact_offset)
            ) / (2 * eps)
        return J

    # ------------------------------------------------------------------ #
    # gravity / potential
    # ------------------------------------------------------------------ #
    def potential(self, q: np.ndarray) -> float:
        p = 0.0
        for chain in self.chains:
            a, b = self.chain_slices[chain.name]
            transforms = self.fk_chain(chain, q[a:b])
            for link in chain.links:
                if link.mass <= 0.0:
                    continue
                com = self.link_com_world(chain, link, transforms)
                p += link.mass * float(self.gravity @ com)
        return p

    def gravity_torque(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        g = np.zeros(self.n_active)
        for i in range(self.n_active):
            qp = np.asarray(q, dtype=float).copy()
            qm = np.asarray(q, dtype=float).copy()
            qp[i] += eps
            qm[i] -= eps
            g[i] = (self.potential(qp) - self.potential(qm)) / (2 * eps)
        return g

    def potential_from_params(self, q: np.ndarray, theta: np.ndarray) -> float:
        """theta = [m_i, hx_i, hy_i, hz_i] for each inertial link (base excluded)."""
        p = 0.0
        k = 0
        for chain in self.chains:
            a, b = self.chain_slices[chain.name]
            transforms = self.fk_chain(chain, q[a:b])
            for link in chain.links:
                if link.mass <= 0.0:
                    continue
                m, hx, hy, hz = theta[k : k + 4]
                k += 4
                T_link = transforms[link.name]
                origin = T_link[:3, 3]
                R = T_link[:3, :3]
                h = np.array([hx, hy, hz])
                p += m * float(self.gravity @ origin) + float(self.gravity @ (R @ h))
        return p

    def gravity_torque_from_params(
        self, q: np.ndarray, theta: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        g = np.zeros(self.n_active)
        for i in range(self.n_active):
            qp = np.asarray(q, dtype=float).copy()
            qm = np.asarray(q, dtype=float).copy()
            qp[i] += eps
            qm[i] -= eps
            g[i] = (self.potential_from_params(qp, theta) - self.potential_from_params(qm, theta)) / (2 * eps)
        return g

    @property
    def n_gravity_params(self) -> int:
        return 4 * len(self.inertial_links)

    def gravity_param_index(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        k = 0
        for chain in self.chains:
            for link in chain.links:
                if link.mass > 0.0:
                    out[link.name] = k
                    k += 4
        return out
