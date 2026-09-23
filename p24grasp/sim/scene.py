"""Interactive simulation scene: free object + contact filtering.

Shared by the RL environment, both viewers and the PyRoki glue.  Contains the
contact bitmask scheme (object {1,2}, palm {1}, fingers {2} -> finger-to-finger
contact on, palm-finger off), the placement fitting and the placement probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mujoco
from scipy.optimize import minimize

from p24grasp.kinematics.ik import (  # noqa: E402
    AXIS,
    CENTER,
    HALF_LENGTH,
    chain_limits,
    fist_center,
    radial_direction,
    solve_angles,
    tip_center_radius,
)
from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.paths import build_dir  # noqa: E402

TIP_RADIUS = 0.008  # contact sphere attached to every tip link by build_mjcf.py
SHAPES = ("cylinder", "sphere")
OBJECT_BODY = "object"
OBJECT_JOINT = "obj_free"


def object_geom_xml(radius: float, mass: float, shape: str = "cylinder",
                    indent: str = "    ", attrs: str = "") -> str:
    """MJCF for the grasped object's collision/visual geom.

    ``mass`` is only emitted when non-zero; a scene that supplies its own
    ``<inertial>`` passes ``mass=0`` so the geom stays massless.
    """
    mass_attr = f' mass="{mass:.6g}"' if mass else ""
    if shape == "sphere":
        return (
            f'{indent}<geom name="object_geom" type="sphere" size="{radius}"'
            f'{mass_attr} rgba="0.25 0.55 0.85 0.9" friction="1.0 0.02 0.02"{attrs}/>'
        )
    # no quat: the cylinder's local +z is the world +z axis (see AXIS)
    return (
        f'{indent}<geom name="object_geom" type="cylinder" size="{radius} {HALF_LENGTH}"'
        f'{mass_attr} rgba="0.25 0.55 0.85 0.9" friction="1.0 0.02 0.02"{attrs}/>'
    )


def object_inertia(radius: float, mass: float, shape: str = "cylinder"):
    """Solid-object diaginertia in the body frame (body +z = cylinder axis)."""
    if shape == "sphere":
        i = 0.4 * mass * radius ** 2
        return i, i, i
    i_axis = 0.5 * mass * radius ** 2
    i_perp = mass * (3.0 * radius ** 2 + (2.0 * HALF_LENGTH) ** 2) / 12.0
    return i_perp, i_perp, i_axis


def build_interactive_scene(
    hand_xml: str,
    radius: float,
    mass: float,
    center: np.ndarray,
    kp: float | None = None,
    kv: float | None = None,
    force_range: float | None = None,
    timestep: float = 0.002,
    gravity: float = 9.81,
    shape: str = "cylinder",
    pose_sliders: bool = False,
) -> str:
    # hand.xml + a free-floating object at the requested centre.
    #
    # Contact filtering by contype/conaffinity bitmasks (a pair collides iff
    # contype1 & conaffinity2 or conaffinity1 & contype2):
    #   * object .......... {1, 2}  - collides with everything below
    #   * palm ............ {1}     - collides with the object only
    #   * finger capsules
    #     + tip spheres ... {2}     - collide with the object AND each other,
    #                                 so adjacent fingers can touch
    # This keeps the palm out of finger contacts: its convex hull bridges the
    # palm bowl and would swallow the fingers (the original 16-24 mm
    # interpenetration), while finger capsules are tight enough for
    # finger-to-finger contact to be meaningful.
    # build_mjcf already emits the wrist shell and the visual meshes as
    # visual-only; this only re-filters the collision geoms.
    scene = hand_xml.replace('contype="1" conaffinity="1"', 'contype="2" conaffinity="2"')
    scene = scene.replace(
        '<geom name="palm_mesh0" type="mesh" mesh="mesh_palm_0" '
        'rgba="0.12 0.12 0.13 1" density="0" contype="2" conaffinity="2"',
        '<geom name="palm_mesh0" type="mesh" mesh="mesh_palm_0" '
        'rgba="0.12 0.12 0.13 1" density="0" contype="1" conaffinity="1"',
    )
    if kp is not None:
        scene = scene.replace('kp="0.5"', f'kp="{kp:.6g}"')
    if kv is not None:
        scene = scene.replace('kv="0.05"', f'kv="{kv:.6g}"')
    if force_range is not None:
        scene = scene.replace(
            'forcelimited="true" forcerange="-1 1"',
            f'forcelimited="true" forcerange="{-force_range:.6g} {force_range:.6g}"',
        )

    # solid-object inertia.  The cylinder's axis is world +z (see
    # sim_grasp.AXIS); tilting it is done by rotating the body, so no quat is
    # baked into the geom here.
    i_a, i_b, i_c = object_inertia(radius, mass, shape)
    object_body = f"""
    <body name="{OBJECT_BODY}" pos="{center[0]} {center[1]} {center[2]}">
      <freejoint name="{OBJECT_JOINT}"/>
      <inertial pos="0 0 0" mass="{mass:.9g}" diaginertia="{i_a:.9g} {i_b:.9g} {i_c:.9g}"/>
{object_geom_xml(radius, 0.0, shape, indent="      ", attrs=' contype="3" conaffinity="3"')}
    </body>
"""
    scene = scene.replace("</worldbody>", object_body + "  </worldbody>", 1)

    if pose_sliders:
        # MuJoCo's viewer has no way to declare a custom slider, but its Control
        # panel builds one per actuator.  These are zero-gain and bound to an
        # existing joint, so they apply no force - the script reads their ctrl
        # values back as the object's placement and size.
        half = 0.08
        full = 2.0 * np.pi  # full rotation, both directions
        rows = [
            ("pose_x", center[0] - half, center[0] + half, "object x [m]"),
            ("pose_y", center[1] - half, center[1] + half, "object y [m]"),
            ("pose_z", center[2] - half, center[2] + half, "object z [m]"),
            ("pose_tilt_x", -full, full, "axis tilt about x [rad]"),
            ("pose_tilt_y", -full, full, "axis tilt about y [rad]"),
            ("obj_size", 0.015, 0.050, "object radius [m]"),
        ]
        sliders = "\n".join(
            f'    <general name="{name}" joint="index_dh_joint_1" gainprm="0 0 0" '
            f'biastype="none" ctrllimited="true" ctrlrange="{lo:.6g} {hi:.6g}"/>'
            for name, lo, hi, _comment in rows
        )
        scene = scene.replace("  </actuator>", sliders + "\n  </actuator>", 1)
        scene = scene.replace(
            "</mujoco>",
            "  <!-- zero-gain sliders: object placement, read back by sim_viewer -->\n"
            "</mujoco>",
            1,
        )

    extras = f"""
  <option gravity="0 0 {-gravity:.6g}" cone="elliptic" timestep="{timestep}"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
"""
    return scene.replace("</mujoco>", extras + "</mujoco>", 1)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


CENTRE_CACHE = build_dir() / "placement_cache.json"


def _load_centre_cache() -> dict:
    if not CENTRE_CACHE.exists():
        return {}
    try:
        return json.loads(CENTRE_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_centre_cache(key: str, center: np.ndarray) -> None:
    cache = _load_centre_cache()
    cache[key] = [float(v) for v in center]
    try:
        CENTRE_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass
def _radial(hand: HandModel, chain, q: np.ndarray, center: np.ndarray,
            shape: str = "cylinder"):
    """(unit direction from the object centre toward the tip, its length)."""
    return radial_direction(hand.tip_point(chain, q), center, AXIS, shape)


def finger_reach(hand: HandModel, center: np.ndarray, radius: float, weight: float,
                 squeeze_gain: float, tip_radius: float, shape: str = "cylinder"):
    """Per-finger (open-pose surface gap [m], IK residual [m]) for a placement.

    Both are properties of the *requested* grasp, independent of the simulation:
    a negative gap means the fingertip already penetrates the object when the
    hand is fully open, and a large residual means the IK target is out of reach
    and the "expected" angle can never be achieved.
    """
    squeeze = squeeze_gain * weight
    r_tip = tip_center_radius(radius, tip_radius, squeeze)
    gaps, residuals = [], []
    for chain in hand.chains:
        lo, hi = chain_limits(hand, chain)
        q0 = np.zeros(len(lo))
        u, r_open = _radial(hand, chain, q0, center, shape)
        gaps.append(r_open - radius - tip_radius)

        target = center + r_tip * u

        def objective(x, chain=chain, target=target):
            return float(np.sum((hand.tip_point(chain, x) - target) ** 2) + 1e-4 * np.sum(x ** 2))

        sol = minimize(objective, q0, bounds=list(zip(lo, hi)), method="L-BFGS-B")
        residuals.append(float(np.linalg.norm(hand.tip_point(chain, sol.x) - target)))
    return np.asarray(gaps), np.asarray(residuals)


def contact_geometry(hand: HandModel, center: np.ndarray, q_target: np.ndarray,
                     shape: str = "cylinder"):
    """How the five contact directions are arranged around the object.

    For a cylinder this returns the fingertip angles in the cross-section plane
    plus the widest uncovered arc.  A grasp can only hold an object if the
    contacts wrap around it: if every finger plus the thumb sits inside one
    semicircle, they all press from the same side and squeeze the object out
    instead of pinning it.

    For a sphere the same idea needs three dimensions, so instead of an arc we
    report the largest angle between any two contact directions (opposition) and
    whether the directions span the full sphere, i.e. whether the origin lies
    inside their convex hull - the necessary condition for force closure.
    """
    dirs = {}
    for chain in hand.chains:
        a, b = hand.chain_slice(chain.name)
        u, _ = _radial(hand, chain, q_target[a:b], center, shape)
        dirs[chain.name] = u

    if shape == "cylinder":
        angles = {k: float(np.degrees(np.arctan2(v[1], v[2]))) for k, v in dirs.items()}
        ordered = sorted(angles.values())
        gaps = [(ordered[(i + 1) % len(ordered)] - ordered[i]) % 360.0
                for i in range(len(ordered))]
        return angles, max(gaps)

    pts = np.asarray(list(dirs.values()))
    cos = np.clip(pts @ pts.T, -1.0, 1.0)
    np.fill_diagonal(cos, -1.0)
    max_opposition = float(np.degrees(np.arccos(cos.max())))
    wrapped = True
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        # origin inside the hull <=> every facet's offset term is non-positive
        wrapped = bool(np.all(hull.equations[:, 3] <= 1e-9))
    except Exception:
        # degenerate (coplanar) directions cannot enclose the origin
        wrapped = False
    # expose the cylinder-style "uncovered arc" number as 360 - opposition so the
    # same downstream check (> 180 means one-sided) still means the right thing
    return dict(dirs), (360.0 - max_opposition if not wrapped else 0.0)


class PlacementProbe:
    """Scores a candidate object placement against the real contact geometry.

    The analytic fit only knows about the fingertip spheres.  Whether the
    *proximal phalanges* clear the object is a question only the collision
    engine can answer, so compile the scene once and ask MuJoCo for the deepest
    penetration at a candidate centre.
    """

    def __init__(self, hand: HandModel, hand_xml: str, radius: float, mass: float,
                 shape: str = "cylinder", timestep: float = 0.002):
        scene = build_interactive_scene(
            hand_xml, radius=radius, mass=mass, center=CENTER,
            timestep=timestep, shape=shape,
        )
        self._path = build_dir() / f"_scene_probe_{shape}.xml"
        self._path.write_text(scene, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self._path))
        self.data = mujoco.MjData(self.model)
        self.hand = hand
        self.shape = shape
        self.obj_gid = next(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == _body_id(self.model, OBJECT_BODY)
        )
        self.jadr = self.model.jnt_qposadr[_joint_id(self.model, OBJECT_JOINT)]
        self.radius = radius

    def set_radius(self, radius: float) -> None:
        self.radius = radius
        self.model.geom_size[self.obj_gid, 0] = radius
        if self.shape == "cylinder":
            self.model.geom_size[self.obj_gid, 1] = HALF_LENGTH

    def _set_hand(self, q_active: np.ndarray) -> None:
        for chain in self.hand.chains:
            a, b = self.hand.chain_slice(chain.name)
            for name, val in self.hand.joint_values(chain, q_active[a:b]).items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0 and self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = val

    def deepest_penetration(self, center: np.ndarray, q_active: np.ndarray) -> float:
        """Most negative object contact distance [m]; 0.0 means no contact."""
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.jadr:self.jadr + 3] = center
        d.qpos[self.jadr + 3:self.jadr + 7] = (
            [0.707107, 0.0, 0.707107, 0.0] if self.shape == "cylinder"
            else [1.0, 0.0, 0.0, 0.0]
        )
        self._set_hand(q_active)
        mujoco.mj_forward(self.model, d)
        deepest = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.obj_gid or c.geom2 == self.obj_gid:
                deepest = min(deepest, float(c.dist))
        return deepest


def fit_object_center(hand: HandModel, radius: float, weight: float,
                      squeeze_gain: float, tip_radius: float,
                      min_clearance: float = 0.008, shape: str = "cylinder",
                      probe: PlacementProbe | None = None,
                      allow_penetration: float = 0.002):
    """Find an object centre that is clear of the hand and reachable.

    The scene default ``sim_grasp.CENTER`` places the cylinder 5.8 mm *inside*
    the index fingertip at the open pose, while ring/little/thumb are 36-60 mm
    short of it.  The initial state therefore interpenetrates - the object is
    ejected on the first contact solve - and the IK targets are unreachable.

    Search (x, y, z) for a centre that leaves every fingertip at least
    ``min_clearance`` outside the surface when the hand is open, keeps each
    finger's IK residual small, and - when ``probe`` is supplied - lets the
    fingers curl around the object without any phalanx cutting through it.
    """
    def cost(p):
        p = np.asarray(p, dtype=float)
        gaps, res = finger_reach(hand, p, radius, weight, squeeze_gain,
                                 tip_radius, shape)
        too_close = np.clip(min_clearance - gaps, 0.0, None)
        # clearance is a hard requirement, so weight it far above the residuals
        total = float(np.sum(too_close ** 2) * 1e4 + np.sum(res ** 2))
        if probe is not None:
            q = solve_angles(hand, tip_center_radius(radius, tip_radius,
                                                     squeeze_gain * weight),
                             weight, squeeze_gain=squeeze_gain, center=p, shape=shape)
            deep = max(0.0, -probe.deepest_penetration(p, q) - allow_penetration)
            total += deep ** 2 * 1e6
        return total

    key = (f"{shape}|{radius:.6g}|{weight:.6g}|{squeeze_gain:.6g}|"
           f"{tip_radius:.6g}|{min_clearance:.6g}|"
           f"{'probe' if probe is not None else 'plain'}")
    cached = _load_centre_cache().get(key)
    if cached is not None:
        center = np.asarray(cached, dtype=float)
    else:
        print("[sim_viewer] fitting the object placement (cached afterwards)...",
              file=sys.stderr, flush=True)
        best = None
        for x0 in (0.02, 0.055, 0.085):
            sol = minimize(cost, [x0, CENTER[1], CENTER[2]], method="Nelder-Mead",
                           options={"maxiter": 150, "xatol": 1e-4, "fatol": 1e-11})
            if best is None or sol.fun < best.fun:
                best = sol
        center = np.asarray(best.x, dtype=float)
        _save_centre_cache(key, center)

    gaps, res = finger_reach(hand, center, radius, weight, squeeze_gain, tip_radius, shape)
    return center, gaps, res


# kept for callers written before sphere support landed
fit_cylinder_center = fit_object_center



