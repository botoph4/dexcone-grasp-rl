#!/usr/bin/env python3
"""The PyRoki version of the grasp kinematics layer.

PyRoki (UC Berkeley, IROS 2025, https://github.com/chungmin99/pyroki) replaces
the scipy L-BFGS-B solver in ``sim_grasp.solve_angles`` with its JAX
differentiable FK + Levenberg-Marquardt solver, solving all five fingertip
targets in one least-squares problem with joint limits as hard constraints.

MuJoCo remains the dynamics/rendering backend - PyRoki has no physics, so the
pipeline stays split exactly as concluded in ``pyroki_probe.py``.

What this script does:
  1. solve the same per-finger grasp targets as sim_grasp.solve_angles, but
     with PyRoki's LM solver (optionally with collision costs on top),
  2. compare the two solvers (max joint-angle difference),
  3. optionally render the PyRoki solution through the MuJoCo scene (PNG).

Install PyRoki first (it is not on PyPI):
  git clone https://github.com/chungmin99/pyroki.git
  cd pyroki && pip install -e .

Run:
  python pyroki_grasp.py                        # cylinder configs, solver comparison
  python pyroki_grasp.py --shape sphere --render
  python pyroki_grasp.py --collision-aware      # add object-avoidance costs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_dataclasses as jdc  # noqa: E402
import jaxlie  # noqa: E402
import jaxls  # noqa: E402

from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.paths import assets_dir, ensure_hand_xml, outputs_dir  # noqa: E402
from p24grasp.kinematics.ik import (  # noqa: E402
    CENTER,
    HALF_LENGTH,
    chain_limits,
    radial_direction,
    solve_angles,
    tip_center_radius,
)
from p24grasp.sim.demo import render_scene  # noqa: E402
from p24grasp.kinematics.ik import fist_center  # noqa: E402
from p24grasp.sim.scene import TIP_RADIUS, build_interactive_scene  # noqa: E402


def build_pyroki(urdf_path: Path):
    """(robot, hand collision model) for the P24 hand.

    The URDF has no <collision> elements, so PyRoki's automatic capsules come
    back empty.  Instead the hand collision model is built as a sphere
    decomposition of the very same capsules ``build_mjcf`` emits for MuJoCo -
    three spheres per phalanx capsule.  This keeps the two backends consistent.
    """
    import pyroki as pk
    import yourdfpy

    from p24grasp.model import mjcf as build_mjcf

    urdf = yourdfpy.URDF.load(str(urdf_path))
    robot = pk.Robot.from_urdf(urdf)

    conv = build_mjcf.URDF2MJCF(urdf_path, assets_dir() / "p24_meshes")
    decomp = {}
    for link_name, link in conv.links.items():
        if link_name == "palm" or not link["visuals"]:
            continue
        cap = conv._capsule_for(link)
        if cap is None:
            continue
        p0, p1, radius = cap
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            continue
        n = axis / length
        decomp[link_name] = {
            "centers": [list(p0 + n * radius), list(0.5 * (p0 + p1)),
                        list(p1 - n * radius)],
            "radii": [float(radius)] * 3,
        }
    coll = pk.collision.RobotCollision.from_sphere_decomposition(decomp, urdf)
    return robot, coll, decomp


def tilt_quat(tilt_x: float, tilt_y: float) -> np.ndarray:
    """wxyz body quaternion for Rx(tilt_x) then Ry(tilt_y), like sim_viewer."""
    ax, ay = tilt_x / 2.0, tilt_y / 2.0
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


def tilt_axis(tilt_x: float, tilt_y: float) -> np.ndarray:
    """World direction of the cylinder axis (body +z) under the tilts."""
    w, x, y, z = tilt_quat(tilt_x, tilt_y)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)])


def _object_obstacle(shape: str, radius: float, center: np.ndarray,
                     tilt_x: float = 0.0, tilt_y: float = 0.0):
    """PyRoki CollGeom for the grasped object (world obstacle).

    A cylinder is represented by a capsule along its axis (the capsule's local
    z is its axis); ``tilt_x`` / ``tilt_y`` rotate the axis the same way the
    viewer's tilt sliders do, so the obstacle matches the IK targets exactly.
    """
    import pyroki as pk

    if shape == "sphere":
        return pk.collision.Sphere.from_center_and_radius(
            jnp.asarray(center), jnp.asarray(radius)
        )
    return pk.collision.Capsule.from_radius_height(
        radius=jnp.asarray(radius),
        height=jnp.asarray(2.0 * HALF_LENGTH),
        position=jnp.asarray(center),
        wxyz=jnp.asarray(tilt_quat(tilt_x, tilt_y)),
    )


def solve_angles_pyroki(
    robot,
    hand: HandModel,
    radius: float,
    weight: float,
    center: np.ndarray | None = None,
    shape: str = "cylinder",
    squeeze_gain: float = 0.0008,
    tip_radius: float = TIP_RADIUS,
    collision_model=None,
    collision_margin: float = 0.002,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> np.ndarray:
    """Grasp angles (16, MuJoCo actuator order) solved with PyRoki's LM.

    Mirrors ``sim_grasp.solve_angles``: each fingertip targets the point on the
    object surface along its open-pose radial direction, with the tip-sphere
    offset applied (see tip_center_radius).  The cylinder axis is the body +z
    direction rotated by ``tilt_x`` / ``tilt_y`` (same convention as the
    viewer's tilt sliders; both the IK targets and the world-collision
    obstacle use it).  With ``collision_model`` the IK additionally keeps the
    phalanges outside the object (world-collision constraint) and away from
    each other (self-collision cost).
    """
    import pyroki as pk

    center = CENTER if center is None else np.asarray(center, dtype=float)
    r_tip = tip_center_radius(radius, tip_radius, squeeze_gain * weight)
    axis = tilt_axis(tilt_x, tilt_y)

    link_indices = []
    targets = []
    for chain in hand.chains:
        q0 = np.zeros(len(chain_limits(hand, chain)[0]))
        p0 = hand.tip_point(chain, q0)
        u, nv = radial_direction(p0, center, axis, shape)
        targets.append(center + r_tip * u if nv > 1e-9 else p0)
        link_indices.append(jnp.array(robot.links.names.index(chain.tip_link)))

    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    costs = [pk.costs.limit_constraint(robot, joint_var)]
    for link_index, target in zip(link_indices, targets):
        costs.append(
            pk.costs.pose_cost(
                robot,
                joint_var,
                target_pose=jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3.identity(), jnp.asarray(target)
                ),
                target_link_index=link_index,
                pos_weight=50.0,
                ori_weight=0.0,
            )
        )
    costs.append(
        pk.costs.rest_cost(joint_var, rest_pose=jnp.zeros(robot.joints.num_actuated_joints),
                           weight=0.001)
    )
    if collision_model is not None:
        costs.append(
            pk.costs.self_collision_cost(
                robot, robot_coll=collision_model, joint_var=joint_var,
                margin=collision_margin, weight=5.0,
            )
        )
        costs.append(
            pk.costs.world_collision_constraint(
                robot, collision_model, joint_var,
                _object_obstacle(shape, radius, center, tilt_x, tilt_y),
                collision_margin,
            )
        )

    problem = jaxls.LeastSquaresProblem(costs=costs, variables=variables)
    solution = problem.analyze().solve(verbose=False, linear_solver="dense_cholesky")
    cfg = np.asarray(solution[joint_var])
    # PyRoki's actuated order (thumb, index, middle, ring, little) is exactly
    # the MuJoCo actuator order, so the result feeds data.ctrl directly
    return cfg


def compare(scipy_q: np.ndarray, pyroki_q: np.ndarray, hand: HandModel):
    print(f"{'finger':<9}{'scipy':>24}{'pyroki':>24}{'|dq| max':>10}")
    for chain in hand.chains:
        a, b = hand.chain_slice(chain.name)
        # pyroki order is thumb-first; map back through joint names
        names = hand.active_joint_names[a:b]
        idx = [hand.active_joint_names.index(n) for n in names]
        # thumb-first permutation of the 16 active joints
        thumb_first = ["thumb_dh_joint_1", "thumb_dh_joint_2", "thumb_dh_joint_3",
                       "thumb_dh_joint_4", "index_dh_joint_1", "index_dh_joint_2",
                       "index_dh_joint_3", "middle_dh_joint_1", "middle_dh_joint_2",
                       "middle_dh_joint_3", "ring_dh_joint_1", "ring_dh_joint_2",
                       "ring_dh_joint_3", "little_dh_joint_1", "little_dh_joint_2",
                       "little_dh_joint_3"]
        pk_chain = np.array([pyroki_q[thumb_first.index(n)] for n in names])
        d = np.abs(scipy_q[a:b] - pk_chain).max()
        print(f"{chain.name:<9}{np.round(scipy_q[a:b],3)!s:>24}"
              f"{np.round(pk_chain,3)!s:>24}{np.degrees(d):>9.2f} deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=str(assets_dir() / "p24_hand_right.urdf"))
    ap.add_argument("--shape", choices=("cylinder", "sphere"), default="cylinder")
    ap.add_argument("--radius", type=float, default=0.032)
    ap.add_argument("--weight", type=float, default=0.85)
    ap.add_argument("--squeeze-gain", type=float, default=0.0008)
    ap.add_argument("--center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="object centre [m] (default: sim_grasp.CENTER)")
    ap.add_argument("--tilt-x", type=float, default=0.0, metavar="DEG",
                    help="cylinder axis tilt about x [degrees]; -90 gives the "
                         "horizontal bar pose like the viewer default")
    ap.add_argument("--tilt-y", type=float, default=0.0, metavar="DEG",
                    help="cylinder axis tilt about y [degrees]")
    ap.add_argument("--collision-aware", action="store_true",
                    help="add self/world-collision costs so the IK keeps the "
                         "phalanges out of the object")
    ap.add_argument("--render", action="store_true",
                    help="render the PyRoki solution through the MuJoCo scene")
    ap.add_argument("--outdir", default=str(outputs_dir("render")))
    args = ap.parse_args()

    import pyroki  # noqa: F401 - fail early with a clear message
    import yourdfpy  # noqa: F401

    hand = HandModel(args.urdf)
    robot, coll, _ = build_pyroki(Path(args.urdf))
    center = None if args.center is None else np.asarray(args.center, dtype=float)
    tilt_x, tilt_y = np.radians(args.tilt_x), np.radians(args.tilt_y)
    print(f"[pyroki] robot ready: {robot.joints.num_actuated_joints} actuated "
          f"joints, collision model from {len(coll.link_names)} links")

    t0 = __import__("time").time()
    q_pyroki = solve_angles_pyroki(
        robot, hand, args.radius, args.weight, center=center, shape=args.shape,
        squeeze_gain=args.squeeze_gain,
        collision_model=coll if args.collision_aware else None,
        tilt_x=tilt_x, tilt_y=tilt_y,
    )
    print(f"[pyroki] LM solve: {__import__('time').time() - t0:.2f}s "
          f"(includes JIT compile on first run)")

    q_scipy = solve_angles(hand, args.radius, args.weight,
                           squeeze_gain=args.squeeze_gain, shape=args.shape)
    compare(q_scipy, q_pyroki, hand)

    if args.render:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        name = f"pyroki_{args.shape}_r{int(args.radius*1000):03d}.png"
        from p24grasp.sim.demo import build_scene, set_hand_pose
        hand_xml = ensure_hand_xml().read_text(encoding="utf-8")
        scene = build_scene(hand_xml, args.radius, args.weight, shape=args.shape)
        # set_hand_pose expects HandModel-chain order; reorder pyroki's output
        thumb_first = ["thumb_dh_joint_1", "thumb_dh_joint_2", "thumb_dh_joint_3",
                       "thumb_dh_joint_4", "index_dh_joint_1", "index_dh_joint_2",
                       "index_dh_joint_3", "middle_dh_joint_1", "middle_dh_joint_2",
                       "middle_dh_joint_3", "ring_dh_joint_1", "ring_dh_joint_2",
                       "ring_dh_joint_3", "little_dh_joint_1", "little_dh_joint_2",
                       "little_dh_joint_3"]
        q_chain_order = np.array([q_pyroki[thumb_first.index(n)]
                                  for n in hand.active_joint_names])
        path = render_scene(scene, hand, q_chain_order, outdir / name)
        print(f"[pyroki] rendered {path.name}")

    print("[pyroki] done")


if __name__ == "__main__":
    main()
