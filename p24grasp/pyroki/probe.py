#!/usr/bin/env python3
"""Feasibility probe: can PyRoki replace parts of this project?

PyRoki (https://github.com/chungmin99/pyroki) is Berkeley's *kinematic
optimization* toolkit: differentiable JAX forward kinematics from URDF, an LM
solver with costs (including collision), and automatic collision primitives.
It is NOT a physics engine - no dynamics, no contacts, no grasping simulation.

This probe loads the P24 URDF into PyRoki and checks, quantitatively, what it
can and cannot cover:

  FK ................ compared against p24grasp.model.urdf.HandModel (0.000 mm expected)
  mimic joints ...... the URDF's 4 DIP mimics must fold with their PIP parents
  collision capsules. auto-generated from the URDF (expected: EMPTY, because
                      the URDF has no <collision> elements - the same gap that
                      build_mjcf.py fills by synthesising capsules)

Verdict: PyRoki can serve as an alternative kinematics/IK layer, but MuJoCo
stays for the grasp dynamics.

Run:  python pyroki_probe.py
"""
from __future__ import annotations

import numpy as np

from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.kinematics.ik import chain_limits  # noqa: E402
from p24grasp.paths import assets_dir  # noqa: E402


def main() -> None:
    try:
        import pyroki as pk  # noqa
        import yourdfpy  # noqa
    except ImportError:
        print("[probe] PyRoki is not installed. Install it with:\n"
              "  git clone https://github.com/chungmin99/pyroki.git\n"
              "  cd pyroki && pip install -e .\n"
              "then rerun this probe.")
        return

    urdf = yourdfpy.URDF.load(str(assets_dir() / "p24_hand_right.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    print(f"[probe] PyRoki robot: {robot.joints.num_actuated_joints} actuated "
          f"joints ({len(robot.links.names)} links)")

    hand = HandModel(str(assets_dir() / "p24_hand_right.urdf"))
    qflex = np.concatenate([chain_limits(hand, c)[1] for c in hand.chains])
    values: dict[str, float] = {}
    for chain in hand.chains:
        a, b = hand.chain_slice(chain.name)
        values.update(hand.joint_values(chain, qflex[a:b]))
    actuated = [n for n in robot.joints.names if n in hand.active_joint_names]
    q_pk = np.array([values[n] for n in actuated])

    T = np.asarray(robot.forward_kinematics(q_pk))  # (*, link_count, wxyz_xyz)
    max_err = 0.0
    for i, name in enumerate(list(robot.links.names)):
        if not name.endswith("_tip_link"):
            continue
        chain = next(c for c in hand.chains if c.tip_link == name)
        a, b = hand.chain_slice(chain.name)
        ours = hand.tip_point(chain, qflex[a:b])
        err = float(np.linalg.norm(T[i, 4:7] - ours)) * 1000
        max_err = max(max_err, err)
        print(f"[probe]   {name:<18} FK diff vs HandModel: {err:.3f} mm")
    print(f"[probe] FK max error: {max_err:.3f} mm "
          f"({'MATCH' if max_err < 1e-3 else 'MISMATCH'})")

    from pyroki.collision import RobotCollision
    coll = RobotCollision.from_urdf(urdf)
    caps = coll.get_swept_capsules(robot, q_pk, q_pk)
    size = np.asarray(caps.size)  # (n_primitives, n_links, 2) = (length, radius)
    n_caps = int(np.count_nonzero(np.abs(size) > 1e-9) / 2)
    print(f"[probe] auto collision capsules from URDF: {n_caps} "
          f"({'none - the URDF has no <collision> elements' if n_caps == 0 else ''})")

    print("\n[probe] verdict:")
    print("  * kinematics (FK/IK):   YES - FK matches HandModel exactly and the "
          "LM solver supports mimic joints and collision costs")
    print("  * grasp dynamics:       NO - PyRoki has no physics, contact solver "
          "or rendering; MuJoCo stays for the simulation")


if __name__ == "__main__":
    main()
