#!/usr/bin/env python3
"""Anti-slip grasp-angle solver.

Combines the two projects:

  * ``agile_hand_friction_cone``  -> friction-cone anti-slip force requirement
  * ``mujoco_hand_grasp``         -> hand kinematics + fingertip IK + visualisation

Given a cylinder (radius, mass, friction coefficient), the solver computes how
hard each finger must press to avoid slipping, converts that required normal
force into a fingertip penetration using a contact stiffness, then solves IK so
each fingertip reaches the corresponding inward target point.  The resulting
joint angles are the requested "anti-slip grasp angles".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from p24grasp.model.urdf import HandModel          # noqa: E402
from p24grasp.paths import assets_dir, outputs_dir  # noqa: E402
from p24grasp.model.friction_cone import FrictionConeConfig  # noqa: E402
from p24grasp.kinematics.ik import CENTER, AXIS, HALF_LENGTH  # noqa: E402
from p24grasp.sim.demo import preview_scene  # noqa: E402

G = 9.81


def chain_limits(model: HandModel, chain):
    lo, hi = [], []
    for j in chain.joints:
        if j.type == "revolute" and j.mimic_joint is None:
            lo.append(j.lower)
            hi.append(j.upper)
    return np.array(lo), np.array(hi)


def solve_fingertip_ik(model: HandModel, chain, target: np.ndarray) -> np.ndarray:
    lo, hi = chain_limits(model, chain)
    x0 = np.zeros(len(lo))

    def objective(x):
        p = model.tip_point(chain, x)
        return float(np.sum((p - target) ** 2) + 1e-4 * np.sum(x ** 2))

    res = minimize(objective, x0, bounds=list(zip(lo, hi)), method="L-BFGS-B")
    return res.x


def compute_antislip_grasp(
    model: HandModel,
    radius: float,
    mass: float,
    mu: float,
    safety_factor: float,
    contact_stiffness: float,
    n_contacts: int,
    f_margin_per_finger: float = 0.3,
    max_penetration_ratio: float = 0.5,
) -> dict:
    """Return anti-slip grasp angles and intermediate quantities."""
    weight = mass * G
    mu_eff = mu / safety_factor

    # total normal force needed so that friction capacity cancels gravity:
    #   mu_eff * sum(f_n) >= weight
    fn_total_req = weight / mu_eff
    fn_per_finger = fn_total_req / n_contacts + f_margin_per_finger

    # fingertip normal force -> penetration (linear contact stiffness)
    penetration = fn_per_finger / contact_stiffness
    penetration_max = max_penetration_ratio * radius
    saturated = penetration > penetration_max
    if saturated:
        penetration = penetration_max

    r_target = max(radius - penetration, 0.002)

    q0 = np.zeros(model.n_active)
    q_sol = np.zeros(model.n_active)
    per_finger = {}

    for chain in model.chains:
        a, b = model.chain_slice(chain.name)
        p0 = model.tip_point(chain, q0[a:b])
        v = p0 - CENTER
        v -= (v @ AXIS) * AXIS
        nv = np.linalg.norm(v)
        radial = v / nv if nv > 1e-9 else np.array([0.0, 0.0, 1.0])

        target = CENTER + r_target * radial
        q_chain = solve_fingertip_ik(model, chain, target)
        q_sol[a:b] = q_chain

        tip = model.tip_point(chain, q_chain)
        per_finger[chain.name] = {
            "q": q_chain.tolist(),
            "target": target.tolist(),
            "tip": tip.tolist(),
            "ik_error_m": float(np.linalg.norm(tip - target)),
        }

    fn_per_achieved = penetration * contact_stiffness
    friction_capacity = mu_eff * n_contacts * fn_per_achieved
    slip_margin = friction_capacity / weight  # > 1 means enough to hold

    return {
        "radius": radius,
        "mass": mass,
        "weight_N": weight,
        "mu": mu,
        "mu_eff": mu_eff,
        "n_contacts": n_contacts,
        "fn_total_req_N": fn_total_req,
        "fn_per_finger_req_N": fn_per_finger,
        "penetration_m": penetration,
        "penetration_saturated": bool(saturated),
        "friction_capacity_N": friction_capacity,
        "slip_margin": slip_margin,
        "q_angles": q_sol.tolist(),
        "joint_names": model.active_joint_names,
        "per_finger": per_finger,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(assets_dir() / "p24_hand_right.urdf"))
    ap.add_argument("--outdir", default=str(outputs_dir("antislip")))
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--safety", type=float, default=1.4)
    ap.add_argument("--stiffness", type=float, default=500.0)
    args = ap.parse_args()

    model = HandModel(args.urdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mesh_root = assets_dir() / "p24_meshes"

    configs = [
        # radius, mass
        (0.015, 0.2),
        (0.025, 0.5),
        (0.035, 1.0),
        (0.035, 2.0),
    ]

    results = []
    for radius, mass in configs:
        r = compute_antislip_grasp(
            model,
            radius=radius,
            mass=mass,
            mu=args.mu,
            safety_factor=args.safety,
            contact_stiffness=args.stiffness,
            n_contacts=len(model.chains),
        )
        results.append(r)
        name = f"r{int(radius*1000):03d}_m{int(mass*1000):03d}.png"
        preview_scene(
            model, np.asarray(r["q_angles"]), radius, mass,
            outdir / name, args.urdf, mesh_root,
        )
        status = "SATURATED" if r["penetration_saturated"] else "ok"
        print(
            f"[antislip] r={radius*1000:4.0f}mm m={mass:.1f}kg "
            f"fn/finger={r['fn_per_finger_req_N']:.2f}N "
            f"pen={r['penetration_m']*1000:.1f}mm "
            f"margin={r['slip_margin']:.2f} {status}"
        )

    (outdir / "antislip_grasp.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[antislip] wrote {outdir / 'antislip_grasp.json'}")


if __name__ == "__main__":
    main()
