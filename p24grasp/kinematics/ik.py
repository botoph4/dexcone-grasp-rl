"""Geometric grasp IK for the P24 hand (scipy backend).

The radial-target solver aims each fingertip at the object surface along its
open-pose direction - the geometric baseline that RL later improves on.  The
PyRoki backend lives in :mod:`p24grasp.pyroki.ik`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from p24grasp.model.urdf import HandModel

# Object centre and axis.  The fingers flex in the world x-z plane (a 90 deg
# sweep of any finger's flexion joint moves its tip by ~(+0.114, 0, -0.111)), so
# a cylinder is grasped like a handle only when its axis is perpendicular to that
# plane.  An axis along +x lies *in* the curl plane and gets pushed out of the
# hand instead of wrapped, which is why the vertical axis is the one that holds.
AXIS = np.array([0.0, 0.0, 1.0])
CENTER = np.array([0.030, 0.050, 0.180])
HALF_LENGTH = 0.06


def chain_limits(model: HandModel, chain):
    lo, hi = [], []
    for j in chain.joints:
        if j.type == "revolute" and j.mimic_joint is None:
            lo.append(j.lower)
            hi.append(j.upper)
    return np.array(lo), np.array(hi)


def tip_center_radius(obj_radius: float, tip_radius: float, squeeze: float = 0.0):
    """Radius at which a fingertip sphere *centre* must sit.

    ``solve_angles`` places the tip frame origin, which is the centre of the
    contact sphere that ``build_mjcf.py`` attaches to every tip link.  For that
    sphere to just touch the object surface its centre sits one tip-radius
    OUTSIDE the surface, at ``obj_radius + tip_radius``.  ``squeeze`` pulls it
    slightly back in, preloading the contact.

    Getting the sign wrong here (``obj_radius - tip_radius``) buries every
    fingertip ~2*tip_radius inside the object, which ejects it on the first
    contact solve.
    """
    return max(obj_radius + tip_radius - squeeze, 0.003)


def fist_center(model: HandModel) -> np.ndarray:
    """Centroid of the five fingertips at full flexion = the grip volume.

    ``solve_angles`` aims each fingertip radially outward along its *open* pose
    direction, which can only build an open-hand wrap: the object ends up where
    the fingers point (~z=0.2) instead of where they close around (~z=0.07).
    Closing the hand fully instead brings every fingertip plus the opposed thumb
    into a compact cluster - that cluster is where a power grasp actually forms,
    and it is the only configuration found in which the object stays held.
    """
    tips = []
    for chain in model.chains:
        _, hi = chain_limits(model, chain)
        tips.append(model.tip_point(chain, np.asarray(hi, dtype=float)))
    return np.mean(np.asarray(tips), axis=0)


def radial_direction(point, center, axis=AXIS, shape: str = "cylinder"):
    """Unit vector from the object centre toward ``point``, plus its length.

    For a cylinder the component along the axis is dropped, because the fingers
    wrap in the plane perpendicular to the axis.  For a sphere the direction is
    the full 3D one.
    """
    v = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    if shape == "cylinder":
        axis = np.asarray(axis, dtype=float)
        v = v - (v @ axis) * axis
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return v / n, n


def solve_angles(
    model: HandModel,
    radius: float,
    weight: float,
    squeeze_gain: float = 0.0008,
    center: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    shape: str = "cylinder",
):
    """Return active q (in HandModel joint order) wrapping the cylinder/sphere.

    ``center`` / ``axis`` / ``shape`` override the module-level object pose,
    which lets a caller move the object without patching globals.
    """
    center = CENTER if center is None else np.asarray(center, dtype=float)
    axis = AXIS if axis is None else np.asarray(axis, dtype=float)

    # heavier object -> press a little further inside the surface
    r_eff = max(radius - squeeze_gain * weight, 0.003)
    q0 = np.zeros(model.n_active)
    q_sol = np.zeros(model.n_active)

    for chain in model.chains:
        a, b = model.chain_slice(chain.name)
        p0 = model.tip_point(chain, q0[a:b])
        u, nv = radial_direction(p0, center, axis, shape)
        target = p0 if nv < 1e-6 else center + r_eff * u

        lo, hi = chain_limits(model, chain)
        x0 = np.zeros(len(lo))

        def objective(x):
            p = model.tip_point(chain, x)
            return float(np.sum((p - target) ** 2) + 1e-4 * np.sum(x ** 2))

        res = minimize(objective, x0, bounds=list(zip(lo, hi)), method="L-BFGS-B")
        q_sol[a:b] = res.x
    return q_sol

