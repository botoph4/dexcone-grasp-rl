"""Observation layout shared by the training env and the viewer's policy replay.

A policy is only valid if the viewer feeds it exactly the vector it was trained
on, so the encoder lives here and both ``grasp_env.py`` and ``sim_viewer.py``
import it.  (``grasp_env`` depends on ``sim_viewer`` for the scene builder, so
the shared code cannot live in either one.)
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# context normalisation, matched by the trained policies
RADIUS_MID, RADIUS_HALF = 0.030, 0.010
MASS_MID, MASS_HALF = 0.85, 0.65
# task thresholds shared by the env's success rule and the viewer's verdict
SUCCESS_DRIFT = 0.075      # a held object settles 50-70 mm below the clasp centre
SUCCESS_SPEED = 0.05
MIN_SUCCESS_CONTACTS = 3

QVEL_SCALE = 10.0
POS_SCALE = 0.1
TIP_SCALE = 0.15
LINVEL_SCALE = 0.5
ANGVEL_SCALE = 2.0


@dataclass
class ObsSpec:
    """Pre-computed model indices/addresses needed to build an observation."""

    obj_body: int
    obj_joint: int
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    qpos_lo: np.ndarray
    qpos_hi: np.ndarray
    qpos_center: np.ndarray
    qpos_half: np.ndarray
    tip_bodies: np.ndarray
    shape: str
    n_active: int

    @property
    def dim(self) -> int:
        # q(n) + dq(n) + prev_action(n) + rel(3) + quat(4) + linvel(3)
        # + angvel(3) + radius(1) + mass(1) + shape(2) + tips(3*5)
        return 3 * self.n_active + 17 + 3 * len(self.tip_bodies)


def actuator_joint_order(model: mujoco.MjModel):
    """Joint names of the joint-actuators, in actuator order.

    ``HandModel.active_joint_names`` is ordered by finger chain (index, little,
    middle, ring, thumb) while ``hand.xml`` declares its actuators thumb-first.
    The action vector is written straight into ``data.ctrl``, so every consumer
    must use the actuator order or commands land on the wrong joint.

    Actuators that are not bound to a joint are skipped: the viewer adds
    zero-gain "pose" actuators purely so the MuJoCo sidebar grows sliders for
    them, and those must not be mistaken for fingers.
    """
    names = []
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or jid < 0:
            continue
        # A zero-gain, unbiased actuator cannot produce force, so it is not a
        # controller channel - that is exactly how the viewer's placement
        # sliders are built, and they must not be mistaken for fingers.
        if (model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE
                and float(model.actuator_gainprm[a, 0]) == 0.0):
            continue
        names.append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).removeprefix("act_")
        )
    return names


def _name2id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


def make_obs_spec(model: mujoco.MjModel, hand, shape: str,
                  obj_body_name: str, obj_joint_name: str) -> ObsSpec:
    qpos_adr, dof_adr, lo, hi = [], [], [], []
    joint_actuators = actuator_joint_order(model)
    for name in joint_actuators:
        jid = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
        lo.append(model.jnt_range[jid][0])
        hi.append(model.jnt_range[jid][1])
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    return ObsSpec(
        obj_body=_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name),
        obj_joint=_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj_joint_name),
        qpos_adr=np.asarray(qpos_adr),
        dof_adr=np.asarray(dof_adr),
        qpos_lo=lo,
        qpos_hi=hi,
        qpos_center=0.5 * (lo + hi),
        qpos_half=np.maximum(0.5 * (hi - lo), 1e-6),
        tip_bodies=np.asarray([
            _name2id(model, mujoco.mjtObj.mjOBJ_BODY, c.tip_link) for c in hand.chains
        ]),
        shape=shape,
        # count the *joint* actuators, not model.nu: the viewer appends
        # zero-gain placement sliders that are not control channels
        n_active=len(joint_actuators),
    )


def build_obs(model: mujoco.MjModel, data: mujoco.MjData, spec: ObsSpec,
              center: np.ndarray, radius: float, mass: float,
              prev_action: np.ndarray) -> np.ndarray:
    """Assemble the normalized 80-dim observation.

    Layout: joint angles (n) | joint velocities (n) | previous action (n) |
    object position relative to the clasp centre (3) | object quaternion (4) |
    object linear velocity (3) | object angular velocity (3) |
    radius (1) | mass (1) | shape one-hot (2) | fingertip offsets (3*5)
    """
    q = (data.qpos[spec.qpos_adr] - spec.qpos_center) / spec.qpos_half
    dq = np.clip(data.qvel[spec.dof_adr] / QVEL_SCALE, -1.0, 1.0)

    obj_pos = data.xpos[spec.obj_body]
    rel = (obj_pos - center) / POS_SCALE
    ctx = np.array([
        (radius - RADIUS_MID) / RADIUS_HALF,
        (mass - MASS_MID) / MASS_HALF,
        1.0 if spec.shape == "cylinder" else 0.0,
        0.0 if spec.shape == "cylinder" else 1.0,
    ])
    tips = ((data.xpos[spec.tip_bodies] - obj_pos) / TIP_SCALE).ravel()

    obs = np.concatenate([
        q, dq, prev_action,
        rel, np.clip(data.xquat[spec.obj_body], -1.0, 1.0),
        np.clip(data.cvel[spec.obj_body, 3:6] / LINVEL_SCALE, -1.0, 1.0),
        np.clip(data.cvel[spec.obj_body, :3] / ANGVEL_SCALE, -1.0, 1.0),
        ctx, tips,
    ])
    return obs.astype(np.float32)


class ObsNormalizer:
    """Apply saved VecNormalize observation statistics.

    Driving the env directly (rather than through a VecEnv) keeps
    ``reset(seed=...)`` authoritative, and lets the viewer reuse the exact
    scaling the policy was trained with.
    """

    def __init__(self, obs_rms, clip_obs: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.asarray(obs_rms.mean, dtype=np.float64)
        self.var = np.asarray(obs_rms.var, dtype=np.float64)
        self.clip_obs = clip_obs
        self.epsilon = epsilon

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        z = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip_obs, self.clip_obs).astype(np.float32)
