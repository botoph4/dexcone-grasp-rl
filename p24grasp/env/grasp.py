#!/usr/bin/env python3
"""Gymnasium environment: learn P24 hand joint targets that hold an object.

``sim_grasp.solve_angles`` aims every fingertip radially outward along its
*open* pose direction.  That can only build an open-hand wrap, and on this hand
it does not hold anything: after fixing the contact geometry the object is still
ejected, because the fingers close through the object instead of around it.

Closing the hand fully instead brings all five fingertips and the opposed thumb
into one compact cluster (``sim_grasp.fist_center``) - a power grasp.  Spawning
the object into that cluster and driving the fingers closed does hold it, so
that is what this environment starts from and what the policy refines.

Scene, contact filtering, actuator limits and timestep are identical to
``sim_viewer.py`` so a trained policy can be replayed there unchanged.

Observation layout is documented in ``build_obs`` and the reward in
``GraspEnv.step``.
"""
from __future__ import annotations

import numpy as np

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
from gymnasium import spaces  # noqa: E402

from p24grasp.env.obs import (  # noqa: E402
    MIN_SUCCESS_CONTACTS,
    SUCCESS_DRIFT,
    SUCCESS_SPEED,
    actuator_joint_order,
    build_obs,
    make_obs_spec,
)
from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.paths import build_dir, ensure_hand_xml, urdf_path  # noqa: E402
from p24grasp.kinematics.ik import HALF_LENGTH, fist_center  # noqa: E402
from p24grasp.sim.scene import (  # noqa: E402
    OBJECT_BODY,
    OBJECT_JOINT,
    SHAPES,
    build_interactive_scene,
)

# control / episode
CONTROL_HZ = 100.0
FRAME_SKIP = 5           # 5 x 2 ms physics steps per decision
EPISODE_STEPS = 400      # 4.0 s
PHYSICS_DT = 0.002

# domain randomisation.  The clasp volume is ~77 mm across, so smaller objects
# simply fall between the fingers and much larger ones fall out of the hand.
RADIUS_RANGE = (0.025, 0.040)
MASS_RANGE = (0.20, 1.50)
CENTER_JITTER = 0.002    # +-2 mm per axis around the clasp centre
AXIS_TILT_DEG = 8.0      # cylinder only, about y and z
INIT_LINVEL = 0.05
INIT_ANGVEL = 0.1
# The thumb sits inside the clasp, so a *uniform* flexion scale drives it into
# the object and squeezes it out; a grid search over thumb x finger scale shows
# most of the non-uniform region holds the object for the whole episode, so the
# thumb is scaled separately and the episode starts already inside that region.
INIT_THUMB_SCALE = (0.30, 0.90)
INIT_FINGER_SCALE = (0.75, 1.00)
INIT_JOINT_NOISE = 0.03       # rad

# termination.  A held object settles 50-70 mm below the fingertip centroid as
# the palm catches it, so "held" is a 75 mm neighbourhood, not the spawn point.
DROP_Z_MARGIN = 0.05
ESCAPE_DRIFT = 0.13
# reward weights
W_HOLD = 0.6
W_STILL = 0.4
W_CONTACT = 0.3
W_EFFORT = 0.05
W_RATE = 0.01
R_SUCCESS = 20.0
R_DROP = -20.0
HOLD_SIGMA = 0.08
STILL_SIGMA = 0.05


def quat_from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    """wxyz quaternion from small XYZ Euler angles (radians)."""
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


class GraspEnv(gym.Env):
    """Hold a cylinder or sphere against gravity using the P24 hand."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, shape: str = "cylinder", urdf: str | None = None,
                 randomize: bool = True, render_mode: str | None = None,
                 force_range: float | None = None, kp: float | None = None,
                 kv: float | None = None):
        super().__init__()
        assert shape in SHAPES, f"shape must be one of {SHAPES}"
        self.shape = shape
        self.randomize = randomize
        self.render_mode = render_mode

        self.hand = HandModel(urdf or str(urdf_path()))
        hand_xml = ensure_hand_xml().read_text(encoding="utf-8")

        # Full flexion is the reference pose: it is the only configuration in
        # which the fingers and the opposed thumb enclose anything.
        self.center = fist_center(self.hand)

        scene = build_interactive_scene(
            hand_xml, radius=0.032, mass=0.85, center=self.center,
            timestep=PHYSICS_DT, shape=shape, force_range=force_range,
            kp=kp, kv=kv,
        )
        self._scene_path = build_dir() / f"_scene_env_{shape}.xml"
        self._scene_path.write_text(scene, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self.data = mujoco.MjData(self.model)
        self.spec = make_obs_spec(self.model, self.hand, shape, OBJECT_BODY, OBJECT_JOINT)
        self.q_flex = np.array(
            [self.hand.joints[n].upper for n in actuator_joint_order(self.model)]
        )
        self._is_thumb = np.array(
            [n.startswith("thumb") for n in actuator_joint_order(self.model)]
        )

        self._obj_gid = next(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.spec.obj_body
        )
        self._hand_geoms = set(range(self.model.ngeom)) - {self._obj_gid}

        self.action_space = spaces.Box(-1.0, 1.0, (self.spec.n_active,), np.float32)
        self.observation_space = spaces.Box(
            -10.0, 10.0, (self.spec.dim,), np.float32
        )

        self._prev_action = np.zeros(self.spec.n_active, np.float32)
        self._radius = 0.032
        self._mass = 0.85
        self._steps = 0
        self._renderer = None

    # ------------------------------------------------------------------ #
    def _set_object(self, radius: float, mass: float, quat: np.ndarray,
                    center: np.ndarray):
        """Rewrite object size/mass/pose in the model (MuJoCo DR idiom)."""
        self.model.geom_size[self._obj_gid, 0] = radius
        if self.shape == "cylinder":
            self.model.geom_size[self._obj_gid, 1] = HALF_LENGTH
        self.model.body_mass[self.spec.obj_body] = mass
        self.model.body_subtreemass[self.spec.obj_body] = mass
        if self.shape == "sphere":
            i = 0.4 * mass * radius ** 2
            self.model.body_inertia[self.spec.obj_body] = (i, i, i)
        else:
            i_axis = 0.5 * mass * radius ** 2
            i_perp = mass * (3.0 * radius ** 2 + (2.0 * HALF_LENGTH) ** 2) / 12.0
            self.model.body_inertia[self.spec.obj_body] = (i_perp, i_perp, i_axis)
        self.model.body_ipos[self.spec.obj_body] = (0.0, 0.0, 0.0)
        self.model.body_iquat[self.spec.obj_body] = (1.0, 0.0, 0.0, 0.0)

        adr = self.model.jnt_qposadr[self.spec.obj_joint]
        self.data.qpos[adr:adr + 3] = center
        self.data.qpos[adr + 3:adr + 7] = quat

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        radius = float(rng.uniform(*RADIUS_RANGE)) if self.randomize else 0.032
        mass = float(rng.uniform(*MASS_RANGE)) if self.randomize else 0.85
        self._radius, self._mass = radius, mass

        base = self.center
        jitter = rng.uniform(-CENTER_JITTER, CENTER_JITTER, 3) if self.randomize else 0.0
        spawn = base + jitter
        self._spawn = spawn

        if self.shape == "cylinder" and self.randomize:
            tilt = np.radians(AXIS_TILT_DEG)
            quat = quat_from_euler(0.0, *rng.uniform(-tilt, tilt, 2))
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_resetData(self.model, self.data)
        self._set_object(radius, mass, quat, spawn)

        # start the fingers part-way closed, so the object spawns inside the
        # closing clasp rather than falling through it
        if self.randomize:
            t_scale = float(rng.uniform(*INIT_THUMB_SCALE))
            f_scale = float(rng.uniform(*INIT_FINGER_SCALE))
        else:
            t_scale, f_scale = 0.60, 0.90
        scale = np.where(self._is_thumb, t_scale, f_scale)
        q0 = scale * self.q_flex
        if self.randomize:
            q0 = q0 + rng.normal(0.0, INIT_JOINT_NOISE, self.spec.n_active)
        q0 = np.clip(q0, self.spec.qpos_lo, self.spec.qpos_hi)
        self.data.qpos[self.spec.qpos_adr] = q0
        self.data.ctrl[:] = q0

        adr = self.model.jnt_dofadr[self.spec.obj_joint]
        if self.randomize:
            self.data.qvel[adr:adr + 3] = rng.uniform(-INIT_LINVEL, INIT_LINVEL, 3)
            self.data.qvel[adr + 3:adr + 6] = rng.uniform(-INIT_ANGVEL, INIT_ANGVEL, 3)

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._prev_action = np.zeros(self.spec.n_active, np.float32)
        obs = build_obs(self.model, self.data, self.spec, self.center, radius, mass,
                        self._prev_action)
        info = {"radius": radius, "mass": mass, "spawn": spawn.copy()}
        return obs, info

    # ------------------------------------------------------------------ #
    def _contact_geoms(self):
        """Hand geoms currently touching the object (distinct, not raw points)."""
        gids = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._obj_gid:
                gids.add(c.geom2)
            elif c.geom2 == self._obj_gid:
                gids.add(c.geom1)
        return gids

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, -1.0, 1.0)
        return self.spec.qpos_center + a * self.spec.qpos_half

    def step(self, action: np.ndarray):
        """Reward = keep the object in the clasp, still, with contact, cheaply.

        The object settles 5-8 cm below the clasp centre once the palm catches
        it, so "held" cannot mean "still at the spawn point".  It is scored as
        staying near the clasp (``W_HOLD``), not moving (``W_STILL``), still
        being touched (``W_CONTACT``), with small penalties for actuator effort
        and action chatter.  A dropped object ends the episode with ``R_DROP``;
        surviving to the end while still and in contact earns ``R_SUCCESS``.
        """
        action = np.asarray(action, dtype=np.float32)
        ctrl = self._action_to_ctrl(action)
        self.data.ctrl[:] = ctrl

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        obj_pos = self.data.xpos[self.spec.obj_body]
        obj_vel = self.data.cvel[self.spec.obj_body, 3:6]
        drift = float(np.linalg.norm(obj_pos - self.center))
        speed = float(np.linalg.norm(obj_vel))
        n_geoms = len(self._contact_geoms())

        effort = float(np.mean(np.abs(self.data.actuator_force) / 1.0))
        rate = float(np.mean(np.abs(action - self._prev_action)))
        self._prev_action = action

        reward = float(
            W_HOLD * np.exp(-(drift / HOLD_SIGMA) ** 2)
            + W_STILL * np.exp(-(speed / STILL_SIGMA) ** 2)
            + W_CONTACT * min(n_geoms, MIN_SUCCESS_CONTACTS) / MIN_SUCCESS_CONTACTS
            - W_EFFORT * effort
            - W_RATE * rate
        )

        dropped = (obj_pos[2] < self.center[2] - DROP_Z_MARGIN
                   or drift > ESCAPE_DRIFT)
        if dropped:
            reward += R_DROP

        truncated = self._steps >= EPISODE_STEPS
        in_hand = (not dropped) and drift < SUCCESS_DRIFT and n_geoms >= MIN_SUCCESS_CONTACTS
        success = bool(in_hand and speed < SUCCESS_SPEED)
        if truncated and success:
            reward += R_SUCCESS

        obs = build_obs(self.model, self.data, self.spec, self.center,
                        self._radius, self._mass, self._prev_action)
        info = {
            "drift": drift,
            "speed": speed,
            "contacts": n_geoms,
            "success": success,
            "effort": effort,
            "is_success": success,
        }
        return obs, reward, dropped, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = self.center
        cam.distance = 0.42
        cam.azimuth = 180.0
        cam.elevation = -20.0
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="P24Grasp-v0", entry_point=lambda **kw: GraspEnv(**kw))
