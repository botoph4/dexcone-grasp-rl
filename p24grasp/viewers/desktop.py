"""Desktop interactive viewer (mjpython + MuJoCo passive viewer).

Placement mode pins the object and exposes it through the Control-panel
sliders; RUN mode releases it and drives the grasp pose (or a trained policy).
Keys avoid the ones the MuJoCo viewer reserves (Space, +/-, arrows, Tab,
[/], Esc, Page Up).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from p24grasp.env.obs import (  # noqa: E402
    MIN_SUCCESS_CONTACTS,
    SUCCESS_DRIFT,
    actuator_joint_order,
    build_obs,
    make_obs_spec,
)
from p24grasp.kinematics.ik import (  # noqa: E402
    CENTER,
    HALF_LENGTH,
    fist_center,
    solve_angles,
    tip_center_radius,
    wrap_aligned_center,
)
from p24grasp.model.urdf import HandModel  # noqa: E402
from p24grasp.paths import assets_dir, build_dir, ensure_hand_xml  # noqa: E402
from p24grasp.sim.scene import (  # noqa: E402
    OBJECT_BODY,
    OBJECT_JOINT,
    SHAPES,
    TIP_RADIUS,
    object_inertia,
    PlacementProbe,
    _body_id,
    _joint_id,
    build_interactive_scene,
    contact_geometry,
    finger_reach,
    fit_object_center,
)

#
# The MuJoCo viewer reserves these for its own shortcuts (its Help panel lists
# them): Space, + / -, Left/Right arrow, Tab / Shift-Tab, [ / ], Esc, Page Up.
# So every key this script binds is a letter or a digit - binding punctuation
# silently fights the viewer instead of reaching key_callback.
KEY_R = 82                 # reset
KEY_P = 80
KEY_C = 67
KEY_O = 79                 # toggle policy playback
KEY_K = 75                 # toggle the test pull
KEY_N, KEY_M = 78, 77      # pull magnitude down / up

# manual placement mode
KEY_W, KEY_S = 87, 83      # object +z / -z
KEY_A, KEY_D = 65, 68      # object -y / +y
KEY_Q, KEY_E = 81, 69      # object -x / +x
KEY_F, KEY_V = 70, 86      # cylinder axis -> +y / -y
KEY_T, KEY_Y = 84, 89      # cylinder axis -> -x / +x
KEY_Z, KEY_X = 90, 88      # nudge step down / up
KEY_G = 71                 # release the object and run the automatic grasp
KEY_B = 66                 # back to manual placement
POLICY_PERIOD = 5          # run the policy every 5 physics steps (100 Hz)
KEY_1, KEY_2, KEY_3 = 49, 50, 51  # '1' .. '6' select the pull direction
KEY_4, KEY_5, KEY_6 = 52, 53, 54

PULL_DIRECTIONS = {
    KEY_1: (1.0, 0.0, 0.0),
    KEY_2: (-1.0, 0.0, 0.0),
    KEY_3: (0.0, 1.0, 0.0),
    KEY_4: (0.0, -1.0, 0.0),
    KEY_5: (0.0, 0.0, 1.0),
    KEY_6: (0.0, 0.0, -1.0),
}
DIRECTION_LABELS = {
    KEY_1: "+x (along cylinder axis)",
    KEY_2: "-x (along cylinder axis)",
    KEY_3: "+y (out of the palm)",
    KEY_4: "-y (into the palm)",
    KEY_5: "+z (up)",
    KEY_6: "-z (down, with gravity)",
}


class GraspMonitor:
    """Compares simulated joint angles with the IK target angles."""

    def __init__(self, model, data, hand: HandModel, q_target: np.ndarray, radius: float,
                 tip_radius: float = TIP_RADIUS, shape: str = "cylinder"):
        self.model = model
        self.data = data
        self.hand = hand
        self.radius = radius
        self.tip_radius = tip_radius
        self.shape = shape

        # joint name -> expected angle (mimic joints included)
        self.target: dict[str, float] = {}
        self.chain_of: dict[str, str] = {}
        for chain in hand.chains:
            a, b = hand.chain_slice(chain.name)
            self.target.update(hand.joint_values(chain, q_target[a:b]))
            for j in chain.joints:
                if j.type == "revolute":
                    self.chain_of[j.name] = chain.name

        self.joints = []  # (name, qpos_addr, dof_addr, actuator_id or -1)
        for name in self.target:
            jid = _joint_id(model, name)
            if jid < 0:
                continue
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
            self.joints.append((name, model.jnt_qposadr[jid], model.jnt_dofadr[jid], act))

        self.obj = _body_id(model, OBJECT_BODY)
        self.tips = []
        for chain in hand.chains:
            bid = _body_id(model, chain.tip_link)
            if bid >= 0:
                self.tips.append((chain.name, bid))

    # ------------------------------------------------------------------ #
    def object_pose(self):
        center = self.data.xpos[self.obj].copy()
        axis = self.data.xmat[self.obj].reshape(3, 3)[:, 2].copy()
        return center, axis

    # old name, kept for callers outside this file
    cylinder_pose = object_pose

    def tip_gaps(self) -> dict[str, float]:
        """Signed clearance between each tip sphere surface and the object."""
        center, axis = self.object_pose()
        out = {}
        for name, bid in self.tips:
            v = self.data.xpos[bid] - center
            if self.shape == "cylinder":
                dist = float(np.linalg.norm(v - (v @ axis) * axis))
            else:
                dist = float(np.linalg.norm(v))
            out[name] = dist - self.radius - self.tip_radius
        return out

    def joint_rows(self):
        rows = []
        for name, qadr, dadr, act in self.joints:
            tgt = self.target[name]
            act_q = float(self.data.qpos[qadr])
            row = {
                "joint": name,
                "finger": self.chain_of.get(name, "-"),
                "target_deg": np.degrees(tgt),
                "actual_deg": np.degrees(act_q),
                "err_deg": np.degrees(act_q - tgt),
                "mimic": name not in self.hand.active_joint_names,
            }
            if act >= 0:
                row["ctrl_deg"] = np.degrees(float(self.data.ctrl[act]))
                row["force"] = float(self.data.actuator_force[act])
                fmax = self.model.actuator_forcerange[act]
                row["saturated"] = bool(
                    np.abs(row["force"]) >= 0.999 * max(abs(fmax[0]), abs(fmax[1]))
                )
            rows.append(row)
        return rows

    def per_finger_error(self):
        out: dict[str, dict] = {}
        for row in self.joint_rows():
            d = out.setdefault(row["finger"], {"max": 0.0, "mean": 0.0, "n": 0})
            d["max"] = max(d["max"], abs(row["err_deg"]))
            d["mean"] += abs(row["err_deg"])
            d["n"] += 1
        for d in out.values():
            d["mean"] /= max(d["n"], 1)
        return out

    def contact_summary(self):
        """(#contacts on the cylinder, sum of normal-force magnitudes [N])."""
        geom_of_cyl = {
            gid for gid in range(self.model.ngeom)
            if self.model.geom_bodyid[gid] == self.obj
        }
        n = 0
        fn = 0.0
        buf = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in geom_of_cyl or c.geom2 in geom_of_cyl:
                n += 1
                mujoco.mj_contactForce(self.model, self.data, i, buf)
                fn += abs(float(buf[0]))
        return n, fn


# --------------------------------------------------------------------------- #
# application
# --------------------------------------------------------------------------- #
class InteractiveGrasp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        hand_xml = ensure_hand_xml().read_text(encoding="utf-8")
        self.mass = args.weight
        self.radius = args.radius
        self.shape = args.shape

        self.hand = HandModel(args.urdf)

        grasp_mode = args.adjust or args.policy is not None
        if args.center is not None:
            self.center = np.asarray(args.center, dtype=float)
            self.center_mode = "given"
        elif args.ik_fit or not grasp_mode:
            # the IK-fitted placement: where the fingertips *point* when open
            probe = PlacementProbe(self.hand, hand_xml, args.radius, args.weight,
                                   args.shape, args.timestep)
            self.center, _, _ = fit_object_center(
                self.hand, args.radius, args.weight, args.squeeze_gain, args.tip_radius,
                shape=args.shape, probe=probe,
            )
            self.center_mode = "ik-fit"
        else:
            # the clasp, wrap-aligned so the fingertips curl over the far side
            # of the cylinder instead of pushing it forward
            self.center = (wrap_aligned_center(self.hand, args.radius)
                           if args.shape == "cylinder" else fist_center(self.hand))
            self.center_mode = "clasp"

        self.open_gaps, self.ik_residuals = finger_reach(
            self.hand, self.center, args.radius, args.weight,
            args.squeeze_gain, args.tip_radius, args.shape,
        )
        # solve_angles targets the tip *frame origin*, which is the centre of the
        # contact sphere attached to every tip link, so the radius we ask for is
        # the object radius grown by one tip radius (see tip_center_radius).
        self.q_target = solve_angles(
            self.hand,
            tip_center_radius(args.radius, args.tip_radius),
            args.weight,
            squeeze_gain=args.squeeze_gain,
            center=self.center,
            shape=args.shape,
        )

        self.contact_angles, self.contact_gap = contact_geometry(
            self.hand, self.center, self.q_target, args.shape
        )

        scene = build_interactive_scene(
            hand_xml,
            radius=args.radius,
            mass=args.weight,
            center=self.center,
            kp=args.kp,
            kv=args.kv,
            force_range=args.force_range,
            timestep=args.timestep,
            gravity=args.gravity,
            shape=args.shape,
            pose_sliders=bool(args.adjust),
        )
        self._tmp = build_dir() / "_scene_viewer.xml"
        self._tmp.write_text(scene, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self._tmp))
        self.data = mujoco.MjData(self.model)
        self.monitor = GraspMonitor(
            self.model, self.data, self.hand, self.q_target, args.radius,
            args.tip_radius, args.shape,
        )

        self.pull_on = False
        self.pull_dir = np.asarray(PULL_DIRECTIONS[KEY_6], dtype=float)
        self.pull_dir_key = KEY_6
        self.pull_cmd = 0.0     # commanded magnitude from the keys
        self.pull_now = 0.0     # ramped magnitude actually applied
        self.elapsed = 0.0
        self.last_report = 0.0
        self.peak_drift = 0.0

        self.obs_spec = make_obs_spec(self.model, self.hand, args.shape,
                                      OBJECT_BODY, OBJECT_JOINT)
        self.joint_order = actuator_joint_order(self.model)
        # the five zero-gain placement actuators (when present) sit after the
        # finger actuators, so joint commands must stop short of them
        self.n_joint_act = len(self.joint_order)
        self.pose_adr = self.n_joint_act if self.n_joint_act < self.model.nu else None
        self.obj_gid = next(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.monitor.obj
        )
        self.q_flex = np.array([self.hand.joints[n].upper for n in self.joint_order])
        self._is_thumb = np.array([n.startswith("thumb") for n in self.joint_order])
        self.policy = None
        self.policy_norm = None
        self.policy_on = False
        self.policy_name = ""
        self._episode_seed = -1
        self._policy_obs = None
        self.policy_episode_seconds = 4.0
        # manual placement mode
        self.adjusting = bool(getattr(args, "adjust", False))
        self.obj_target = np.asarray(self.center, dtype=float).copy()
        # Horizontal initial pose: the cylinder's axis starts along +y, lying
        # across the fingers like a bar - the fingers flex in the world x-z
        # plane, so each of them wraps around the bar's circular cross-section.
        # tilt_x = -90 deg rotates the body's +z axis onto +y.  A sphere is
        # rotationally symmetric, so it keeps the identity quaternion (the
        # trained policies expect it).
        self.obj_tilt_x = -np.pi / 2.0 if self.shape == "cylinder" else 0.0
        self.obj_tilt_y = 0.0
        self.nudge = float(getattr(args, "nudge", 0.002))
        self.tilt_step = np.radians(float(getattr(args, "tilt_step", 5.0)))
        self._physics_count = 0
        self._prev_action = np.zeros(self.obs_spec.n_active, np.float32)
        if args.policy:
            self._load_policy(args.policy, args.policy_norm)
        self.reset()

    # ------------------------------------------------------------------ #
    def _load_policy(self, model_path: str, norm_path: str | None) -> None:
        """Load an SB3 policy plus the VecNormalize stats it was trained with."""
        try:
            import p24grasp.env.grasp as grasp_env  # noqa: F401 - the pickle references GraspEnv
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from p24grasp.env.obs import ObsNormalizer

            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(model_file)
            # the algorithm lives in the run path (runs/<shape>/<algo>/...) but
            # the model itself may sit in a "best"/"checkpoints" subdirectory
            algo = SAC if "sac" in str(model_file).lower() else PPO
            self.policy = algo.load(str(model_file), device="cpu")
            self.policy_name = model_file.parent.name
            norm_file = norm_path or str(model_file.parent / "vecnormalize.pkl")
            if Path(norm_file).exists():
                vec = VecNormalize.load(
                    norm_file,
                    DummyVecEnv([lambda: grasp_env.GraspEnv(shape=self.shape)]),
                )
                self.policy_norm = ObsNormalizer(vec.obs_rms, vec.clip_obs, vec.epsilon)
                vec.close()
            self.policy_on = True
            # the policy was trained on fixed-length episodes; running past that
            # horizon lets the grasp degrade, so replay in the same window
            self.policy_episode_seconds = (
                grasp_env.EPISODE_STEPS * grasp_env.FRAME_SKIP * grasp_env.PHYSICS_DT
            )
            print(f"[viewer] loaded policy {model_file} "
                  f"(obs norm: {'yes' if self.policy_norm else 'NO - results will be wrong'})")
        except Exception as exc:  # noqa: BLE001 - fall back to the IK behaviour
            print(f"[viewer] could not load policy ({exc}); "
                  f"falling back to the IK grasp targets.", file=sys.stderr)
            self.policy = None
            self.policy_on = False

    def _reset_from_env(self, seed: int) -> np.ndarray:
        """Sample one episode from GraspEnv and adopt its state."""
        import p24grasp.env.grasp as grasp_env
        src = getattr(self, "_env_source", None)
        if src is None:
            src = grasp_env.GraspEnv(shape=self.shape, randomize=True)
            self._env_source = src
        obs, _info = src.reset(seed=seed)
        self.model.geom_size[:] = src.model.geom_size
        self.model.body_mass[:] = src.model.body_mass
        self.model.body_subtreemass[:] = src.model.body_subtreemass
        self.model.body_inertia[:] = src.model.body_inertia
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = src.data.qpos
        self.data.qvel[:] = src.data.qvel
        self.data.ctrl[:self.n_joint_act] = src.data.ctrl[:self.n_joint_act]
        self.center = np.asarray(src.center, dtype=float)
        self.radius = float(src._radius)
        self.mass = float(src._mass)
        self.monitor.radius = self.radius
        mujoco.mj_forward(self.model, self.data)
        return obs

    def _policy_action(self) -> np.ndarray:
        obs = build_obs(self.model, self.data, self.obs_spec, self.center,
                        self.radius, self.mass, self._prev_action)
        if self.policy_norm is not None:
            obs = self.policy_norm(obs)
        action, _ = self.policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).ravel()
        self._prev_action = action
        return self.obs_spec.qpos_center + np.clip(action, -1, 1) * self.obs_spec.qpos_half

    # ------------------------------------------------------------------ #
    def preflight(self) -> str:
        """Report whether the *requested* grasp is physically meaningful.

        This is computed from hand kinematics only, before any simulation, so it
        separates "the IK target is unreachable" from "the controller cannot
        hold the joint at the target".
        """
        c = self.center
        lines = [
            "=" * 78,
            f"grasp request: shape={self.shape}  radius={self.radius * 1000:.0f} mm  "
            f"mass={self.mass:.3f} kg  centre=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) "
            f"[{self.center_mode}]",
            f"scene default centre was ({CENTER[0]:.4f}, {CENTER[1]:.4f}, {CENTER[2]:.4f})",
            "",
            f"  {'finger':<9}{'open gap':>11}{'IK residual':>13}   verdict",
        ]
        bad = 0
        for chain, gap, res in zip(self.hand.chains, self.open_gaps, self.ik_residuals):
            notes = []
            if gap < 0:
                notes.append("open pose ALREADY penetrates")
            if res > 0.004:
                notes.append("target out of reach")
            bad += bool(notes)
            lines.append(
                f"  {chain.name:<9}{gap * 1000:>9.1f}mm{res * 1000:>11.1f}mm   "
                + ("; ".join(notes) if notes else "ok")
            )
        lines.append("")

        info, uncovered = self.contact_angles, self.contact_gap
        wrapped = uncovered <= 180.0
        if self.shape == "cylinder":
            lines.append(
                "contact directions around the cylinder axis (0 deg = +z): "
                + "  ".join(f"{k}={info[k]:.0f}" for k in info)
            )
            lines.append(
                f"widest uncovered arc = {uncovered:.0f} deg"
                + (": contacts wrap the object." if wrapped else
                   " (> 180): every contact is on one side, so the hand presses the "
                   "object outward\ninstead of pinning it - expect it to be squeezed "
                   "out along the axis rather than held.")
            )
        else:
            pts = np.asarray(list(info.values()))
            cos = np.clip(pts @ pts.T, -1.0, 1.0)
            np.fill_diagonal(cos, -1.0)
            lines.append(
                "contact directions (unit vectors from the object centre): "
                + "  ".join(k for k in info)
            )
            lines.append(
                f"max opposition between any two fingers = "
                f"{np.degrees(np.arccos(cos.max())):.0f} deg; "
                + ("directions span the sphere (force closure possible)." if wrapped else
                   "directions all lie in one hemisphere, so the hand pushes the "
                   "sphere away\nrather than pinning it - it will roll out.")
            )

        if bad:
            lines.append(
                f"{bad} finger(s) cannot follow the requested grasp; the remaining "
                "angle error is an IK/placement\nproblem, not an actuator problem - "
                "raising --kp will not fix it."
            )
        elif not wrapped:
            lines.append(
                "all fingers reach their targets, so any residual angle error under "
                "load is an actuator problem\n(raise --kp), but the grasp is not "
                "force-closed so it will not hold."
            )
        else:
            lines.append(f"all fingers can reach the {self.shape} and start clear of it.")
        lines.append("=" * 78)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.adjusting:
            # keep the object where the operator put it and bring the fingers to
            # the grasp pose so the clearance is visible while placing it
            self.obj_target = np.asarray(self.center, dtype=float).copy()
            # horizontal initial pose: axis along +y, like a bar across the palm
            self.obj_tilt_x = -np.pi / 2.0 if self.shape == "cylinder" else 0.0
            self.obj_tilt_y = 0.0
            self._write_pose_sliders()
            self._pin_object()
            self._apply_grasp_pose()
            mujoco.mj_forward(m, d)
            self.elapsed = 0.0
            self.pull_now = 0.0
            self.pull_cmd = 0.0
            self.peak_drift = 0.0
            self._physics_count = 0
            return
        if self.policy is not None:
            # Replay a genuine episode: a policy trained with domain
            # randomisation only behaves as evaluated when the episode is drawn
            # from the same distribution, so sample one from the real env and
            # copy its state rather than hand-building a "typical" start.
            self._episode_seed += 1
            obs = self._reset_from_env(seed=self._episode_seed)
            self._prev_action = np.zeros(self.obs_spec.n_active, np.float32)
            self._policy_obs = obs
            self.elapsed = 0.0
            self.pull_now = 0.0
            self.pull_cmd = 0.0
            self.peak_drift = 0.0
            self._physics_count = 0
            return
        else:
            frac = self.args.pregrasp
            for name, qadr, _dadr, _act in self.monitor.joints:
                d.qpos[qadr] = frac * self.monitor.target[name]
        # object at the clasp centre.  The cylinder's axis is set by the quat on
        # its geom, so the body quaternion stays identity - applying the same
        # rotation here as well would compose them and tip the axis sideways.
        jid = _joint_id(m, OBJECT_JOINT)
        d.qpos[m.jnt_qposadr[jid]: m.jnt_qposadr[jid] + 7] = [*self.center, 1.0, 0.0, 0.0, 0.0]
        for aid in range(self.n_joint_act):
            aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            jname = aname.removeprefix("act_")
            if self.policy is not None:
                d.ctrl[aid] = d.qpos[m.jnt_qposadr[_joint_id(m, jname)]]
            elif jname in self.monitor.target:
                d.ctrl[aid] = self.monitor.target[jname]
        mujoco.mj_forward(m, d)
        self.elapsed = 0.0
        self.pull_now = 0.0
        self.pull_cmd = 0.0
        self.peak_drift = 0.0
        self._prev_action = np.zeros(self.obs_spec.n_active, np.float32)
        self._physics_count = 0

    # ------------------------------------------------------------------ #
    def _grasp_pose(self) -> np.ndarray:
        """The pose the hand closes to for a power grasp.

        Uniform flexion over-drives the thumb, which sits inside the clasp and
        would push the object out, so the thumb is scaled separately (see
        grasp_env.INIT_THUMB_SCALE / INIT_FINGER_SCALE).
        """
        return np.where(self._is_thumb, 0.60 * self.q_flex, 0.90 * self.q_flex)

    def _apply_grasp_pose(self):
        n = self.n_joint_act
        self.data.ctrl[:n] = np.clip(
            self._grasp_pose(),
            self.model.actuator_ctrlrange[:n, 0], self.model.actuator_ctrlrange[:n, 1],
        )

    def _set_radius(self, radius: float):
        """Resize the object (radius slider); mass stays, inertia follows."""
        radius = float(np.clip(radius, 0.010, 0.060))
        if abs(radius - self.radius) < 1e-9:
            return
        self.radius = radius
        self.monitor.radius = radius
        m = self.model
        m.geom_size[self.obj_gid, 0] = radius
        if self.shape == "cylinder":
            m.geom_size[self.obj_gid, 1] = HALF_LENGTH
        i_a, i_b, i_c = object_inertia(radius, self.mass, self.shape)
        m.body_inertia[self.monitor.obj] = (i_a, i_b, i_c)

    def _read_pose_sliders(self) -> bool:
        """Adopt the MuJoCo sidebar sliders as the object placement + size."""
        if self.pose_adr is None:
            return False
        vals = self.data.ctrl[self.pose_adr:self.pose_adr + 6]
        self.obj_target = np.asarray(vals[:3], dtype=float).copy()
        self.obj_tilt_x, self.obj_tilt_y = float(vals[3]), float(vals[4])
        self._set_radius(float(vals[5]))
        return True

    def _write_pose_sliders(self):
        """Push a keyboard nudge back so the sliders stay in sync."""
        if self.pose_adr is None:
            return
        self.data.ctrl[self.pose_adr:self.pose_adr + 6] = [
            *self.obj_target, self.obj_tilt_x, self.obj_tilt_y, self.radius,
        ]

    def _object_axis(self) -> np.ndarray:
        """World direction of the cylinder's axis under the current tilts."""
        w, x, y, z = self._object_quat()
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    def _object_quat(self) -> np.ndarray:
        """Body quaternion for the placement tilts: Rx(tilt_x) then Ry(tilt_y)."""
        ax, ay = self.obj_tilt_x / 2.0, self.obj_tilt_y / 2.0
        qx = np.array([np.cos(ax), np.sin(ax), 0.0, 0.0])
        qy = np.array([np.cos(ay), 0.0, np.sin(ay), 0.0])
        return self._quat_mul(qx, qy)

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _start_run(self):
        """Release the object and start the automatic grasp from a clean state.

        While placing the object the fingers are parked on the grasp pose but
        the pinned object deflects them, so simply unpinning would let the
        servos snap back and flick the object away.  Re-seat the hand and the
        object the way ``grasp_env.GraspEnv.reset`` does instead, so the run
        starts from the same state the policy/pose was validated on.
        """
        m, d = self.model, self.data
        # _reset_from_env overwrites the model geometry with the sampled env
        # state, so remember the operator's size choice and restore it after
        want_radius = self.radius
        if self.policy is not None:
            # The policy was trained on randomised finger starts and is
            # sensitive to them, so draw one from the real env (pose *and*
            # controller targets) and keep only the operator's object placement
            # on top of it.
            self._episode_seed += 1
            self._reset_from_env(seed=self._episode_seed)
            self._set_radius(want_radius)
            # keep the sampled jitter/tilt/velocity (the policy is sensitive to
            # them) and shift the whole grasp by the operator's offset, so
            # leaving the object at the clasp reproduces a normal episode
            jid0 = _joint_id(m, OBJECT_JOINT)
            a0 = m.jnt_qposadr[jid0]
            sampled_pos = d.qpos[a0:a0 + 3].copy()
            sampled_quat = d.qpos[a0 + 3:a0 + 7].copy()
            offset = self.obj_target - np.asarray(self.center, dtype=float)
        else:
            n = self.n_joint_act
            q = np.clip(self._grasp_pose(),
                        self.model.actuator_ctrlrange[:n, 0],
                        self.model.actuator_ctrlrange[:n, 1])
            d.qpos[self.obs_spec.qpos_adr] = q
            d.qvel[self.obs_spec.dof_adr] = 0.0
            d.ctrl[:self.n_joint_act] = q
        jid = _joint_id(m, OBJECT_JOINT)
        adr = m.jnt_qposadr[jid]
        if self.policy is not None:
            d.qpos[adr:adr + 3] = sampled_pos + offset
            d.qpos[adr + 3:adr + 7] = self._quat_mul(self._object_quat(), sampled_quat)
        else:
            d.qpos[adr:adr + 3] = self.obj_target
            d.qpos[adr + 3:adr + 7] = self._object_quat()
            d.qvel[m.jnt_dofadr[jid]:m.jnt_dofadr[jid] + 6] = 0.0
        mujoco.mj_forward(m, d)
        self.adjusting = False
        self.elapsed = 0.0
        self._physics_count = 0
        self._prev_action = np.zeros(self.obs_spec.n_active, np.float32)
        self.peak_drift = 0.0

    def _pin_object(self):
        """Hold the object kinematically at the placement target."""
        m, d = self.model, self.data
        jid = _joint_id(m, OBJECT_JOINT)
        adr = m.jnt_qposadr[jid]
        d.qpos[adr:adr + 3] = self.obj_target
        d.qpos[adr + 3:adr + 7] = self._object_quat()
        dadr = m.jnt_dofadr[jid]
        d.qvel[dadr:dadr + 6] = 0.0

    def apply_pull(self):
        d = self.data
        m = self.model
        d.xfrc_applied[:] = 0.0
        if self.pull_now > 1e-6:
            d.xfrc_applied[self.monitor.obj, :3] = self.pull_now * self.pull_dir

    def step(self, frame_dt: float):
        m, d = self.model, self.data
        nsub = max(1, int(round(frame_dt / m.opt.timestep)))

        ramp = self.args.pull_ramp * frame_dt
        want = self.pull_cmd if self.pull_on else 0.0
        self.pull_now += np.clip(want - self.pull_now, -ramp, ramp)

        # the policy must run at a fixed 100 Hz.  Indexing the substep loop
        # directly would give 8 steps per 1/60 s frame, so a counter carries
        # across frames instead.
        for _ in range(nsub):
            if self.adjusting:
                # object pinned, fingers parked on the grasp pose: the operator
                # is dialling in the starting pose, not simulating a grasp yet
                self._read_pose_sliders()
                self._apply_grasp_pose()
                self._pin_object()
            elif self.policy_on and self._physics_count % POLICY_PERIOD == 0:
                d.ctrl[:self.n_joint_act] = self._policy_action()
            elif not self.policy_on:
                self._apply_grasp_pose()
            self._physics_count += 1
            self.apply_pull()
            mujoco.mj_step(m, d)
            self.elapsed += m.opt.timestep

        self.peak_drift = max(
            self.peak_drift, float(np.linalg.norm(d.xpos[self.monitor.obj] - self.center))
        )

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def status_lines(self):
        fingers = self.monitor.per_finger_error()
        ncon, _fn = self.monitor.contact_summary()
        center, _axis = self.monitor.cylinder_pose()
        drift = float(np.linalg.norm(center - self.center))

        rows = [
            (f"t={self.elapsed:5.2f}s   mode: "
             + ("ADJUST (object pinned)" if self.adjusting else
                ("RUN (policy)" if self.policy_on else "RUN (grasp pose)"))
             + f"   object ({self.obj_target[0]:.4f}, {self.obj_target[1]:.4f}, "
               f"{self.obj_target[2]:.4f})  r={self.radius * 1000:.1f}mm"
             + (f"  axis ({self._object_axis()[0]:+.2f}, "
                f"{self._object_axis()[1]:+.2f}, {self._object_axis()[2]:+.2f})"
                f"  step {self.nudge * 1000:.1f} mm / "
                f"{np.degrees(self.tilt_step):.1f} deg"
                if self.shape == "cylinder" else "")),
            f"t={self.elapsed:5.2f}s   pull {self.pull_now:5.2f} N  "
            f"[{DIRECTION_LABELS[self.pull_dir_key]}]  {'ON ' if self.pull_on else 'off'}"
            + (f"   policy {'ON' if self.policy_on else 'off'}" if self.policy else ""),
            f"{self.shape} drift {drift * 1000:6.2f} mm   contacts {ncon}   "
            f"peak drift {self.peak_drift * 1000:6.2f} mm",
            "",
            f"{'finger':<8}{'max err':>10}{'mean err':>10}{'tip gap':>10}",
        ]
        gaps = self.monitor.tip_gaps()
        for chain in self.hand.chains:
            e = fingers.get(chain.name, {"max": 0.0, "mean": 0.0})
            rows.append(
                f"{chain.name:<8}{e['max']:>9.1f}°{e['mean']:>9.1f}°"
                f"{gaps.get(chain.name, float('nan')) * 1000:>8.2f}mm"
            )
        return rows

    def help_lines(self):
        return [
            "k  toggle pull      n/m  pull magnitude (now %.2f N)" % self.pull_cmd,
            "1-6  pull dir  +x -x +y -y +z -z   (now: %s)" % DIRECTION_LABELS[self.pull_dir_key],
            "r  reset            p  print full joint table",
            "c  clear force",
            ("PLACEMENT  move: w/s +z-  a/d -y+  q/e -x+   z/x step   g start"
             if self.adjusting else
             "b  back to placement mode   g  restart grasp"),
            ("           sliders: pose_x/y/z, pose_tilt_x/y (+-360 deg), obj_size"
             if self.adjusting else ""),
            ("           tilt keys: f->+y  v->-y  t->-x  y->+x"
             if self.shape == "cylinder" and self.adjusting else ""),
            "episodes auto-repeat every %.1f s" % self.policy_episode_seconds,
            (f"o  policy: {self.policy_name} {'ON' if self.policy_on else 'off'}"
             if self.policy is not None else "o  policy: none loaded"),
        ]

    def full_report(self, title: str = "joint angle tracking") -> str:
        rows = self.monitor.joint_rows()
        lines = ["", f"=== {title} ===",
                 f"shape={self.shape}  radius={self.radius * 1000:.0f} mm  mass={self.mass:.3f} kg  "
                 f"pull={self.pull_now:.2f} N {DIRECTION_LABELS[self.pull_dir_key]}",
                 f"{'joint':<24}{'target°':>9}{'actual°':>9}{'err°':>9}"
                 f"{'force Nm':>10}{'sat':>5}"]
        worst = ("", 0.0)
        for r in rows:
            flag = "M" if r["mimic"] else " "
            sat = "YES" if r.get("saturated") else ""
            lines.append(
                f"{r['joint']:<24}{r['target_deg']:>9.2f}{r['actual_deg']:>9.2f}"
                f"{r['err_deg']:>9.2f}{r.get('force', float('nan')):>10.4f}{sat:>5} {flag}"
            )
            if abs(r["err_deg"]) > worst[1] and not r["mimic"]:
                worst = (r["joint"], abs(r["err_deg"]))

        act_rows = [r for r in rows if not r["mimic"]]
        mean_err = float(np.mean([abs(r["err_deg"]) for r in act_rows])) if act_rows else 0.0
        max_err = float(np.max([abs(r["err_deg"]) for r in act_rows])) if act_rows else 0.0
        lines.append("")
        lines.append(
            f"active joints: mean |err| {mean_err:.2f}°   max |err| {max_err:.2f}° "
            f"({worst[0]})"
        )

        ncon, _fn = self.monitor.contact_summary()
        center, _axis = self.monitor.cylinder_pose()
        drift = float(np.linalg.norm(center - self.center))
        holding = ncon >= MIN_SUCCESS_CONTACTS and drift < SUCCESS_DRIFT
        lines.append(
            f"{self.shape}: {ncon} contact(s), {drift * 1000:.1f} mm from the nominal "
            f"grasp pose -> {'held' if holding else 'NOT held'}"
        )
        if self.policy_on:
            lines.append(
                "verdict: a learned policy is driving the joints, so the errors above "
                "measure its distance\nfrom the geometric IK target, not from its own "
                "intent - the policy deliberately departs\nfrom that pose."
            )
            if holding:
                lines.append("        the object is held.")
            lines.append("")
            return "\n".join(lines)
        if max_err > 5.0:
            lines.append(
                "verdict: angles do NOT reach the IK target. If raising --kp (and "
                "--force-range) shrinks the error, the actuators are too weak; if it "
                "does not, the IK target itself is not reachable."
            )
        elif not holding:
            lines.append(
                "verdict: the joints track the IK target, but the cylinder is not in "
                "the hand - the angle agreement\nbelow only shows the hand holding its "
                "commanded pose, not a successful grasp."
            )
            if self.contact_gap > 180.0:
                lines.append(
                    "cause: the contacts do not wrap the cylinder (widest uncovered "
                    "arc %.0f deg), so it is squeezed out." % self.contact_gap
                )
        else:
            lines.append("verdict: the joints track the IK target and the cylinder is held.")
        lines.append("")
        return "\n".join(lines)

    def console_status(self):
        fingers = self.monitor.per_finger_error()
        ncon, _ = self.monitor.contact_summary()
        center, _ = self.monitor.cylinder_pose()
        parts = " ".join(f"{c.name}:{fingers.get(c.name, {'max': 0})['max']:4.1f}"
                         for c in self.hand.chains)
        print(
            f"[t={self.elapsed:5.2f}] pull={self.pull_now:5.2f}N "
            f"drift={np.linalg.norm(center - self.center) * 1000:5.2f}mm "
            f"con={ncon:2d}  max|err| {parts}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # loops
    # ------------------------------------------------------------------ #
    def on_key(self, keycode: int):
        if keycode == KEY_K:
            self.pull_on = not self.pull_on
        elif keycode == KEY_M:
            self.pull_cmd += self.args.pull_step
            self.pull_on = True
        elif keycode == KEY_N:
            self.pull_cmd = max(0.0, self.pull_cmd - self.args.pull_step)
        elif keycode in PULL_DIRECTIONS:
            self.pull_dir = np.asarray(PULL_DIRECTIONS[keycode], dtype=float)
            self.pull_dir_key = keycode
        elif keycode == KEY_R:
            self.reset()
        elif keycode == KEY_P:
            print(self.full_report(), flush=True)
        elif keycode == KEY_C:
            self.pull_on = False
            self.pull_cmd = 0.0
        elif keycode == KEY_G and self.adjusting:
            self._start_run()
            print(f"[viewer] released at ({self.obj_target[0]:.4f}, "
                  f"{self.obj_target[1]:.4f}, {self.obj_target[2]:.4f}), "
                  f"tilt x{np.degrees(self.obj_tilt_x):+.0f}/y{np.degrees(self.obj_tilt_y):+.0f} deg"
                  f" -> running")
        elif keycode == KEY_B and not self.adjusting:
            self.adjusting = True
            self.obj_target = self.data.qpos[
                self.model.jnt_qposadr[_joint_id(self.model, OBJECT_JOINT)]:
                self.model.jnt_qposadr[_joint_id(self.model, OBJECT_JOINT)] + 3
            ].copy()
            print("[viewer] back to placement mode (object re-pinned)")
        elif keycode in (KEY_W, KEY_S, KEY_A, KEY_D, KEY_Q, KEY_E,
                         KEY_F, KEY_V, KEY_T, KEY_Y):
            delta = {
                KEY_W: (0, 0, 1), KEY_S: (0, 0, -1),
                KEY_A: (0, -1, 0), KEY_D: (0, 1, 0),
                KEY_Q: (-1, 0, 0), KEY_E: (1, 0, 0),
            }.get(keycode)
            if not self.adjusting:
                # RUN mode: nudge the *live* object relative to its current
                # pose instead of snapping it back to the placement target
                jid = _joint_id(self.model, OBJECT_JOINT)
                adr = self.model.jnt_qposadr[jid]
                dadr = self.model.jnt_dofadr[jid]
                if delta is not None:
                    self.data.qpos[adr:adr + 3] = (
                        self.data.qpos[adr:adr + 3] + np.asarray(delta) * self.nudge
                    )
                elif self.shape == "cylinder":
                    # rotate the live orientation about x/y by one step
                    step = self.tilt_step if keycode in (KEY_F, KEY_Y) else -self.tilt_step
                    dq = (np.array([np.cos(step / 2), np.sin(step / 2), 0.0, 0.0])
                          if keycode in (KEY_F, KEY_V) else
                          np.array([np.cos(step / 2), 0.0, np.sin(step / 2), 0.0]))
                    self.data.qpos[adr + 3:adr + 7] = self._quat_mul(
                        dq, self.data.qpos[adr + 3:adr + 7]
                    )
                self.data.qvel[dadr:dadr + 6] = 0.0
                return
            if delta is not None:
                self.obj_target = self.obj_target + np.asarray(delta) * self.nudge
                self._write_pose_sliders()
            # the keys are labelled by where the cylinder *axis* ends up, which
            # is the opposite sign of the body rotation that gets it there.
            # Tilting a sphere is physically meaningless (rotational symmetry),
            # so those keys are no-ops for it.
            if self.shape != "cylinder":
                return
            elif keycode == KEY_T:
                self.obj_tilt_y -= self.tilt_step      # axis -> -x
            elif keycode == KEY_Y:
                self.obj_tilt_y += self.tilt_step      # axis -> +x
            elif keycode == KEY_F:
                self.obj_tilt_x -= self.tilt_step      # axis -> +y
            else:
                self.obj_tilt_x += self.tilt_step      # axis -> -y
            self._write_pose_sliders()
            self._pin_object()
        elif keycode == KEY_Z:
            self.nudge = max(0.0002, self.nudge / 2)
        elif keycode == KEY_X:
            self.nudge = min(0.02, self.nudge * 2)
        elif keycode == KEY_O and self.policy is not None:
            self.policy_on = not self.policy_on
            self._prev_action = np.zeros(self.obs_spec.n_active, np.float32)

    def run_viewer(self):
        print("[viewer] ctrl+right-drag drags the object with the mouse; "
              "'k' applies the keyboard pull force.")
        if self.adjusting:
            print("[viewer] PLACEMENT mode: set the object pose with the "
                  "pose_x / pose_y / pose_z / pose_tilt_x / pose_tilt_y / obj_size "
                  f"sliders in the Control panel (ctrl[{self.pose_adr}..{self.model.nu - 1}]), "
                  "then press 'g' to release and grasp.")
        try:
            handle = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.on_key
            )
        except RuntimeError as exc:
            # On macOS the passive viewer needs a process with a proper Cocoa
            # main loop, which only mjpython provides.
            print(f"\n[viewer] cannot open the GUI window: {exc}\n"
                  f"[viewer] run this script with mjpython instead:\n"
                  f"           mjpython {Path(__file__).name} "
                  f"--radius {self.radius} --weight {self.mass}\n"
                  f"[viewer] falling back to the headless pull test.\n", flush=True)
            self.run_headless()
            return

        with handle as viewer:
            viewer.cam.lookat[:] = self.center
            viewer.cam.distance = 0.42
            viewer.cam.azimuth = 180.0
            viewer.cam.elevation = -20.0
            frame_dt = 1.0 / 60.0
            while viewer.is_running():
                t0 = time.time()
                if not self.adjusting and self.elapsed >= self.policy_episode_seconds:
                    # re-run the grasp from the same placement, so the operator
                    # can watch it repeatedly without re-dialling the pose
                    self._start_run()
                self.step(frame_dt)
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "\n".join(self.help_lines()), ""),
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     "\n".join(self.status_lines()), ""),
                ])
                viewer.sync()
                if self.elapsed - self.last_report > self.args.report_period:
                    self.last_report = self.elapsed
                    self.console_status()
                dt = frame_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        print(self.full_report("final state at viewer close"))

    def run_headless(self):
        frame_dt = 1.0 / 60.0
        if self.adjusting:
            # headless placement mode: settle the fingers on the grasp pose, then
            # release and run, mirroring what 'g' does interactively
            for _ in range(int(self.args.settle_time * 60)):
                self.step(frame_dt)
            self._start_run()
        # only pull if one was asked for; a policy episode is about holding, and
        # a default 5 N yank would just knock the object out of the hand
        self.pull_on = self.args.pull_force > 0
        self.pull_cmd = self.args.pull_force
        # With a policy in charge, run whole episodes back to back (like the
        # interactive mode) so a headless run samples the policy's distribution
        # instead of always testing episode seed 0.  Without a policy the run
        # is a single window of the requested duration.
        if self.policy is None:
            limit = self.args.duration
            while self.elapsed < limit:
                self.step(frame_dt)
                if self.elapsed - self.last_report > self.args.report_period:
                    self.last_report = self.elapsed
                    self.console_status()
        else:
            wall = 0.0
            episodes = 0
            while wall < self.args.duration:
                while self.elapsed < self.policy_episode_seconds:
                    self.step(frame_dt)
                episodes += 1
                wall += self.policy_episode_seconds
                self._start_run()
                print(f"[viewer] episode {episodes} finished", flush=True)
            print(f"[viewer] ran {episodes} policy episodes", flush=True)
        print(self.full_report(f"headless pull test ({self.pull_cmd:.1f} N)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=str(assets_dir() / "p24_hand_right.urdf"))
    ap.add_argument("--shape", choices=SHAPES, default="cylinder",
                    help="grasped object: cylinder (axis +x) or sphere")
    ap.add_argument("--radius", type=float, default=0.026,
                    help="object radius [m]; 0.026 is the largest cylinder the "
                         "fingers can actually wrap over (see docs/FINDINGS.md)")
    ap.add_argument("--weight", type=float, default=0.85, help="object mass [kg]")
    ap.add_argument("--center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="object centre [m]; default is the clasp centre, the only "
                         "placement in which the hand actually holds the object")
    ap.add_argument("--ik-fit", action="store_true",
                    help="place the object where the per-finger IK reaches instead "
                         "(shows what the geometric solver produces; it does not hold)")
    ap.add_argument("--squeeze-gain", type=float, default=0.0008,
                    help="IK target depth per kg, passed to solve_angles")
    ap.add_argument("--tip-radius", type=float, default=TIP_RADIUS,
                    help="radius of the contact sphere attached to each tip link")
    ap.add_argument("--pregrasp", type=float, default=0.6,
                    help="initial joint angle as a fraction of the IK target")

    ap.add_argument("--kp", type=float, default=None, help="override actuator kp")
    ap.add_argument("--kv", type=float, default=None, help="override actuator kv")
    ap.add_argument("--force-range", type=float, default=None,
                    help="override symmetric actuator forcerange [N*m]")
    ap.add_argument("--timestep", type=float, default=0.002)
    ap.add_argument("--gravity", type=float, default=9.81,
                    help="gravity magnitude [m/s^2]; use 0 to isolate the grip from "
                         "the object simply falling")

    ap.add_argument("--pull-force", type=float, default=0.0,
                    help="headless perturbation force [N]; 0 (default) just checks "
                         "that the grasp holds, >0 tests it against a pull")
    ap.add_argument("--pull-step", type=float, default=0.5, help="pull increment per key [N]")
    ap.add_argument("--pull-ramp", type=float, default=10.0, help="pull ramp rate [N/s]")

    ap.add_argument("--adjust", action="store_true",
                    help="start in manual placement mode: the object is pinned and "
                         "positioned from the viewer's Control sliders (or the "
                         "keyboard), 'g' releases it and runs the automatic grasp")
    ap.add_argument("--nudge", type=float, default=0.002,
                    help="initial placement step size [m] (-/= halve/double it)")
    ap.add_argument("--tilt-step", type=float, default=5.0,
                    help="placement tilt increment [degrees]; keys are labelled by "
                         "where the cylinder axis ends up: f ->+y, v ->-y, "
                         "t ->-x, y ->+x")
    ap.add_argument("--policy", default=None,
                    help="SB3 model.zip to drive the actuators (trained by "
                         "train_grasp_rl.py); 'o' toggles it at runtime")
    ap.add_argument("--policy-norm", default=None,
                    help="vecnormalize.pkl saved next to the policy (required: the "
                         "policy was trained on normalized observations)")
    ap.add_argument("--headless", action="store_true",
                    help="no viewer: settle, then apply a constant pull for --duration")
    ap.add_argument("--settle-time", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--report-period", type=float, default=0.5)
    args = ap.parse_args()

    app = InteractiveGrasp(args)
    print(app.preflight(), flush=True)
    app.console_status()
    if args.headless:
        app.run_headless()
    else:
        app.run_viewer()


if __name__ == "__main__":
    main()
