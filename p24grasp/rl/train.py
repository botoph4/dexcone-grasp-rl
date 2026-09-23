#!/usr/bin/env python3
"""Train PPO / SAC to hold a cylinder or sphere with the P24 hand.

The geometric IK cannot grasp: it aims each fingertip outward along its *open*
pose direction, so the object ends up where the fingers point rather than where
they close.  ``grasp_env.GraspEnv`` instead exposes the 16 actuator targets as a
continuous action and rewards keeping the object in the clasp, so a policy can
find the closure the IK misses.

Usage
-----
    python train_grasp_rl.py --algo both --shape cylinder --timesteps 1000000
    python train_grasp_rl.py --algo ppo  --shape sphere   --timesteps 500000

Artefacts land in ``runs/<shape>/<algo>/``: ``model.zip`` and ``vecnormalize.pkl``
(both are needed for evaluation and viewer playback), ``best/``, ``checkpoints/``
and a tensorboard log.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


import torch  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from p24grasp.env.grasp import GraspEnv  # noqa: E402
from p24grasp.paths import outputs_dir  # noqa: E402

RUNS = outputs_dir("runs")

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.002,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
)

SAC_KWARGS = dict(
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256], log_std_init=-2,
                       activation_fn=torch.nn.Tanh),
)


class SyncNormEvalCallback(EvalCallback):
    """EvalCallback that first copies the running obs statistics.

    The evaluation env is a separate ``VecNormalize``; without syncing
    ``obs_rms`` from training the policy sees differently scaled observations
    and the reported success rate is meaningless.
    """

    def _on_step(self) -> bool:
        sync_envs_normalization(self.model.get_env(), self.eval_env)
        cont = super()._on_step()
        if self.eval_env is not None and self.best_model_save_path is not None:
            best_dir = Path(self.best_model_save_path)
            train_env = self.model.get_env()
            if isinstance(train_env, VecNormalize) and best_dir.exists():
                train_env.save(str(best_dir / "vecnormalize.pkl"))
        return cont


def make_env(shape: str, seed: int, randomize: bool = True):
    def _init():
        env = GraspEnv(shape=shape, randomize=randomize)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def build_model(algo: str, env, seed: int, tensorboard_log: Path):
    if algo == "ppo":
        return PPO("MlpPolicy", env, seed=seed, verbose=1,
                   tensorboard_log=str(tensorboard_log), device="cpu", **PPO_KWARGS)
    return SAC("MlpPolicy", env, seed=seed, verbose=1,
               tensorboard_log=str(tensorboard_log), device="cpu", **SAC_KWARGS)


def train(algo: str, shape: str, timesteps: int, seed: int,
          eval_freq: int, n_eval: int) -> Path:
    outdir = RUNS / shape / algo
    outdir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = outdir / "tb"

    train_env = VecNormalize(
        DummyVecEnv([make_env(shape, seed, randomize=True)]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )
    eval_env = VecNormalize(
        DummyVecEnv([make_env(shape, seed + 10_000, randomize=True)]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )

    model = build_model(algo, train_env, seed, tensorboard_log)

    callbacks = [
        SyncNormEvalCallback(
            eval_env, best_model_save_path=str(outdir / "best"),
            log_path=str(outdir / "eval"), eval_freq=eval_freq,
            n_eval_episodes=n_eval, deterministic=True, render=False,
        ),
        CheckpointCallback(
            save_freq=max(eval_freq, 50_000), save_path=str(outdir / "checkpoints"),
            name_prefix=algo,
            save_vecnormalize=True,
        ),
    ]

    print(f"[train] {algo} on {shape}: {timesteps} steps -> {outdir}", flush=True)
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    model.save(str(outdir / "model"))
    train_env.save(str(outdir / "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()

    best = outdir / "best" / "best_model.zip"
    if best.exists() and not (outdir / "best" / "vecnormalize.pkl").exists():
        shutil.copy(outdir / "vecnormalize.pkl", outdir / "best" / "vecnormalize.pkl")
    print(f"[train] wrote {outdir / 'model.zip'}", flush=True)
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=("ppo", "sac", "both"), default="both")
    ap.add_argument("--shape", choices=("cylinder", "sphere", "both"), default="cylinder")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--n-eval-episodes", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    algos = ("ppo", "sac") if args.algo == "both" else (args.algo,)
    shapes = ("cylinder", "sphere") if args.shape == "both" else (args.shape,)
    for shape in shapes:
        for algo in algos:
            train(algo, shape, args.timesteps, args.seed, args.eval_freq,
                  args.n_eval_episodes)


if __name__ == "__main__":
    main()
