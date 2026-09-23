#!/usr/bin/env python3
"""Evaluate trained grasp policies and compare them.

Loads ``runs/<shape>/<algo>/model.zip`` plus its ``vecnormalize.pkl`` (the
policy was trained on normalized observations, so both are required), runs the
same seeded domain-randomisation distribution for every algorithm, and reports
success rate, how long the object stayed held, contact count and actuator
effort.  Optionally writes offscreen PNG snapshots and an MP4 per model.

Usage
-----
    python eval_grasp_rl.py --shape cylinder --algo both --episodes 50
    python eval_grasp_rl.py --shape sphere --algo ppo --episodes 20 --media 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


import mujoco  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from p24grasp.env.obs import ObsNormalizer  # noqa: E402
from p24grasp.paths import outputs_dir  # noqa: E402
from p24grasp.env.grasp import (  # noqa: E402
    EPISODE_STEPS,
    MIN_SUCCESS_CONTACTS,
    SUCCESS_DRIFT,
    GraspEnv,
)

RUNS = outputs_dir("runs")
RESULTS = outputs_dir("eval")
ALGOS = {"ppo": PPO, "sac": SAC}


def load_policy(shape: str, algo: str, which: str = "best"):
    """(model, obs normalizer, GraspEnv) for a saved policy."""
    run_dir = RUNS / shape / algo
    if which == "best" and (run_dir / "best" / "best_model.zip").exists():
        model_path = run_dir / "best" / "best_model.zip"
        norm_path = run_dir / "best" / "vecnormalize.pkl"
    else:
        model_path = run_dir / "model.zip"
        norm_path = run_dir / "vecnormalize.pkl"
    if not model_path.exists() or not norm_path.exists():
        raise FileNotFoundError(
            f"no policy at {model_path} (need model.zip and vecnormalize.pkl); "
            f"train it first with train_grasp_rl.py"
        )
    model = ALGOS[algo].load(str(model_path), device="cpu")
    vec = VecNormalize.load(str(norm_path), DummyVecEnv([lambda: GraspEnv(shape=shape)]))
    normalizer = ObsNormalizer(vec.obs_rms, vec.clip_obs, vec.epsilon)
    vec.close()
    return model, normalizer, GraspEnv(shape=shape)


class FrameRecorder:
    """Offscreen renderer that grabs frames from the live simulation."""

    def __init__(self, env: GraspEnv, width: int = 640, height: int = 480):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self.cam)
        self.cam.lookat[:] = env.center
        self.cam.distance = 0.40
        self.cam.azimuth = 180.0
        self.cam.elevation = -20.0

    def grab(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.cam)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def rollout(model, normalize, env: GraspEnv, seed: int, record: bool = False):
    """Run one episode; return per-episode metrics and (optionally) frames."""
    obs, info = env.reset(seed=seed)
    obs = normalize(obs)

    recorder = FrameRecorder(env) if record else None
    frames = [recorder.grab()] if recorder else []

    held_steps = 0
    contact_sum = 0
    effort_sum = 0.0
    steps = 0
    for t in range(EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        obs = normalize(obs)
        steps = t + 1
        in_hand = (info["drift"] < SUCCESS_DRIFT
                   and info["contacts"] >= MIN_SUCCESS_CONTACTS)
        held_steps += int(in_hand)
        contact_sum += info["contacts"]
        effort_sum += info["effort"]
        if recorder and t % 4 == 0:
            frames.append(recorder.grab())
        if terminated or truncated:
            break

    if recorder:
        recorder.close()
    return {
        "seed": seed,
        "steps": steps,
        "held_seconds": held_steps * 0.01,
        "held_fraction": held_steps / max(steps, 1),
        "mean_contacts": contact_sum / max(steps, 1),
        "effort": effort_sum * 0.01,
        "final_drift": float(info.get("drift", np.nan)),
        "final_contacts": int(info.get("contacts", 0)),
        "success": bool(info.get("success", False)),
    }, frames


def summarise(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "episodes": len(rows),
        "success_rate": mean("success"),
        "held_seconds_mean": mean("held_seconds"),
        "held_fraction_mean": mean("held_fraction"),
        "mean_contacts": mean("mean_contacts"),
        "effort_mean": mean("effort"),
        "final_drift_mm_mean": mean("final_drift") * 1000.0,
    }


def save_media(frames: list[np.ndarray], outdir: Path, tag: str):
    if not frames:
        return
    import imageio.v2 as imageio
    outdir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
        imageio.imwrite(outdir / f"{tag}_t{i}.png", frame)
    imageio.mimsave(outdir / f"{tag}.mp4", frames, fps=25)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=("cylinder", "sphere"), default="cylinder")
    ap.add_argument("--algo", choices=("ppo", "sac", "both"), default="both")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--which", choices=("best", "final"), default="best")
    ap.add_argument("--media", type=int, default=0,
                    help="number of episodes to render PNGs/MP4 for")
    ap.add_argument("--outdir", default=str(RESULTS))
    args = ap.parse_args()

    algos = ("ppo", "sac") if args.algo == "both" else (args.algo,)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {}

    for algo in algos:
        try:
            model, normalizer, env = load_policy(args.shape, algo, args.which)
        except FileNotFoundError as exc:
            print(f"[eval] skipping {algo}: {exc}", flush=True)
            continue
        print(f"[eval] {args.shape}/{algo} over {args.episodes} episodes", flush=True)
        rows = []
        for i in range(args.episodes):
            record = i < args.media
            row, frames = rollout(model, normalizer, env, seed=i, record=record)
            rows.append(row)
            if record:
                tag = f"{args.shape}_{algo}_seed{i}"
                save_media(frames, outdir / "media", tag)
            if (i + 1) % 10 == 0:
                print(f"   {i + 1}/{args.episodes} done, "
                      f"running success {np.mean([r['success'] for r in rows]):.2f}",
                      flush=True)
        summary = summarise(rows)
        report[f"{args.shape}/{algo}"] = {"summary": summary, "episodes": rows}
        print(f"[eval] {args.shape}/{algo}: success {summary['success_rate']:.2f}  "
              f"held {summary['held_seconds_mean']:.2f}s  "
              f"contacts {summary['mean_contacts']:.1f}  "
              f"effort {summary['effort_mean']:.2f}  "
              f"final drift {summary['final_drift_mm_mean']:.1f}mm", flush=True)
        env.close()

    (outdir / f"eval_{args.shape}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if len(report) > 1:
        keys = sorted(report)
        print("\n| model | success | held (s) | contacts | effort | final drift (mm) |")
        print("| --- | --- | --- | --- | --- | --- |")
        for k in keys:
            s = report[k]["summary"]
            print(f"| {k} | {s['success_rate']:.2f} | {s['held_seconds_mean']:.2f} | "
                  f"{s['mean_contacts']:.1f} | {s['effort_mean']:.2f} | "
                  f"{s['final_drift_mm_mean']:.1f} |")
    print(f"\n[eval] wrote {outdir / f'eval_{args.shape}.json'}")


if __name__ == "__main__":
    main()
