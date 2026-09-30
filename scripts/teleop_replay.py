"""Thin launcher: python scripts/teleop_replay.py --dir ..."""
from p24grasp.teleop.run import replay_main

if __name__ == "__main__":
    raise SystemExit(replay_main())
