"""Thin launcher: python scripts/teleop_record.py --out DIR --frames N"""
from p24grasp.teleop.run import record_main

if __name__ == "__main__":
    raise SystemExit(record_main())
