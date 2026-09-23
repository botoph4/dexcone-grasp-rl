#!/usr/bin/env python3
"""Thin launcher for the URDF -> MJCF generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.model.mjcf import main

if __name__ == "__main__":
    main()
