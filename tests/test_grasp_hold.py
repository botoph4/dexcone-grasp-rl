"""The nominal grasp pose holds the cylinder (headless physics)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.viewers.web import WebGrasp


def test_cylinder_held_4s():
    g = WebGrasp("cylinder", 0.026, 0.85)
    g.start_run()
    for _ in range(240):
        g.step()
    drift = np.linalg.norm(g.data.xpos[g.obj_body] - g.obj_target) * 1000
    assert drift < 75 and g.contact_count() >= 3


def test_pull_3n_still_held():
    g = WebGrasp("cylinder", 0.026, 0.85)
    g.start_run()
    for _ in range(60):
        g.step()
    for _ in range(180):
        g.step(pull=np.array([0.0, 0.0, -3.0]))
    drift = np.linalg.norm(g.data.xpos[g.obj_body] - g.obj_target) * 1000
    assert drift < 75
