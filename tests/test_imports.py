"""Every package module imports cleanly."""
import importlib

MODULES = [
    "p24grasp",
    "p24grasp.paths",
    "p24grasp.model.urdf",
    "p24grasp.model.mjcf",
    "p24grasp.model.friction_cone",
    "p24grasp.kinematics.ik",
    "p24grasp.sim.scene",
    "p24grasp.sim.demo",
    "p24grasp.env.obs",
    "p24grasp.env.grasp",
    "p24grasp.viewers.desktop",
    "p24grasp.viewers.web",
]


def test_import_all():
    import pytest
    for name in MODULES:
        if name in ("p24grasp.env.grasp", "p24grasp.viewers.desktop",
                    "p24grasp.viewers.web"):
            # these import gymnasium / viser at module level
            if name.startswith("p24grasp.env"):
                pytest.importorskip("gymnasium")
        importlib.import_module(name)


def test_rl_and_pyroki_import_when_deps_present():
    import importlib.util
    if importlib.util.find_spec("stable_baselines3") is not None:
        importlib.import_module("p24grasp.rl.train")
        importlib.import_module("p24grasp.rl.eval")
    if importlib.util.find_spec("pyroki") is not None:
        importlib.import_module("p24grasp.pyroki.probe")
        importlib.import_module("p24grasp.pyroki.ik")
