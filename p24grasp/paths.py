"""Central path resolution: assets, build artifacts, outputs.

Everything generated at runtime goes to ``build/`` (gitignored); everything the
user cares about goes to ``outputs/`` (gitignored).  Asset paths are absolute
so scenes written into ``build/`` keep working - MuJoCo resolves mesh paths
relative to the XML location, so relative paths would break the moment the XML
moves out of the repo root.
"""
from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent


def assets_dir() -> Path:
    return PKG / "assets"


def build_dir() -> Path:
    d = ROOT / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir(*sub: str) -> Path:
    d = ROOT / "outputs"
    for part in sub:
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def urdf_path() -> Path:
    return assets_dir() / "p24_hand_right.urdf"


def hand_xml_path() -> Path:
    return build_dir() / "hand.xml"


def ensure_hand_xml(force: bool = False) -> Path:
    """Return the generated hand.xml, building it on demand."""
    path = hand_xml_path()
    if force or not path.exists():
        from p24grasp.model import mjcf
        mjcf.build(path)
    return path
