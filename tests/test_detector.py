"""Detector helpers: visibility sanitization for model variants."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.detector import sanitize_visibility  # noqa: E402


def test_nan_visibility_becomes_visible():
    out = sanitize_visibility(np.array([np.nan, 0.0, 0.7, np.nan]))
    np.testing.assert_allclose(out, [1.0, 0.0, 0.7, 1.0])


def test_finite_visibility_passthrough():
    values = np.array([0.2, 0.9, 0.5])
    np.testing.assert_allclose(sanitize_visibility(values), values)
