"""Friction-cone modelling for point contacts on the dexterous hand."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def orthonormal_tangents(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def contact_frame(n: np.ndarray) -> np.ndarray:
    """Return R = [t1, t2, n]."""
    t1, t2 = orthonormal_tangents(n)
    return np.column_stack([t1, t2, n])


@dataclass
class FrictionConeConfig:
    mu: float = 0.7
    safety_factor: float = 1.4  # mu_eff = mu / safety_factor
    f_min: float = 0.1          # [N]
    f_max: float = 8.0          # [N]
    f_margin: float = 0.3       # [N]
    rho_max: float = 0.65
    normal_local: Tuple[float, float, float] = (0.0, 0.0, 1.0)


class FrictionCone:
    def __init__(self, cfg: FrictionConeConfig):
        self.cfg = cfg
        self.mu_eff = cfg.mu / cfg.safety_factor

    def decompose(self, n: np.ndarray, f_apply: np.ndarray) -> Tuple[float, np.ndarray]:
        n = n / np.linalg.norm(n)
        fn = float(n @ f_apply)
        ft = f_apply - fn * n
        return fn, ft

    def slip_margin(self, n: np.ndarray, f_apply: np.ndarray) -> Tuple[float, float, np.ndarray]:
        fn, ft = self.decompose(n, f_apply)
        rho = float(np.linalg.norm(ft) / max(self.mu_eff * fn, 1e-9))
        return rho, fn, ft

    def desired_normal_force(self, n: np.ndarray, f_apply: np.ndarray) -> float:
        _, fn, ft = self.slip_margin(n, f_apply)
        fn_des = float(np.linalg.norm(ft) / self.mu_eff + self.cfg.f_margin)
        fn_des = max(fn_des, self.cfg.f_min)
        fn_des = min(fn_des, self.cfg.f_max)
        return fn_des

    def polyhedral_matrix(self, m: int = 4) -> np.ndarray:
        """Inner polyhedral approximation matrix F so that F @ [fx, fy, fz] <= 0.

        Columns are ordered [f_t1, f_t2, f_n] in the contact frame.
        """
        mu = self.mu_eff
        if m == 4:
            c = mu / np.sqrt(2.0)
            return np.array(
                [
                    [1.0, 0.0, -c],
                    [-1.0, 0.0, -c],
                    [0.0, 1.0, -c],
                    [0.0, -1.0, -c],
                    [0.0, 0.0, -1.0],
                ]
            )
        # generic regular polygon inscribed in the friction disk
        rows = []
        for k in range(m):
            phi = 2.0 * np.pi * k / m + np.pi / m
            u = np.array([np.cos(phi), np.sin(phi)])
            rows.append([u[0], u[1], -mu * np.cos(np.pi / m)])
        rows.append([0.0, 0.0, -1.0])
        return np.array(rows)
