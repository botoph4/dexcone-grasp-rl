"""GeoRT-style neural retargeter: an MLP trained in simulation.

Training is unsupervised in the geometric sense: no paired human->robot
demonstrations.  Poses are sampled in HUMAN angle space; each pose's
fingertip targets are produced by the "custom hand" trick (Qin et al.):
the P24 kinematics itself plays the virtual human hand, scaled down to
human proportions, so the learned mapping must compensate for the size
difference exactly as the live system does.  The HybridRetargeter serves
as the solver/teacher; the MLP distills it into a 1 kHz-class mapping.

Runtime inference is pure numpy (no torch on the deployment board).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from p24grasp.teleop.angles import HUMAN_ABDUCTION_ROM, HUMAN_FLEXION_ROM
from p24grasp.teleop.retarget import (
    HybridRetargeter,
    scaling_q16,
)

DEFAULT_WEIGHTS = Path.home() / ".cache" / "p24grasp" / "geort_retarget.npz"

N_INPUT = 20  # 15 flexion + 5 abduction (human angle space)
N_OUTPUT = 16  # active DOFs, radians
HIDDEN = (256, 256)
HUMAN_SCALE = 0.72  # virtual human hand size relative to the P24 reach


def sample_human_angles(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 20) human-angle vectors (deg): 15 flexion (thumb CMC/MP/IP,
    fingers MCP/PIP/DIP) + 5 abduction.  Fingers flex together (grasp-like),
    DIP ~= 0.7*PIP (anatomical coupling)."""
    flexion = np.zeros((n, 15))
    # per-pose grasp amount shared across fingers, with per-finger noise
    grasp = rng.beta(2.0, 2.0, size=(n, 5))
    for row in range(5):
        base = row * 3 if row == 0 else 3 + (row - 1) * 3
        rom = HUMAN_FLEXION_ROM[row]
        for k in range(3):
            flexion[:, base + k] = grasp[:, row] * rom[k] * (0.7 + 0.3 * rng.random(n))
    # thumb opposition correlates with index flexion
    flexion[:, 0] = np.clip(flexion[:, 0] + 0.3 * flexion[:, 3], 0, 60.0)
    # anatomical DIP ~= 0.7 * PIP for the four fingers
    for row in range(1, 5):
        base = 3 + (row - 1) * 3
        flexion[:, base + 2] = 0.7 * flexion[:, base + 1]
    abduction = rng.normal(0.0, 8.0, size=(n, 5))
    abduction = np.clip(abduction, -HUMAN_ABDUCTION_ROM, HUMAN_ABDUCTION_ROM)
    abduction[:, 2] = 0.0  # middle finger is the reference
    return np.hstack([flexion, abduction])


def make_dataset(n: int, seed: int = 0, teacher: HybridRetargeter | None = None):
    """(X, Y): human angles (n, 20) deg -> teacher q16 (n, 16) rad.

    Virtual-human targets: the P24 FK at the scaled human pose, then the
    whole tip cloud shrunk by HUMAN_SCALE, mimicking a smaller human hand.
    """
    from p24grasp.teleop.angles import HandAngles  # noqa: E402

    rng = np.random.default_rng(seed)
    teacher = teacher or HybridRetargeter(scale=1.0, max_nfev=120)
    angles_all = sample_human_angles(rng, n)
    x_all, y_all = [], []
    for i in range(n):
        flexion = angles_all[i, :15].reshape(5, 3)
        abduction = angles_all[i, 15:20]
        angles = HandAngles(flexion=flexion, abduction=abduction,
                            flexion_vis=np.ones((5, 3)),
                            abduction_vis=np.ones(5), presence=1.0)
        q_virtual = scaling_q16(teacher.hand, angles)
        targets = teacher._tips(q_virtual) * HUMAN_SCALE  # pylint: disable=protected-access
        q_sol = teacher._q20_to_active(  # pylint: disable=protected-access
            teacher.solve(targets, q_virtual))
        x_all.append(angles_all[i])
        y_all.append(q_sol)
    return np.stack(x_all), np.stack(y_all)


def _build_mlp():
    import torch  # noqa: E402

    layers = []
    sizes = [N_INPUT, *HIDDEN, N_OUTPUT]
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def train_geort(weights_path: str | Path | None = None, n_samples: int = 4000,
                epochs: int = 120, batch_size: int = 256, seed: int = 0):
    """Train the MLP and save the weights; returns (holdout tip error mm)."""
    import torch  # noqa: E402

    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    x_all, y_all = make_dataset(n_samples, seed=seed)
    split = int(0.9 * n_samples)
    x_train = torch.tensor(x_all[:split], dtype=torch.float32)
    y_train = torch.tensor(y_all[:split], dtype=torch.float32)
    x_val = torch.tensor(x_all[split:], dtype=torch.float32)
    y_val = torch.tensor(y_all[split:], dtype=torch.float32)

    torch.manual_seed(seed)
    model = _build_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train))
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start:start + batch_size]
            loss = torch.nn.functional.mse_loss(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(model(x_val), y_val).item()
            print(f"geort epoch {epoch:3d}  val mse(rad^2)={val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        prediction = model(x_val).numpy()
        tip_error = _tip_error_mm(prediction, y_val.numpy())
    print(f"geort holdout tip error: {tip_error:.1f} mm")

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            arrays[f"w{i}"] = layer.weight.detach().numpy()
            arrays[f"b{i}"] = layer.bias.detach().numpy()
    np.savez_compressed(path, **arrays)
    print(f"geort weights saved to {path}")
    return float(tip_error)


def _tip_error_mm(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """Mean fingertip position error (mm) between FK(predicted q) and
    FK(reference q): the honest geometric metric for retargeting quality."""
    teacher = HybridRetargeter(scale=1.0, max_nfev=1)
    errors = []
    for q_pred, q_ref in zip(y_pred, y_ref):
        tips_pred = teacher._tips(q_pred)  # pylint: disable=protected-access
        tips_ref = teacher._tips(q_ref)  # pylint: disable=protected-access
        errors.append(np.mean(np.linalg.norm(tips_pred - tips_ref, axis=1)))
    return float(np.mean(errors)) * 1000.0


class GeoRtRetargeter:
    """1 kHz-class MLP retargeter: human angles -> 16 DOF (deg, URDF order).

    Loads the weights saved by train_geort; pure numpy inference (no torch
    on the deployment board).  Falls back to the hybrid retargeter until a
    model has been trained.
    """

    def __init__(self, weights_path: str | Path | None = None):
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self._layers = None
        self._fallback = HybridRetargeter()
        self._error = None
        try:
            data = np.load(path)
            # pair w{i}/b{i} by their actual keys: the indices skip the
            # ReLU layers in the Sequential (w0, w2, w4, ...), and the
            # alphabetical file order puts all b* first.
            w_keys = sorted(key for key in data.files if key.startswith("w"))
            weights = []
            for w_key in w_keys:
                weights.append(data[w_key])
                weights.append(data["b" + w_key[1:]])
            if not weights:
                raise ValueError(f"no weights in {path}")
            self._layers = weights
            self._q20_prev = np.zeros(20)
        except Exception as exc:  # model not trained yet: hybrid fallback
            self._error = exc

    def _forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for i in range(0, len(self._layers), 2):
            weight = self._layers[i]
            bias = self._layers[i + 1]
            out = out @ weight.T + bias
            if i < len(self._layers) - 2:
                out = np.maximum(out, 0.0)
        return out

    def retarget(self, angles, keypoints3d: np.ndarray | None = None) -> np.ndarray:
        if self._layers is None:
            return self._fallback.retarget(angles, keypoints3d)
        flexion = np.nan_to_num(angles.flexion, nan=0.0).ravel()
        abduction = np.nan_to_num(angles.abduction, nan=0.0)
        x = np.concatenate([flexion, abduction]).astype(np.float32)
        q16 = self._forward(x)
        teacher = self._fallback  # reuse its URDF conversion + limits
        q16 = np.clip(q16, teacher._lb, teacher._ub)  # pylint: disable=protected-access
        self._q20_prev = teacher._to_urdf_degrees(q16)  # pylint: disable=protected-access
        return self._q20_prev.copy()
