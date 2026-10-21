"""GeoRT neural retargeter: dataset, training, inference."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.angles import HUMAN_FLEXION_ROM  # noqa: E402
from p24grasp.teleop.geort import (  # noqa: E402
    GeoRtRetargeter,
    make_dataset,
    sample_human_angles,
)
from p24grasp.teleop.retarget import P24_LIMITS  # noqa: E402


def test_sample_human_angles_anatomy():
    rng = np.random.default_rng(0)
    samples = sample_human_angles(rng, 200)
    assert samples.shape == (200, 20)
    flexion = samples[:, :15].reshape(-1, 5, 3)
    assert np.all(flexion >= 0)
    assert np.all(flexion <= HUMAN_FLEXION_ROM[None, :, :] + 1e-6)
    # DIP ~= 0.7 * PIP for the fingers
    np.testing.assert_allclose(flexion[:, 1:, 2], 0.7 * flexion[:, 1:, 1],
                               atol=1e-9)
    # middle abduction is the zero reference
    np.testing.assert_allclose(samples[:, 17], 0.0, atol=1e-9)


def test_make_dataset_shapes_and_validity():
    x, y = make_dataset(30, seed=0)
    assert x.shape == (30, 20)
    assert y.shape == (30, 16)
    assert np.all(np.isfinite(y))


def test_geort_retargeter_without_model_falls_back():
    retargeter = GeoRtRetargeter(weights_path="/nonexistent/geort.npz")
    assert retargeter._error is not None  # pylint: disable=protected-access
    from p24grasp.teleop.angles import HandAngles  # noqa: E402

    angles = HandAngles(flexion=np.full((5, 3), 30.0), abduction=np.zeros(5),
                        flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                        presence=1.0)
    q20 = retargeter.retarget(angles, keypoints3d=np.full((21, 3), np.nan))
    assert q20.shape == (20,)
    assert np.all(np.isfinite(q20))


def test_weight_pairs_load_by_name_not_alphabetical(tmp_path):
    # regression: alphabetical file order puts all b* before w*, which the
    # old loader paired wrongly; weights must be paired by name.
    rng = np.random.default_rng(0)
    arrays = {}
    for i, (in_dim, out_dim) in enumerate(((20, 256), (256, 256), (256, 16))):
        arrays[f"w{i * 2}"] = rng.normal(0, 0.05, (out_dim, in_dim))
        arrays[f"b{i * 2}"] = rng.normal(0, 0.05, (out_dim,))
    path = tmp_path / "fake.npz"
    np.savez(path, **arrays)

    retargeter = GeoRtRetargeter(weights_path=path)
    assert retargeter._error is None  # pylint: disable=protected-access
    from p24grasp.teleop.angles import HandAngles  # noqa: E402

    angles = HandAngles(flexion=np.full((5, 3), 30.0), abduction=np.zeros(5),
                        flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                        presence=1.0)
    q20 = retargeter.retarget(angles, keypoints3d=np.full((21, 3), np.nan))
    assert q20.shape == (20,)
    assert np.all(np.isfinite(q20))


def test_geort_train_and_infer(tmp_path):
    torch = pytest.importorskip("torch")
    assert torch  # torch installed: run the real training path
    from p24grasp.teleop.geort import train_geort  # noqa: E402

    weights = tmp_path / "geort.npz"
    error_mm = train_geort(weights_path=weights, n_samples=300, epochs=30,
                           batch_size=128, seed=0)
    assert weights.exists()
    assert error_mm < 50.0  # rough sanity: distillation converges

    retargeter = GeoRtRetargeter(weights_path=weights)
    from p24grasp.teleop.angles import HandAngles  # noqa: E402

    angles = HandAngles(flexion=np.full((5, 3), 45.0), abduction=np.zeros(5),
                        flexion_vis=np.ones((5, 3)), abduction_vis=np.ones(5),
                        presence=1.0)
    q20 = retargeter.retarget(angles, keypoints3d=np.full((21, 3), np.nan))
    assert q20.shape == (20,)
    assert np.all(q20 >= P24_LIMITS[:, 0] - 1e-9)
    assert np.all(q20 <= P24_LIMITS[:, 1] + 1e-9)
