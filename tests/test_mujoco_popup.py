"""The MuJoCo popup's model + skeleton updates (headless -- no viewer)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mujoco_popup import apply_frame, build_model  # noqa: E402


def test_build_model_has_hand_and_skeleton_geoms():
    _model, data, qpos_ids, joint_ids, bone_ids = build_model()
    assert len(qpos_ids) == 20
    assert len(joint_ids) == 21
    assert len(bone_ids) == 21  # one ellipsoid per BONES entry
    assert data.qpos.shape[0] >= 20


def test_apply_frame_updates_robot_and_skeleton():
    model, data, qpos_ids, joint_ids, bone_ids = build_model()
    keypoints = np.full((21, 3), np.nan)
    keypoints[5] = (0.01, -0.01, 0.30)  # index MCP visible
    keypoints[6] = (0.01, -0.01, 0.34)
    apply_frame(model, data, qpos_ids=qpos_ids, joint_ids=joint_ids,
                bone_ids=bone_ids, command=np.full(20, 30.0),
                keypoints3d=keypoints)
    # robot command applied
    np.testing.assert_allclose(data.qpos[qpos_ids[4]], np.radians(30.0))
    # the visible joint sphere moved to the scene position
    expected = np.array([0.30 + 0.9 * 0.30, -0.9 * 0.01, 0.42 + 0.9 * 0.01])
    np.testing.assert_allclose(data.geom_xpos[joint_ids[5]], expected, atol=1e-9)
    # the missing joints are hidden (alpha 0)
    assert model.geom_rgba[joint_ids[4], 3] == 0.0
    # the index MCP->PIP bone (BONES[5] = (5, 6)) is visible, stretched
    assert model.geom_rgba[bone_ids[5], 3] > 0.0
    np.testing.assert_allclose(model.geom_size[bone_ids[5], 2],
                               0.5 * 0.9 * 0.04, atol=1e-6)
