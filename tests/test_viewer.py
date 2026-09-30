"""Viewer helpers: keypoint overlay and depth colorization."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p24grasp.teleop.state_machine import State  # noqa: E402
from p24grasp.teleop.viewer import (  # noqa: E402
    BONES,
    P24MeshRenderer,
    _brighten_background,
    colorize_depth,
    compose_display,
    overlay_keypoints,
    skeleton_arrays,
)


def test_overlay_draws_keypoints_on_image():
    color = np.zeros((60, 80, 3), np.uint8)
    uv = np.full((21, 2), np.nan)
    uv[0] = (40.0, 45.0)  # below the 34 px banner strip
    uv[5] = (40.0, 20.0)
    vis = np.zeros(21)
    vis[0] = 1.0
    out = overlay_keypoints(color, uv, vis, presence=0.9, state=State.TRACKING)
    assert out.shape == color.shape
    # a green keypoint pixel was drawn near (40, 45)
    assert (out[43:48, 38:43] == [0, 220, 80]).all(axis=-1).any()


def test_overlay_hides_missing_keypoints():
    color = np.zeros((60, 80, 3), np.uint8)
    uv = np.full((21, 2), np.nan)
    out = overlay_keypoints(color, uv, np.zeros(21), presence=0.0,
                            state=State.HOLD)
    # banner strip at the top is drawn; everything below stays untouched
    np.testing.assert_array_equal(out[35:], color[35:])


def test_colorize_depth_maps_scale_and_marks_invalid():
    depth = np.full((30, 40), 0.3)
    depth[0, 0] = np.nan
    out = colorize_depth(depth)
    assert out.shape == (30, 40, 3)
    # constant depth -> uniform jet(0) = strong blue channel
    assert out[5, 5][2] > 100
    assert np.all(out[0, 0] <= 15)  # invalid region near-black


def test_skeleton_arrays_drop_invalid_and_axis_flip():
    kp = np.zeros((21, 3))
    kp[:, 2] = 0.4  # forward
    kp[5] = np.nan  # index MCP missing: its bones must be dropped
    vis = np.ones(21)
    points, _, segments, scolors, has_hand = skeleton_arrays(kp, vis)
    assert has_hand
    assert len(points) == 20  # 21 - 1 invalid
    # camera (x, y, z) -> viser (x, z, -y)
    np.testing.assert_allclose(points[0], [0.0, 0.4, 0.0])
    # bones touching keypoint 5 are skipped: (0,5) and (5,6) and (5,9)
    assert len(segments) == len(BONES) - 3
    assert scolors.shape == (len(segments), 2, 3)  # viser per-endpoint colors


def test_skeleton_arrays_empty_when_no_hand():
    points, _, segments, _, has_hand = skeleton_arrays(
        np.full((21, 3), np.nan), np.ones(21))
    assert not has_hand
    assert len(points) == 0
    assert len(segments) == 0


def test_compose_display_2x2_grid():
    color = np.zeros((48, 64, 3), np.uint8)
    depth = np.zeros((30, 40, 3), np.uint8)
    out = compose_display(color, depth, None, ["line one", "line two"])
    # 2x2 grid: top-left is the color panel, bottom-right is the status panel
    assert out.shape == (96, 128, 3)
    np.testing.assert_array_equal(out[:48, :64], color)
    assert (out[48:, 64:] == 20).any()  # status panel background drawn


def test_compose_display_grid_with_p24_panel():
    color = np.zeros((48, 64, 3), np.uint8)
    depth = np.zeros((48, 64, 3), np.uint8)
    p24 = np.zeros((220, 320, 3), np.uint8)
    out = compose_display(color, depth, p24, ["status"])
    assert out.shape == (96, 128, 3)
    np.testing.assert_array_equal(out[48:, :64], 0)  # P24 panel in bottom-left


def test_brighten_background_replaces_dark_pixels():
    image = np.zeros((30, 40, 3), np.uint8)
    image[10:20, 10:20] = (180, 180, 180)  # lit mesh region
    out = _brighten_background(image)
    assert out[0, 0].sum() > 600  # light gradient background
    assert out[0, 0][2] > out[0, 0][0]  # MuJoCo-style blue tint
    assert (out[15, 15] == [255, 255, 255]).all()  # mesh brightened, clipped


def test_p24_mesh_renderer_never_raises():
    # With mujoco present this renders the hand; without it (e.g. this test
    # env) construction must still succeed and render() must return None.
    renderer = P24MeshRenderer()
    image = renderer.render(np.zeros(20))
    if renderer._error is None:  # pylint: disable=protected-access
        assert image is not None and image.ndim == 3
    else:
        assert image is None
