import numpy as np
import pytest

from avdata import geometry
from avdata.fixture import CAM_INTRINSIC, SENSORS


def test_identity_quaternion():
    assert np.allclose(geometry.quat_to_rot([1, 0, 0, 0]), np.eye(3))


def test_rotation_is_orthonormal():
    r = geometry.quat_to_rot([0.3, -0.2, 0.5, 0.7])
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_point_ahead_projects_to_principal_point():
    """A point straight ahead of the camera lands on the principal point."""
    _, cam_t, cam_q, _ = SENSORS["CAM_FRONT"]
    ahead_global = np.array([[20.0 + cam_t[0]], [cam_t[1]], [cam_t[2]]])  # ego at origin
    cam = geometry.global_to_sensor(ahead_global, [0, 0, 0], [1, 0, 0, 0], cam_t, cam_q)
    assert cam[2, 0] == pytest.approx(20.0)  # depth along camera z
    uv = geometry.project_to_image(cam, np.array(CAM_INTRINSIC))
    assert uv[:, 0] == pytest.approx([CAM_INTRINSIC[0][2], CAM_INTRINSIC[1][2]])


def test_box_behind_camera_is_dropped():
    _, cam_t, cam_q, _ = SENSORS["CAM_FRONT"]
    corners = geometry.box_corners(
        np.array([-10.0, 0, 1]), np.array([2, 4, 1.5]), np.array([1, 0, 0, 0])
    )
    cam = geometry.global_to_sensor(corners, [0, 0, 0], [1, 0, 0, 0], cam_t, cam_q)
    assert geometry.box_to_2d(cam, np.array(CAM_INTRINSIC), 1600, 900) is None


def test_box_is_clipped_to_image():
    _, cam_t, cam_q, _ = SENSORS["CAM_FRONT"]
    corners = geometry.box_corners(
        np.array([6.0, 3.0, 1]), np.array([2, 4, 1.5]), np.array([1, 0, 0, 0])
    )
    cam = geometry.global_to_sensor(corners, [0, 0, 0], [1, 0, 0, 0], cam_t, cam_q)
    x1, y1, x2, y2 = geometry.box_to_2d(cam, np.array(CAM_INTRINSIC), 1600, 900)
    assert 0 <= x1 < x2 <= 1600 and 0 <= y1 < y2 <= 900
