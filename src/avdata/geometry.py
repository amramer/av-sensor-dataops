"""3D geometry for nuScenes: quaternions, frame transforms, box projection.

nuScenes conventions
- quaternions are [w, x, y, z]
- boxes (sample_annotation) live in the GLOBAL frame; size is [width, length, height]
- chain: global -> ego (ego_pose) -> sensor (calibrated_sensor)
- camera frame: x right, y down, z forward
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def quat_to_rot(q: list[float] | Array) -> Array:
    """Rotation matrix from a unit quaternion [w, x, y, z]."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def box_corners(center: Array, size: Array, rotation: Array) -> Array:
    """8 box corners (3x8) in the box's frame of reference.

    size = [width, length, height]; length runs along the box's x axis.
    """
    w, length, h = size
    x = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack([x, y, z])
    return quat_to_rot(rotation) @ corners + np.asarray(center, dtype=np.float64).reshape(3, 1)


def global_to_sensor(
    points: Array,
    ego_translation: Array,
    ego_rotation: Array,
    sensor_translation: Array,
    sensor_rotation: Array,
) -> Array:
    """Transform 3xN points from the global frame into a sensor frame."""
    p = points - np.asarray(ego_translation, dtype=np.float64).reshape(3, 1)
    p = quat_to_rot(ego_rotation).T @ p
    p = p - np.asarray(sensor_translation, dtype=np.float64).reshape(3, 1)
    return quat_to_rot(sensor_rotation).T @ p


def project_to_image(points_cam: Array, intrinsic: Array) -> Array:
    """Pinhole projection of 3xN camera-frame points to 2xN pixel coordinates."""
    uvw = np.asarray(intrinsic, dtype=np.float64) @ points_cam
    return uvw[:2] / uvw[2:3]


def box_to_2d(
    corners_cam: Array, intrinsic: Array, width: int, height: int, min_depth: float = 0.1
) -> tuple[float, float, float, float] | None:
    """Tight 2D box (x1, y1, x2, y2) around a 3D box, clipped to the image.

    Returns None when the box is behind the camera or entirely outside the image.
    Uses only corners in front of the camera (same approach as the nuScenes devkit
    `post_process_coords`), which is exact enough for 2D detection labels.
    """
    in_front = corners_cam[2] > min_depth
    if not in_front.any():
        return None
    uv = project_to_image(corners_cam[:, in_front], intrinsic)
    x1, y1 = uv.min(axis=1)
    x2, y2 = uv.max(axis=1)
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    return (
        float(np.clip(x1, 0, width)),
        float(np.clip(y1, 0, height)),
        float(np.clip(x2, 0, width)),
        float(np.clip(y2, 0, height)),
    )
